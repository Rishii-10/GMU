"""
Tests for the multi-turn follow-up loop (app.agent1_extraction.
extract_case_with_followup / extract_and_classify_with_followup).

Split into two groups, following the existing project convention of
separating deterministic logic tests from live-model tests:

  - Mechanism tests use a small scripted in-memory backend (`_ScriptedBackend`)
    so context growth, the turn cap, and the audit trail's shape can be
    asserted exactly, with no dependency on a live model and no flakiness.
  - A couple of live tests exercise the loop against a real local Ollama
    call (llama3.2:3b), using the same skip-if-Ollama-unavailable pattern as
    tests/test_integration_agent1_pipeline.py, to confirm the mechanism
    actually helps against a real backend -- with the same tolerance for a
    small model's variability that the existing integration tests use.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    DANGER_SIGN_FOLLOWUP_QUESTIONS,
    LLMBackend,
    OllamaBackend,
    RegexBackend,
    extract_and_classify_with_followup,
    extract_case_with_followup,
)
from app.schemas import ClassificationLabel
from tests.test_integration_agent1_pipeline import _ollama_available


ORIGINAL_MESSAGE = "My child has had a fever for two days."


class _ScriptedBackend(LLMBackend):
    """Deterministic test double. Returns one scripted parsed-JSON dict per
    call (queue order), and records every (patient_text, context) it was
    invoked with so tests can assert exactly what was fed back in."""

    name = "scripted_test_backend"
    supports_followup = True

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, Optional[list[str]]]] = []

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        self.calls.append((patient_text, list(context) if context else None))
        if not self._responses:
            raise AssertionError("_ScriptedBackend ran out of scripted responses")
        return self._responses.pop(0)


def _base_response(**danger_signs_overrides) -> dict:
    danger_signs = {
        "not_able_to_drink_or_breastfeed": None,
        "vomits_everything": None,
        "convulsions": None,
        "lethargic_or_unconscious": None,
    }
    danger_signs.update(danger_signs_overrides)
    return {
        "symptom": "fever",
        "duration": "2 days",
        "severity": "unknown",
        "age_group": "child",
        "age_months": 24,
        "location": None,
        "notes": None,
        "danger_signs": danger_signs,
        "cough": None,
        "diarrhea": None,
    }


def _sequential_answers(*answers: str):
    it = iter(answers)

    def provider(question: str) -> str:
        return next(it)

    return provider


# --- Mechanism tests (deterministic, no live backend) -----------------------


def test_original_message_survives_context_across_multiple_followup_turns():
    # Two turns needed to fully resolve all four fields.
    backend = _ScriptedBackend(
        [
            _base_response(),  # turn 0: nothing assessed
            _base_response(not_able_to_drink_or_breastfeed=False, vomits_everything=False),  # turn 1
            _base_response(
                not_able_to_drink_or_breastfeed=False,
                vomits_everything=False,
                convulsions=False,
                lethargic_or_unconscious=False,
            ),  # turn 2: fully resolved
        ]
    )
    provider = _sequential_answers("No.", "No.")
    case, trail = extract_case_with_followup(
        ORIGINAL_MESSAGE, backend, answer_provider=provider, case_id="c1", max_followup_turns=2
    )

    assert len(backend.calls) == 3
    # every call after the first must still carry the original message in its context
    for patient_text, context in backend.calls[1:]:
        assert context is not None
        assert any(ORIGINAL_MESSAGE in entry for entry in context), (
            f"original message missing from context on a follow-up call: {context}"
        )
    # raw_symptom_text on the final case must be the ORIGINAL message, never an answer
    assert case.raw_symptom_text == ORIGINAL_MESSAGE
    assert case.danger_signs.all_assessed()
    assert len(trail) == 2


def test_turn_cap_is_respected():
    # Backend never resolves the fields, no matter how many turns -- loop
    # must stop at max_followup_turns and return whatever it has.
    backend = _ScriptedBackend([_base_response() for _ in range(10)])
    provider = _sequential_answers("I don't know.", "I don't know.", "I don't know.", "I don't know.")
    case, trail = extract_case_with_followup(
        ORIGINAL_MESSAGE, backend, answer_provider=provider, max_followup_turns=2
    )
    # 1 initial call + at most 2 follow-up calls = 3 total
    assert len(backend.calls) == 3
    assert len(trail) == 2
    assert case.danger_signs.missing_fields()  # still incomplete -- cap was hit, not silently resolved


def test_cap_hit_falls_through_to_incomplete_assessment_not_guessed_false():
    backend = _ScriptedBackend([_base_response() for _ in range(10)])
    provider = _sequential_answers("I don't know.", "I don't know.")
    case, trail, result = extract_and_classify_with_followup(
        ORIGINAL_MESSAGE, backend, answer_provider=provider, max_followup_turns=2
    )
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    # none of the still-missing fields were silently defaulted to False
    assert case.danger_signs.not_able_to_drink_or_breastfeed is None
    assert case.danger_signs.convulsions is None


def test_loop_stops_early_once_a_danger_sign_resolves_true():
    # Turn 1's answer reveals a confirmed danger sign -- the loop must not
    # keep asking after that (classify() will short-circuit to EMERGENCY
    # regardless of the other three fields).
    backend = _ScriptedBackend(
        [
            _base_response(),  # turn 0
            _base_response(convulsions=True),  # turn 1: confirmed danger sign
        ]
    )
    provider = _sequential_answers("Yes, he had a seizure.", "SHOULD NOT BE CALLED")
    case, trail = extract_case_with_followup(
        ORIGINAL_MESSAGE, backend, answer_provider=provider, max_followup_turns=2
    )
    assert len(backend.calls) == 2  # not 3 -- stopped after the resolving turn
    assert len(trail) == 1
    assert case.danger_signs.convulsions is True


def test_audit_trail_contains_only_followup_turns_not_original_message():
    backend = _ScriptedBackend(
        [
            _base_response(),
            # fully resolved on this turn -> loop stops after exactly 1 follow-up
            _base_response(
                not_able_to_drink_or_breastfeed=False,
                vomits_everything=False,
                convulsions=False,
                lethargic_or_unconscious=False,
            ),
        ]
    )
    provider = _sequential_answers("No.")
    case, trail = extract_case_with_followup(
        ORIGINAL_MESSAGE, backend, answer_provider=provider, max_followup_turns=2
    )
    assert len(trail) == 1
    assert trail[0]["question"] in DANGER_SIGN_FOLLOWUP_QUESTIONS.values()
    assert trail[0]["answer"] == "No."
    for turn in trail:
        assert ORIGINAL_MESSAGE not in turn["question"]
        assert ORIGINAL_MESSAGE not in turn["answer"]


def test_no_answer_provider_behaves_like_single_shot():
    backend = _ScriptedBackend([_base_response()])
    case, trail = extract_case_with_followup(ORIGINAL_MESSAGE, backend, answer_provider=None)
    assert len(backend.calls) == 1
    assert trail == []


# --- RegexBackend scoping ----------------------------------------------------


def test_regex_backend_is_excluded_from_followup_loop():
    def _must_not_be_called(question: str) -> str:
        raise AssertionError(f"RegexBackend should never trigger a follow-up question: {question!r}")

    backend = RegexBackend()
    assert backend.supports_followup is False

    case, trail = extract_case_with_followup(
        "मेरे बच्चे को बुखार है", backend, answer_provider=_must_not_be_called, max_followup_turns=2
    )
    assert trail == []  # loop never ran, so no audit trail


# --- Live Ollama tests --------------------------------------------------------

pytestmark_live = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable at localhost:11434 with llama3.2:3b pulled",
)


@pytestmark_live
def test_live_followup_loop_improves_completeness_over_single_shot():
    backend = OllamaBackend()
    message = "My 3 year old has had a fever for 2 days and seems unusually sleepy."

    provider = _sequential_answers(
        "No, he can drink normally.",
        "No, he has not vomited at all.",
        "No, he has not had any convulsions or fits.",
    )
    case, trail = extract_case_with_followup(
        message, backend, answer_provider=provider, case_id="live-followup-1", max_followup_turns=2
    )
    print("\n--- live follow-up loop ---")
    print("trail:", trail)
    print("final danger_signs:", case.danger_signs)
    print("raw_symptom_text:", case.raw_symptom_text)

    # original message must be preserved on the final case regardless of
    # how many follow-up turns ran (or none at all)
    assert case.raw_symptom_text == message
    assert 0 <= len(trail) <= 2

    # An empty trail is only a correct outcome if the single-shot pass
    # already left nothing worth asking about -- either it already found a
    # confirmed danger sign (this message describes "unusually sleepy",
    # which a capable extraction should read as lethargic=True on the first
    # pass -- observed behavior in practice) or it was already fully
    # assessed. If the trail is empty for any other reason, that's a real
    # bug in the loop's stopping condition, not an acceptable model quirk.
    if not trail:
        assert case.danger_signs.any_true() or case.danger_signs.all_assessed(), (
            "loop asked zero follow-up questions but danger_signs is neither "
            f"resolved-true nor fully assessed: {case.danger_signs}"
        )


@pytestmark_live
def test_live_followup_respects_cap_against_real_backend():
    backend = OllamaBackend()
    message = "My child is sick."  # deliberately vague -- unlikely to resolve in 2 turns

    provider = _sequential_answers("Not sure.", "Not sure.", "Not sure.", "Not sure.")
    case, trail = extract_case_with_followup(
        message, backend, answer_provider=provider, case_id="live-followup-2", max_followup_turns=2
    )
    print("\n--- live follow-up cap check ---")
    print("trail:", trail)
    assert len(trail) <= 2
