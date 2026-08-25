"""
Drives the fabricated follow-up conversation scripts
(tests/fixtures/followup_scripts.py) through the real
app.agent1_extraction.extract_case_with_followup(), asserting each
script's expected outcome. Complements tests/test_agent1_followup.py
(pediatric mechanism tests) and tests/test_followup_policy.py (adult
mechanism tests) with a BALANCED end-to-end sweep across both routes and a
mix of outcomes (resolves early, resolves after full budget, hits the turn
cap unresolved) -- per the user's test-data requirement to test all cases,
not just the easy ones.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import LLMBackend, extract_and_classify_with_followup
from app.schemas import ClassificationLabel
from tests.fixtures.followup_scripts import SCRIPTS, FollowupScript


class _ScriptedBackend(LLMBackend):
    name = "fixture_scripted_backend"
    supports_followup = True

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls = 0

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        self.calls += 1
        return self._responses.pop(0)


def _run_script(script: FollowupScript):
    backend = _ScriptedBackend(script["responses"])
    answers = iter(script["answers"])

    def provider(question: str) -> str:
        return next(answers)

    case, trail, result = extract_and_classify_with_followup(
        f"[{script['name']}] initial message", backend, answer_provider=provider, max_followup_turns=2
    )
    return case, trail, result


@pytest.mark.parametrize("script", SCRIPTS, ids=[s["name"] for s in SCRIPTS])
def test_script_uses_exactly_expected_turns(script: FollowupScript):
    _case, trail, _result = _run_script(script)
    assert len(trail) == script["expected_turns"], (
        f"{script['name']}: expected {script['expected_turns']} follow-up turns, got {len(trail)}"
    )


def test_pediatric_resolves_to_emergency_turn1():
    script = next(s for s in SCRIPTS if s["name"] == "pediatric_resolves_to_emergency_turn1")
    case, trail, result = _run_script(script)
    assert result.label == ClassificationLabel.EMERGENCY
    assert case.danger_signs.convulsions is True


def test_pediatric_resolves_complete_no_danger_sign():
    script = next(s for s in SCRIPTS if s["name"] == "pediatric_resolves_complete_no_danger_sign")
    case, trail, result = _run_script(script)
    assert result.label != ClassificationLabel.EMERGENCY
    assert result.label != ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert case.danger_signs.all_assessed()


def test_pediatric_hits_turn_cap_unresolved():
    script = next(s for s in SCRIPTS if s["name"] == "pediatric_hits_turn_cap_unresolved")
    case, trail, result = _run_script(script)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert case.danger_signs.missing_fields()
    # never silently defaulted to False
    assert case.danger_signs.not_able_to_drink_or_breastfeed is None


def test_adult_resolves_symptom_turn1():
    script = next(s for s in SCRIPTS if s["name"] == "adult_resolves_symptom_turn1")
    case, trail, result = _run_script(script)
    assert case.symptom is not None
    assert "chest" in case.symptom.lower() or "pain" in case.symptom.lower()


def test_adult_hits_turn_cap_unresolved():
    script = next(s for s in SCRIPTS if s["name"] == "adult_hits_turn_cap_unresolved")
    case, trail, result = _run_script(script)
    assert case.symptom is None
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "INSUFFICIENT_SYMPTOM_DATA"


def test_adult_no_followup_needed_symptom_already_present():
    script = next(s for s in SCRIPTS if s["name"] == "adult_no_followup_needed_symptom_already_present")
    case, trail, result = _run_script(script)
    assert trail == []
    assert case.symptom == "fever and body ache"


def test_all_scripts_produce_a_valid_classification_result():
    # Broad sanity sweep: every script, regardless of outcome, must
    # produce a well-formed ClassificationResult -- no exceptions, no None
    # label, for every route/outcome combination in the fixture.
    for script in SCRIPTS:
        case, trail, result = _run_script(script)
        assert result.label in ClassificationLabel
        assert case.raw_symptom_text.startswith(f"[{script['name']}]")
