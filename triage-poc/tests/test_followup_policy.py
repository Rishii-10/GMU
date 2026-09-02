"""
Tests for app/followup_policy.py (Phase 4: minimum-viable-info follow-up
gate) and its wiring into the adult/out-of-band branch of
app.agent1_extraction.extract_case_with_followup(). The pediatric branch's
existing behavior is already pinned by tests/test_agent1_followup.py
(unmodified, still green) -- this file adds the new adult-route coverage
and direct unit tests for the policy functions themselves.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    ADULT_SYMPTOM_FOLLOWUP_QUESTION,
    LLMBackend,
    extract_case_with_followup,
)
from app.followup_policy import missing_required, missing_required_adult, missing_required_pediatric
from app.schemas import AgeGroup, DangerSigns, ExtractedCase


def make_case(**kwargs) -> ExtractedCase:
    defaults = dict(raw_symptom_text="test", danger_signs=DangerSigns())
    defaults.update(kwargs)
    return ExtractedCase(**defaults)


# --- missing_required_pediatric / missing_required_adult: direct unit tests -


def test_pediatric_all_none_returns_all_four_fields():
    case = make_case(age_months=24)
    assert set(missing_required_pediatric(case)) == {
        "not_able_to_drink_or_breastfeed",
        "vomits_everything",
        "convulsions",
        "lethargic_or_unconscious",
    }


def test_pediatric_any_true_returns_empty_even_with_others_unassessed():
    case = make_case(age_months=24, danger_signs=DangerSigns(convulsions=True))
    assert missing_required_pediatric(case) == []


def test_pediatric_all_assessed_returns_empty():
    ds = DangerSigns(
        not_able_to_drink_or_breastfeed=False,
        vomits_everything=False,
        convulsions=False,
        lethargic_or_unconscious=False,
    )
    case = make_case(age_months=24, danger_signs=ds)
    assert missing_required_pediatric(case) == []


def test_adult_no_symptom_returns_sentinel_field():
    case = make_case(age_group=AgeGroup.ADULT)
    assert missing_required_adult(case) == ["symptom_tokens"]


def test_adult_with_symptom_text_returns_empty():
    case = make_case(age_group=AgeGroup.ADULT, symptom="chest pain")
    assert missing_required_adult(case) == []


# --- missing_required(): route dispatch --------------------------------------


def test_route_dispatch_pediatric_for_in_band_age():
    case = make_case(age_months=24)
    assert missing_required(case) == missing_required_pediatric(case)


def test_route_dispatch_adult_for_age_months_60_plus():
    case = make_case(age_months=780)
    assert missing_required(case) == missing_required_adult(case)


def test_route_dispatch_adult_for_age_group_adult_no_age_months():
    case = make_case(age_group=AgeGroup.ADULT)
    assert missing_required(case) == missing_required_adult(case)


def test_route_dispatch_adult_for_age_group_elderly_no_age_months():
    case = make_case(age_group=AgeGroup.ELDERLY)
    assert missing_required(case) == missing_required_adult(case)


def test_route_dispatch_asks_for_age_when_age_fully_unresolved():
    # age_months=None and age_group=None (the new honest default). The gate
    # must NOT silently fall through to the pediatric danger-sign checklist
    # (the old behavior) -- age is the gap, so it asks for age. Mirrors
    # rules_engine.classify()'s AGE_UNKNOWN_CANNOT_ROUTE branch, which runs
    # before either route is chosen.
    assert make_case().age_group is None  # pin the new default
    assert missing_required(make_case()) == ["age_months"]


def test_route_dispatch_asks_for_age_when_age_group_explicitly_unknown():
    # A caller that explicitly set UNKNOWN is treated the same as None here:
    # still no usable age band to route on.
    case = make_case(age_group=AgeGroup.UNKNOWN)
    assert missing_required(case) == ["age_months"]


def test_route_dispatch_pediatric_when_age_group_gives_a_band_but_no_months():
    # age_group=INFANT/CHILD is a usable pediatric signal -> pediatric gate,
    # not an age question.
    case = make_case(age_group=AgeGroup.CHILD)
    assert missing_required(case) == missing_required_pediatric(case)


# --- Never asks exam-only/unknowable fields -----------------------------------


def test_pediatric_never_names_an_exam_only_field():
    case = make_case(age_months=24)
    exam_only = {"breaths_per_minute", "chest_indrawing", "stridor_when_calm",
                 "skin_pinch_goes_back_slowly", "skin_pinch_goes_back_very_slowly"}
    assert not (set(missing_required(case)) & exam_only)


# --- Wiring: adult follow-up loop in extract_case_with_followup -------------


class _AdultScriptedBackend(LLMBackend):
    """Scripted backend for the adult/out-of-band follow-up path."""

    name = "adult_scripted_test_backend"
    supports_followup = True

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, Optional[list[str]]]] = []

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        self.calls.append((patient_text, list(context) if context else None))
        return self._responses.pop(0)


def _adult_response(symptom: Optional[str]) -> dict:
    return {
        "symptom": symptom,
        "duration": None,
        "severity": "unknown",
        "age_group": "adult",
        "age_months": None,
        "location": None,
        "notes": None,
        "danger_signs": {
            "not_able_to_drink_or_breastfeed": None,
            "vomits_everything": None,
            "convulsions": None,
            "lethargic_or_unconscious": None,
        },
        "cough": None,
        "diarrhea": None,
    }


def test_adult_followup_asks_symptom_clarifier_when_symptom_missing():
    backend = _AdultScriptedBackend(
        [
            _adult_response(symptom=None),  # turn 0: nothing yet
            _adult_response(symptom="chest pain"),  # turn 1: resolved
        ]
    )
    provider = lambda q: "It's my chest, it hurts."
    case, trail = extract_case_with_followup(
        "I don't feel well", backend, answer_provider=provider, max_followup_turns=2
    )
    assert len(trail) == 1
    assert trail[0]["question"] == ADULT_SYMPTOM_FOLLOWUP_QUESTION
    assert case.symptom == "chest pain" or case.symptom is not None


def test_adult_followup_does_not_ask_pediatric_danger_sign_questions():
    backend = _AdultScriptedBackend([_adult_response(symptom=None) for _ in range(5)])
    asked_questions = []

    def provider(q: str) -> str:
        asked_questions.append(q)
        return "still not sure"

    extract_case_with_followup("I don't feel well", backend, answer_provider=provider, max_followup_turns=2)
    from app.agent1_extraction import DANGER_SIGN_FOLLOWUP_QUESTIONS

    for q in asked_questions:
        assert q not in DANGER_SIGN_FOLLOWUP_QUESTIONS.values()
        assert q == ADULT_SYMPTOM_FOLLOWUP_QUESTION


def test_adult_followup_skips_entirely_when_symptom_already_present():
    backend = _AdultScriptedBackend([_adult_response(symptom="fever and chills")])

    def _must_not_be_called(q: str) -> str:
        raise AssertionError("should not ask a follow-up when symptom is already reported")

    case, trail = extract_case_with_followup(
        "I have fever and chills", backend, answer_provider=_must_not_be_called, max_followup_turns=2
    )
    assert trail == []
    assert case.symptom == "fever and chills"


def test_adult_followup_respects_turn_cap():
    backend = _AdultScriptedBackend([_adult_response(symptom=None) for _ in range(5)])
    provider = lambda q: "not sure"
    case, trail = extract_case_with_followup(
        "I don't feel well", backend, answer_provider=provider, max_followup_turns=2
    )
    assert len(trail) == 2
    assert len(backend.calls) == 3
