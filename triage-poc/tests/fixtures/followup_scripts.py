"""
Fabricated multi-turn follow-up conversation scripts (Phase 8) -- named,
scripted (question -> canned patient answer -> next LLM extraction)
sequences covering both routes (pediatric IMNCI, adult dataset classifier)
and a BALANCED spread of outcomes: resolves early (danger sign confirmed /
symptom obtained), resolves after the full turn budget, and genuinely hits
the turn cap unresolved -- not just the easy "it works" case.

Each script is a list of scripted backend response dicts (same shape
LLMBackend.extract() returns) consumed in order by a small scripted
backend double, paired with the canned answers a caregiver would give.
tests/test_followup_scripts.py drives these through the real
extract_case_with_followup() and asserts the expected final outcome.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class FollowupScript(TypedDict):
    name: str
    route: str  # "pediatric" | "adult"
    responses: list[dict]
    answers: list[str]
    expected_turns: int
    expected_outcome: str  # free-text description, asserted against in the test


def _ds(**overrides) -> dict:
    base = {
        "not_able_to_drink_or_breastfeed": None,
        "vomits_everything": None,
        "convulsions": None,
        "lethargic_or_unconscious": None,
    }
    base.update(overrides)
    return base


def _pediatric_response(symptom: str = "fever", **ds_overrides) -> dict:
    return {
        "symptom": symptom,
        "duration": "2 days",
        "severity": "unknown",
        "age_group": "child",
        "age_months": 18,
        "location": "Denkanikottai",
        "notes": None,
        "danger_signs": _ds(**ds_overrides),
        "cough": None,
        "diarrhea": None,
    }


def _adult_response(symptom: Optional[str]) -> dict:
    return {
        "symptom": symptom,
        "duration": None,
        "severity": "unknown",
        "age_group": "adult",
        "age_months": None,
        "location": "Thally",
        "notes": None,
        "danger_signs": _ds(),
        "cough": None,
        "diarrhea": None,
    }


SCRIPTS: list[FollowupScript] = [
    {
        "name": "pediatric_resolves_to_emergency_turn1",
        "route": "pediatric",
        "responses": [
            _pediatric_response(),
            _pediatric_response(convulsions=True),
        ],
        "answers": ["Yes, he had a seizure this morning."],
        "expected_turns": 1,
        "expected_outcome": "EMERGENCY",
    },
    {
        "name": "pediatric_resolves_complete_no_danger_sign",
        "route": "pediatric",
        "responses": [
            _pediatric_response(),
            _pediatric_response(not_able_to_drink_or_breastfeed=False, vomits_everything=False),
            _pediatric_response(
                not_able_to_drink_or_breastfeed=False,
                vomits_everything=False,
                convulsions=False,
                lethargic_or_unconscious=False,
            ),
        ],
        "answers": ["No, drinking fine.", "No vomiting, no fits, alert and active."],
        "expected_turns": 2,
        "expected_outcome": "COMPLETE_NOT_EMERGENCY",
    },
    {
        "name": "pediatric_hits_turn_cap_unresolved",
        "route": "pediatric",
        "responses": [_pediatric_response() for _ in range(5)],
        "answers": ["I'm not sure.", "I don't know.", "Maybe, not sure.", "I can't tell."],
        "expected_turns": 2,  # max_followup_turns default
        "expected_outcome": "INCOMPLETE_ASSESSMENT",
    },
    {
        "name": "adult_resolves_symptom_turn1",
        "route": "adult",
        "responses": [
            _adult_response(symptom=None),
            _adult_response(symptom="chest pain and sweating"),
        ],
        "answers": ["My chest hurts and I'm sweating a lot."],
        "expected_turns": 1,
        "expected_outcome": "SYMPTOM_OBTAINED",
    },
    {
        "name": "adult_hits_turn_cap_unresolved",
        "route": "adult",
        "responses": [_adult_response(symptom=None) for _ in range(5)],
        "answers": ["I don't feel good.", "Just unwell.", "Not sure how to describe it.", "I don't know."],
        "expected_turns": 2,
        "expected_outcome": "STILL_NO_SYMPTOM",
    },
    {
        "name": "adult_no_followup_needed_symptom_already_present",
        "route": "adult",
        "responses": [_adult_response(symptom="fever and body ache")],
        "answers": [],
        "expected_turns": 0,
        "expected_outcome": "SYMPTOM_OBTAINED",
    },
]
