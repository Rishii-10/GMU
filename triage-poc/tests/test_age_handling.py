"""
Age-handling regression tests (Phase 2 of the "eliminate hardcoded age
defaults + fix age-phrasing normalization" task).

Two concerns, both covered here:

PART A -- no hardcoded/implicit age default anywhere in intake->classify:
  - ExtractedCase.age_group defaults to None, not AgeGroup.UNKNOWN.
  - RegexBackend leaves age_group None (was a hardcoded "unknown").
  - _sanitize_enum_fields coerces junk age_group to None, not "unknown".
  - rules_engine.classify() returns INCOMPLETE_ASSESSMENT /
    AGE_UNKNOWN_CANNOT_ROUTE when age is fully unresolved, instead of
    silently taking the pediatric IMNCI route.
  - the follow-up layer (both entry points) asks an age-clarifier for it.

PART B -- age-phrasing normalization + the <2-month deterministic safety net:
  - _infant_age_months_from_text / _apply_infant_age_floor unit behavior.
  - Every phrasing from the Phase 1 B.3 table, replayed through the
    pipeline as a fixed backend double (so the test is deterministic and
    pins the SAFETY NET's job -- "correct the <2mo boundary regardless of
    what the LLM extracted" -- not the LLM's behavior).
  - The four B.4 week/newborn cases reaching YOUNG_INFANT_NO_VALIDATED_RULESET
    end to end (extraction -> classify()), asserted explicitly.
  - An Ollama-gated live replay of the same table (skipped when no local
    model), same convention as tests/test_integration_agent1_pipeline.py.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    AGE_CLARIFIER_FOLLOWUP_QUESTION,
    LLMBackend,
    OllamaBackend,
    RegexBackend,
    _apply_infant_age_floor,
    _infant_age_months_from_text,
    _sanitize_enum_fields,
    extract_and_classify,
    extract_case,
    question_for_incomplete_result,
)
from app.followup_policy import missing_required
from app.rules_engine import classify
from app.schemas import AgeGroup, ClassificationLabel, DangerSigns, ExtractedCase
from tests.test_integration_agent1_pipeline import _ollama_available


# ===========================================================================
# PART A -- hardcoded / implicit age defaults are gone
# ===========================================================================


def test_extracted_case_age_group_defaults_to_none():
    c = ExtractedCase(raw_symptom_text="x")
    assert c.age_group is None
    assert c.age_months is None


def test_extracted_case_age_group_accepts_explicit_null():
    # The prompt tells the model to emit null for absent fields; that must
    # now validate rather than crash (the A.b prompt/schema contradiction).
    c = ExtractedCase(raw_symptom_text="x", age_group=None)
    assert c.age_group is None


def test_extracted_case_age_group_still_accepts_explicit_unknown():
    c = ExtractedCase(raw_symptom_text="x", age_group=AgeGroup.UNKNOWN)
    assert c.age_group is AgeGroup.UNKNOWN


def test_regex_backend_leaves_age_group_none_not_unknown():
    out = RegexBackend().extract("my child has a fever")
    assert out["age_group"] is None
    assert out["age_months"] is None


@pytest.mark.parametrize("junk", ["teenager", "baby", "ADULT", "vao nien", "42"])
def test_sanitize_coerces_junk_age_group_to_none(junk):
    assert _sanitize_enum_fields({"age_group": junk})["age_group"] is None


@pytest.mark.parametrize("good", ["infant", "child", "adult", "elderly", "unknown"])
def test_sanitize_leaves_valid_age_group_untouched(good):
    assert _sanitize_enum_fields({"age_group": good})["age_group"] == good


def test_sanitize_leaves_absent_age_group_absent():
    assert "age_group" not in _sanitize_enum_fields({"severity": "mild"})


def _min_case(**kw) -> ExtractedCase:
    d = dict(raw_symptom_text="test", danger_signs=DangerSigns())
    d.update(kw)
    return ExtractedCase(**d)


def test_classify_age_fully_unknown_does_not_fall_through_to_pediatric():
    # danger signs all assessed False + cough present: under the old behavior
    # this returned MILD/COUGH_OR_COLD via the pediatric route. Now it must
    # refuse to route.
    case = _min_case(
        raw_symptom_text="patient has a cough",
        danger_signs=DangerSigns(
            not_able_to_drink_or_breastfeed=False, vomits_everything=False,
            convulsions=False, lethargic_or_unconscious=False,
        ),
        cough={"present": True},
    )
    result = classify(case)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "AGE_UNKNOWN_CANNOT_ROUTE"
    assert result.missing_fields == ["age_months"]
    assert result.condition not in {"YOUNG_INFANT_NO_VALIDATED_RULESET", "AGE_OUT_OF_MODULE_SCOPE"}


def test_classify_age_group_unknown_enum_also_cannot_route():
    result = classify(_min_case(age_group=AgeGroup.UNKNOWN))
    assert result.condition == "AGE_UNKNOWN_CANNOT_ROUTE"


def test_classify_age_group_child_still_routes_pediatric():
    # A lay band with no exact months is still a usable pediatric signal.
    result = classify(_min_case(age_group=AgeGroup.CHILD))
    assert result.condition != "AGE_UNKNOWN_CANNOT_ROUTE"


def test_age_unknown_triggers_age_clarifier_followup_both_entry_points():
    # Flow B (post-classification): rules_engine result -> question.
    case, result = extract_and_classify("मेरे बच्चे को बुखार है", RegexBackend())
    assert result.condition == "AGE_UNKNOWN_CANNOT_ROUTE"
    assert question_for_incomplete_result(result) == AGE_CLARIFIER_FOLLOWUP_QUESTION
    # Flow A (pre-classification): followup_policy gate -> same field key.
    assert missing_required(_min_case()) == ["age_months"]


# ===========================================================================
# PART B -- deterministic <2-month infant-age safety net (unit level)
# ===========================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the baby is 3 weeks old and not feeding", 0),
        ("6 week old infant, lethargic", 1),
        ("4 week old, fever", 0),
        ("baby is 8 weeks old", 1),          # 8*7/30.44 = 1.84 -> 1, still <2
        ("newborn with yellow eyes", 0),
        ("just born, not crying", 0),
        ("neonate, poor feeding", 0),
        ("10 days old, jaundice", 0),
        ("infant 45 days old", 1),
        ("she is a new-born", 0),
    ],
)
def test_infant_age_from_text_hits(text, expected):
    assert _infant_age_months_from_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "she is 6 months old and has a fever",   # months -> LLM's job, not this net
        "my son is 2 years old with a bad cough",
        "cough for 3 weeks",                     # duration, not age ("old" absent)
        "born 3 weeks ago",                      # KNOWN GAP -- not covered by design
        "10 week old baby",                      # 10w -> 2.3mo, >= 2, out of net scope
        "patient is about 70 years old",
        "no age mentioned at all",
    ],
)
def test_infant_age_from_text_misses(text):
    assert _infant_age_months_from_text(text) is None


def test_apply_infant_age_floor_overrides_wrong_llm_value():
    parsed = {"age_months": 3, "age_group": "infant", "notes": None}
    out = _apply_infant_age_floor("the baby is 3 weeks old and not feeding", parsed)
    assert out["age_months"] == 0
    assert "deterministic infant-age rule" in out["notes"]


def test_apply_infant_age_floor_fills_when_llm_gave_none():
    out = _apply_infant_age_floor("newborn with yellow eyes", {"age_months": None})
    assert out["age_months"] == 0


def test_apply_infant_age_floor_leaves_consistent_low_llm_value_alone():
    parsed = {"age_months": 1, "notes": "x"}
    out = _apply_infant_age_floor("6 week old infant", parsed)
    assert out["age_months"] == 1
    assert out["notes"] == "x"  # untouched, no audit note appended


def test_apply_infant_age_floor_noop_when_no_infant_phrasing():
    parsed = {"age_months": 60, "notes": None}
    assert _apply_infant_age_floor("my daughter turned 5 last month", parsed) == parsed


# ===========================================================================
# PART B -- full pipeline replay of the Phase 1 B.3 / B.4 table
# ===========================================================================


class _ReplayBackend(LLMBackend):
    """Emits a fixed parsed-JSON dict -- used to replay the EXACT
    (age_months, age_group) the Phase 1 audit observed llama3.2:3b produce
    for a given phrasing, so we can assert what the pipeline does with it
    deterministically. The point is to pin the safety net, not the LLM."""

    name = "replay_test_backend"

    def __init__(self, age_months, age_group):
        self._age_months = age_months
        self._age_group = age_group

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        return {
            "symptom": "fever", "duration": None, "severity": "unknown",
            "age_group": self._age_group, "age_months": self._age_months,
            "location": None, "notes": None,
            "danger_signs": {k: None for k in (
                "not_able_to_drink_or_breastfeed", "vomits_everything",
                "convulsions", "lethargic_or_unconscious")},
            "cough": None, "diarrhea": None,
        }


# (raw_text, LLM's observed age_months, LLM's observed age_group,
#  expected age_months after the pipeline's safety net, is-young-infant)
B3_TABLE = [
    ("she is 6 months old and has a fever",        6,    "infant",  6,   False),
    ("my son is 2 years old with a bad cough",     24,   "child",   24,  False),
    ("patient is about 70 years old, chest pain",  None, "elderly", None, False),
    ("grandmother, maybe 65, difficulty breathing",None, "elderly", None, False),
    ("aged one and a half years, fever",           18,   "infant",  18,  False),
    ("child is 18 months, loose motions",          18,   "child",   18,  False),
    ("my daughter turned 5 last month, vomiting",  5,    "child",   5,   False),  # KNOWN GAP: not fixed at this layer
    ("the baby is 3 weeks old and not feeding",    3,    "infant",  0,   True),
    ("6 week old infant, lethargic",               6,    "infant",  1,   True),
    ("4 week old, fever",                          4,    "infant",  0,   True),
    ("newborn with yellow eyes",                   None, "infant",  0,   True),
    ("2 month old baby coughing",                  2,    "infant",  2,   False),
]


@pytest.mark.parametrize("raw,llm_am,llm_ag,exp_am,young", B3_TABLE)
def test_b3_table_pipeline_normalization(raw, llm_am, llm_ag, exp_am, young):
    case = extract_case(raw, _ReplayBackend(llm_am, llm_ag))
    assert case.age_months == exp_am, f"{raw!r}: expected age_months={exp_am}, got {case.age_months}"
    if young:
        assert case.age_months is not None and case.age_months < 2


@pytest.mark.parametrize(
    "raw,llm_am,llm_ag",
    [
        ("the baby is 3 weeks old and not feeding", 3, "infant"),
        ("6 week old infant, lethargic", 6, "infant"),
        ("4 week old, fever", 4, "infant"),
        ("newborn with yellow eyes", None, "infant"),
    ],
)
def test_b4_week_and_newborn_cases_reach_young_infant_escalation_end_to_end(raw, llm_am, llm_ag):
    case, result = extract_and_classify(raw, _ReplayBackend(llm_am, llm_ag))
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "YOUNG_INFANT_NO_VALIDATED_RULESET", (
        f"{raw!r} must escalate as a young infant, got condition={result.condition!r} "
        f"(age_months={case.age_months})"
    )


# ===========================================================================
# PART B -- live Ollama replay (skipped when no local model)
# ===========================================================================

_LIVE_CASES = [
    ("she is 6 months old and has a fever",          lambda am: am == 6),
    ("my son is 2 years old with a bad cough",       lambda am: am in (23, 24, 25)),
    ("the baby is 3 weeks old and not feeding",      lambda am: am is not None and am < 2),
    ("6 week old infant, lethargic",                 lambda am: am is not None and am < 2),
    ("4 week old baby, fever",                       lambda am: am is not None and am < 2),
    ("newborn with yellow eyes",                     lambda am: am is not None and am < 2),
    ("10 day old, not feeding",                      lambda am: am is not None and am < 2),
    ("2 month old baby coughing",                    lambda am: am == 2),
    ("child is 18 months, loose motions",            lambda am: am in (17, 18, 19)),
]


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable with llama3.2:3b")
@pytest.mark.parametrize("raw,ok", _LIVE_CASES)
def test_live_ollama_age_normalization(raw, ok):
    case = extract_case(raw, OllamaBackend())
    assert ok(case.age_months), f"{raw!r}: age_months={case.age_months} failed expectation"


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable with llama3.2:3b")
@pytest.mark.parametrize("raw", [
    "the baby is 3 weeks old and not feeding",
    "6 week old infant, lethargic",
    "newborn with yellow eyes",
])
def test_live_ollama_young_infant_reaches_escalation(raw):
    _, result = extract_and_classify(raw, OllamaBackend())
    assert result.condition == "YOUNG_INFANT_NO_VALIDATED_RULESET"
