"""Unit tests for app/rules_engine.py in isolation (no LLM, no Agent 1).

These are WHO IMCI-*pattern* cases I constructed to exercise the standard,
published IMCI classification algorithm (danger signs; age-based
fast-breathing cutoffs of 50/min under 12mo and 40/min 12mo-5yr; the
two-of-the-following dehydration logic) -- they are original test fixtures
following that public clinical algorithm, not a transcription of the WHO
IMCI chart booklet's own named worked examples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import (
    ClassificationLabel,
    CoughDifficultBreathing,
    DangerSigns,
    DiarrheaAssessment,
    ExtractedCase,
)
from app.rules_engine import classify, most_severe_wins
from app.rules_engine import (
    classify_cough_or_difficult_breathing,
    classify_danger_signs,
    classify_diarrhea_dehydration,
    check_danger_sign_completeness,
)


def make_case(**kwargs) -> ExtractedCase:
    defaults = dict(raw_symptom_text="test", age_months=8, danger_signs=DangerSigns())
    defaults.update(kwargs)
    return ExtractedCase(**defaults)


def test_incomplete_danger_signs_blocks_classification():
    case = make_case(danger_signs=DangerSigns(not_able_to_drink_or_breastfeed=False))
    result = classify(case)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert "vomits_everything" in result.missing_fields
    assert "convulsions" in result.missing_fields
    assert "lethargic_or_unconscious" in result.missing_fields


def test_all_danger_signs_false_is_complete_not_incomplete():
    ds = DangerSigns(
        not_able_to_drink_or_breastfeed=False,
        vomits_everything=False,
        convulsions=False,
        lethargic_or_unconscious=False,
    )
    assert check_danger_sign_completeness(make_case(danger_signs=ds)) is None
    assert classify_danger_signs(make_case(danger_signs=ds)) is None


def test_any_danger_sign_true_is_emergency():
    ds = DangerSigns(
        not_able_to_drink_or_breastfeed=False,
        vomits_everything=False,
        convulsions=True,
        lethargic_or_unconscious=False,
    )
    case = make_case(danger_signs=ds)
    result = classify(case)
    assert result.label == ClassificationLabel.EMERGENCY
    assert result.condition == "GENERAL_DANGER_SIGN"
    assert "convulsions" in result.reasoning[0]


ALL_DS_NEGATIVE = DangerSigns(
    not_able_to_drink_or_breastfeed=False,
    vomits_everything=False,
    convulsions=False,
    lethargic_or_unconscious=False,
)


def test_fast_breathing_under_12mo_is_pneumonia():
    # WHO IMCI cutoff for 2-<12 months is 50 breaths/min.
    case = make_case(
        age_months=8,
        danger_signs=ALL_DS_NEGATIVE,
        cough=CoughDifficultBreathing(present=True, breaths_per_minute=55, chest_indrawing=False, stridor_when_calm=False),
    )
    result = classify(case)
    assert result.label == ClassificationLabel.MODERATE
    assert result.condition == "PNEUMONIA"


def test_borderline_breathing_under_cutoff_is_not_pneumonia():
    case = make_case(
        age_months=8,
        danger_signs=ALL_DS_NEGATIVE,
        cough=CoughDifficultBreathing(present=True, breaths_per_minute=48, chest_indrawing=False, stridor_when_calm=False),
    )
    result = classify(case)
    assert result.label == ClassificationLabel.MILD
    assert result.condition == "COUGH_OR_COLD"


def test_fast_breathing_cutoff_shifts_at_12_months():
    # 45/min is NOT fast for a 14-month-old (cutoff 40 would actually flag this as fast: 45>=40)
    # so use a value that is fast under the <12mo cutoff (50) but not under the >=12mo cutoff (40)
    # -- there is no such value since 40 < 50, cutoffs only get stricter with age, so instead
    # check that 42 breaths/min is NOT pneumonia for an 8-month-old (needs >=50) but IS for a
    # 14-month-old (needs >=40).
    young = make_case(
        age_months=8,
        danger_signs=ALL_DS_NEGATIVE,
        cough=CoughDifficultBreathing(present=True, breaths_per_minute=42, chest_indrawing=False, stridor_when_calm=False),
    )
    older = make_case(
        age_months=14,
        danger_signs=ALL_DS_NEGATIVE,
        cough=CoughDifficultBreathing(present=True, breaths_per_minute=42, chest_indrawing=False, stridor_when_calm=False),
    )
    assert classify(young).condition == "COUGH_OR_COLD"
    assert classify(older).condition == "PNEUMONIA"


def test_chest_indrawing_overrides_breathing_rate_to_severe():
    case = make_case(
        age_months=8,
        danger_signs=ALL_DS_NEGATIVE,
        cough=CoughDifficultBreathing(present=True, breaths_per_minute=42, chest_indrawing=True, stridor_when_calm=False),
    )
    result = classify(case)
    assert result.label == ClassificationLabel.SEVERE
    assert result.condition == "SEVERE_PNEUMONIA_OR_SEVERE_DISEASE"


def test_severe_dehydration_needs_two_severe_signs():
    d = DiarrheaAssessment(
        present=True,
        sunken_eyes=True,
        drinks_poorly_or_not_able=True,
        skin_pinch_goes_back_very_slowly=False,
    )
    case = make_case(danger_signs=ALL_DS_NEGATIVE, diarrhea=d)
    result = classify(case)
    assert result.label == ClassificationLabel.SEVERE
    assert result.condition == "SEVERE_DEHYDRATION"


def test_one_severe_sign_alone_is_not_severe_dehydration():
    d = DiarrheaAssessment(present=True, sunken_eyes=True, drinks_poorly_or_not_able=False)
    case = make_case(danger_signs=ALL_DS_NEGATIVE, diarrhea=d)
    result = classify(case)
    assert result.label != ClassificationLabel.SEVERE


def test_some_dehydration_two_signs():
    d = DiarrheaAssessment(present=True, restless_or_irritable=True, drinks_eagerly_thirsty=True)
    case = make_case(danger_signs=ALL_DS_NEGATIVE, diarrhea=d)
    result = classify(case)
    assert result.label == ClassificationLabel.MODERATE
    assert result.condition == "SOME_DEHYDRATION"


def test_no_dehydration_signs_gives_mild():
    d = DiarrheaAssessment(present=True, restless_or_irritable=False, sunken_eyes=False,
                            drinks_eagerly_thirsty=False, drinks_poorly_or_not_able=False)
    case = make_case(danger_signs=ALL_DS_NEGATIVE, diarrhea=d)
    result = classify(case)
    assert result.label == ClassificationLabel.MILD
    assert result.condition == "NO_DEHYDRATION"


def test_most_severe_wins_across_axes():
    case = make_case(
        age_months=30,
        danger_signs=ALL_DS_NEGATIVE,
        cough=CoughDifficultBreathing(present=True, breaths_per_minute=20, chest_indrawing=False, stridor_when_calm=False),  # MILD
        diarrhea=DiarrheaAssessment(present=True, sunken_eyes=True, drinks_poorly_or_not_able=True),  # SEVERE
    )
    result = classify(case)
    assert result.label == ClassificationLabel.SEVERE
    assert result.condition == "SEVERE_DEHYDRATION"
    assert any("overridden" in r for r in result.reasoning)


def test_most_severe_wins_function_directly_with_tie_keeps_first():
    from app.schemas import ClassificationResult

    a = ClassificationResult(label=ClassificationLabel.MODERATE, condition="A")
    b = ClassificationResult(label=ClassificationLabel.MODERATE, condition="B")
    winner = most_severe_wins([a, b])
    assert winner.condition == "A"


def test_confirmed_danger_sign_short_circuits_even_if_others_unassessed():
    # convulsions=True is confirmed; the other 3 fields were never asked
    # about (None). This must still be EMERGENCY immediately -- one
    # confirmed danger sign is sufficient to act on and must not be gated
    # behind completeness of the other three.
    ds = DangerSigns(convulsions=True)
    case = make_case(danger_signs=ds)
    result = classify(case)
    assert result.label == ClassificationLabel.EMERGENCY
    assert result.condition == "GENERAL_DANGER_SIGN"


def test_young_infant_escalated_not_generic_out_of_scope():
    # Young infant (<2mo) is a real, tracked clinical gap (no validated WHO
    # IMCI young-infant ruleset yet) -- distinct from a generic out-of-scope
    # age, hence its own condition string rather than AGE_OUT_OF_MODULE_SCOPE.
    case = make_case(age_months=1, danger_signs=ALL_DS_NEGATIVE)
    result = classify(case)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "YOUNG_INFANT_NO_VALIDATED_RULESET"
    assert result.condition != "AGE_OUT_OF_MODULE_SCOPE"


def test_no_cough_no_diarrhea_no_danger_signs_defaults_mild_with_caveat():
    case = make_case(danger_signs=ALL_DS_NEGATIVE)
    result = classify(case)
    assert result.label == ClassificationLabel.MILD
    assert result.condition == "NO_DANGER_SIGNS_NO_SPECIFIC_ILLNESS_CLASSIFIED"
