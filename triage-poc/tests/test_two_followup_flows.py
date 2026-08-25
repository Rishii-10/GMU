"""
Tests both valid entry points into the follow-up module, per the intended
architecture: Agent 1's extraction can flow into the follow-up module in
two ways --

  (A) Agent 1 -> follow-up module directly, PRE-classification:
      extract_case_with_followup()'s loop, gated by
      app.followup_policy.missing_required() -- already covered by
      tests/test_agent1_followup.py, tests/test_followup_policy.py,
      tests/test_followup_scripts.py.

  (B) Rules Engine -> follow-up module, POST-classification:
      app.agent1_extraction.question_for_incomplete_result() -- takes
      rules_engine.classify()'s own INCOMPLETE_ASSESSMENT result (with its
      real `missing_fields`) and returns the next question, for a caller
      using the single-shot extract_and_classify() (no loop) who wants to
      decide "should I ask a follow-up" AFTER seeing the real
      classification, not before.

This file specifically exercises (B) end-to-end against the REAL
rules_engine.classify() (no test doubles for the classification step) to
prove the Rules Engine's own missing_fields output genuinely drives the
follow-up module, not just a duplicated policy computation that happens to
agree with it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    DANGER_SIGN_FOLLOWUP_QUESTIONS,
    ADULT_SYMPTOM_FOLLOWUP_QUESTION,
    question_for_incomplete_result,
)
from app.rules_engine import classify
from app.schemas import AgeGroup, ClassificationLabel, DangerSigns, ExtractedCase


def make_case(**kwargs) -> ExtractedCase:
    defaults = dict(raw_symptom_text="test", danger_signs=DangerSigns())
    defaults.update(kwargs)
    return ExtractedCase(**defaults)


def test_flow_b_pediatric_incomplete_produces_a_danger_sign_question():
    case = make_case(age_months=18, danger_signs=DangerSigns(not_able_to_drink_or_breastfeed=False))
    result = classify(case)  # real Rules Engine, no test double
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.missing_fields  # real output drives what's asked next

    question = question_for_incomplete_result(result)
    assert question is not None
    assert question in DANGER_SIGN_FOLLOWUP_QUESTIONS.values()
    # Specifically the question for the FIRST missing field, matching
    # _followup_question_for()'s own "first field with a template" rule.
    assert question == DANGER_SIGN_FOLLOWUP_QUESTIONS[result.missing_fields[0]]


def test_flow_b_adult_incomplete_produces_the_symptom_clarifier():
    case = make_case(age_group=AgeGroup.ADULT)  # no symptom_tokens at all
    result = classify(case)  # real Rules Engine
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "INSUFFICIENT_SYMPTOM_DATA"
    assert result.missing_fields == ["symptom_tokens"]

    question = question_for_incomplete_result(result)
    assert question == ADULT_SYMPTOM_FOLLOWUP_QUESTION


def test_flow_b_returns_none_when_classification_is_not_incomplete():
    case = make_case(
        age_months=18,
        danger_signs=DangerSigns(
            not_able_to_drink_or_breastfeed=False,
            vomits_everything=False,
            convulsions=False,
            lethargic_or_unconscious=False,
        ),
    )
    result = classify(case)
    assert result.label != ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert question_for_incomplete_result(result) is None


def test_flow_b_returns_none_for_emergency_even_though_others_unassessed():
    # A confirmed danger sign short-circuits to EMERGENCY, not
    # INCOMPLETE_ASSESSMENT, even though the other three fields were never
    # asked about -- question_for_incomplete_result() must not manufacture
    # a follow-up question for an already-actionable EMERGENCY result.
    case = make_case(age_months=18, danger_signs=DangerSigns(convulsions=True))
    result = classify(case)
    assert result.label == ClassificationLabel.EMERGENCY
    assert question_for_incomplete_result(result) is None


def test_flow_a_and_flow_b_agree_on_which_question_to_ask_first():
    # Both entry points must recommend the SAME first question for the
    # same underlying gap -- they are two doors into the same follow-up
    # module, not two independently-drifting implementations.
    from app.followup_policy import missing_required_pediatric
    from app.agent1_extraction import _followup_question_for

    case = make_case(age_months=18, danger_signs=DangerSigns(not_able_to_drink_or_breastfeed=False))

    # Flow A's pre-classification check:
    flow_a_missing = missing_required_pediatric(case)
    flow_a_question = _followup_question_for(flow_a_missing)

    # Flow B's post-classification check, against the REAL classify() output:
    result = classify(case)
    flow_b_question = question_for_incomplete_result(result)

    assert flow_a_question == flow_b_question


def test_flow_b_works_through_extract_and_classify_single_shot():
    # The realistic caller shape for Flow B: single-shot extraction (no
    # follow-up loop at all), then decide whether to ask based on the
    # Rules Engine's own verdict.
    from app.agent1_extraction import RegexBackend, extract_and_classify

    case, result = extract_and_classify("मेरे बच्चे को बुखार है", RegexBackend())
    if result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT:
        question = question_for_incomplete_result(result)
        assert question is not None
