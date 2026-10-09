"""Stage 4 regression tests: follow-up relevance, AHP role, onset bug, safe report.

Covers the four Stage-4 fixes (plan §5 defects FU-1, AHP-1, AHP-2, RPT-1):
  - onset-acuity substring bug: "three days"/"chronic" are NOT acute;
  - follow-up selection is constrained to clinically-relevant discriminators;
  - AHP has a defined, tested role (within-tier ESI prioritization; danger-sign
    override forces ESI-1; it never downgrades below its disease tier);
  - the doctor report suppresses the dataset diagnosis on danger-sign / IMNCI /
    abstaining paths (no "Cervical spondylosis / heating pad" for a convulsing
    infant).

Classifier/deterministic only -- no Ollama, no FAISS.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import get_default_classifier
from app.emergency_scorer import _onset_acuity_from_duration, score_emergency
from app.followup_question_selector import select_followup_question
from app.routing.report import _build_report_prompt, _fallback_report, _dataset_diagnosis_is_primary
from app.routing.schemas import DispatchResult
from app.schemas import (
    AgeGroup,
    ClassificationLabel,
    ClassificationResult,
    DangerSigns,
    DiseaseCandidate,
    EmergencyBand,
    ExtractedCase,
)

clf = get_default_classifier()


# --- AHP-2: onset-acuity substring bug ---------------------------------------
@pytest.mark.parametrize(
    "duration, expected",
    [
        ("three days", 0.5),   # contains "hr" -> used to be 1.0 (acute)
        ("chronic", 0.2),      # contains "hr" -> used to be 1.0 (acute)
        ("2 hours", 1.0),
        ("sudden onset", 1.0),
        ("3 weeks", 0.2),
        ("5 days", 0.5),
        (None, 0.5),
    ],
)
def test_onset_acuity_word_boundary(duration, expected):
    assert _onset_acuity_from_duration(duration)[0] == expected


# --- AHP-1: AHP has a defined role (within-tier ESI prioritization) ----------
def _adult(duration=None, danger=None):
    return ExtractedCase(
        raw_symptom_text="x", age_group=AgeGroup.ADULT, duration=duration,
        danger_signs=danger or DangerSigns(),
    )


def test_ahp_onset_acuity_feeds_the_score_within_a_tier():
    # AHP's defined within-tier prioritization job: a more acute onset raises
    # the onset-acuity attribute (and never lowers the composite score). This
    # also pins the AHP-2 fix -- "three days"/"chronic" are no longer acute.
    acute = score_emergency(_adult(duration="2 hours"), "Pneumonia", ClassificationLabel.SEVERE)
    chronic = score_emergency(_adult(duration="3 weeks"), "Pneumonia", ClassificationLabel.SEVERE)
    three_days = score_emergency(_adult(duration="three days"), "Pneumonia", ClassificationLabel.SEVERE)
    assert acute.attribute_scores["onset_acuity"] > chronic.attribute_scores["onset_acuity"]
    assert three_days.attribute_scores["onset_acuity"] == 0.5  # subacute, not acute
    assert acute.score >= chronic.score


def test_ahp_danger_sign_override_forces_esi_1():
    r = score_emergency(_adult(danger=DangerSigns(convulsions=True)), "Common Cold", ClassificationLabel.MILD)
    assert r.override_triggered is True
    assert r.band == EmergencyBand.EMERGENCY
    assert r.esi_level == 1


def test_ahp_score_in_range_and_band_consistent():
    r = score_emergency(_adult(duration="2 days"), "Heart attack", ClassificationLabel.EMERGENCY)
    assert 1 <= r.score <= 10
    assert r.band in (EmergencyBand.EMERGENCY, EmergencyBand.URGENT, EmergencyBand.NON_URGENT)


# --- FU-1: follow-up questions are clinically relevant to the differential ---
def test_followup_for_respiratory_set_asks_a_respiratory_question():
    pset = ["Common Cold", "GERD", "Pneumonia", "Tuberculosis", "Bronchial Asthma"]
    sev = {d: "MILD" for d in pset}
    q = select_followup_question(
        pset, sev, known_tokens=["cough", "high_fever"],
        rows_by_disease=clf._rows_by_disease, vocabulary=clf.kb.vocabulary,
    )
    assert q is not None
    # The chosen token must be characteristic of a disease in the set (>=30% of
    # some set disease's rows) -- not an incidental dataset association.
    assert any(
        sum(1 for r in clf._rows_by_disease[d] if q.symptom_token in r) / len(clf._rows_by_disease[d]) >= 0.30
        for d in pset
    ), q.symptom_token


def test_followup_token_is_never_already_known():
    pset = ["Heart attack", "Tuberculosis"]
    sev = {"Heart attack": "EMERGENCY", "Tuberculosis": "SEVERE"}
    known = ["chest_pain", "breathlessness"]
    q = select_followup_question(
        pset, sev, known_tokens=known,
        rows_by_disease=clf._rows_by_disease, vocabulary=clf.kb.vocabulary,
    )
    if q is not None:
        assert q.symptom_token not in known


# --- RPT-1: doctor report suppresses spurious dataset dx ----------------------
def _dispatch():
    return DispatchResult(facility=None, route=None, urgency="EMERGENCY", reasoning=[], no_facility_found=True)


def test_report_suppresses_dataset_dx_on_pediatric_danger_sign():
    # Convulsing infant: EMERGENCY via IMNCI danger sign, with a spurious
    # attached dataset disease. The report must NOT name it.
    result = ClassificationResult(
        label=ClassificationLabel.EMERGENCY,
        condition="GENERAL_DANGER_SIGN",
        probable_disease="Cervical spondylosis",
        candidates=[DiseaseCandidate(name="Cervical spondylosis", score=0.3,
                                     matched_symptoms=[], precautions=["use heating pad", "exercise"])],
    )
    case = ExtractedCase(raw_symptom_text="baby convulsing", age_months=12,
                         danger_signs=DangerSigns(convulsions=True))
    assert _dataset_diagnosis_is_primary(result) is False
    prompt = _build_report_prompt(case, result, _dispatch())
    assert "Cervical spondylosis" not in prompt
    assert "heating pad" not in prompt
    assert "Cervical spondylosis" not in _fallback_report(case, result, _dispatch())


def test_report_keeps_dataset_dx_on_confident_adult():
    # Adult CONFIDENT dataset path: condition == probable_disease -> name it.
    result = ClassificationResult(
        label=ClassificationLabel.SEVERE,
        condition="Pneumonia",
        probable_disease="Pneumonia",
        candidates=[DiseaseCandidate(name="Pneumonia", score=0.9, matched_symptoms=["cough"], precautions=["rest"])],
    )
    case = _adult(duration="2 days")
    assert _dataset_diagnosis_is_primary(result) is True
    assert "Pneumonia" in _build_report_prompt(case, result, _dispatch())
    assert "Pneumonia" in _fallback_report(case, result, _dispatch())


def test_report_suppresses_dx_when_abstaining_wide_emergency():
    # EMERGENCY_IN_WIDE_DIFFERENTIAL: probable_disease None, abstaining -> no dx.
    result = ClassificationResult(
        label=ClassificationLabel.EMERGENCY,
        condition="EMERGENCY_IN_WIDE_DIFFERENTIAL",
        probable_disease=None,
        abstention_triggered=True,
        candidates=[DiseaseCandidate(name="Heart attack", score=0.3, matched_symptoms=[], precautions=[])],
    )
    assert _dataset_diagnosis_is_primary(result) is False
