"""
Tests for the adult-route Stage 1 / Stage 2 / adaptive follow-up stack:

  - app.disease_classifier.DiseaseClassifier.classify_with_cp()  (CP gate)
  - app.emergency_scorer.score_emergency()                       (AHP Stage 2)
  - app.followup_question_selector.select_followup_question()
  - app.agent1_extraction.classify_with_adaptive_followup()

None of these four had a single test before this file, even though the
design doc listed them as "Complete, fully tested". Several tests here are
regression tests for crashes that were live on main:

  * DiseaseClassifier() raised NameError (`held_out_rows`) after the
    Sep 19 merge -- every adult classification died on construction.
  * classify_with_adaptive_followup() and the Streamlit adaptive UI used
    `clf._kb`, which does not exist (the attribute is `clf.kb`).
  * A "no" answer in the adaptive loop re-selected the same token, so the
    same question was asked up to max_turns times.

No LLM, no network: the backend used here is a stub whose generate_text()
raises, so generate_followup_question_text() always takes its template
fallback path.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import followup_question_selector as fqs
from app.agent1_extraction import (
    LLMBackend,
    classify_with_adaptive_followup,
    generate_followup_question_text,
)
from app.disease_classifier import (
    _RED_FLAG_TOKENS,
    DiseaseClassifier,
    get_default_classifier,
    severity_for_disease,
)
from app.disease_kb import DiseaseKB
from app.emergency_scorer import AHP_WEIGHTS, score_emergency
from app.rules_engine import classify
from app.schemas import (
    AgeGroup,
    ClassificationLabel,
    DangerSigns,
    EmergencyBand,
    ExtractedCase,
)


def make_case(**kwargs) -> ExtractedCase:
    defaults = dict(raw_symptom_text="test", danger_signs=DangerSigns())
    defaults.update(kwargs)
    return ExtractedCase(**defaults)


class _OfflineBackend(LLMBackend):
    """Stub: extraction unsupported, text generation always fails, so the
    adaptive loop must fall back to the static question templates."""

    name = "offline_stub"
    supports_followup = True

    def extract(self, patient_text, context=None):  # pragma: no cover
        raise RuntimeError("not used in these tests")

    def generate_text(self, system_prompt, user_prompt):
        raise RuntimeError("LLM unavailable")


# ---------------------------------------------------------------------------
# Construction / regression
# ---------------------------------------------------------------------------


def test_classifier_constructs_without_error():
    # Regression: NameError('held_out_rows') on main after the Sep 19 merge.
    clf = DiseaseClassifier(DiseaseKB.load())
    assert clf.kb is not None
    assert len(clf.kb.diseases) == 41


def test_cp_calibration_scores_are_out_of_fold_and_well_formed():
    clf = get_default_classifier()
    scores = clf._cp_cal_nonconformity_scores
    # One score per unique (disease, profile) held out across the k folds.
    assert len(scores) > 0
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Not every held-out profile is a memorised duplicate: if calibration
    # were leaking (full-data model scoring its own training rows) nearly
    # every score would sit at exactly 0.0.
    assert any(s > 0.0 for s in scores)


# ---------------------------------------------------------------------------
# Stage 1: Conformal Prediction gate
# ---------------------------------------------------------------------------


def test_cp_empty_tokens_abstains_with_empty_set():
    cp = get_default_classifier().classify_with_cp([])
    assert cp.decision == "ABSTAIN"
    assert cp.prediction_set == []
    assert cp.set_size == 0


def test_cp_unrecognised_tokens_only_abstains():
    cp = get_default_classifier().classify_with_cp(["not_a_real_token"])
    assert cp.decision == "ABSTAIN"
    assert cp.candidates == []


def test_cp_decision_matches_set_size():
    clf = get_default_classifier()
    for tokens in (
        ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
        ["chest_pain", "breathlessness", "sweating"],
        ["high_fever", "headache", "vomiting"],
        ["fatigue"],
    ):
        cp = clf.classify_with_cp(tokens)
        assert cp.set_size == len(cp.prediction_set)
        if cp.set_size == 1:
            assert cp.decision == "CONFIDENT"
        elif 2 <= cp.set_size <= 3:
            assert cp.decision == "UNCERTAIN"
        else:
            assert cp.decision == "ABSTAIN"
        # Every member of the set has a calibrated posterior recorded.
        assert set(cp.posteriors) == set(cp.prediction_set)


def test_cp_full_fungal_infection_profile_is_confident():
    cp = get_default_classifier().classify_with_cp(
        ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"]
    )
    assert cp.decision == "CONFIDENT"
    assert cp.prediction_set == ["Fungal infection"]


def test_cp_heart_attack_profile_never_excludes_heart_attack():
    # The whole point of the x3 emergency boost (F1): Heart attack shares
    # chest_pain/breathlessness/sweating with Tuberculosis and must at
    # minimum stay in the prediction set, and the label the engine surfaces
    # must be EMERGENCY (worst-case rule on UNCERTAIN, F3).
    cp = get_default_classifier().classify_with_cp(["chest_pain", "breathlessness", "sweating", "vomiting"])
    assert "Heart attack" in cp.prediction_set
    result = classify(make_case(age_months=600, symptom_tokens=["chest_pain", "breathlessness", "sweating", "vomiting"]))
    assert result.label == ClassificationLabel.EMERGENCY


def test_every_red_flag_token_exists_in_vocabulary():
    # F2 can only fire for tokens the dataset vocabulary actually contains.
    vocab = set(get_default_classifier().kb.vocabulary)
    missing = sorted(t for t in _RED_FLAG_TOKENS if t not in vocab)
    assert missing == [], f"red-flag tokens not in dataset vocabulary: {missing}"


def test_red_flag_override_collapses_to_emergency_when_emergency_in_set():
    # altered_sensorium appears in ~95% of Paralysis (brain hemorrhage) rows
    # and in no other disease's rows, so with the set containing Paralysis
    # the override must collapse the set to it.
    cp = get_default_classifier().classify_with_cp(["headache", "vomiting", "altered_sensorium"])
    assert cp.decision == "CONFIDENT"
    assert cp.prediction_set == ["Paralysis (brain hemorrhage)"]


# ---------------------------------------------------------------------------
# Stage 2: AHP emergency scorer
# ---------------------------------------------------------------------------


def test_ahp_weights_sum_to_one():
    assert abs(sum(AHP_WEIGHTS.values()) - 1.0) < 0.005


def test_ahp_score_in_range_and_band_consistent():
    for disease in ("Acne", "Malaria", "Pneumonia", "Heart attack"):
        case = make_case(age_months=420, symptom_tokens=["fatigue"])
        er = score_emergency(case, disease, severity_for_disease(disease))
        assert 1 <= er.score <= 10
        if er.score <= 3:
            assert er.band == EmergencyBand.NON_URGENT
        elif er.score <= 7:
            assert er.band == EmergencyBand.URGENT
        else:
            assert er.band == EmergencyBand.EMERGENCY
        assert 1 <= er.esi_level <= 5


def test_ahp_emergency_disease_scores_higher_than_mild_disease():
    case = make_case(age_months=420)
    mild = score_emergency(case, "Acne", ClassificationLabel.MILD)
    emerg = score_emergency(case, "Heart attack", ClassificationLabel.EMERGENCY)
    assert emerg.score > mild.score
    assert emerg.band == EmergencyBand.EMERGENCY
    assert mild.band == EmergencyBand.NON_URGENT


def test_ahp_danger_sign_override_forces_score_10():
    case = make_case(age_months=420, danger_signs=DangerSigns(convulsions=True))
    er = score_emergency(case, "Acne", ClassificationLabel.MILD)
    assert er.override_triggered is True
    assert er.score == 10
    assert er.band == EmergencyBand.EMERGENCY
    assert er.esi_level == 1


def test_ahp_unassessed_danger_signs_do_not_trigger_override():
    # None = not assessed, never treated as present.
    case = make_case(age_months=420, danger_signs=DangerSigns())
    er = score_emergency(case, "Acne", ClassificationLabel.MILD)
    assert er.override_triggered is False


def test_confident_adult_result_carries_emergency_result():
    result = classify(
        make_case(age_months=420, symptom_tokens=["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"])
    )
    assert result.label == ClassificationLabel.MILD
    assert result.emergency_result is not None
    assert result.emergency_result.band == EmergencyBand.NON_URGENT


# ---------------------------------------------------------------------------
# Follow-up question selector
# ---------------------------------------------------------------------------


def _selector_inputs():
    clf = get_default_classifier()
    return clf._rows_by_disease, list(clf.kb.vocabulary)


def test_selector_emergency_first_when_tiers_differ():
    rows, vocab = _selector_inputs()
    fq = fqs.select_followup_question(
        prediction_set=["Heart attack", "Tuberculosis"],
        severity_map={"Heart attack": "EMERGENCY", "Tuberculosis": "SEVERE"},
        known_tokens=["chest_pain", "breathlessness", "sweating"],
        rows_by_disease=rows,
        vocabulary=vocab,
    )
    assert fq is not None
    assert fq.severity_urgency == "EMERGENCY_PRIORITY"
    assert fq.symptom_token not in {"chest_pain", "breathlessness", "sweating"}
    assert fq.template_question  # never empty


def test_selector_info_gain_when_same_tier():
    rows, vocab = _selector_inputs()
    fq = fqs.select_followup_question(
        prediction_set=["Dengue", "Typhoid"],
        severity_map={"Dengue": "SEVERE", "Typhoid": "SEVERE"},
        known_tokens=["high_fever", "headache"],
        rows_by_disease=rows,
        vocabulary=vocab,
    )
    assert fq is not None
    assert fq.severity_urgency == "ROUTINE"
    assert fq.symptom_token not in {"high_fever", "headache"}


def test_selector_never_returns_a_known_token():
    rows, vocab = _selector_inputs()
    known = ["high_fever", "headache", "vomiting", "nausea"]
    fq = fqs.select_followup_question(
        prediction_set=["Dengue", "Typhoid", "Malaria"],
        severity_map={"Dengue": "SEVERE", "Typhoid": "SEVERE", "Malaria": "MODERATE"},
        known_tokens=known,
        rows_by_disease=rows,
        vocabulary=vocab,
    )
    assert fq is None or fq.symptom_token not in known


def test_template_fallback_covers_every_vocabulary_token():
    _, vocab = _selector_inputs()
    for token in vocab:
        q = fqs.template_for(token)
        assert q.endswith("?")


def test_generate_question_text_falls_back_to_template_when_llm_fails():
    text = generate_followup_question_text(
        "chest_pain", ["Heart attack", "Tuberculosis"],
        {"Heart attack": "EMERGENCY", "Tuberculosis": "SEVERE"},
        backend=_OfflineBackend(),
    )
    assert text == fqs.template_for("chest_pain")


# ---------------------------------------------------------------------------
# Adaptive follow-up loop (programmatic API)
# ---------------------------------------------------------------------------


def _uncertain_case() -> ExtractedCase:
    # Finds a token set the current calibration treats as UNCERTAIN so the
    # loop has something to do. Skips if the fitted CP gate happens to be
    # CONFIDENT/ABSTAIN on every probe (calibration-dependent).
    clf = get_default_classifier()
    probes = [
        ["chest_pain", "breathlessness", "sweating"],
        ["high_fever", "headache", "vomiting"],
        ["fatigue", "weight_loss", "cough"],
        ["abdominal_pain", "vomiting", "nausea"],
        ["itching", "skin_rash"],
    ]
    for tokens in probes:
        if clf.classify_with_cp(tokens).decision == "UNCERTAIN":
            return make_case(age_months=420, age_group=AgeGroup.ADULT, symptom_tokens=tokens)
    pytest.skip("no probe produced an UNCERTAIN prediction set under current calibration")


def test_adaptive_loop_runs_without_attribute_error():
    # Regression: `clf._kb` AttributeError on first use.
    case = _uncertain_case()
    result, trail = classify_with_adaptive_followup(case, _OfflineBackend(), answer_provider=lambda q: "no", max_turns=1)
    assert result is not None
    assert len(trail) == 1


def test_adaptive_loop_skipped_entirely_without_answer_provider():
    case = _uncertain_case()
    result, trail = classify_with_adaptive_followup(case, _OfflineBackend(), answer_provider=None)
    assert trail == []
    assert result.label in set(ClassificationLabel)


def test_adaptive_loop_does_not_repeat_a_question_after_no():
    # Regression: a "no" left the token out of symptom_tokens, so the
    # selector picked it again and the CHW saw the same question up to
    # max_turns times.
    case = _uncertain_case()
    _, trail = classify_with_adaptive_followup(case, _OfflineBackend(), answer_provider=lambda q: "no", max_turns=3)
    tokens_asked = [t["token"] for t in trail]
    assert len(tokens_asked) == len(set(tokens_asked)), f"repeated question: {tokens_asked}"


def test_adaptive_loop_yes_adds_token_and_respects_max_turns():
    case = _uncertain_case()
    result, trail = classify_with_adaptive_followup(case, _OfflineBackend(), answer_provider=lambda q: "yes", max_turns=3)
    assert len(trail) <= 3
    # Loop stops early once the gate is CONFIDENT (set size 1), or at the cap.
    if len(trail) < 3:
        assert len(result.prediction_set) <= 1 or len(result.prediction_set) >= 4


def test_adaptive_loop_never_downgrades_below_worst_case_while_uncertain():
    # While the set is still open, the surfaced label must be the worst-case
    # severity of the set (F3), never the top candidate's own severity.
    case = _uncertain_case()
    result, _ = classify_with_adaptive_followup(case, _OfflineBackend(), answer_provider=lambda q: "no", max_turns=1)
    if len(result.prediction_set) >= 2:
        worst = min(
            (severity_for_disease(d) for d in result.prediction_set),
            key=lambda lbl: {ClassificationLabel.EMERGENCY: 0, ClassificationLabel.SEVERE: 1,
                             ClassificationLabel.MODERATE: 2, ClassificationLabel.MILD: 3}[lbl],
        )
        assert result.label == worst
