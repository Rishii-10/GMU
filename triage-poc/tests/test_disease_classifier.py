"""
Unit tests for app/disease_kb.py and app/disease_classifier.py (Phase 1/2).

No LLM, no network -- pure CSV loading and arithmetic against the real
data/disease_symptoms.csv and data/disease_precautions.csv files, mirroring
tests/test_rules_engine.py's "deterministic layer tested in isolation"
convention.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_kb import DiseaseKB, normalize_disease_name, normalize_symptom_token
from app.disease_classifier import (
    DEFAULT_UNKNOWN_DISEASE_SEVERITY,
    DiseaseClassifier,
    MIN_CANDIDATE_SCORE,
    severity_for_disease,
)
from app.schemas import ClassificationLabel


@pytest.fixture(scope="module")
def kb() -> DiseaseKB:
    return DiseaseKB.load()


@pytest.fixture(scope="module")
def classifier(kb: DiseaseKB) -> DiseaseClassifier:
    return DiseaseClassifier(kb)


# --- normalization -----------------------------------------------------------


def test_normalize_symptom_token_strips_and_lowercases():
    assert normalize_symptom_token("  Skin_Rash ") == "skin_rash"


def test_normalize_symptom_token_collapses_stray_internal_space():
    # Real artifacts from the CSV (see app/disease_kb.py module docstring).
    assert normalize_symptom_token("dischromic _patches") == "dischromic_patches"
    assert normalize_symptom_token("foul_smell_of urine") == "foul_smell_of_urine"
    assert normalize_symptom_token("spotting_ urination") == "spotting_urination"


def test_normalize_disease_name_strips_but_does_not_lowercase():
    assert normalize_disease_name("Diabetes ") == "Diabetes"


# --- DiseaseKB ----------------------------------------------------------------


def test_kb_loads_all_41_diseases(kb: DiseaseKB):
    assert len(kb.diseases) == 41


def test_kb_vocabulary_has_131_unique_tokens(kb: DiseaseKB):
    assert len(kb.vocabulary) == 131


def test_kb_every_disease_has_precautions(kb: DiseaseKB):
    for disease in kb.diseases:
        assert kb.precautions_for(disease), f"{disease} has no precautions"


def test_kb_malaria_symptoms_and_precautions(kb: DiseaseKB):
    symptoms = kb.symptoms_for("Malaria")
    assert "high_fever" in symptoms
    assert "chills" in symptoms
    precautions = kb.precautions_for("Malaria")
    assert "avoid oily food" in precautions


def test_kb_unknown_disease_returns_empty(kb: DiseaseKB):
    assert kb.symptoms_for("Not A Real Disease") == set()
    assert kb.precautions_for("Not A Real Disease") == []


# --- DiseaseClassifier: basic behavior ----------------------------------------


def test_empty_symptom_tokens_returns_no_candidates(classifier: DiseaseClassifier):
    assert classifier.classify_diseases([]) == []


def test_unrecognized_tokens_only_returns_no_candidates(classifier: DiseaseClassifier):
    assert classifier.classify_diseases(["not_a_real_symptom_xyz"]) == []


def test_recognized_and_unrecognized_mixed_uses_only_recognized(classifier: DiseaseClassifier):
    with_junk = classifier.classify_diseases(["high_fever", "chills", "sweating", "not_a_real_symptom"])
    without_junk = classifier.classify_diseases(["high_fever", "chills", "sweating"])
    assert [c.name for c in with_junk] == [c.name for c in without_junk]


def test_scores_are_positive_and_sum_to_one(classifier: DiseaseClassifier):
    candidates = classifier.classify_diseases(["high_fever", "chills", "sweating", "vomiting"])
    assert candidates
    for c in candidates:
        assert c.score >= MIN_CANDIDATE_SCORE
    assert sum(c.score for c in candidates) == pytest.approx(1.0, abs=1e-3)


def test_candidates_ranked_most_probable_first(classifier: DiseaseClassifier):
    candidates = classifier.classify_diseases(["high_fever", "chills", "sweating", "vomiting"])
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_matched_symptoms_is_intersection_with_reported(classifier: DiseaseClassifier):
    candidates = classifier.classify_diseases(["high_fever", "chills", "sweating", "vomiting"])
    reported = {"high_fever", "chills", "sweating", "vomiting"}
    for c in candidates:
        assert set(c.matched_symptoms) <= reported


def test_candidate_carries_its_disease_precautions(classifier: DiseaseClassifier, kb: DiseaseKB):
    candidates = classifier.classify_diseases(["high_fever", "chills", "sweating", "vomiting"])
    for c in candidates:
        assert c.precautions == kb.precautions_for(c.name)


# --- The user-flagged edge case: 3-way probabilistic tie-break ---------------


def test_three_way_overlapping_symptom_set_ranks_by_probability_not_first_seen(
    classifier: DiseaseClassifier,
):
    """Chest pain + breathlessness + sweating genuinely overlaps three
    diseases in the real dataset (Heart attack, Pneumonia, Tuberculosis) --
    three DIFFERENT clinical/emergency situations, exactly the edge case the
    user flagged. Assert the classifier returns a real ranked, scored list
    (not an arbitrary single pick), that all three real candidates surface,
    and that the ranking is driven by score, not dict/CSV iteration order.
    """
    candidates = classifier.classify_diseases(["chest_pain", "breathlessness", "sweating"])
    names = [c.name for c in candidates]

    assert "Heart attack" in names
    assert "Pneumonia" in names
    assert "Tuberculosis" in names

    # Ranking must follow score, not insertion/alphabetical order (Heart
    # attack alphabetically precedes Pneumonia/Tuberculosis but must not be
    # assumed to win just from that).
    for earlier, later in zip(candidates, candidates[1:]):
        assert earlier.score >= later.score

    # Every returned candidate must be auditable: real matched symptoms and
    # a real score, not a placeholder.
    for c in candidates:
        assert c.matched_symptoms
        assert c.score > 0


def test_classification_is_deterministic(classifier: DiseaseClassifier):
    a = classifier.classify_diseases(["chest_pain", "breathlessness", "sweating"])
    b = classifier.classify_diseases(["chest_pain", "breathlessness", "sweating"])
    assert [(c.name, c.score) for c in a] == [(c.name, c.score) for c in b]


def test_top_n_limits_returned_candidates(classifier: DiseaseClassifier):
    all_candidates = classifier.classify_diseases(["chest_pain", "breathlessness", "sweating"])
    top_1 = classifier.classify_diseases(["chest_pain", "breathlessness", "sweating"], top_n=1)
    assert len(top_1) == 1
    assert top_1[0].name == all_candidates[0].name


# --- Severity data: data/disease_severity.csv, not a hardcoded table --------


def test_kb_loads_severity_csv(kb: DiseaseKB):
    assert kb.severity_for("Heart attack") == "EMERGENCY"
    assert kb.severity_for("Paralysis (brain hemorrhage)") == "EMERGENCY"
    assert kb.severity_for("Common Cold") == "MILD"


def test_kb_severity_covers_all_41_diseases(kb: DiseaseKB):
    for disease in kb.diseases:
        assert kb.severity_for(disease) is not None, f"{disease} has no severity on file"


def test_kb_severity_for_unknown_disease_is_none(kb: DiseaseKB):
    assert kb.severity_for("Not A Real Disease") is None


def test_kb_load_degrades_gracefully_when_severity_file_missing(tmp_path):
    import shutil

    from app.disease_kb import DEFAULT_DATA_DIR

    # Copy only the two required CSVs into a scratch dir, deliberately
    # omitting disease_severity.csv -- DiseaseKB.load() must not error.
    shutil.copy(DEFAULT_DATA_DIR / "disease_symptoms.csv", tmp_path / "disease_symptoms.csv")
    shutil.copy(DEFAULT_DATA_DIR / "disease_precautions.csv", tmp_path / "disease_precautions.csv")

    kb = DiseaseKB.load(data_dir=tmp_path)
    assert kb.severity_by_disease == {}
    assert kb.severity_for("Heart attack") is None
    assert len(kb.diseases) == 41  # the two real files still load fully


def test_severity_for_disease_reads_from_csv_data(kb: DiseaseKB):
    assert severity_for_disease("Heart attack", kb=kb) == ClassificationLabel.EMERGENCY
    assert severity_for_disease("Common Cold", kb=kb) == ClassificationLabel.MILD
    assert severity_for_disease("Malaria", kb=kb) == ClassificationLabel.MODERATE


def test_severity_for_disease_unknown_disease_falls_back(kb: DiseaseKB):
    assert severity_for_disease("Not A Real Disease", kb=kb) == DEFAULT_UNKNOWN_DISEASE_SEVERITY


def test_severity_for_disease_is_genuinely_swappable():
    # The core "not hardcoded" proof: build a DiseaseKB from ENTIRELY
    # different severity data and confirm severity_for_disease()'s output
    # changes accordingly -- nothing about the severity mapping is baked
    # into disease_classifier.py's own logic.
    custom_kb = DiseaseKB(
        symptoms_by_disease={"Common Cold": {"runny_nose"}},
        precautions_by_disease={"Common Cold": ["rest"]},
        severity_by_disease={"Common Cold": "EMERGENCY"},  # deliberately absurd, to prove the point
    )
    assert severity_for_disease("Common Cold", kb=custom_kb) == ClassificationLabel.EMERGENCY


def test_severity_for_disease_rejects_incomplete_assessment_as_a_severity():
    # A malformed CSV row with "INCOMPLETE_ASSESSMENT" as its severity
    # value must NOT be accepted -- that's a real ClassificationLabel
    # member but a distinct state, not a severity tier.
    custom_kb = DiseaseKB(
        symptoms_by_disease={"X": {"fever"}},
        precautions_by_disease={"X": ["rest"]},
        severity_by_disease={"X": "INCOMPLETE_ASSESSMENT"},
    )
    assert severity_for_disease("X", kb=custom_kb) == DEFAULT_UNKNOWN_DISEASE_SEVERITY


def test_severity_for_disease_rejects_garbage_severity_value():
    custom_kb = DiseaseKB(
        symptoms_by_disease={"X": {"fever"}},
        precautions_by_disease={"X": ["rest"]},
        severity_by_disease={"X": "NOT_A_REAL_SEVERITY"},
    )
    assert severity_for_disease("X", kb=custom_kb) == DEFAULT_UNKNOWN_DISEASE_SEVERITY
