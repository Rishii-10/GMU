"""
Tests for the age-based routing between the pediatric IMNCI path and the
dataset-driven disease classifier path in app/rules_engine.py::classify()
(Phase 2 of the "broaden to all ages" plan).

Kept separate from tests/test_rules_engine.py, which is scoped to the
original pediatric-only classification rules -- this file is specifically
about the routing decision and the two new code paths
(classify_via_dataset, _attach_dataset_context) it introduces. No LLM, no
network: pure deterministic logic against the real CSV-backed classifier.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import AgeGroup, ClassificationLabel, DangerSigns, ExtractedCase
from app.rules_engine import classify, classify_via_dataset


def make_case(**kwargs) -> ExtractedCase:
    defaults = dict(raw_symptom_text="test", danger_signs=DangerSigns())
    defaults.update(kwargs)
    return ExtractedCase(**defaults)


ALL_DS_NEGATIVE = DangerSigns(
    not_able_to_drink_or_breastfeed=False,
    vomits_everything=False,
    convulsions=False,
    lethargic_or_unconscious=False,
)


# --- Young infant (<2mo): unchanged, NOT rerouted to the dataset classifier --


def test_young_infant_still_incomplete_not_rerouted_to_dataset():
    case = make_case(
        age_months=1,
        danger_signs=ALL_DS_NEGATIVE,
        symptom_tokens=["high_fever", "chills"],  # even with dataset-recognizable symptoms
    )
    result = classify(case)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "AGE_OUT_OF_MODULE_SCOPE"
    assert result.candidates == []  # must not have used the dataset path
    assert result.probable_disease is None


# --- Adult/elderly routing -----------------------------------------------


def test_age_months_60_plus_routes_to_dataset():
    case = make_case(age_months=780, symptom_tokens=["chest_pain", "breathlessness", "sweating"])
    result = classify(case)
    assert result.probable_disease is not None
    assert result.candidates
    # Confirms the dataset path actually ran, not the pediatric axes.
    assert result.condition in {c.name for c in result.candidates}


def test_age_group_adult_with_no_age_months_routes_to_dataset():
    case = make_case(age_group=AgeGroup.ADULT, symptom_tokens=["high_fever", "chills", "sweating", "vomiting"])
    result = classify(case)
    assert result.probable_disease is not None
    assert result.label in (
        ClassificationLabel.EMERGENCY,
        ClassificationLabel.SEVERE,
        ClassificationLabel.MODERATE,
        ClassificationLabel.MILD,
    )


def test_age_group_elderly_with_no_age_months_routes_to_dataset():
    case = make_case(age_group=AgeGroup.ELDERLY, symptom_tokens=["chest_pain", "breathlessness", "sweating"])
    result = classify(case)
    assert result.probable_disease == "Heart attack" or result.probable_disease in {
        "Pneumonia", "Tuberculosis", "Heart attack"
    }


def test_dataset_route_with_no_symptom_tokens_is_incomplete_not_a_guess():
    case = make_case(age_months=800, symptom_tokens=[])
    result = classify(case)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "INSUFFICIENT_SYMPTOM_DATA"
    assert result.probable_disease is None


def test_dataset_route_maps_emergency_disease_to_emergency_label():
    case = make_case(age_months=780, symptom_tokens=["chest_pain", "vomiting", "breathlessness", "sweating"])
    result = classify_via_dataset(case)
    top = result.candidates[0]
    if top.name == "Heart attack":
        assert result.label == ClassificationLabel.EMERGENCY


def test_dataset_route_does_not_consult_danger_signs():
    # danger_signs left entirely unassessed (all None) -- must NOT trigger
    # the pediatric completeness gate or block classification; that gate is
    # a WHO IMCI pediatric construct this route does not inherit (see
    # classify_via_dataset()'s docstring).
    case = make_case(age_months=780, symptom_tokens=["chest_pain", "breathlessness", "sweating"])
    result = classify(case)
    assert result.condition != "DANGER_SIGNS_NOT_FULLY_ASSESSED"


# --- Pediatric in-band path: dataset candidates attached as supplementary --


def test_pediatric_case_keeps_imnci_label_with_dataset_context_attached():
    case = make_case(
        age_months=8,
        danger_signs=ALL_DS_NEGATIVE,
        symptom_tokens=["high_fever", "chills", "sweating", "vomiting"],
    )
    result = classify(case)
    # IMNCI's own outcome (no cough/diarrhea reported, no danger signs) --
    # label must stay MILD/NO_DANGER_SIGNS..., never overridden by the
    # dataset classifier's opinion.
    assert result.label == ClassificationLabel.MILD
    assert result.condition == "NO_DANGER_SIGNS_NO_SPECIFIC_ILLNESS_CLASSIFIED"
    # But supplementary context IS attached.
    assert result.candidates
    assert result.probable_disease is not None


def test_pediatric_case_without_symptom_tokens_has_no_dataset_context():
    case = make_case(age_months=8, danger_signs=ALL_DS_NEGATIVE)  # symptom_tokens defaults to []
    result = classify(case)
    assert result.candidates == []
    assert result.probable_disease is None


def test_pediatric_emergency_danger_sign_still_wins_with_context_attached():
    ds = DangerSigns(
        not_able_to_drink_or_breastfeed=False,
        vomits_everything=False,
        convulsions=True,
        lethargic_or_unconscious=False,
    )
    case = make_case(age_months=8, danger_signs=ds, symptom_tokens=["high_fever", "chills"])
    result = classify(case)
    assert result.label == ClassificationLabel.EMERGENCY
    assert result.condition == "GENERAL_DANGER_SIGN"  # IMNCI condition, not a disease name
