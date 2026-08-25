"""
Tests for FAISSDisambiguator.match_dataset_symptom() (Phase 3: FAISS
vocabulary rebuilt from data/disease_symptoms.csv) and its wiring into
app.agent1_extraction._apply_disambiguation() -> ExtractedCase.symptom_tokens.

Kept separate from tests/test_disambiguation.py (which pins the original,
UNTOUCHED pediatric two-category disambiguate()/VOCABULARY behavior) so
this file can freely exercise the new second index without any risk of the
two suites' fixtures/assumptions colliding.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import LLMBackend, extract_case_with_followup
from app.disambiguation import Disambiguator, DisambiguationResult, FAISSDisambiguator, StubDisambiguator
from app.disease_kb import DiseaseKB


@pytest.fixture(scope="module")
def faiss_disambiguator() -> FAISSDisambiguator:
    try:
        return FAISSDisambiguator()
    except ImportError as e:
        pytest.skip(f"faiss-cpu / sentence-transformers not installed: {e}")


# --- match_dataset_symptom(): direct tests -----------------------------------


def test_dataset_vocabulary_covers_all_131_csv_tokens(faiss_disambiguator: FAISSDisambiguator):
    kb = DiseaseKB.load()
    dataset_tokens = {token for _surface, token in faiss_disambiguator.dataset_vocabulary}
    assert dataset_tokens == set(kb.vocabulary)


def test_exact_dataset_token_matches_itself(faiss_disambiguator: FAISSDisambiguator):
    result = faiss_disambiguator.match_dataset_symptom("high fever")
    assert result.matched_category == "high_fever"
    assert result.confidence >= faiss_disambiguator.dataset_confidence_threshold


def test_snake_case_query_also_matches(faiss_disambiguator: FAISSDisambiguator):
    result = faiss_disambiguator.match_dataset_symptom("chest_pain")
    assert result.matched_category == "chest_pain"


def test_curated_hindi_alias_matches(faiss_disambiguator: FAISSDisambiguator):
    result = faiss_disambiguator.match_dataset_symptom("बुखार")
    assert result.matched_category == "high_fever"
    assert result.confidence >= faiss_disambiguator.dataset_confidence_threshold


def test_curated_hinglish_alias_matches(faiss_disambiguator: FAISSDisambiguator):
    result = faiss_disambiguator.match_dataset_symptom("chhati mein dard")
    assert result.matched_category == "chest_pain"


def test_below_threshold_returns_no_match_with_real_score(faiss_disambiguator: FAISSDisambiguator):
    result = faiss_disambiguator.match_dataset_symptom("the weather is nice today")
    assert result.matched_category is None
    assert result.confidence is not None
    assert result.confidence < faiss_disambiguator.dataset_confidence_threshold


def test_disabled_dataset_matching_is_honest_noop():
    try:
        d = FAISSDisambiguator(enable_dataset_matching=False)
    except ImportError as e:
        pytest.skip(str(e))
    result = d.match_dataset_symptom("high fever")
    assert result.matched_category is None
    assert result.confidence is None


def test_pediatric_disambiguate_unaffected_by_dataset_index(faiss_disambiguator: FAISSDisambiguator):
    # "cough" is both a pediatric-category surface form (-> cough_or_
    # difficult_breathing) AND a literal dataset token in its own right --
    # the exact collision the module docstring flags. disambiguate() must
    # still return the pediatric category, unaffected by the second index.
    result = faiss_disambiguator.disambiguate("cough")
    assert result.matched_category == "cough_or_difficult_breathing"
    assert result.confidence > 0.99


def test_stub_disambiguator_dataset_match_is_noop():
    stub = StubDisambiguator()
    result = stub.match_dataset_symptom("high fever")
    assert result.matched_category is None
    assert result.confidence is None


def test_abc_default_match_dataset_symptom_is_noop():
    class _MinimalDisambiguator(Disambiguator):
        def disambiguate(self, symptom_text: str) -> DisambiguationResult:
            return DisambiguationResult(original_text=symptom_text, matched_category=None, confidence=None)

    d = _MinimalDisambiguator()
    result = d.match_dataset_symptom("anything")
    assert result.matched_category is None
    assert result.confidence is None


def test_deterministic_dataset_match(faiss_disambiguator: FAISSDisambiguator):
    a = faiss_disambiguator.match_dataset_symptom("chhati mein dard")
    b = faiss_disambiguator.match_dataset_symptom("chhati mein dard")
    assert a.matched_category == b.matched_category
    assert a.confidence == b.confidence


# --- Wiring into extract_case_with_followup -> ExtractedCase.symptom_tokens --


class _FixedResponseBackend(LLMBackend):
    """Small local test double -- always returns the same parsed dict.
    Not shared with other test files' backends, per this repo's existing
    per-file test-double convention."""

    name = "fixed_response_test_backend"
    supports_followup = False

    def __init__(self, symptom_text: Optional[str]):
        self._symptom_text = symptom_text

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        return {
            "symptom": self._symptom_text,
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


def test_pipeline_populates_symptom_tokens_from_dataset_match(faiss_disambiguator: FAISSDisambiguator):
    backend = _FixedResponseBackend(symptom_text="chest pain")
    case, _trail = extract_case_with_followup("chest pain", backend, disambiguator=faiss_disambiguator)
    assert "chest_pain" in case.symptom_tokens


def test_pipeline_symptom_tokens_empty_when_no_confident_dataset_match(faiss_disambiguator: FAISSDisambiguator):
    backend = _FixedResponseBackend(symptom_text="the weather is nice today")
    case, _trail = extract_case_with_followup("irrelevant", backend, disambiguator=faiss_disambiguator)
    assert case.symptom_tokens == []


def test_pipeline_symptom_tokens_stays_empty_with_stub_disambiguator():
    backend = _FixedResponseBackend(symptom_text="chest pain")
    case, _trail = extract_case_with_followup("chest pain", backend, disambiguator=StubDisambiguator())
    assert case.symptom_tokens == []


def test_pipeline_symptom_tokens_empty_when_symptom_is_none(faiss_disambiguator: FAISSDisambiguator):
    backend = _FixedResponseBackend(symptom_text=None)
    case, _trail = extract_case_with_followup("test", backend, disambiguator=faiss_disambiguator)
    assert case.symptom_tokens == []


def test_pipeline_multi_symptom_message_populates_multiple_tokens(faiss_disambiguator: FAISSDisambiguator):
    # A single free-text symptom string reporting several symptoms at once
    # (very common in real caregiver messages) must match EACH symptom
    # against the dataset vocabulary, not just whichever one dominates the
    # whole-string embedding -- see _split_symptom_clauses() in
    # app/agent1_extraction.py.
    backend = _FixedResponseBackend(symptom_text="chest pain and breathlessness and sweating")
    case, _trail = extract_case_with_followup(
        "chest pain and breathlessness and sweating", backend, disambiguator=faiss_disambiguator
    )
    assert set(case.symptom_tokens) == {"chest_pain", "breathlessness", "sweating"}


def test_pipeline_multi_symptom_comma_separated_also_splits(faiss_disambiguator: FAISSDisambiguator):
    backend = _FixedResponseBackend(symptom_text="high fever, vomiting, joint pain")
    case, _trail = extract_case_with_followup(
        "high fever, vomiting, joint pain", backend, disambiguator=faiss_disambiguator
    )
    assert set(case.symptom_tokens) == {"high_fever", "vomiting", "joint_pain"}


def test_pipeline_multi_symptom_deduplicates_tokens(faiss_disambiguator: FAISSDisambiguator):
    backend = _FixedResponseBackend(symptom_text="cough and cough and cough")
    case, _trail = extract_case_with_followup(
        "cough and cough and cough", backend, disambiguator=faiss_disambiguator
    )
    assert case.symptom_tokens == ["cough"]


def test_pipeline_dataset_tokens_feed_disease_classifier_end_to_end(faiss_disambiguator: FAISSDisambiguator):
    # Full pipeline: raw text -> extraction -> FAISS dataset match ->
    # symptom_tokens -> rules_engine.classify() routes adult age_group to
    # the dataset classifier and uses symptom_tokens as its evidence.
    from app.rules_engine import classify as rules_classify

    backend = _FixedResponseBackend(symptom_text="chest pain")
    case, _trail = extract_case_with_followup("chest pain", backend, disambiguator=faiss_disambiguator)
    result = rules_classify(case)
    assert result.probable_disease is not None
    assert result.candidates
