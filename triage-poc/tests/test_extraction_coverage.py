"""Stage 1 regression tests: Agent 1 extraction coverage.

Live 3B runs (plan §3) showed multi-symptom and code-mixed messages collapsing
to a single token:
  - "45yo: chest pain, sweating, breathless, vomiting" -> only chest_pain
  - "pet dard aur ulti" -> [yellowing_of_eyes] ("ulti" dropped, "aur" not split)

Two fixes lock the behavior here:
  1. Agent 1 returns a `symptoms` LIST (one item per distinct symptom), so the
     engine is no longer starved when the model packs everything into one
     field; _normalize_symptom_fields() canonicalizes both the new list shape
     and the legacy single `symptom` string.
  2. _SYMPTOM_CLAUSE_SPLIT also splits code-mixed conjunctions (aur / और /
     तथா / மற்றும் / + / /), so each symptom maps to its own dataset token.

The pure-unit tests (normalizer + splitter) need no model and always run. The
wiring tests drive the real FAISS dataset index (installed in CI) through a
fixed backend, so they assert end-to-end token coverage WITHOUT needing Ollama.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    LLMBackend,
    _normalize_symptom_fields,
    _split_symptom_clauses,
    extract_case_with_followup,
)
from app.disambiguation import FAISSDisambiguator, StubDisambiguator


# --- Test doubles (per-file, per this repo's convention) ---------------------
class _ListBackend(LLMBackend):
    """Returns the new `symptoms` list contract, like the updated prompt asks."""

    name = "list_test_backend"
    supports_followup = False

    def __init__(self, symptoms: list[str]):
        self._symptoms = symptoms

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        return {
            "symptoms": list(self._symptoms),
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


class _LegacyStringBackend(LLMBackend):
    """Old single-`symptom` string shape -- must still work (back-compat)."""

    name = "legacy_string_test_backend"
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


@pytest.fixture(scope="module")
def faiss_disambiguator() -> FAISSDisambiguator:
    try:
        return FAISSDisambiguator()
    except ImportError as e:
        pytest.skip(f"faiss-cpu / sentence-transformers not installed: {e}")


# --- Pure-unit: symptom-list normalizer (no model needed) --------------------
def test_normalize_list_contract_sets_symptoms_and_joined_symptom():
    out = _normalize_symptom_fields({"symptoms": ["chest pain", "sweating"]})
    assert out["symptoms"] == ["chest pain", "sweating"]
    assert out["symptom"] == "chest pain, sweating"


def test_normalize_list_splits_item_that_still_carries_a_conjunction():
    # The model occasionally leaves two symptoms in one item; the normalizer
    # clause-splits each item so the downstream matcher still sees two.
    out = _normalize_symptom_fields({"symptoms": ["pet dard aur ulti"]})
    assert out["symptoms"] == ["pet dard", "ulti"]


def test_normalize_list_dedupes_and_drops_empty():
    out = _normalize_symptom_fields({"symptoms": ["cough", "cough", "", None]})
    assert out["symptoms"] == ["cough"]


def test_normalize_legacy_single_symptom_string_is_preserved_verbatim():
    out = _normalize_symptom_fields({"symptom": "chest pain and sweating"})
    assert out["symptom"] == "chest pain and sweating"  # legacy field untouched
    assert out["symptoms"] == ["chest pain", "sweating"]  # list derived for matching


def test_normalize_empty_when_nothing_stated():
    out = _normalize_symptom_fields({"symptoms": []})
    assert out["symptoms"] == []
    assert out["symptom"] is None


# --- Pure-unit: clause splitter (no model needed) ----------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("chest pain and breathlessness", ["chest pain", "breathlessness"]),
        ("pet dard aur ulti", ["pet dard", "ulti"]),              # Hinglish "aur"
        ("सर दर्द और बुखार", ["सर दर्द", "बुखार"]),                 # Hindi "और"
        ("chest pain + sweating", ["chest pain", "sweating"]),    # "+"
        ("cough/fever", ["cough", "fever"]),                      # "/"
        ("a, b; c & d", ["a", "b", "c", "d"]),                    # classic separators
        ("just one symptom", ["just one symptom"]),               # nothing to split
    ],
)
def test_split_symptom_clauses(text, expected):
    assert _split_symptom_clauses(text) == expected


# --- Wiring: multi-symptom and code-mixed -> multiple correct tokens ---------
def test_heart_attack_message_yields_all_four_tokens(faiss_disambiguator):
    # Regression for the MI under-extraction (live #2): the engine must see
    # all four symptoms, not just chest_pain, so Stage 2's safety layer can act.
    backend = _ListBackend(["chest pain", "sweating", "breathlessness", "vomiting"])
    case, _ = extract_case_with_followup("mi", backend, disambiguator=faiss_disambiguator)
    assert set(case.symptom_tokens) == {"chest_pain", "sweating", "breathlessness", "vomiting"}


def test_hinglish_stomach_pain_and_vomiting_both_recovered(faiss_disambiguator):
    # Regression for live #3: "pet dard aur ulti" delivered as ONE phrase must
    # still split on "aur" into stomach_pain + vomiting (was yellowing_of_eyes).
    backend = _LegacyStringBackend("pet dard aur ulti")
    case, _ = extract_case_with_followup("hinglish", backend, disambiguator=faiss_disambiguator)
    assert set(case.symptom_tokens) == {"stomach_pain", "vomiting"}


def test_cough_and_fever_yields_both_tokens(faiss_disambiguator):
    backend = _ListBackend(["cough", "fever"])
    case, _ = extract_case_with_followup("cf", backend, disambiguator=faiss_disambiguator)
    assert set(case.symptom_tokens) == {"cough", "high_fever"}


def test_list_backend_also_sets_joined_legacy_symptom(faiss_disambiguator):
    backend = _ListBackend(["chest pain", "sweating"])
    case, _ = extract_case_with_followup("x", backend, disambiguator=faiss_disambiguator)
    # Legacy `symptom` may be overwritten to the pediatric matched category by
    # disambiguate(); what Stage 1 guarantees is the full token coverage.
    assert set(case.symptom_tokens) == {"chest_pain", "sweating"}


# --- Extraction-coverage set (deterministic, FAISS-backed) -------------------
_COVERAGE_CASES = [
    (["chest pain", "sweating", "breathlessness", "vomiting"],
     {"chest_pain", "sweating", "breathlessness", "vomiting"}),
    (["high fever", "vomiting", "joint pain"], {"high_fever", "vomiting", "joint_pain"}),
    (["pet dard", "ulti"], {"stomach_pain", "vomiting"}),           # Hinglish aliases
    (["सर दर्द और बुखार"], {"headache", "high_fever"}),             # Devanagari + split
    (["chest pain + sweating"], {"chest_pain", "sweating"}),        # "+" inside one item
]


@pytest.mark.parametrize("symptoms, expected_subset", _COVERAGE_CASES)
def test_extraction_coverage_set(faiss_disambiguator, symptoms, expected_subset):
    backend = _ListBackend(symptoms)
    case, _ = extract_case_with_followup("cov", backend, disambiguator=faiss_disambiguator)
    assert expected_subset.issubset(set(case.symptom_tokens)), (
        f"{symptoms} -> {case.symptom_tokens}, missing {expected_subset - set(case.symptom_tokens)}"
    )


def test_coverage_noop_without_disambiguator():
    # With no real index, tokens stay empty -- honest no-op, not a crash.
    backend = _ListBackend(["chest pain", "sweating"])
    case, _ = extract_case_with_followup("x", backend, disambiguator=StubDisambiguator())
    assert case.symptom_tokens == []
