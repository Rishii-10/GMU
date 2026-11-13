"""Denied symptoms in the FIRST message stay denied (paper §3: three states).

Live llama3.2:3b (2026-11-13) dropped English denials ("fever but no cough" ->
["fever"]) and copied the noun out of Hinglish ones ("ulti nahi hai" -> ["ulti"]
-> vomiting PRESENT), and the matcher alone maps "no cough" -> cough at 0.879.
These tests lock:
  - app.negation.negated_core: what counts as a denial (and that inabilities
    such as "not able to drink" never do);
  - app.negation.split_denials: moving/recovering denials from the raw text,
    and dropping LLM denials that are not grounded in the message;
  - extract_case -> _apply_disambiguation: denials reach
    `absent_symptom_tokens`, never `symptom_tokens`, and present wins on conflict.
Pure-unit tests need no model; the wiring tests use the FAISS index (no Ollama).
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    LLMBackend,
    _apply_disambiguation,
    _apply_negation_scan,
    _normalize_symptom_fields,
    extract_case,
)
from app.disambiguation import DisambiguationResult, Disambiguator, FAISSDisambiguator
from app.negation import negated_core, split_denials
from app.rules_engine import classify


class _FixedBackend(LLMBackend):
    name = "fixed_test_double"

    def __init__(self, symptoms, denied=None, age_group="adult"):
        self._symptoms, self._denied, self._age = symptoms, denied, age_group

    def extract(self, raw_text: str, context: Optional[list[str]] = None) -> dict:
        out = {"symptoms": list(self._symptoms), "age_group": self._age}
        if self._denied is not None:
            out["denied_symptoms"] = list(self._denied)
        return out


class _DictDisambiguator(Disambiguator):
    """Deterministic matcher: phrase -> (token, confidence)."""

    def __init__(self, table):
        self.table = table

    def disambiguate(self, symptom_text: str) -> DisambiguationResult:
        return DisambiguationResult(original_text=symptom_text)

    def match_dataset_symptom(self, symptom_text: str) -> DisambiguationResult:
        tok, conf = self.table.get(symptom_text.strip().lower(), (None, 0.0))
        return DisambiguationResult(original_text=symptom_text, matched_category=tok, confidence=conf)


@pytest.fixture(scope="module")
def faiss_disambiguator() -> FAISSDisambiguator:
    try:
        return FAISSDisambiguator()
    except ImportError as e:
        pytest.skip(f"faiss-cpu / sentence-transformers not installed: {e}")


# --- negated_core ------------------------------------------------------------
@pytest.mark.parametrize(
    "clause, core",
    [
        ("no cough", "cough"),
        ("not breathless", "breathless"),
        ("no sweating", "sweating"),
        ("denies chest pain", "chest pain"),
        ("he has no fever", "fever"),
        ("there is no rash", "rash"),
        ("does not have diarrhoea", "diarrhoea"),
        ("he is not vomiting", "vomiting"),
        ("not having headache", "headache"),
        ("no more fever", "fever"),
        ("never had seizures", "seizures"),
        ("ulti nahi hai", "ulti"),
        ("koi dard nahi", "dard"),
        ("bukhar nahi", "bukhar"),
        ("उल्टी नहीं है", "उल्टी"),
    ],
)
def test_denials_are_recognised(clause, core):
    assert negated_core(clause) == core


@pytest.mark.parametrize(
    "clause",
    [
        # inabilities and losses of function are SYMPTOMS, never denials
        "not able to drink",
        "cannot breathe",
        "not breathing",
        "not feeding well",
        "not eating",
        "is not drinking",
        "not feeling well",
        "no appetite",
        "khana nahi kha raha",
        "bhookh nahi lagti",
        "neend nahi aati",
        # plain positives
        "vomiting",
        "fever",
        "pet dard",
        "vomits everything",
    ],
)
def test_inabilities_and_positives_are_not_denials(clause):
    assert negated_core(clause) is None


# --- split_denials -------------------------------------------------------------
def test_dropped_english_denial_is_recovered():
    # the live model returned only ["fever"] for this message
    assert split_denials("I am 40, I have fever but no cough", ["fever"]) == (["fever"], ["cough"])


def test_hinglish_noun_copied_out_of_a_denial_is_moved_to_denied():
    # the live model returned ["dard", "ulti"]; "ulti" was read as vomiting PRESENT
    present, denied = split_denials("35 saal, sir dard hai, ulti nahi hai", ["dard", "ulti"])
    assert present == ["dard"] and denied == ["ulti"]


def test_negated_item_in_symptom_list_is_moved():
    assert split_denials("fever", ["no cough", "fever"]) == (["fever"], ["cough"])


def test_ungrounded_llm_denials_are_dropped():
    # the live model copied the prompt's example denials into this message
    _, denied = split_denials(
        "35 saal, sir dard hai, ulti nahi hai", ["sir dard"], ["cough", "breathless", "sweating", "ulti"]
    )
    assert denied == ["ulti"]


def test_grounded_llm_denial_is_kept():
    _, denied = split_denials("chest pain, not breathless", ["chest pain"], ["breathlessness"])
    assert denied == ["breathlessness"]


def test_present_wins_when_both_negated_and_reported():
    present, denied = split_denials("no fever now but cough and fever yesterday", ["cough", "fever"])
    assert "fever" in present and "fever" not in denied


def test_no_negation_leaves_list_unchanged():
    assert split_denials("I am 30, headache and vomiting", ["headache", "vomiting"]) == (
        ["headache", "vomiting"],
        [],
    )


def test_inability_is_kept_present():
    assert split_denials("baby not able to drink, loose motions", ["not able to drink", "loose motions"]) == (
        ["not able to drink", "loose motions"],
        [],
    )


def test_negation_scan_rebuilds_legacy_symptom_string_only_when_changed():
    parsed = _normalize_symptom_fields({"symptoms": ["fever", "ulti"]})
    out = _apply_negation_scan("fever, ulti nahi hai", parsed)
    assert out["symptoms"] == ["fever"] and out["symptom"] == "fever"
    assert out["denied_symptoms"] == ["ulti"]
    same = _apply_negation_scan("fever and ulti", _normalize_symptom_fields({"symptoms": ["fever", "ulti"]}))
    assert same["symptoms"] == ["fever", "ulti"] and same["symptom"] == "fever, ulti"


# --- wiring: denials reach absent_symptom_tokens only ----------------------------
_TABLE = {
    "fever": ("high_fever", 0.98),
    "cough": ("cough", 1.0),
    "ulti": ("vomiting", 1.0),
    "sir dard": ("headache", 1.0),
    "pain": ("joint_pain", 0.858),
}


def test_denial_goes_to_absent_tokens_not_present():
    case = extract_case("I have fever but no cough", _FixedBackend(["fever"]))
    case = _apply_disambiguation(case, _DictDisambiguator(_TABLE))
    assert case.symptom_tokens == ["high_fever"]
    assert case.absent_symptom_tokens == ["cough"]


def test_low_confidence_denial_is_not_recorded():
    # a wrong ABSENT token can rule a disease out, so it needs >= 0.90
    case = extract_case("headache, no pain", _FixedBackend(["headache"]))
    case = _apply_disambiguation(case, _DictDisambiguator({**_TABLE, "headache": ("headache", 1.0)}))
    assert case.absent_symptom_tokens == []


def test_token_both_present_and_denied_stays_present_only():
    case = extract_case("cough, no cough", _FixedBackend(["cough"], denied=["cough"]))
    case = _apply_disambiguation(case, _DictDisambiguator(_TABLE))
    assert "cough" in case.symptom_tokens and "cough" not in case.absent_symptom_tokens


def test_denial_reaches_the_classifier():
    with_denial = _apply_disambiguation(
        extract_case("fever and cough, ulti nahi hai", _FixedBackend(["fever", "cough", "ulti"])),
        _DictDisambiguator(_TABLE),
    )
    assert with_denial.absent_symptom_tokens == ["vomiting"]
    assert "vomiting" not in with_denial.symptom_tokens
    # the result is computed from present + absent evidence without error
    assert classify(with_denial).label is not None


@pytest.mark.parametrize(
    "raw, llm_symptoms, present, absent",
    [
        ("I am 40, I have fever but no cough", ["fever"], ["high_fever"], ["cough"]),
        ("45 year old man, chest pain, not breathless, no sweating", ["chest pain"], ["chest_pain"], ["breathlessness", "sweating"]),
        ("35 saal, sir dard hai, ulti nahi hai", ["sir dard", "ulti"], ["headache"], ["vomiting"]),
    ],
)
def test_live_failure_messages_end_to_end_with_faiss(faiss_disambiguator, raw, llm_symptoms, present, absent):
    """The exact LLM outputs measured live, through the real FAISS index."""
    case = _apply_disambiguation(extract_case(raw, _FixedBackend(llm_symptoms)), faiss_disambiguator)
    assert sorted(case.absent_symptom_tokens) == sorted(absent)
    for tok in absent:
        assert tok not in case.symptom_tokens
    for tok in present:
        assert tok in case.symptom_tokens


@pytest.mark.parametrize("alias, token", [("sir dard", "headache"), ("khansi", "cough"), ("khasi", "cough")])
def test_new_hinglish_aliases(faiss_disambiguator, alias, token):
    assert faiss_disambiguator.match_dataset_symptom(alias).matched_category == token
