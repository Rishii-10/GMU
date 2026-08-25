"""
Tests for app/disambiguation.py and its wiring into
app.agent1_extraction.extract_case_with_followup().

Split the same way as the rest of this repo's test suite: fast/deterministic
checks first (StubDisambiguator, the boolean-field-isolation wiring test),
then tests against the real FAISSDisambiguator, which loads a real
sentence-transformers model. That's not a live network call once the model
is cached locally, but it isn't instantaneous either, so the disambiguator
is built ONCE per test module via a module-scoped fixture rather than once
per test.

This file is intentionally self-contained -- it does not import anything
from tests/test_agent1_followup.py, per the task instruction not to modify
that file; `_FixedResponseBackend` below is a small local double, not a
reuse of that file's `_ScriptedBackend`.
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    LLMBackend,
    apply_disambiguation_fallback,
    extract_case_with_followup,
)
from app.disambiguation import (
    DisambiguationResult,
    Disambiguator,
    FAISSDisambiguator,
    StubDisambiguator,
)
from app.schemas import CoughDifficultBreathing, DangerSigns, DiarrheaAssessment, ExtractedCase


# --- StubDisambiguator (no dependencies, always runs) ------------------------


def test_stub_disambiguator_returns_input_unchanged_with_none_confidence():
    stub = StubDisambiguator()
    result = stub.disambiguate("पेट दर्द")
    assert isinstance(result, DisambiguationResult)
    assert result.matched_category is None
    assert result.confidence is None
    assert result.original_text == "पेट दर्द"


# --- FAISSDisambiguator: real model, built once for this module -------------


@pytest.fixture(scope="module")
def faiss_disambiguator() -> FAISSDisambiguator:
    try:
        return FAISSDisambiguator()
    except ImportError as e:
        pytest.skip(f"faiss-cpu / sentence-transformers not installed: {e}")


def test_exact_match(faiss_disambiguator):
    result = faiss_disambiguator.disambiguate("cough")
    assert result.matched_category == "cough_or_difficult_breathing"
    assert result.confidence >= faiss_disambiguator.confidence_threshold
    assert result.confidence > 0.99  # near-1.0 for a literal vocabulary entry


def test_fuzzy_match_above_threshold():
    # Fresh instance (not the shared fixture) so we can also assert on the
    # instance's own confidence_threshold without coupling to other tests.
    d = FAISSDisambiguator()
    result = d.disambiguate("having a lot of trouble breathing")
    assert result.matched_category == "cough_or_difficult_breathing"
    assert result.confidence >= d.confidence_threshold


def test_below_threshold_returns_original_text_not_forced_guess(faiss_disambiguator):
    result = faiss_disambiguator.disambiguate("the sky is blue today")
    assert result.matched_category is None
    assert result.original_text == "the sky is blue today"
    # confidence is reported (even if low), not hidden -- this is what lets
    # a caller tell "we tried and found nothing good enough" (a real score)
    # apart from StubDisambiguator's "never tried" (None).
    assert result.confidence is not None
    assert result.confidence < faiss_disambiguator.confidence_threshold


def test_hinglish_cough_matches_correctly(faiss_disambiguator):
    # Romanized Hindi (Hinglish), not Devanagari script -- exercises the
    # multilingual embedding space's handling of transliterated text, not
    # just native-script Hindi.
    result = faiss_disambiguator.disambiguate("khaansi aa rahi hai")
    assert result.matched_category == "cough_or_difficult_breathing"
    assert result.confidence >= faiss_disambiguator.confidence_threshold


def test_devanagari_diarrhea_matches_correctly(faiss_disambiguator):
    result = faiss_disambiguator.disambiguate("पेट में दस्त")
    assert result.matched_category == "diarrhea"
    assert result.confidence >= faiss_disambiguator.confidence_threshold


@pytest.mark.xfail(
    reason=(
        "Observed, reproducible finding: paraphrase-multilingual-MiniLM-L12-v2 "
        "correctly matches pure-Devanagari Hindi diarrhea phrases (see "
        "test_devanagari_diarrhea_matches_correctly) and Hinglish COUGH phrases "
        "(see test_hinglish_cough_matches_correctly), but this specific "
        "romanized-Hindi (Hinglish) diarrhea phrasing gets misrouted to "
        "'cough_or_difficult_breathing' instead of 'diarrhea' -- 'saans lene "
        "mein takleef' (a cough vocabulary entry) scores higher against it than "
        "any actual diarrhea vocabulary entry does. This looks like a real "
        "weakness in how this model's multilingual embedding space handles "
        "code-switched/romanized text specifically (as opposed to native-script "
        "text, which it handles fine) -- the same category of backend-fidelity "
        "finding as the llama3.2:3b negation xfail in "
        "test_integration_agent1_pipeline.py, not a disambiguator or vocabulary "
        "bug. Worth carrying into the paper's disambiguation accuracy results "
        "rather than papering over by hand-picking vocabulary to force this "
        "exact phrase to pass."
    ),
    strict=False,
)
def test_hinglish_diarrhea_known_limitation(faiss_disambiguator):
    result = faiss_disambiguator.disambiguate("pet mein dast ho raha hai")
    assert result.matched_category == "diarrhea"


def test_deterministic_same_input_same_output(faiss_disambiguator):
    a = faiss_disambiguator.disambiguate("khaansi aa rahi hai")
    b = faiss_disambiguator.disambiguate("khaansi aa rahi hai")
    assert a.matched_category == b.matched_category
    assert a.confidence == b.confidence


# --- apply_disambiguation_fallback: pure unit tests, no model needed --------
#
# These construct DisambiguationResult directly rather than going through a
# real/stub Disambiguator -- the function under test only ever consumes the
# result object, so this exercises every case-shape deterministically and
# fast, independent of FAISSDisambiguator/StubDisambiguator's own behavior
# (which is covered separately above).


def _bare_case(symptom: str = "irrelevant", **overrides) -> ExtractedCase:
    return ExtractedCase(raw_symptom_text="raw", symptom=symptom, **overrides)


def test_fallback_no_match_is_noop():
    case = _bare_case()
    result = DisambiguationResult(original_text="x", matched_category=None, confidence=0.2)
    out = apply_disambiguation_fallback(case, result)
    assert out is case  # not even a copy -- nothing to do
    assert out.cough is None
    assert out.diarrhea is None


def test_fallback_creates_cough_block_when_block_entirely_missing():
    case = _bare_case(cough=None)
    result = DisambiguationResult(
        original_text="khaansi", matched_category="cough_or_difficult_breathing", confidence=0.9
    )
    out = apply_disambiguation_fallback(case, result)
    assert out.cough == CoughDifficultBreathing(present=True)
    assert out.cough.duration_days is None
    assert out.cough.chest_indrawing is None
    # unrelated axis and the original case object untouched
    assert out.diarrhea is None
    assert case.cough is None  # non-mutating -- original left alone


def test_fallback_creates_diarrhea_block_when_block_entirely_missing():
    case = _bare_case(diarrhea=None)
    result = DisambiguationResult(original_text="dast", matched_category="diarrhea", confidence=0.9)
    out = apply_disambiguation_fallback(case, result)
    assert out.diarrhea == DiarrheaAssessment(present=True)
    assert out.diarrhea.blood_in_stool is None
    assert out.cough is None


def test_fallback_sets_present_but_preserves_sibling_fields_when_block_partial():
    # Backend built the block (captured duration_days from context) but
    # never resolved `present` itself -- fallback must fill `present`
    # WITHOUT discarding duration_days.
    case = _bare_case(cough=CoughDifficultBreathing(present=None, duration_days=3))
    result = DisambiguationResult(
        original_text="khaansi", matched_category="cough_or_difficult_breathing", confidence=0.9
    )
    out = apply_disambiguation_fallback(case, result)
    assert out.cough.present is True
    assert out.cough.duration_days == 3  # preserved, not wiped out


def test_fallback_never_overrides_present_true():
    case = _bare_case(cough=CoughDifficultBreathing(present=True, duration_days=5))
    result = DisambiguationResult(
        original_text="khaansi", matched_category="cough_or_difficult_breathing", confidence=0.9
    )
    out = apply_disambiguation_fallback(case, result)
    assert out is case  # untouched -- backend already assessed this axis
    assert out.cough.present is True
    assert out.cough.duration_days == 5


def test_fallback_never_overrides_present_false():
    # This is the core safety case: the backend explicitly assessed cough
    # as ABSENT. A disambiguated symptom string must never flip that to
    # True, no matter how confident the match.
    case = _bare_case(cough=CoughDifficultBreathing(present=False))
    result = DisambiguationResult(
        original_text="khaansi", matched_category="cough_or_difficult_breathing", confidence=0.99
    )
    out = apply_disambiguation_fallback(case, result)
    assert out is case
    assert out.cough.present is False


def test_fallback_matched_category_never_touches_the_other_axis():
    # matched_category="diarrhea" must only ever be able to affect
    # case.diarrhea, never case.cough, even if cough is missing/eligible.
    case = _bare_case(cough=None, diarrhea=DiarrheaAssessment(present=False))
    result = DisambiguationResult(original_text="dast", matched_category="diarrhea", confidence=0.9)
    out = apply_disambiguation_fallback(case, result)
    assert out.cough is None  # not created, wrong axis
    assert out.diarrhea.present is False  # already assessed -- untouched


# --- Wiring: disambiguation must never touch boolean clinical fields --------


class _FixedResponseBackend(LLMBackend):
    """Minimal local test double -- always returns the same parsed dict.
    Deliberately not shared with test_agent1_followup.py's _ScriptedBackend
    (that file is left untouched per the task instructions)."""

    name = "fixed_response_test_backend"
    supports_followup = False

    def __init__(
        self,
        symptom_text: Optional[str],
        danger_signs: dict,
        cough: Optional[dict] = None,
        diarrhea: Optional[dict] = None,
    ):
        self._symptom_text = symptom_text
        self._danger_signs = danger_signs
        self._cough = cough
        self._diarrhea = diarrhea

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        return {
            "symptom": self._symptom_text,
            "duration": "3 days",
            "severity": "mild",
            "age_group": "child",
            "age_months": 30,
            "location": None,
            "notes": None,
            "danger_signs": dict(self._danger_signs),
            "cough": dict(self._cough) if self._cough is not None else None,
            "diarrhea": dict(self._diarrhea) if self._diarrhea is not None else None,
        }


ALL_DS_MIXED = {
    "not_able_to_drink_or_breastfeed": False,
    "vomits_everything": False,
    "convulsions": True,
    "lethargic_or_unconscious": False,
}


def test_disambiguation_never_alters_danger_signs_or_the_other_axis(faiss_disambiguator):
    # Backend never populates cough/diarrhea at all -- the block-missing
    # fallback case. symptom "khaansi" disambiguates to
    # cough_or_difficult_breathing with real confidence, so the fallback
    # SHOULD create a cough block (that's the point of this task) -- but
    # danger_signs and the unrelated diarrhea axis must stay untouched.
    backend = _FixedResponseBackend(symptom_text="khaansi", danger_signs=ALL_DS_MIXED)
    case, trail = extract_case_with_followup(
        "मेरे बच्चे को खांसी है", backend, disambiguator=faiss_disambiguator
    )
    assert case.disambiguation_confidence is not None
    assert case.symptom == "cough_or_difficult_breathing"
    # danger_signs is a different axis entirely -- disambiguation must
    # never touch it, no matter what.
    assert case.danger_signs == DangerSigns(**ALL_DS_MIXED)
    # the matched category is cough, not diarrhea -- the diarrhea axis must
    # stay exactly as the backend left it (untouched, still None).
    assert case.diarrhea is None
    # the matched, targeted axis: fallback fires because cough was never
    # assessed by the backend at all (block is None).
    assert case.cough is not None
    assert case.cough.present is True
    assert case.cough.duration_days is None  # never invented


def test_disambiguation_wired_in_even_without_followup_support():
    # backend.supports_followup=False (like RegexBackend) must still get
    # disambiguation applied -- it's on the single shared exit path, not
    # gated behind follow-up eligibility. Uses StubDisambiguator so this
    # test has no dependency on the real model.
    backend = _FixedResponseBackend(
        symptom_text="loose motions",
        danger_signs={k: None for k in ALL_DS_MIXED},
    )
    stub = StubDisambiguator()
    case, trail = extract_case_with_followup("test", backend, disambiguator=stub)
    assert trail == []
    # stub never changes symptom, but it DOES run (sets the confidence
    # field, even if to None) -- proving the wiring reaches every backend,
    # not just followup-capable ones.
    assert case.disambiguation_confidence is None
    assert case.symptom == "loose motions"  # unchanged by stub, as designed


def test_default_disambiguator_is_stub_when_none_passed():
    # No disambiguator argument at all -- must behave exactly as before
    # this task, since every pre-existing caller (including all of
    # tests/test_agent1_followup.py) relies on this default.
    backend = _FixedResponseBackend(symptom_text="fever", danger_signs={k: False for k in ALL_DS_MIXED})
    case, trail = extract_case_with_followup("test", backend)
    assert case.symptom == "fever"
    assert case.disambiguation_confidence is None


def test_disambiguation_skipped_when_symptom_is_none():
    backend = _FixedResponseBackend(symptom_text=None, danger_signs={k: False for k in ALL_DS_MIXED})
    stub = StubDisambiguator()
    case, trail = extract_case_with_followup("test", backend, disambiguator=stub)
    assert case.symptom is None
    assert case.disambiguation_confidence is None


# --- End-to-end: fallback through the real pipeline, real model -------------


def test_pipeline_never_flips_an_explicit_false_even_with_confident_match(faiss_disambiguator):
    # The core safety case, exercised through the full pipeline rather than
    # the unit-level function above: backend explicitly assessed cough as
    # ABSENT (present=False), but the caregiver's free-text symptom string
    # still says "khaansi" (e.g. reporting a past/unrelated symptom, or a
    # translation quirk). A highly confident disambiguation match must NOT
    # flip that already-assessed False to True.
    backend = _FixedResponseBackend(
        symptom_text="khaansi",
        danger_signs={k: False for k in ALL_DS_MIXED},
        cough={"present": False, "duration_days": None},
    )
    case, trail = extract_case_with_followup(
        "मेरे बच्चे को खांसी नहीं है", backend, disambiguator=faiss_disambiguator
    )
    assert case.symptom == "cough_or_difficult_breathing"
    assert case.disambiguation_confidence >= faiss_disambiguator.confidence_threshold
    assert case.cough.present is False  # untouched, exactly as the backend assessed it


def test_pipeline_backfills_present_without_discarding_sibling_fields(faiss_disambiguator):
    # Backend captured duration_days from context but never resolved
    # `present` -- the full pipeline must fill `present=True` without
    # discarding duration_days.
    backend = _FixedResponseBackend(
        symptom_text="khaansi",
        danger_signs={k: False for k in ALL_DS_MIXED},
        cough={"present": None, "duration_days": 4},
    )
    case, trail = extract_case_with_followup("test", backend, disambiguator=faiss_disambiguator)
    assert case.cough.present is True
    assert case.cough.duration_days == 4


def test_pipeline_below_threshold_does_not_backfill(faiss_disambiguator):
    backend = _FixedResponseBackend(
        symptom_text="the sky is blue today",
        danger_signs={k: False for k in ALL_DS_MIXED},
        cough=None,
    )
    case, trail = extract_case_with_followup("test", backend, disambiguator=faiss_disambiguator)
    assert case.disambiguation_confidence < faiss_disambiguator.confidence_threshold
    assert case.cough is None  # no confident match -- fallback must not fire
