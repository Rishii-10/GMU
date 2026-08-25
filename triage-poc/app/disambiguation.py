"""
Symptom disambiguation: maps Agent 1's free-text `symptom` string (lay
terms, multilingual) onto app.rules_engine's own controlled vocabulary of
symptom categories.

DATASET SYMPTOM-TOKEN MATCHING -- a SECOND, SEPARATE index (added
alongside the original two-category VOCABULARY/disambiguate() below, not
merged into it)
-----------------------------------------------------------------------------
The original vocabulary/index above is deliberately small (2 categories)
and stays exactly as originally built: it is what app.agent1_extraction's
apply_disambiguation_fallback() keys off to backfill case.cough.present /
case.diarrhea.present, and ~20 existing tests pin its exact matched_category
/ confidence behavior for phrases like "cough" and "khaansi aa rahi hai".

Separately, app.disease_kb.DiseaseKB.vocabulary (131 normalized symptom
tokens from data/disease_symptoms.csv) needs its own FAISS index so free
text can be matched onto the dataset's disease-classifier vocabulary and
populate ExtractedCase.symptom_tokens (app.disease_classifier consumes
that list). This is kept as a SECOND index inside FAISSDisambiguator
(`match_dataset_symptom()`), not folded into the existing VOCABULARY/
disambiguate() pair, for a concrete correctness reason: the dataset
vocabulary and the pediatric-category vocabulary overlap on some literal
surface forms (e.g. "cough" is both a pediatric-category surface form
mapping to "cough_or_difficult_breathing" AND a literal dataset token in
its own right). Merging them into one index would make disambiguate()'s
top-1 result for "cough" depend on FAISS's internal tie-breaking between
two near-identical embeddings instead of always returning the pediatric
category the existing tests (and apply_disambiguation_fallback) depend on.
Two independent indexes -- one deliberately small and pediatric-scoped,
one covering the full dataset -- avoids that collision entirely, at the
cost of one extra (still local, still free) embedding-model encode pass at
construction time.

VOCABULARY SOURCE -- decision, stated explicitly
--------------------------------------------------
No disease/symptom CSV exists anywhere in this repo (checked: no `*.csv`
file anywhere in the tree), and none is referenced by ARCHITECTURE.md.

AGENT1_README.md *describes* a `disambiguation.py` already existing with a
`FAISSDisambiguator` "correctly shaped and wired in, but stubbed with
NotImplementedError" -- that description does not match the actual
repository state. Checked directly: no `disambiguation.py` exists anywhere
in this repo's git history (`git log --all -S"FAISSDisambiguator"` matches
only the README text itself, never a source file), and it isn't on disk.
Flagging this discrepancy rather than trusting the doc, per this project's
own running principle of verifying claims against actual code.

Given no external vocabulary source exists, the vocabulary is built
directly from what app/rules_engine.py actually branches on -- checked with
`grep -n "symptom" app/rules_engine.py`, which returns zero matches.
rules_engine.py NEVER reads the free-text `symptom` field at all. Its two
symptom-shaped classification axes are driven by dedicated structured
booleans instead:
  - `classify_cough_or_difficult_breathing` reads `case.cough.present`
    (plus the exam-only chest-indrawing/stridor/breathing-rate signs)
  - `classify_diarrhea_dehydration` reads `case.diarrhea.present` (plus
    the dehydration signs)
Danger signs are their own dedicated boolean checklist
(`case.danger_signs.*`), not a symptom category to disambiguate a free-text
string into.

So the only two categories a disambiguation match can currently be *useful*
for, given the rules engine's real scope, are:
    - "cough_or_difficult_breathing"
    - "diarrhea"

"fever" is deliberately EXCLUDED even though it is extremely common in real
messages (it appeared in nearly every hand-written test case in this repo
so far) -- rules_engine.py has no fever/malaria-risk classification axis at
all yet (that's Step 2 territory, alongside the young-infant and
malnutrition charts). Including "fever" in the vocabulary would produce a
confident-looking match the rules engine cannot act on, which is exactly
the false-precision failure mode this whole exercise exists to prevent. A
message whose symptom is "fever" should honestly disambiguate to "no
confident match" today -- that is a correct reflection of a real, tracked
gap in the rules engine, not a disambiguator defect.

Each of the two categories has several curated lay-term surface forms
(English, Hindi, Hinglish) so FAISS has more than one phrasing per category
to match against. This is a small, deliberately-scoped starting list, not a
claim of vocabulary completeness -- in the same spirit as the RegexBackend
keyword table in agent1_extraction.py.

EMBEDDING MODEL -- decision, stated explicitly
------------------------------------------------
`paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, ~470MB,
384-dim). Chosen because: (a) it's free and runs fully locally, no API
calls, satisfying the zero-cost-stack constraint; (b) it's trained across
50+ languages including Hindi via multilingual knowledge distillation, so
it can compare a Hindi/Hinglish phrase against an English vocabulary entry
in the same embedding space without a separate translation step; (c) at
~470MB it sits inside the requested 100-500MB budget -- the larger
`paraphrase-multilingual-mpnet-base-v2` (~1.1GB) or `LaBSE` (~1.8GB) would
give marginally better multilingual quality but blow that budget for a POC
running on a laptop with limited RAM (the same "roughly 8GB+ RAM" resource
constraint the Implementation Plan already flags for local Ollama).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class DisambiguationResult(BaseModel):
    original_text: str
    # None means "no confident match" (FAISS: below threshold) OR
    # "disambiguation was never actually run" (StubDisambiguator) --
    # `confidence` disambiguates between those two cases (see below).
    matched_category: Optional[str] = None
    # None only when disambiguation did not run at all (StubDisambiguator).
    # A real (possibly low) float means "we tried, and this was the best
    # score" -- reported even when below threshold, so a caller can tell
    # "we tried and found nothing good enough" apart from "never tried."
    confidence: Optional[float] = None


class Disambiguator(ABC):
    """Single-method interface so future disambiguator implementations
    (a different embedding model, a hosted service, whatever) stay
    swappable behind this one call."""

    @abstractmethod
    def disambiguate(self, symptom_text: str) -> DisambiguationResult:
        """Map free-text symptom_text onto the controlled vocabulary, or
        report no confident match. Must never raise on ordinary input -- an
        honest "no match" is always preferable to a crash or a forced
        guess."""

    def match_dataset_symptom(self, symptom_text: str) -> DisambiguationResult:
        """Map free-text symptom_text onto app.disease_kb.DiseaseKB's
        dataset symptom-token vocabulary (see module docstring for why this
        is a separate index/method from disambiguate()). Concrete default
        here (not abstract) is an honest no-op -- "never actually run" --
        matching StubDisambiguator's own philosophy for disambiguate();
        implementations that can't do this (or don't want to) simply don't
        override it. FAISSDisambiguator below is the one real
        implementation."""
        return DisambiguationResult(original_text=symptom_text, matched_category=None, confidence=None)


class StubDisambiguator(Disambiguator):
    """Honest placeholder: returns the input text unchanged with
    confidence=None ("disambiguation not actually run"), never a fake
    confident answer. This is the default disambiguator in
    extract_case_with_followup() when the caller doesn't pass one, so every
    existing caller (including all tests written before this task) gets
    exactly the pre-this-task behavior: `symptom` stays as whatever the
    backend produced, `disambiguation_confidence` stays None."""

    def disambiguate(self, symptom_text: str) -> DisambiguationResult:
        return DisambiguationResult(original_text=symptom_text, matched_category=None, confidence=None)


# (surface form, canonical category recognized by rules_engine.py) pairs.
# See the module docstring for why only these two categories exist and why
# "fever" is deliberately absent.
VOCABULARY: list[tuple[str, str]] = [
    ("cough", "cough_or_difficult_breathing"),
    ("coughing", "cough_or_difficult_breathing"),
    ("cold and cough", "cough_or_difficult_breathing"),
    ("difficult breathing", "cough_or_difficult_breathing"),
    ("difficulty breathing", "cough_or_difficult_breathing"),
    ("breathing problem", "cough_or_difficult_breathing"),
    ("shortness of breath", "cough_or_difficult_breathing"),
    ("chest congestion", "cough_or_difficult_breathing"),
    ("trouble breathing", "cough_or_difficult_breathing"),
    ("खांसी", "cough_or_difficult_breathing"),
    ("खाँसी", "cough_or_difficult_breathing"),
    ("साँस लेने में तकलीफ", "cough_or_difficult_breathing"),
    ("साँस फूलना", "cough_or_difficult_breathing"),
    ("khaansi", "cough_or_difficult_breathing"),
    ("saans lene mein takleef", "cough_or_difficult_breathing"),
    ("diarrhea", "diarrhea"),
    ("diarrhoea", "diarrhea"),
    ("loose motions", "diarrhea"),
    ("loose motion", "diarrhea"),
    ("watery stool", "diarrhea"),
    ("watery stools", "diarrhea"),
    ("loose potty", "diarrhea"),
    ("दस्त", "diarrhea"),
    ("पतले दस्त", "diarrhea"),
    ("दस्त लग रहे हैं", "diarrhea"),
    ("dast", "diarrhea"),
    ("pet mein dast", "diarrhea"),
]

# UNCALIBRATED. Carried over from the Implementation Plan's original (also
# uncalibrated) 0.65 suggestion for FAISS confidence -- it has not been
# validated against any labelled sample. Calibrating it (Milestone 3a in
# the plan) requires a labelled dataset of symptom-text -> correct-category
# pairs and a threshold sweep prioritising minimum false negatives; neither
# exists in this repo yet. Do not present this default as validated, and do
# not quote it in the paper as a calibrated result.
DEFAULT_CONFIDENCE_THRESHOLD = 0.65


# Curated Hindi/Hinglish lay-term aliases for the dataset symptom tokens a
# rural caregiver is most likely to actually say -- a deliberately small
# starting list (same "not a claim of completeness" status as VOCABULARY
# above and the RegexBackend keyword table in agent1_extraction.py), not an
# attempt to cover all 131 tokens. Each entry maps a lay surface form to the
# exact normalized dataset token (app.disease_kb.normalize_symptom_token
# output) it should resolve to.
_DATASET_LAY_TERM_ALIASES: list[tuple[str, str]] = [
    ("bukhar", "high_fever"),
    ("बुखार", "high_fever"),
    ("tez bukhar", "high_fever"),
    ("halka bukhar", "mild_fever"),
    ("khaansi", "cough"),
    ("खांसी", "cough"),
    ("saans phoolna", "breathlessness"),
    ("साँस फूलना", "breathlessness"),
    ("ulti", "vomiting"),
    ("उल्टी", "vomiting"),
    ("dast", "diarrhoea"),
    ("दस्त", "diarrhoea"),
    ("loose motion", "diarrhoea"),
    ("sar dard", "headache"),
    ("सर दर्द", "headache"),
    ("chhati mein dard", "chest_pain"),
    ("छाती में दर्द", "chest_pain"),
    ("pet dard", "stomach_pain"),
    ("पेट दर्द", "stomach_pain"),
    ("badan dard", "muscle_pain"),
    ("बदन दर्द", "muscle_pain"),
    ("jodo mein dard", "joint_pain"),
    ("जोड़ों में दर्द", "joint_pain"),
    ("kamar dard", "back_pain"),
    ("कमर दर्द", "back_pain"),
    ("kamzori", "weakness_in_limbs"),
    ("कमज़ोरी", "weakness_in_limbs"),
    ("thakan", "fatigue"),
    ("थकान", "fatigue"),
    ("chakkar aana", "dizziness"),
    ("चक्कर आना", "dizziness"),
    ("jee michlana", "nausea"),
    ("जी मिचलाना", "nausea"),
    ("thand lagna", "chills"),
    ("ठंड लगना", "chills"),
    ("paseena aana", "sweating"),
    ("पसीना आना", "sweating"),
    ("khujli", "itching"),
    ("खुजली", "itching"),
    ("charmrog", "skin_rash"),
    ("chamdi par dane", "skin_rash"),
    ("aankhen peeli", "yellowing_of_eyes"),
    ("आँखें पीली", "yellowing_of_eyes"),
    ("bhookh na lagna", "loss_of_appetite"),
    ("भूख न लगना", "loss_of_appetite"),
]


def _build_dataset_vocabulary(kb: Optional["DiseaseKB"] = None) -> list[tuple[str, str]]:  # noqa: F821 -- DiseaseKB imported lazily below
    """Builds the (surface form, dataset symptom token) pairs FAISS embeds
    for match_dataset_symptom(). For every token in DiseaseKB.vocabulary
    (131 tokens from data/disease_symptoms.csv), adds both a "humanized"
    surface form (underscores -> spaces, e.g. "high fever") and the raw
    snake_case token itself (e.g. "high_fever") as separate entries mapping
    to that same token -- so a caregiver's own snake_case-shaped phrasing
    (unlikely) and natural phrasing both score well, and an exact-token
    query embeds near-identically to its own vocabulary entry. Then appends
    the curated Hindi/Hinglish aliases above.

    Lazy import of app.disease_kb here (not at module top) keeps this
    module's only hard dependency at import time being pydantic, matching
    the existing "heavy stuff imported lazily" pattern in this file --
    disease_kb itself is lightweight (stdlib csv only) but there is no
    reason to force the import graph at module load for callers who only
    want the original two-category disambiguate().
    """
    from app.disease_kb import DiseaseKB as _DiseaseKB

    kb = kb or _DiseaseKB.load()
    entries: list[tuple[str, str]] = []
    for token in kb.vocabulary:
        humanized = token.replace("_", " ")
        entries.append((humanized, token))
        if humanized != token:
            entries.append((token, token))
    entries.extend(_DATASET_LAY_TERM_ALIASES)
    return entries


# CALIBRATED (unlike DEFAULT_CONFIDENCE_THRESHOLD above, which has no
# labelled sample to calibrate against yet): this is the actual output of
# calibrate_dataset_threshold() run against
# tests/fixtures/symptom_calibration.py's 80-item labelled sample --
# 97.5% accuracy at 0.45 vs. 93.8% at the previous hand-picked 0.65 (full
# sweep: tests/test_calibration.py::test_shipped_default_matches_calibration).
# Reproducible, not a guess: re-run calibrate_dataset_threshold() against
# that fixture (or a larger/better labelled sample, should one become
# available) to verify or update this value -- it is not meant to be
# hand-edited independently of that computation.
DEFAULT_DATASET_CONFIDENCE_THRESHOLD = 0.45


class FAISSDisambiguator(Disambiguator):
    """Real implementation: embeds VOCABULARY once at construction into a
    flat (exact, not approximate) FAISS index, then embeds each query
    symptom_text at call time and does a cosine-similarity nearest-neighbor
    search against it.

    Deterministic: SentenceTransformer.encode() runs the model in eval mode
    (no dropout, no sampling) so the same input always produces the same
    embedding, and IndexFlatIP does an exact nearest-neighbor search (no
    approximate/randomized index structure) -- same input + same index
    always gives the same output.

    faiss / sentence-transformers / numpy are imported lazily, inside
    __init__, specifically so that importing this module -- or using
    StubDisambiguator, or the Disambiguator ABC -- never requires these
    (large, ~470MB+) dependencies to be installed. Only code that actually
    constructs a FAISSDisambiguator needs them.
    """

    def __init__(
        self,
        vocabulary: Optional[list[tuple[str, str]]] = None,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        enable_dataset_matching: bool = True,
        dataset_vocabulary: Optional[list[tuple[str, str]]] = None,
        dataset_confidence_threshold: float = DEFAULT_DATASET_CONFIDENCE_THRESHOLD,
    ):
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "FAISSDisambiguator requires faiss-cpu and sentence-transformers "
                "(see requirements.txt). Use StubDisambiguator if these aren't "
                "installed, or install them and retry."
            ) from e

        self._np = np
        self.vocabulary = list(vocabulary) if vocabulary is not None else list(VOCABULARY)
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        self._model = SentenceTransformer(model_name)

        surface_forms = [surface for surface, _category in self.vocabulary]
        embeddings = self._model.encode(surface_forms, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        # Cosine similarity via inner product on L2-normalized vectors.
        # Flat = exact search, not an approximate/randomized index -- this
        # is what makes results deterministic and, at this vocabulary size
        # (tens of entries), exact search costs nothing worth trading away.
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        # Second, independent index over the dataset symptom-token
        # vocabulary -- see module docstring for why this is kept separate
        # from self._index/self.vocabulary above rather than merged.
        self.dataset_confidence_threshold = dataset_confidence_threshold
        self.dataset_vocabulary: list[tuple[str, str]] = []
        self._dataset_index = None
        if enable_dataset_matching:
            self.dataset_vocabulary = (
                list(dataset_vocabulary) if dataset_vocabulary is not None else _build_dataset_vocabulary()
            )
            dataset_surface_forms = [surface for surface, _token in self.dataset_vocabulary]
            dataset_embeddings = self._model.encode(dataset_surface_forms, normalize_embeddings=True)
            dataset_embeddings = np.asarray(dataset_embeddings, dtype="float32")
            self._dataset_index = faiss.IndexFlatIP(dataset_embeddings.shape[1])
            self._dataset_index.add(dataset_embeddings)

    def disambiguate(self, symptom_text: str) -> DisambiguationResult:
        np = self._np
        query = self._model.encode([symptom_text], normalize_embeddings=True)
        query = np.asarray(query, dtype="float32")
        scores, indices = self._index.search(query, 1)
        top_score = float(scores[0][0])
        top_idx = int(indices[0][0])

        if top_idx < 0 or top_score < self.confidence_threshold:
            # Below threshold: report the original text unchanged and the
            # real (low) score -- NOT a forced best-guess match. A wrong
            # confident match is worse than an honest "couldn't
            # disambiguate" in a clinical triage pipeline.
            return DisambiguationResult(original_text=symptom_text, matched_category=None, confidence=top_score)

        return DisambiguationResult(
            original_text=symptom_text,
            matched_category=self.vocabulary[top_idx][1],
            confidence=top_score,
        )

    def raw_dataset_top1(self, symptom_text: str) -> tuple[Optional[str], Optional[float]]:
        """Unthresholded top-1 nearest-neighbor search against the dataset
        symptom-token index: always returns the best match and its score,
        with NO accept/reject decision applied -- that decision
        (score >= self.dataset_confidence_threshold) is exactly what
        match_dataset_symptom() layers on top of this. Split out
        specifically so calibrate_dataset_threshold() below can compute
        every candidate threshold's accuracy from ONE pass over a labelled
        sample (one embedding-model call per sample item) rather than
        re-running the model once per threshold value swept.

        Returns (None, None) if this instance was constructed with
        enable_dataset_matching=False -- an honest "not available."
        """
        if self._dataset_index is None:
            return None, None
        np = self._np
        query = self._model.encode([symptom_text], normalize_embeddings=True)
        query = np.asarray(query, dtype="float32")
        scores, indices = self._dataset_index.search(query, 1)
        top_score = float(scores[0][0])
        top_idx = int(indices[0][0])
        if top_idx < 0:
            return None, top_score
        return self.dataset_vocabulary[top_idx][1], top_score

    def match_dataset_symptom(self, symptom_text: str) -> DisambiguationResult:
        """Top-1 nearest-neighbor search against the dataset symptom-token
        index (see module docstring / __init__ for why this is a separate
        index from disambiguate()'s). `matched_category` here is a
        normalized dataset token (e.g. "high_fever"), suitable for direct
        use as an ExtractedCase.symptom_tokens entry -- not a pediatric
        rules-engine category.

        Returns a no-match result (matched_category=None, confidence=None)
        if this instance was constructed with enable_dataset_matching=False
        -- an honest "not available," not a crash or a forced guess.
        Below-threshold behavior mirrors disambiguate(): the real (low)
        score is still reported, distinguishing "tried and found nothing
        good enough" from "never tried."
        """
        token, top_score = self.raw_dataset_top1(symptom_text)
        if token is None or top_score is None:
            return DisambiguationResult(original_text=symptom_text, matched_category=None, confidence=top_score)
        if top_score < self.dataset_confidence_threshold:
            return DisambiguationResult(original_text=symptom_text, matched_category=None, confidence=top_score)
        return DisambiguationResult(original_text=symptom_text, matched_category=token, confidence=top_score)


class ThresholdCalibrationResult(BaseModel):
    """One candidate threshold's measured performance against a labelled
    sample -- see calibrate_dataset_threshold() below."""

    threshold: float
    accuracy: float
    positive_recall: float
    negative_precision: float
    sample_size: int


def calibrate_dataset_threshold(
    labelled_sample: list[tuple[str, Optional[str]]],
    disambiguator: "FAISSDisambiguator",
    candidate_thresholds: Optional[list[float]] = None,
) -> tuple[float, list[ThresholdCalibrationResult]]:
    """Real, complete calibration: sweeps `candidate_thresholds` against
    `labelled_sample` (a list of (text, expected_token_or_None) pairs --
    see tests/fixtures/symptom_calibration.py for the shipped fabricated
    sample) and returns (best_threshold, every_candidate's_metrics).

    This is what turns DEFAULT_DATASET_CONFIDENCE_THRESHOLD from a
    permanently-uncalibrated guess into something a caller can actually
    compute an evidence-based value for -- not just measure accuracy at a
    fixed number and stop there. It does NOT mutate
    DEFAULT_DATASET_CONFIDENCE_THRESHOLD, `disambiguator`, or any other
    module/instance state -- purely a read-only measurement over
    `disambiguator`'s raw_dataset_top1() (unthresholded, see that method)
    -- so shipping a different default remains an explicit, separate
    decision a caller makes by constructing
    FAISSDisambiguator(dataset_confidence_threshold=best_threshold), not
    something this function does on its own.

    Efficient: raw_dataset_top1() (one embedding-model call per sample
    item) runs exactly once per sample item, regardless of how many
    threshold candidates are swept -- accept/reject at each candidate
    threshold is then pure Python arithmetic over the cached raw scores.

    `candidate_thresholds` defaults to 0.05-spaced values from 0.30 to
    0.95 inclusive. best_threshold is the candidate with the highest
    accuracy; ties are broken toward the LOWER threshold (higher recall)
    -- an explicit, stated policy: in a clinical triage context, missing a
    real symptom (false negative) is judged worse than one extra
    questionable match (false positive), so a tie leans permissive rather
    than picking an arbitrary/first-seen value.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30..0.95

    raw: list[tuple[Optional[str], Optional[float], Optional[str]]] = [
        (*disambiguator.raw_dataset_top1(text), expected) for text, expected in labelled_sample
    ]

    results: list[ThresholdCalibrationResult] = []
    for threshold in candidate_thresholds:
        correct = 0
        positive_total = positive_correct = 0
        negative_total = negative_correct = 0
        for token, score, expected in raw:
            predicted = token if (token is not None and score is not None and score >= threshold) else None
            is_correct = predicted == expected
            correct += int(is_correct)
            if expected is not None:
                positive_total += 1
                positive_correct += int(is_correct)
            else:
                negative_total += 1
                negative_correct += int(is_correct)
        results.append(
            ThresholdCalibrationResult(
                threshold=threshold,
                accuracy=correct / len(raw),
                positive_recall=(positive_correct / positive_total) if positive_total else 1.0,
                negative_precision=(negative_correct / negative_total) if negative_total else 1.0,
                sample_size=len(raw),
            )
        )

    best = max(results, key=lambda r: (r.accuracy, -r.threshold))
    return best.threshold, results
