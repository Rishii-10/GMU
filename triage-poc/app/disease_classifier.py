"""
Dataset-driven disease classifier: symptom set -> ranked disease candidates.

Deterministic, no LLM -- same rule as app/rules_engine.py. Consumes
app.disease_kb.DiseaseKB (built from the user-supplied CSVs) and never
touches raw patient text; its only input is the normalized symptom-token
set app.disambiguation's FAISS step has already produced.

SCORING METHOD -- decision, stated explicitly
------------------------------------------------
Naive-Bayes-style posterior over diseases, estimated from per-disease
symptom-row frequencies in data/disease_symptoms.csv (~4920 rows across 41
diseases):

    P(disease | symptoms) ~= P(disease) * prod( P(symptom_i | disease) )

- P(disease): prior, estimated as (row count for this disease) / (total
  rows). Diseases with more rows in the CSV (i.e. more documented
  symptom-combination variants) get a proportionally higher prior -- this
  reflects the dataset's own row distribution, not a claim about real-world
  disease prevalence.
- P(symptom | disease): likelihood, estimated as (rows for this disease
  containing this symptom) / (rows for this disease), Laplace-smoothed
  (add-1 over the full |vocabulary| token space) so a symptom that never
  co-occurred with a disease in the CSV does not zero out that disease's
  entire posterior -- one absent-from-training symptom should reduce a
  candidate's score, not eliminate it outright. This smoothing is exactly
  what keeps scoring "unbiased/balanced across classes" per the task's test
  requirement: without it, diseases with rich/varied training rows would
  systematically out-score diseases with few rows whenever the caller's
  exact symptom combination wasn't literally in the CSV.
- Only symptoms actually reported (`case.symptom_tokens`, i.e. only
  positive evidence) are scored -- there is no "symptom absent" signal from
  Agent 1's extraction to multiply in P(not symptom | disease) for, unlike
  rules_engine.py's DangerSigns which are explicitly Optional[bool] with a
  real False state. A disease is never penalized for a symptom the patient
  simply wasn't asked about.
- Scores are normalized to sum to 1 across the returned candidates (a
  relative ranking aid), NOT a claim of calibrated real-world probability --
  same "uncalibrated, not validated" caveat that already applies to
  DEFAULT_CONFIDENCE_THRESHOLD in app/disambiguation.py.

This directly implements the user-flagged edge case: when a symptom set
overlaps several different diseases (including diseases in different
emergency tiers), the top-ranked candidate is chosen by this probabilistic
score, not by iteration order/first-match, and every candidate's score is
returned so the choice is auditable.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

from app.disease_kb import DEFAULT_DATA_DIR, DiseaseKB, normalize_disease_name, normalize_symptom_token
from app.schemas import ClassificationLabel, DiseaseCandidate

# Below this posterior share, a candidate is noise, not a real signal --
# excluded from the returned ranking entirely. Uncalibrated, same status as
# app.disambiguation.DEFAULT_CONFIDENCE_THRESHOLD: a placeholder pending a
# labelled-sample threshold sweep (see Open work).
MIN_CANDIDATE_SCORE = 0.01

# Laplace smoothing constant (add-1). See module docstring.
_LAPLACE_ALPHA = 1.0


class DiseaseClassifier:
    """Estimates a Naive-Bayes-style model once at construction from the raw
    CSV rows (not just DiseaseKB's deduplicated symptom sets -- row-level
    co-occurrence frequency is what the likelihoods are estimated from), then
    answers classify_diseases() queries as plain arithmetic, no I/O.
    """

    def __init__(self, kb: DiseaseKB, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.kb = kb
        self.vocabulary = kb.vocabulary
        self._vocab_size = len(self.vocabulary)

        rows_by_disease, total_rows = _load_symptom_rows(Path(data_dir) / "disease_symptoms.csv")
        self._rows_by_disease = rows_by_disease
        self._total_rows = total_rows

        # Precompute log-priors and log-likelihoods once; classify_diseases()
        # is then just a sum over reported symptoms per disease.
        self._log_prior: dict[str, float] = {}
        self._log_likelihood: dict[str, dict[str, float]] = {}
        for disease, rows in rows_by_disease.items():
            n_rows = len(rows)
            self._log_prior[disease] = math.log(n_rows / total_rows)

            symptom_counts: dict[str, int] = {}
            for row_symptoms in rows:
                for token in row_symptoms:
                    symptom_counts[token] = symptom_counts.get(token, 0) + 1

            likelihoods: dict[str, float] = {}
            denom = n_rows + _LAPLACE_ALPHA * self._vocab_size
            for token in self.vocabulary:
                count = symptom_counts.get(token, 0)
                likelihoods[token] = math.log((count + _LAPLACE_ALPHA) / denom)
            self._log_likelihood[disease] = likelihoods

    def classify_diseases(
        self, symptom_tokens: list[str], top_n: Optional[int] = None
    ) -> list[DiseaseCandidate]:
        """Ranks all 41 diseases by posterior given the reported symptom
        tokens (already-normalized dataset vocabulary, e.g. from FAISS
        disambiguation). Unrecognized tokens (not in self.vocabulary) are
        silently ignored -- they carry no signal this model can use, and
        raising here would make an honest "didn't recognize that symptom"
        into a pipeline failure.

        Returns candidates with score >= MIN_CANDIDATE_SCORE, most probable
        first, normalized to sum to 1 across exactly the candidates
        returned (not across all 41 diseases) -- if only two diseases clear
        the noise floor, their two scores are what's compared.

        Empty symptom_tokens -> empty list: no evidence, no ranking to
        report, rather than falling back to priors alone (which would
        silently reflect "which disease has the most CSV rows" instead of
        an honest "no signal").
        """
        vocab_set = set(self.vocabulary)
        recognized = sorted({t for t in symptom_tokens if t in vocab_set})
        if not recognized:
            return []

        log_posteriors: dict[str, float] = {}
        for disease in self.kb.diseases:
            score = self._log_prior[disease]
            likelihoods = self._log_likelihood[disease]
            for token in recognized:
                score += likelihoods.get(token, 0.0)
            log_posteriors[disease] = score

        # Normalize via log-sum-exp for numerical stability, then convert to
        # linear probabilities summing to 1 across all 41 diseases first...
        max_log = max(log_posteriors.values())
        exp_scores = {d: math.exp(s - max_log) for d, s in log_posteriors.items()}
        total = sum(exp_scores.values())
        probs = {d: v / total for d, v in exp_scores.items()}

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        # ...then filter to the noise floor and re-normalize across just the
        # surviving candidates, per the docstring above.
        surviving = [(d, p) for d, p in ranked if p >= MIN_CANDIDATE_SCORE]
        if not surviving:
            return []
        survivors_total = sum(p for _, p in surviving)

        candidates: list[DiseaseCandidate] = []
        for disease, p in surviving:
            renormalized = p / survivors_total
            matched = sorted(set(recognized) & self.kb.symptoms_for(disease))
            candidates.append(
                DiseaseCandidate(
                    name=disease,
                    score=round(renormalized, 4),
                    matched_symptoms=matched,
                    precautions=self.kb.precautions_for(disease),
                )
            )
        if top_n is not None:
            candidates = candidates[:top_n]
        return candidates


# DISEASE -> SEVERITY TIER
# --------------------------------------------------------------------------
# Turns a top-ranked disease candidate into this schema's 4-tier scale
# (EMERGENCY/SEVERE/MODERATE/MILD) so app.rules_engine.py can produce a
# triage label for the adult/out-of-band route, the same way IMNCI's fixed
# clinical rules produce a label for the pediatric route.
#
# Data-driven, not a Python dict: severity_for_disease() below reads
# app.disease_kb.DiseaseKB.severity_for(), which loads
# data/disease_severity.csv at DiseaseKB.load() time. See that CSV's
# loader (app/disease_kb.py, _load_severity) for why severity is kept as
# an editable data file rather than hardcoded in this module -- briefly: a
# naive derivation from the precaution-text CSV (e.g. "call ambulance" ->
# EMERGENCY) was tried and rejected because it silently misses genuinely
# emergency diseases whose precaution wording doesn't happen to contain an
# emergency-sounding phrase (Paralysis/brain hemorrhage being the
# concrete counterexample found). This repo ships a curated default
# covering all 41 dataset diseases in data/disease_severity.csv; replacing
# that file with better-sourced values requires no code change here.
#
# A disease with no row in the CSV (or a value that isn't one of
# EMERGENCY/SEVERE/MODERATE/MILD) falls back to
# DEFAULT_UNKNOWN_DISEASE_SEVERITY, not an extreme -- MILD would
# understate real risk, EMERGENCY would over-trigger dispatch, so the
# unknown-severity default sits in the middle.
DEFAULT_UNKNOWN_DISEASE_SEVERITY = ClassificationLabel.MODERATE

_VALID_SEVERITY_LABELS = frozenset(
    {ClassificationLabel.EMERGENCY, ClassificationLabel.SEVERE, ClassificationLabel.MODERATE, ClassificationLabel.MILD}
)


def severity_for_disease(disease_name: str, kb: Optional[DiseaseKB] = None) -> ClassificationLabel:
    """Looks up data/disease_severity.csv (via `kb`, or the shared default
    DiseaseClassifier's DiseaseKB when `kb` is omitted -- see
    get_default_classifier() below), falling back to
    DEFAULT_UNKNOWN_DISEASE_SEVERITY for a disease with no row on file, or
    a row whose value isn't one of the four valid severity tiers (guards
    specifically against "INCOMPLETE_ASSESSMENT" -- a real
    ClassificationLabel member, but a distinct state, not a severity tier;
    a malformed CSV row must not silently produce that as a "severity").

    `kb` is an explicit override, not required by any production caller
    (app.rules_engine.classify_via_dataset always omits it, using the
    shared singleton) -- it exists so severity data is genuinely swappable
    and testable: pass a DiseaseKB built from different severity data and
    this function's output changes accordingly, proving severity isn't
    baked into this function's logic.
    """
    kb = kb or get_default_classifier().kb
    raw = kb.severity_for(disease_name)
    if raw is None:
        return DEFAULT_UNKNOWN_DISEASE_SEVERITY
    try:
        label = ClassificationLabel(raw)
    except ValueError:
        return DEFAULT_UNKNOWN_DISEASE_SEVERITY
    return label if label in _VALID_SEVERITY_LABELS else DEFAULT_UNKNOWN_DISEASE_SEVERITY


_default_classifier: Optional["DiseaseClassifier"] = None


def get_default_classifier() -> "DiseaseClassifier":
    """Module-level lazy singleton: DiseaseClassifier construction loads
    and scans the ~4920-row CSV once (cheap, but not free) -- callers like
    app.rules_engine.classify() that may run many times per process should
    share one instance rather than rebuilding it per call. Mirrors the
    lazy-construction spirit of FAISSDisambiguator (heavy setup deferred to
    first real use), just without the try/except since this module's only
    dependency is the stdlib csv module."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = DiseaseClassifier(DiseaseKB.load())
    return _default_classifier


def _load_symptom_rows(path: Path) -> tuple[dict[str, list[list[str]]], int]:
    """Row-level (not deduplicated) symptom sets per disease -- this is
    what likelihood/prior estimation needs, distinct from
    DiseaseKB.symptoms_for() which returns the deduplicated union used for
    vocabulary/matched_symptoms display."""
    rows_by_disease: dict[str, list[list[str]]] = {}
    total_rows = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            disease = normalize_disease_name(row[0])
            tokens = [normalize_symptom_token(c) for c in row[1:] if c.strip()]
            rows_by_disease.setdefault(disease, []).append(tokens)
            total_rows += 1
    return rows_by_disease, total_rows
