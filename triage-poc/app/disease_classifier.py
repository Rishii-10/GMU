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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.disease_kb import DEFAULT_DATA_DIR, DiseaseKB, normalize_disease_name, normalize_symptom_token
from app.schemas import ClassificationLabel, DiseaseCandidate

# Below this posterior share, a candidate is noise, not a real signal.
MIN_CANDIDATE_SCORE = 0.01

# Laplace smoothing constant (add-1). See module docstring.
_LAPLACE_ALPHA = 1.0

# Calibration/threshold derivation: stratified k-fold cross-validation over
# UNIQUE (disease, symptom-set) profiles, not raw duplicated CSV rows --
# see _fit_calibration_and_thresholds()'s docstring for why a row-level
# split leaks (data/disease_symptoms.csv repeats the same profile up to
# 90x) and why that leakage, not a hand-pickable constant, was the actual
# bug behind an earlier τ=1.0 threshold collapse.
#
# k=5 chosen because: every one of the 41 diseases has between 5 and 10
# unique profiles (checked directly against the CSV -- min=5 for Fungal
# infection, max=10 for Hepatitis D, mean=7.4, 304 unique profiles total).
# k=5 is the largest fold count where even the sparsest disease (5
# profiles) still contributes at least one profile to every fold's
# TRAINING side (5 profiles / 5 folds = exactly 1 held out per fold, 4
# retained) -- no fold ever trains on zero rows for any disease. It also
# keeps each fold's holdout share close to the prior single-split's 20%,
# the ratio _TARGET_SELECTIVE_ERROR/_MIN_COVERAGE below were already
# chosen relative to. k=10 was considered and rejected: with a minimum of
# 5 profiles per disease, most (disease, fold) combinations would hold out
# zero profiles for that disease, unevenly thinning the aggregated
# evaluation sample without adding real held-out diversity given how few
# unique profiles exist in total (304).
_CALIBRATION_N_FOLDS = 5
_CALIBRATION_SEED = 42

# Risk-coverage operating point: target selective error ≤ 3 % (30 in 1000).
# The δ chosen is the smallest gap threshold that keeps selective error below
# this target while maintaining coverage ≥ 60 %.  Both τ and δ are derived
# from held-out calibration data; they are not hand-picked.
_TARGET_SELECTIVE_ERROR = 0.03
_MIN_COVERAGE = 0.60


def _fit_naive_bayes(
    rows_by_disease: dict[str, list[list[str]]],
    vocabulary: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Estimate log-priors and Laplace-smoothed log-likelihoods from a set
    of rows (row-level, i.e. duplication-weighted -- see module docstring
    for why the PRODUCTION model deliberately keeps this weighting).

    Factored out of DiseaseClassifier.__init__() so the exact same
    estimation procedure can be re-run, scoped to only one fold's training
    rows, during calibration fitting (_fit_calibration_and_thresholds())
    without duplicating the math or letting the two drift out of sync.

    `total_rows` is derived internally from `rows_by_disease` (not passed
    in) so a caller scoping this to a training-fold subset can never
    accidentally pass a mismatched total.

    A disease with zero rows in `rows_by_disease` (not present as a key,
    or present with an empty list) is silently excluded from the returned
    tables rather than raising on log(0) -- defensive: given this
    project's k=5 fold count and the minimum 5-profile-per-disease floor
    (see _CALIBRATION_N_FOLDS's comment), no disease should ever actually
    lose all its rows in one fold, but a data change that violated that
    assumption should degrade to "this disease can't win this fold" rather
    than crash classifier construction.
    """
    log_prior: dict[str, float] = {}
    log_likelihood: dict[str, dict[str, float]] = {}
    vocab_size = len(vocabulary)
    total_rows = sum(len(rows) for rows in rows_by_disease.values())
    for disease, rows in rows_by_disease.items():
        n_rows = len(rows)
        if n_rows == 0 or total_rows == 0:
            continue
        log_prior[disease] = math.log(n_rows / total_rows)

        symptom_counts: dict[str, int] = {}
        for row_symptoms in rows:
            for token in row_symptoms:
                symptom_counts[token] = symptom_counts.get(token, 0) + 1

        likelihoods: dict[str, float] = {}
        denom = n_rows + _LAPLACE_ALPHA * vocab_size
        for token in vocabulary:
            count = symptom_counts.get(token, 0)
            likelihoods[token] = math.log((count + _LAPLACE_ALPHA) / denom)
        log_likelihood[disease] = likelihoods
    return log_prior, log_likelihood


def _nb_posteriors(
    recognized_tokens: list[str],
    diseases: list[str],
    log_prior: dict[str, float],
    log_likelihood: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Softmax the log-posteriors for `diseases` given `recognized_tokens`,
    against the supplied (possibly fold-restricted) log_prior/
    log_likelihood tables. Shared by DiseaseClassifier._raw_posteriors()
    (production, full-data model) and _fit_calibration_and_thresholds()'s
    per-fold scoring (fold-restricted model) so the two can never silently
    drift out of sync with each other."""
    if not diseases:
        return {}
    log_posts: dict[str, float] = {}
    for disease in diseases:
        score = log_prior[disease]
        likelihoods = log_likelihood[disease]
        for token in recognized_tokens:
            score += likelihoods.get(token, 0.0)
        log_posts[disease] = score
    max_log = max(log_posts.values())
    exp_scores = {d: math.exp(s - max_log) for d, s in log_posts.items()}
    total = sum(exp_scores.values())
    return {d: v / total for d, v in exp_scores.items()}


def _unique_profiles_by_disease(
    rows_by_disease: dict[str, list[list[str]]],
) -> dict[str, list[tuple[str, ...]]]:
    """Collapse each disease's (possibly heavily duplicated) rows to its
    distinct sorted symptom-token tuples, order-preserving on first
    occurrence (stable for the deterministic seeded shuffle downstream).

    Why this exists: data/disease_symptoms.csv repeats the same (disease,
    symptom-set) combination up to 90 times (checked directly -- only 304
    unique combinations across 4,920 rows). Treating raw rows as 4,920
    independent samples for calibration purposes massively overweights
    whichever profiles happen to be duplicated most, and -- more
    importantly -- lets a row-level train/holdout split leak near-exact
    duplicates across the split boundary. See
    _fit_calibration_and_thresholds()'s docstring for the full mechanism.
    """
    profiles_by_disease: dict[str, list[tuple[str, ...]]] = {}
    for disease, rows in rows_by_disease.items():
        seen: set[tuple[str, ...]] = set()
        ordered: list[tuple[str, ...]] = []
        for row in rows:
            profile = tuple(sorted(row))
            if profile not in seen:
                seen.add(profile)
                ordered.append(profile)
        profiles_by_disease[disease] = ordered
    return profiles_by_disease


@dataclass
class DiagnosisOutput:
    """Full Stage-1 output from classify_with_abstention()."""
    candidates: list[DiseaseCandidate]
    calibrated_confidence: float    # isotonic-calibrated P(top is correct)
    raw_confidence: float           # raw NB posterior for top disease
    gap_to_second: float            # top - second calibrated posterior
    abstain: bool                   # True = engine refuses to answer
    abstain_reason: Optional[str]   # human-readable reason
    tau: float                      # Youden-J-derived threshold used
    delta: float                    # risk-coverage-derived gap used


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

        # Precompute log-priors and log-likelihoods once, from ALL rows
        # (row-level duplication weighting preserved -- see module
        # docstring); classify_diseases() is then just a sum over reported
        # symptoms per disease. This PRODUCTION model is unaffected by the
        # calibration-leakage fix below -- only the abstention thresholds
        # derived FROM it change, not this estimation itself.
        self._log_prior, self._log_likelihood = _fit_naive_bayes(rows_by_disease, self.vocabulary)

        # Fit isotonic calibration and derive abstention thresholds via
        # leak-free k-fold cross-validation over unique profiles. Done once
        # at construction. See _fit_calibration_and_thresholds().
        self._calibrator, self._tau, self._delta = self._fit_calibration_and_thresholds()

    # ------------------------------------------------------------------
    # Calibration and threshold derivation
    # ------------------------------------------------------------------
    def _raw_posteriors(self, symptom_tokens: list[str]) -> dict[str, float]:
        """Return raw NB posteriors (sum to 1) for a given symptom token list."""
        vocab_set = set(self.vocabulary)
        recognized = sorted({t for t in symptom_tokens if t in vocab_set})
        if not recognized:
            return {}
        return _nb_posteriors(recognized, self.kb.diseases, self._log_prior, self._log_likelihood)

    def _fit_calibration_and_thresholds(self):
        """Fit isotonic calibration and derive τ (Youden's J) and δ
        (risk-coverage) via leak-free stratified k-fold cross-validation
        over UNIQUE (disease, symptom-set) profiles.

        WHY (superseding the prior single 80/20-row-split approach): this
        codebase's disease_symptoms.csv has 4,920 rows but only 304 unique
        (disease, symptom-set) combinations -- 94% of rows are exact
        duplicates of another row for the same disease (checked directly;
        one profile repeats up to 90 times). A row-level 80/20 split leaks
        near-identical duplicate rows across the train/holdout boundary:
        ~95% of held-out rows turned out to be memorized copies scoring
        ~1.0 raw confidence with near-perfect accuracy, while genuinely
        novel-to-training profiles (the ~5% isolated by chance) scored
        much lower and were wrong 94% of the time. Youden's J, fit against
        that artificial bimodal reality, collapsed τ to exactly 1.0 -- an
        operating point that only ever accepted an exact-duplicate-of-
        training query. Compounding this: the prior implementation also
        scored held-out rows using a model trained on ALL rows (including
        the held-out rows themselves) -- direct leakage the code used to
        excuse as "negligible"; it demonstrably was not.

        Method:
          1. Collapse each disease's rows to its unique (disease,
             symptom-set) profiles (_unique_profiles_by_disease()).
          2. Stratified k-fold (k=_CALIBRATION_N_FOLDS) over those
             profiles, per disease, fixed seed for reproducibility.
          3. Per fold: refit log-priors/log-likelihoods via
             _fit_naive_bayes() on ONLY the rows belonging to this fold's
             TRAINING profiles -- a genuinely separate model per fold, not
             the full-data model.
          4. Score every held-out profile in that fold with that fold's
             own model -- one out-of-fold (raw_top, is_correct, gap)
             triple per unique profile, 304 total across all folds, each
             scored by a model that never saw that exact profile during
             its own training.
          5. Fit IsotonicRegression on the aggregated 304 out-of-fold
             (raw_posterior, is_correct) pairs.
          6. Derive τ via Youden's J index (Youden 1950; BMC Med Res Meth
             2024) over the calibrated scores.
          7. Derive δ via risk-coverage curve (Chow 1970; Geifman &
             El-Yaniv 2017) at target selective error _TARGET_SELECTIVE_ERROR,
             minimum coverage _MIN_COVERAGE.

        The PRODUCTION model (self._log_prior/self._log_likelihood, set in
        __init__) is untouched by this method -- still fit once on ALL
        rows. Only the calibration curve and τ/δ derived from it change.

        Returns (calibrator, tau, delta).
        """
        from sklearn.isotonic import IsotonicRegression

        profiles_by_disease = _unique_profiles_by_disease(self._rows_by_disease)

        rng = random.Random(_CALIBRATION_SEED)
        # Assign each disease's unique profiles to folds round-robin after
        # a per-disease shuffle -- stratified, so every disease
        # contributes to every fold's training side (see
        # _CALIBRATION_N_FOLDS's comment for why k=5 specifically
        # guarantees this given the 5-10 profile range).
        fold_of_profile: dict[tuple[str, tuple[str, ...]], int] = {}
        for disease, profiles in profiles_by_disease.items():
            shuffled = list(profiles)
            rng.shuffle(shuffled)
            for i, profile in enumerate(shuffled):
                fold_of_profile[(disease, profile)] = i % _CALIBRATION_N_FOLDS

        raw_tops: list[float] = []
        is_correct: list[int] = []
        gaps: list[float] = []

        vocab_set = set(self.vocabulary)
        for fold_idx in range(_CALIBRATION_N_FOLDS):
            held_out = [
                key for key, f in fold_of_profile.items() if f == fold_idx
            ]
            if not held_out:
                continue

            # Training rows for this fold: every row EXCEPT rows whose
            # (disease, profile) is held out this fold. Row-level
            # duplication weighting is preserved WITHIN the training set
            # (matches the production model's own methodology) -- only the
            # held-out side is deduplicated to unique profiles.
            held_out_profiles_by_disease: dict[str, set[tuple[str, ...]]] = {}
            for disease, profile in held_out:
                held_out_profiles_by_disease.setdefault(disease, set()).add(profile)

            train_rows_by_disease: dict[str, list[list[str]]] = {}
            for disease, rows in self._rows_by_disease.items():
                excluded = held_out_profiles_by_disease.get(disease, set())
                kept = [row for row in rows if tuple(sorted(row)) not in excluded]
                if kept:
                    train_rows_by_disease[disease] = kept

            fold_log_prior, fold_log_likelihood = _fit_naive_bayes(
                train_rows_by_disease, self.vocabulary
            )
            fold_diseases = list(fold_log_prior.keys())

            for true_disease, profile in held_out:
                recognized = sorted({t for t in profile if t in vocab_set})
                if not recognized:
                    continue
                probs = _nb_posteriors(recognized, fold_diseases, fold_log_prior, fold_log_likelihood)
                if not probs:
                    continue
                sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                top_d, top_p = sorted_probs[0]
                second_p = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
                raw_tops.append(top_p)
                is_correct.append(1 if top_d == true_disease else 0)
                gaps.append(top_p - second_p)

        if not raw_tops:
            return None, 0.5, 0.1

        # Fit isotonic calibration on (raw_posterior, is_correct)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_tops, is_correct)

        # Apply calibration to get calibrated scores
        cal_tops = list(calibrator.transform(raw_tops))

        # Derive τ via Youden's J (maximise Sensitivity + Specificity - 1)
        unique_thresholds = sorted(set(cal_tops))
        best_j = -2.0
        best_tau = 0.5
        for thresh in unique_thresholds:
            tp = sum(1 for s, c in zip(cal_tops, is_correct) if s >= thresh and c == 1)
            fn = sum(1 for s, c in zip(cal_tops, is_correct) if s < thresh  and c == 1)
            fp = sum(1 for s, c in zip(cal_tops, is_correct) if s >= thresh and c == 0)
            tn = sum(1 for s, c in zip(cal_tops, is_correct) if s < thresh  and c == 0)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens + spec - 1.0
            if j > best_j:
                best_j = j
                best_tau = thresh

        # Derive δ via risk-coverage curve
        # Sweep gap thresholds; pick smallest δ achieving selective error ≤ target
        # while coverage ≥ _MIN_COVERAGE.
        n_total = len(is_correct)
        sorted_gaps = sorted(set(gaps))
        best_delta = 0.0
        for gap_thresh in sorted_gaps:
            answered = [
                c for raw_t, c, g in zip(raw_tops, is_correct, gaps)
                if g >= gap_thresh
            ]
            if not answered:
                continue
            coverage = len(answered) / n_total
            if coverage < _MIN_COVERAGE:
                continue
            err = sum(1 for c in answered if c == 0) / len(answered)
            if err <= _TARGET_SELECTIVE_ERROR:
                best_delta = gap_thresh
                break   # first (smallest) gap threshold meeting the target

        return calibrator, best_tau, best_delta

    def _apply_calibration(self, raw_top_posterior: float) -> float:
        """Apply isotonic calibration to a raw NB top posterior."""
        if self._calibrator is None:
            return raw_top_posterior
        return float(self._calibrator.transform([raw_top_posterior])[0])

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

    def classify_with_abstention(
        self, symptom_tokens: list[str]
    ) -> DiagnosisOutput:
        """Stage-1 entry point with calibration and reject-option abstention.

        Steps:
          1. Compute raw NB posteriors.
          2. Apply isotonic calibration to the top posterior.
          3. Check abstention conditions (Chow 1970 / Geifman & El-Yaniv 2017):
               top_calibrated_posterior >= τ  (Youden's J threshold)
               gap(top − second) >= δ         (risk-coverage threshold)
          4. If both pass → answer.  If either fails → abstain.

        Returns a DiagnosisOutput with full audit trail.
        No LLM call anywhere in this path.
        """
        candidates = self.classify_diseases(symptom_tokens)

        if not candidates:
            return DiagnosisOutput(
                candidates=[],
                calibrated_confidence=0.0,
                raw_confidence=0.0,
                gap_to_second=0.0,
                abstain=True,
                abstain_reason="no recognized symptom tokens — cannot rank diseases",
                tau=self._tau,
                delta=self._delta,
            )

        raw_top = candidates[0].score
        cal_top = self._apply_calibration(raw_top)
        raw_second = candidates[1].score if len(candidates) > 1 else 0.0
        gap = raw_top - raw_second

        if cal_top < self._tau:
            return DiagnosisOutput(
                candidates=candidates,
                calibrated_confidence=cal_top,
                raw_confidence=raw_top,
                gap_to_second=gap,
                abstain=True,
                abstain_reason=(
                    f"calibrated confidence {cal_top:.3f} < τ={self._tau:.3f} "
                    f"(Youden's J threshold) — insufficient certainty"
                ),
                tau=self._tau,
                delta=self._delta,
            )

        if gap < self._delta:
            top_name = candidates[0].name
            second_name = candidates[1].name if len(candidates) > 1 else "?"
            return DiagnosisOutput(
                candidates=candidates,
                calibrated_confidence=cal_top,
                raw_confidence=raw_top,
                gap_to_second=gap,
                abstain=True,
                abstain_reason=(
                    f"gap {gap:.3f} < δ={self._delta:.3f} (risk-coverage threshold) — "
                    f"'{top_name}' and '{second_name}' are too close to distinguish safely"
                ),
                tau=self._tau,
                delta=self._delta,
            )

        return DiagnosisOutput(
            candidates=candidates,
            calibrated_confidence=cal_top,
            raw_confidence=raw_top,
            gap_to_second=gap,
            abstain=False,
            abstain_reason=None,
            tau=self._tau,
            delta=self._delta,
        )


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
