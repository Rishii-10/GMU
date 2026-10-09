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
# keeps each fold's holdout share close to the prior single-split's 20%.
# k=10 was considered and rejected: with a minimum of
# 5 profiles per disease, most (disease, fold) combinations would hold out
# zero profiles for that disease, unevenly thinning the aggregated
# evaluation sample without adding real held-out diversity given how few
# unique profiles exist in total (304).
_CALIBRATION_N_FOLDS = 5
_CALIBRATION_SEED = 42

# Conformal miscoverage budget α. 0.10 (nominal 90% coverage) was tuned on the
# out-of-fold calibration data (Stage 3): it gives a non-vacuous q̂, keeps
# under-triage lowest, and leaves a meaningful UNCERTAIN rate so the
# abstain-and-ask follow-up loop has a real role. Empirical held-out coverage
# is ~0.99 (the gate is conservative). (The previous τ/δ reject option — a
# Youden-J threshold and a risk-coverage gap — was never called by anything and
# has been removed: Conformal Prediction is the sole uncertainty gate.)
DEFAULT_CP_ALPHA = 0.10

# Fix 2 — Emergency safety boost.
# The dataset has only 2 EMERGENCY diseases out of 41, giving them a raw prior
# of ~2.4 % each.  Their symptoms overlap heavily with SEVERE diseases (e.g.
# Heart attack shares chest_pain/breathlessness/sweating with Tuberculosis).
# Multiplying their pre-calibration posterior by this factor compensates for
# the class imbalance.  In a clinical triage system, over-triaging an emergency
# is always preferable to missing one.  Value of 3.0 chosen so that a tie
# between an EMERGENCY and SEVERE candidate resolves toward EMERGENCY.
EMERGENCY_SAFETY_FACTOR = 3.0

# Fix 3 — Red-flag symptom override.
# If an EMERGENCY disease is already in the CP prediction set AND one of these
# discriminating tokens is present in the patient's reported symptoms, collapse
# the prediction set to that disease immediately without waiting for the
# adaptive follow-up loop.
#
# Membership rule (checked against data/disease_symptoms.csv, enforced by
# tests/test_cp_ahp_adaptive.py::test_every_red_flag_token_exists_in_vocabulary):
# the token must exist in the 131-token vocabulary, appear in >= 80 % of an
# EMERGENCY disease's rows, and in 0 % of every other disease's rows.
#
# Only Paralysis (brain hemorrhage) has such tokens. Heart attack's rows
# contain exactly four tokens (chest_pain, breathlessness, sweating,
# vomiting), every one of which also appears in >= 90 % of Tuberculosis
# rows, so NO red-flag token exists for Heart attack in this dataset; the
# Heart-attack-vs-TB separation rests entirely on EMERGENCY_SAFETY_FACTOR
# and the worst-case-severity rule on UNCERTAIN sets (rules_engine F3).
# An earlier version of this set listed radiating_pain,
# loss_of_consciousness and sudden_severe_headache (not in the vocabulary,
# could never match) and weakness_in_limbs (in the vocabulary, but a
# Cervical spondylosis token: 0 % of Paralysis rows).
_RED_FLAG_TOKENS: frozenset[str] = frozenset({
    "altered_sensorium",          # Paralysis: 95 % of rows, 0 % elsewhere
    "weakness_of_one_body_side",  # Paralysis: 90 % of rows, 0 % elsewhere
})


def _clean_candidate_scores(probs: dict[str, float]) -> dict[str, float]:
    """Renormalize raw NB softmax posteriors over the >=MIN_CANDIDATE_SCORE
    survivors -- the exact score classify_diseases() exposes as each
    candidate's `score`. Computing CP nonconformity from this same quantity
    (out-of-fold at calibration, and on live candidates at inference) keeps the
    conformal calibration and inference scores exchangeable. Returns {} when no
    token is recognised / nothing survives the floor."""
    surviving = {d: p for d, p in probs.items() if p >= MIN_CANDIDATE_SCORE}
    total = sum(surviving.values())
    if total <= 0:
        return {}
    return {d: round(p / total, 4) for d, p in surviving.items()}


def _boost_emergency_scores(
    scores: dict[str, float],
    emergency_names: "frozenset[str]",
    factor: float = EMERGENCY_SAFETY_FACTOR,
) -> dict[str, float]:
    """Fix F1 -- multiply EMERGENCY-tier diseases' scores by
    EMERGENCY_SAFETY_FACTOR and renormalize, to compensate for the 2-of-41
    EMERGENCY class imbalance (over-triaging an emergency beats missing one).
    No-op when no EMERGENCY disease is present. Applied identically at CP
    calibration and inference so the conformity score is the same quantity.

    `emergency_names` is passed in (precomputed once from the classifier's own
    DiseaseKB) rather than looked up via severity_for_disease(): this runs
    INSIDE DiseaseClassifier.__init__ (CP calibration), and the kb-less
    severity_for_disease() would call get_default_classifier() and recurse into
    construction.

    `factor` defaults to the production constant and exists so the sensitivity
    analysis (tests/evaluate_sensitivity.py) can RE-CALIBRATE the conformal gate
    at a different boost rather than changing it only at inference, which would
    break exchangeability between calibration and inference. factor = 1.0
    disables the boost."""
    if factor == 1.0 or not any(d in emergency_names for d in scores):
        return dict(scores)
    boosted = {
        d: (s * factor if d in emergency_names else s)
        for d, s in scores.items()
    }
    total = sum(boosted.values())
    return {d: s / total for d, s in boosted.items()} if total > 0 else dict(scores)


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
class ConformalResult:
    """Output of classify_with_cp() — Conformal Prediction prediction set.

    prediction_set: diseases the engine cannot formally rule out at the
      chosen coverage level (default α=0.05 → ≥95% guarantee).
    decision:
      "CONFIDENT"  — set_size = 1; proceed to Stage 2 AHP.
      "UNCERTAIN"  — set_size 2-3; trigger adaptive follow-up loop.
      "ABSTAIN"    — set_size ≥ 4 or empty; refer to higher facility.
    q_hat: the quantile threshold applied (from holdout calibration scores).
    posteriors: calibrated probability for each disease in the set (useful
      for Info Gain computation in the follow-up question selector).
    """
    prediction_set: list[str]
    set_size: int
    decision: str            # "CONFIDENT" | "UNCERTAIN" | "ABSTAIN"
    alpha: float
    q_hat: float
    candidates: list[DiseaseCandidate]
    posteriors: dict[str, float]   # disease -> calibrated P within prediction set


class DiseaseClassifier:
    """Estimates a Naive-Bayes-style model once at construction from the raw
    CSV rows (not just DiseaseKB's deduplicated symptom sets -- row-level
    co-occurrence frequency is what the likelihoods are estimated from), then
    answers classify_diseases() queries as plain arithmetic, no I/O.
    """

    def __init__(
        self,
        kb: DiseaseKB,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        emergency_factor: float = EMERGENCY_SAFETY_FACTOR,
    ):
        self.kb = kb
        self.vocabulary = kb.vocabulary
        self._vocab_size = len(self.vocabulary)
        # Injectable so the sensitivity analysis can rebuild a fully
        # re-calibrated classifier at a different boost. Production always
        # uses the module default.
        self._emergency_factor = emergency_factor

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

        # EMERGENCY-tier disease names, resolved from THIS classifier's own kb
        # (kb passed explicitly so it never calls get_default_classifier(),
        # which would recurse while we are still inside __init__). Used by the
        # F1 emergency boost at both CP calibration (below) and inference.
        self._emergency_disease_names: frozenset[str] = frozenset(
            d for d in self.kb.diseases
            if severity_for_disease(d, self.kb) == ClassificationLabel.EMERGENCY
        )

        # Fit the isotonic top-1 confidence calibrator (for a reported,
        # non-degenerate confidence + ECE/Brier) and store the Conformal
        # Prediction calibration nonconformity scores, via leak-free k-fold
        # cross-validation over unique profiles. Done once at construction.
        # See _fit_calibration_and_thresholds().
        (
            self._calibrator,
            self._cp_cal_nonconformity_scores,
            self.ece_,
            self.brier_,
        ) = self._fit_calibration_and_thresholds()

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
        """Fit the isotonic top-1 confidence calibrator, compute the Conformal
        Prediction calibration nonconformity scores, and report calibration
        quality (ECE/Brier) -- all via leak-free stratified k-fold
        cross-validation over UNIQUE (disease, symptom-set) profiles.

        Stage 3 change -- the conformal gate is now a non-vacuous split
        conformal (LAC) on the NB posterior:
          - Nonconformity on a class d is s(d) = 1 - p(d), where p(d) is the
            emergency-boosted, renormalized NB candidate score (exactly what
            classify_diseases() + the F1 boost expose at inference), computed
            OUT OF FOLD here so calibration and inference score the same
            quantity. This replaced the old s(d) = 1 - isotonic(d): the
            isotonic was fit on TOP-1 correctness (raw_top -> is_correct) but
            applied PER CLASS, collapsing almost every calibrated posterior to
            {0, 1}. That made >=5% of calibration nonconformity scores exactly
            1.0, pinning q̂ to 1.0 at α=0.05 (CP-1: the gate admitted every
            candidate >=1%, so its coverage claim was trivially true) and the
            displayed posteriors degenerate at 0.000 (CP-3).
          - The isotonic calibrator is KEPT, but only to report a calibrated
            top-1 confidence and the ECE/Brier below -- it no longer gates.

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
             own model -- one out-of-fold (raw_top, is_correct) pair plus
             one CP nonconformity score (1 − boosted true-class score) per
             unique profile, 304 total across all folds, each scored by a
             model that never saw that exact profile during its own training.
          5. Fit IsotonicRegression on the aggregated 304 out-of-fold
             (raw_posterior, is_correct) pairs (for the reported top-1
             confidence + ECE/Brier only -- it does not gate).
          6. Store the out-of-fold CP nonconformity scores; classify_with_cp()
             takes their ⌈(n+1)(1−α)⌉/n-th quantile as q̂ at inference.

        The PRODUCTION model (self._log_prior/self._log_likelihood, set in
        __init__) is untouched by this method -- still fit once on ALL rows.

        Returns (calibrator, cp_cal_nonconformity_scores, ece, brier).
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
        # Conformal Prediction nonconformity score per held-out profile:
        # s = 1 - p(true_class), where p is the emergency-boosted renormalized
        # NB candidate score -- EXACTLY what classify_diseases() + the F1 boost
        # expose at inference, so calibration and inference score the same
        # quantity (exchangeability). A true class filtered out below
        # MIN_CANDIDATE_SCORE scores 0.0 -> maximal nonconformity 1.0
        # (conservative, widens q̂). Collected INSIDE the fold loop (leak-free).
        cp_cal_nonconformity_scores: list[float] = []

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
                raw_tops.append(top_p)
                is_correct.append(1 if top_d == true_disease else 0)
                # Emergency-boosted renormalized score of the TRUE class,
                # computed exactly as classify_diseases()+F1 do at inference.
                scored = _boost_emergency_scores(
                    _clean_candidate_scores(probs),
                    self._emergency_disease_names,
                    self._emergency_factor,
                )
                cp_cal_nonconformity_scores.append(1.0 - scored.get(true_disease, 0.0))

        if not raw_tops:
            return None, [], None, None

        # Fit isotonic calibration on (raw_top_posterior, is_correct). KEPT only
        # to report a calibrated top-1 confidence and the ECE/Brier below -- it
        # no longer feeds the conformal gate (see the Stage 3 note above).
        # Isotonic is still the best calibrator here empirically (ECE ~0.019 vs
        # raw ~0.109 vs Platt ~0.463).
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_tops, is_correct)

        # Calibration quality on the out-of-fold top-1 predictions, reported for
        # the paper / UI (never used to gate). Brier = mean squared error of the
        # calibrated confidence vs correctness; ECE = 10-bin expected calibration
        # error.
        cal_tops = [float(v) for v in calibrator.transform(raw_tops)]
        n_total = len(is_correct)
        brier = sum((p - c) ** 2 for p, c in zip(cal_tops, is_correct)) / n_total
        n_bins = 10
        ece = 0.0
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            idx = [
                i for i, p in enumerate(cal_tops)
                if (p > lo or (b == 0 and p >= lo)) and p <= hi
            ]
            if not idx:
                continue
            conf = sum(cal_tops[i] for i in idx) / len(idx)
            acc = sum(is_correct[i] for i in idx) / len(idx)
            ece += (len(idx) / n_total) * abs(acc - conf)

        return calibrator, cp_cal_nonconformity_scores, ece, brier

    def calibrated_top_confidence(self, symptom_tokens: list[str]) -> Optional[float]:
        """Isotonic-calibrated confidence that the top-1 NB disease is correct.

        This is the one place the isotonic calibrator is used at inference --
        for a reported, non-degenerate top-1 confidence (and it backs the
        ECE/Brier in `ece_`/`brier_`). It does NOT gate: the conformal gate
        (classify_with_cp) scores on the NB posterior directly. Returns None
        when no token is recognised or no calibrator was fit.
        """
        if self._calibrator is None:
            return None
        probs = self._raw_posteriors(symptom_tokens)
        if not probs:
            return None
        raw_top = max(probs.values())
        return float(self._calibrator.transform([raw_top])[0])

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

    def classify_with_cp(
        self, symptom_tokens: list[str], alpha: float = DEFAULT_CP_ALPHA
    ) -> ConformalResult:
        """Conformal Prediction gate (split conformal / LAC).

        Coverage guarantee: P(true disease ∈ prediction_set) ≥ 1 − α under
        exchangeability of calibration and test samples (Vovk 2005;
        Angelopoulos & Bates 2021).

        Algorithm (Stage 3 — non-vacuous):
          1. candidates = classify_diseases() — NB softmax renormalized over
             the >=MIN_CANDIDATE_SCORE survivors.
          2. F1 emergency boost: EMERGENCY diseases ×EMERGENCY_SAFETY_FACTOR,
             renormalized. The result `score` is the conformity probability.
          3. Nonconformity s(d) = 1 − score(d).
          4. q̂ = the ⌈(n+1)(1−α)⌉/n-th smallest of the stored OUT-OF-FOLD
             calibration nonconformity scores (same boosted-score quantity).
          5. prediction_set = {d : s(d) ≤ q̂}. If that is empty, the posterior
             is flat/uncertain, so the set becomes the whole plausible
             differential (all candidates ≥ MIN_CANDIDATE_SCORE) — an honest
             wide ABSTAIN rather than a forced-confident top-1.
          6. set_size → decision: 1 CONFIDENT, 2-3 UNCERTAIN, ≥4 ABSTAIN.
          7. F2 red-flag override can still collapse to a single EMERGENCY.

        Unlike the pre-Stage-3 gate, q̂ is no longer pinned to 1.0 (the old
        isotonic-collapse bug): the set now genuinely shrinks as the posterior
        concentrates with discriminating tokens, and `posteriors` carries the
        real (boosted) NB probabilities, not degenerate 0.000 values.

        No LLM call.  Returns ConformalResult with full audit trail.
        """
        candidates = self.classify_diseases(symptom_tokens)

        if not candidates:
            return ConformalResult(
                prediction_set=[],
                set_size=0,
                decision="ABSTAIN",
                alpha=alpha,
                q_hat=0.0,
                candidates=[],
                posteriors={},
            )

        # Fix F1 — emergency safety boost. Applied via the same helper the CP
        # calibration uses, so the conformity score is the same quantity in
        # both places (exchangeability).
        boosted_map = _boost_emergency_scores(
            {c.name: c.score for c in candidates},
            self._emergency_disease_names,
            self._emergency_factor,
        )
        candidates = sorted(
            [c.model_copy(update={"score": round(boosted_map[c.name], 4)}) for c in candidates],
            key=lambda c: c.score,
            reverse=True,
        )

        # q̂ from stored out-of-fold calibration nonconformity scores.
        cal_scores = self._cp_cal_nonconformity_scores
        n = len(cal_scores)
        if n == 0:
            # Degenerate: no calibration data — fall back to top-1.
            top = candidates[0]
            return ConformalResult(
                prediction_set=[top.name],
                set_size=1,
                decision="CONFIDENT",
                alpha=alpha,
                q_hat=1.0,
                candidates=candidates,
                posteriors={top.name: top.score},
            )
        q_idx = min(int(math.ceil((n + 1) * (1.0 - alpha))), n) - 1
        q_hat = sorted(cal_scores)[q_idx]

        # LAC prediction set: include d iff nonconformity(d) = 1 − score(d) ≤ q̂.
        in_set = [c for c in candidates if (1.0 - c.score) <= q_hat]
        if not in_set:
            # Flat posterior, nothing clears the bar: the honest answer is the
            # whole plausible differential (wide/uncertain), NOT a confident
            # top-1. rules_engine's Stage-2 safety then escalates if the wide
            # set contains an EMERGENCY/SEVERE disease.
            in_set = list(candidates)
        prediction_set = [c.name for c in in_set]
        posteriors = {c.name: c.score for c in in_set}

        set_size = len(prediction_set)
        if set_size == 1:
            decision = "CONFIDENT"
        elif set_size <= 3:
            decision = "UNCERTAIN"
        else:
            decision = "ABSTAIN"

        # Fix 3 — Red-flag symptom override (applied after prediction set is built).
        # If an EMERGENCY disease is already in the prediction set AND the patient
        # has reported a symptom that is highly specific to that emergency disease
        # (near-zero frequency in all SEVERE/MODERATE diseases), collapse the
        # prediction set immediately — no follow-up question needed.
        # Safety: the CP gate already established the true disease COULD be an
        # emergency (it's in the set); the red-flag token confirms it.
        token_set = set(symptom_tokens)
        if decision != "CONFIDENT" and token_set & _RED_FLAG_TOKENS:
            emergency_in_set = [
                d for d in prediction_set if d in self._emergency_disease_names
            ]
            if emergency_in_set:
                top_em = max(emergency_in_set, key=lambda d: posteriors.get(d, 0.0))
                prediction_set = [top_em]
                posteriors = {top_em: posteriors[top_em]}
                set_size = 1
                decision = "CONFIDENT"

        return ConformalResult(
            prediction_set=prediction_set,
            set_size=set_size,
            decision=decision,
            alpha=alpha,
            q_hat=q_hat,
            candidates=candidates,
            posteriors=posteriors,
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
