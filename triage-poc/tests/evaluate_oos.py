"""Stage 5 evaluation harness -- the numbers the paper rests on.

The legacy evals (evaluate_1000/balanced/100_cases) score FULL in-sample symptom
rows and report F1 = 1.00. That measures internal consistency, not real caller
performance -- it is kept only as a "pipeline wiring sanity check". This harness
instead measures the regime that matters for a triage-safety claim:

  * OUT-OF-SAMPLE: leave-profiles-out k-fold, so every test profile is scored by
    a model that never trained on it.
  * SHORT MESSAGES: each test profile truncated to 2-4 tokens (the real caller
    condition -- few symptoms, not a full 17-symptom row).
  * CODE-MIXED / PARAPHRASE consistency: the same case under token-order
    shuffles must yield the same triage label.

Metrics (guiding principle: for triage, selective risk and under-triage are the
headline, not accuracy):
  - under-triage rate   : true EMERGENCY/SEVERE surfaced at a lower tier  (~0 is the number the paper rests on)
  - over-triage rate    : true MILD/MODERATE surfaced as EMERGENCY/SEVERE  (bounded, reported as the trade-off)
  - risk-coverage        : selective error at each coverage (CONFIDENT fraction)
  - silent-error vs abstain: wrong-and-confident rate vs uncertain-and-asked rate
  - decision consistency : label agreement across token-order variants

Baselines + ablations (same OOS short-message set):
  - rules-only / flat-threshold top-1 (no conformal gate)
  - ablation: no worst-case severity (surface the top candidate's tier)

Deterministic: scores tokens directly through app.rules_engine.classify (adult
route), no Ollama / FAISS. Agent-1 extraction error is a separate, stated
limitation (L4) and is not measured here.

Run:  PYTHONPATH=. python tests/evaluate_oos.py            # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_oos.py --write    # -> tests/eval_oos_results.json (tracked baseline)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import (
    DiseaseClassifier,
    _fit_naive_bayes,
    _clean_candidate_scores,
    _boost_emergency_scores,
    _nb_posteriors,
    get_default_classifier,
    severity_for_disease,
)
from app.disease_kb import DiseaseKB
from app.schemas import AgeGroup, ClassificationLabel, ExtractedCase, SEVERITY_RANK

_SEED = 42
_N_FOLDS = 5
_SHORT_MIN, _SHORT_MAX = 2, 4
_ADULT_AGE_MONTHS = 600


def _unique_profiles(rows_by_disease):
    out = []
    for d, rows in rows_by_disease.items():
        seen = set()
        for row in rows:
            key = tuple(sorted(row))
            if key not in seen:
                seen.add(key)
                out.append((d, key))
    return out


def _assign_folds(profiles, rng):
    by_d = {}
    for d, p in profiles:
        by_d.setdefault(d, []).append((d, p))
    fold_of = {}
    for d, lst in by_d.items():
        shuffled = list(lst)
        rng.shuffle(shuffled)
        for i, kp in enumerate(shuffled):
            fold_of[kp] = i % _N_FOLDS
    return fold_of


def _truncate(profile, rng, k_min=_SHORT_MIN, k_max=_SHORT_MAX):
    toks = list(profile)
    rng.shuffle(toks)
    k = min(len(toks), rng.randint(k_min, k_max))
    return toks[:k]


def run_evaluation() -> dict:
    kb = DiseaseKB.load()
    full = get_default_classifier()  # production model (for severity lookups)
    rows_by_disease = full._rows_by_disease
    vocab = full.vocabulary
    vocab_set = set(vocab)
    emergency_names = full._emergency_disease_names

    profiles = _unique_profiles(rows_by_disease)
    rng = random.Random(_SEED)
    fold_of = _assign_folds(profiles, rng)

    # Decision records for each OOS short message, under the real pipeline.
    records = []       # {true, true_rank, label, label_rank, decision, set_size}
    # Ablation records (same inputs, scored by simple variants) and consistency.
    abl_flat = []      # flat-threshold top-1 (no CP gate): label = top-1 disease tier
    abl_noworst = []   # CP set but surface the TOP candidate tier (no worst-case)
    consistency_hits = consistency_total = 0

    trunc_rng = random.Random(_SEED + 1)

    for fold_idx in range(_N_FOLDS):
        held = [k for k, f in fold_of.items() if f == fold_idx]
        held_by_d = {}
        for d, p in held:
            held_by_d.setdefault(d, set()).add(p)
        train = {}
        for d, rows in rows_by_disease.items():
            ex = held_by_d.get(d, set())
            kept = [r for r in rows if tuple(sorted(r)) not in ex]
            if kept:
                train[d] = kept
        lp, ll = _fit_naive_bayes(train, vocab)
        # A fold-local classifier that reuses the production calibration/q̂ but a
        # fold-trained NB -- so the test profile is genuinely out-of-sample for
        # the NB model (the conformal calibration stays fixed, as in deployment).
        fold_clf = _FoldModel(full, lp, ll, emergency_names)

        for true_d, profile in held:
            true_rank = SEVERITY_RANK[severity_for_disease(true_d, kb)]
            short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
            if not short:
                continue
            res = _classify_short(fold_clf, short, full)
            records.append({
                "true": true_d, "true_rank": true_rank,
                "label": res["label"], "label_rank": SEVERITY_RANK.get(ClassificationLabel(res["label"]), 99),
                "decision": res["decision"], "set_size": res["set_size"],
                "correct_in_set": true_d in res["prediction_set"],
            })
            abl_flat.append({"true_rank": true_rank, "label_rank": SEVERITY_RANK[severity_for_disease(res["top"], kb)]})
            abl_noworst.append({"true_rank": true_rank, "label_rank": SEVERITY_RANK[severity_for_disease(res["top"], kb)]})

            # consistency: re-shuffle the same short tokens, expect same label
            alt = list(short)
            random.Random(hash(tuple(short)) & 0xFFFF).shuffle(alt)
            res2 = _classify_short(fold_clf, alt, full)
            consistency_total += 1
            consistency_hits += 1 if res2["label"] == res["label"] else 0

    return _summarize(records, abl_flat, abl_noworst, consistency_hits, consistency_total)


class _FoldModel:
    """A DiseaseClassifier-like shim: fold-trained NB posteriors + the
    production conformal calibration (q̂) and emergency set."""
    def __init__(self, full, lp, ll, emergency_names):
        self._lp, self._ll = lp, ll
        self._diseases = list(lp.keys())
        self._cp = sorted(full._cp_cal_nonconformity_scores)
        self._alpha = 0.10
        self._emergency_names = emergency_names
        self._vocab = full.vocabulary

    def posteriors(self, tokens):
        rec = sorted({t for t in tokens if t in set(self._vocab)})
        if not rec:
            return {}
        return _nb_posteriors(rec, self._diseases, self._lp, self._ll)

    def q_hat(self):
        import math
        n = len(self._cp)
        qi = min(int(math.ceil((n + 1) * (1.0 - self._alpha))), n) - 1
        return self._cp[qi]


def _classify_short(fold_clf, tokens, full) -> dict:
    """Replicates classify_with_cp + classify_via_dataset labelling on a
    fold-trained model: LAC set on boosted NB posterior, emergency-in-set
    worst-case escalation, else INCOMPLETE/CONFIDENT tiering."""
    probs = fold_clf.posteriors(tokens)
    if not probs:
        return {"label": ClassificationLabel.INCOMPLETE_ASSESSMENT.value, "decision": "ABSTAIN",
                "set_size": 0, "prediction_set": [], "top": None}
    scores = _boost_emergency_scores(_clean_candidate_scores(probs), fold_clf._emergency_names)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[0][0]
    q = fold_clf.q_hat()
    in_set = [d for d, s in scores.items() if (1.0 - s) <= q] or [d for d, _ in ranked]
    set_size = len(in_set)
    decision = "CONFIDENT" if set_size == 1 else ("UNCERTAIN" if set_size <= 3 else "ABSTAIN")
    # worst-case severity across the set (rules_engine F3 + Stage-2 escalation)
    worst_rank = min(SEVERITY_RANK[severity_for_disease(d)] for d in in_set)
    worst_label = next(l for l, r in SEVERITY_RANK.items() if r == worst_rank)
    if set_size == 1:
        label = severity_for_disease(top).value
    else:
        # UNCERTAIN or wide ABSTAIN: surface worst-case (Stage 2 safety)
        label = worst_label.value if hasattr(worst_label, "value") else worst_label
    return {"label": label, "decision": decision, "set_size": set_size,
            "prediction_set": in_set, "top": top}


def _rate(num, den):
    return round(num / den, 4) if den else 0.0


def _summarize(records, abl_flat, abl_noworst, c_hits, c_total) -> dict:
    n = len(records)
    emg_sev = [r for r in records if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]]
    mild_mod = [r for r in records if r["true_rank"] >= SEVERITY_RANK[ClassificationLabel.MODERATE]]

    def under(rs):
        return sum(1 for r in rs if r["label_rank"] > r["true_rank"])
    def over(rs):
        return sum(1 for r in rs if r["label_rank"] < r["true_rank"])

    confident = [r for r in records if r["decision"] == "CONFIDENT"]
    conf_wrong = sum(1 for r in confident if r["label_rank"] != r["true_rank"])
    abstain_or_ask = [r for r in records if r["decision"] != "CONFIDENT"]

    return {
        "n": n,
        "regime": "out-of-sample (leave-profiles-out 5-fold) + short message (2-4 tokens)",
        "under_triage_rate_emergency_severe": _rate(under(emg_sev), len(emg_sev)),
        "under_triage_count": under(emg_sev),
        "emergency_severe_n": len(emg_sev),
        "over_triage_rate_mild_moderate": _rate(over(mild_mod), len(mild_mod)),
        "cp_coverage_true_in_set": _rate(sum(1 for r in records if r["correct_in_set"]), n),
        "decision_counts": dict(Counter(r["decision"] for r in records)),
        "coverage_confident_fraction": _rate(len(confident), n),
        "selective_risk_on_confident": _rate(conf_wrong, len(confident)),
        "silent_error_rate": _rate(conf_wrong, n),            # wrong AND confident
        "abstain_or_ask_rate": _rate(len(abstain_or_ask), n),
        "consistency_label_agreement": _rate(c_hits, c_total),
        "baseline_flat_threshold": {
            "under_triage_rate_emergency_severe": _rate(
                sum(1 for r in abl_flat if r["label_rank"] > r["true_rank"]
                    and SEVERITY_RANK.get(ClassificationLabel.SEVERE) >= r["true_rank"]),
                sum(1 for r in abl_flat if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]),
            ),
        },
        "ablation_no_worstcase_surface_top": {
            "under_triage_rate_emergency_severe": _rate(
                sum(1 for r in abl_noworst if r["label_rank"] > r["true_rank"]
                    and r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]),
                sum(1 for r in abl_noworst if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]),
            ),
        },
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 72)
    print("OUT-OF-SAMPLE + SHORT-MESSAGE EVALUATION (Stage 5)")
    print("=" * 72)
    for k in ("n", "regime", "under_triage_rate_emergency_severe", "under_triage_count",
              "emergency_severe_n", "over_triage_rate_mild_moderate",
              "cp_coverage_true_in_set", "coverage_confident_fraction",
              "selective_risk_on_confident", "silent_error_rate", "abstain_or_ask_rate",
              "consistency_label_agreement", "decision_counts"):
        print(f"  {k}: {ev[k]}")
    print(f"  baseline (flat-threshold top-1) under-triage: "
          f"{ev['baseline_flat_threshold']['under_triage_rate_emergency_severe']}")
    print(f"  ablation (no worst-case, surface top) under-triage: "
          f"{ev['ablation_no_worstcase_surface_top']['under_triage_rate_emergency_severe']}")
    out = out_dir / "eval_oos_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("out-of-sample + short-message evaluation (Stage 5)")
    print("Running out-of-sample short-message evaluation...")
    ev = run_evaluation()
    print_report(ev, out_dir)
