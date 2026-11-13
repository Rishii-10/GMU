"""Naive Bayes vs logistic regression on the short messages the system actually sees.

Backs (or refutes) the paper's §4.2 claim that Naive Bayes is "both more
accurate and far better calibrated than richer alternatives such as logistic
regression ... on the short, few-symptom messages this use case produces".
Before this harness the repo contained no logistic-regression code at all.

Same regime as tests/evaluate_oos.py, reproduced exactly: leave-profiles-out
5-fold over the 304 unique (disease, profile) pairs, each held-out profile
truncated to the SAME 2-4 token "short message" (same seeds, same draw order).
Every model sees the same training rows and the same test tokens.

Comparators (LR is given every reasonable chance, so this is not a strawman):
  naive_bayes            production presence-only multinomial-style NB (app.disease_classifier)
  logreg_full_C=<c>      sklearn LogisticRegression on binary bag-of-tokens, trained on the
                         full dataset rows, C in C_GRID
  logreg_short_C=<c>     same, trained on short-subset AUGMENTED rows (each training row ->
                         N_AUG random 2-4 token subsets) -- LR trained on inputs that look like
                         the test inputs
Metrics: top-1 / top-3 accuracy, ECE (10 equal-width bins on top-1 confidence),
multi-class Brier, NLL, mean top-1 confidence, overconfidence (= confidence - accuracy).
For each LR family the row with the best TEST accuracy is compared with NB -- an
optimistic upper bound for LR, since C is chosen on the test folds.

RESULT (2026-11-13): NB beats LR trained on full rows, but LR trained on short
subsets beats NB on accuracy (0.905 vs 0.812 at C=1) -- see `finding` in the
results JSON. The paper's §4.2 sentence must be narrowed accordingly.

Run (from triage-poc/):  PYTHONPATH=. python tests/evaluate_nb_vs_lr.py [--write]
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import _fit_naive_bayes, _nb_posteriors, get_default_classifier
from tests.evaluate_oos import _N_FOLDS, _SEED, _assign_folds, _truncate, _unique_profiles

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_AUG = 3
_ECE_BINS = 10


def _features(tokens, index):
    x = [0.0] * len(index)
    for t in tokens:
        i = index.get(t)
        if i is not None:
            x[i] = 1.0
    return x


def _metrics(records):
    n = len(records)
    top1 = sum(r["rank"] == 0 for r in records) / n
    top3 = sum(r["rank"] is not None and r["rank"] < 3 for r in records) / n
    conf = [r["conf"] for r in records]
    correct = [r["rank"] == 0 for r in records]
    ece = 0.0
    for b in range(_ECE_BINS):
        lo, hi = b / _ECE_BINS, (b + 1) / _ECE_BINS
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc_b = sum(correct[i] for i in idx) / len(idx)
            conf_b = sum(conf[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(acc_b - conf_b)
    brier = sum(r["brier"] for r in records) / n
    nll = sum(-math.log(max(r["p_true"], 1e-12)) for r in records) / n
    mean_conf = sum(conf) / n
    return {
        "n": n,
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "nll": round(nll, 4),
        "mean_top1_confidence": round(mean_conf, 4),
        "overconfidence": round(mean_conf - top1, 4),
    }


def _record(probs: dict, true_d: str, diseases: list[str]) -> dict:
    full = {d: float(probs.get(d, 0.0)) for d in diseases}
    ranked = sorted(full, key=lambda d: -full[d])
    rank = ranked.index(true_d)
    brier = sum((full[d] - (1.0 if d == true_d else 0.0)) ** 2 for d in diseases)
    return {"rank": rank, "conf": full[ranked[0]], "p_true": full[true_d], "brier": float(brier)}


def run_nb_vs_lr_evaluation() -> dict:
    from sklearn.linear_model import LogisticRegression

    full = get_default_classifier()
    rows_by_disease = full._rows_by_disease
    vocab = list(full.vocabulary)
    vocab_set = set(vocab)
    index = {t: i for i, t in enumerate(vocab)}
    diseases = sorted(rows_by_disease)

    profiles = _unique_profiles(rows_by_disease)
    fold_of = _assign_folds(profiles, random.Random(_SEED))
    trunc_rng = random.Random(_SEED + 1)  # identical draw order to evaluate_oos
    aug_rng = random.Random(_SEED + 3)

    names = ["naive_bayes"] + [f"logreg_full_C={c:g}" for c in C_GRID] + [f"logreg_short_C={c:g}" for c in C_GRID]
    recs = {n: [] for n in names}

    for fold_idx in range(_N_FOLDS):
        held = [k for k, f in fold_of.items() if f == fold_idx]
        held_by_d = {}
        for d, p in held:
            held_by_d.setdefault(d, set()).add(p)
        train = {}
        for d, rows in rows_by_disease.items():
            kept = [r for r in rows if tuple(sorted(r)) not in held_by_d.get(d, set())]
            if kept:
                train[d] = kept
        lp, ll = _fit_naive_bayes(train, vocab)

        X_full, y_full, X_short, y_short = [], [], [], []
        for d, rows in train.items():
            for r in rows:
                X_full.append(_features(r, index))
                y_full.append(d)
                for _ in range(N_AUG):
                    X_short.append(_features(_truncate(r, aug_rng), index))
                    y_short.append(d)
        models = {}
        for c in C_GRID:
            models[f"logreg_full_C={c:g}"] = LogisticRegression(C=c, max_iter=2000).fit(X_full, y_full)
            models[f"logreg_short_C={c:g}"] = LogisticRegression(C=c, max_iter=2000).fit(X_short, y_short)

        tests = []
        for true_d, profile in held:
            short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
            if short:
                tests.append((true_d, sorted(set(short))))
        if not tests:
            continue
        for true_d, toks in tests:
            recs["naive_bayes"].append(_record(_nb_posteriors(toks, list(lp), lp, ll), true_d, diseases))
        X_test = [_features(t, index) for _, t in tests]
        for name, m in models.items():
            for (true_d, _), row in zip(tests, m.predict_proba(X_test)):
                recs[name].append(_record(dict(zip(m.classes_, row)), true_d, diseases))

    results = {n: _metrics(r) for n, r in recs.items()}
    nb = results["naive_bayes"]

    def _vs(family: str) -> dict:
        fam = [n for n in names if n.startswith(family)]
        best = max(fam, key=lambda n: (results[n]["top1_accuracy"], -results[n]["ece"]))
        b = results[best]
        return {
            "best_by_test_accuracy": best,
            "nb_more_accurate": bool(nb["top1_accuracy"] > b["top1_accuracy"]),
            "nb_lower_ece": bool(nb["ece"] < b["ece"]),
            "nb_lower_brier": bool(nb["brier"] < b["brier"]),
            "lr_direction_of_miscalibration": "overconfident" if b["overconfidence"] > 0.02
            else ("underconfident" if b["overconfidence"] < -0.02 else "calibrated"),
        }

    return {
        "regime": "out-of-sample (leave-profiles-out 5-fold) + short message (2-4 tokens); same "
                  "test tokens as tests/evaluate_oos.py",
        "n_test_cases": nb["n"],
        "lr_note": "C is selected on the test folds -- an optimistic upper bound for LR",
        "models": results,
        "nb_vs_lr_trained_on_full_rows": _vs("logreg_full"),
        "nb_vs_lr_trained_on_short_subsets": _vs("logreg_short"),
        "finding": "NB beats LR trained on full rows on accuracy, ECE and Brier (that LR is "
                   "UNDER-confident, not overconfident). LR trained on short-subset augmented rows "
                   "beats NB on accuracy at every C >= 0.1, on Brier at C >= 1, and matches or beats its ECE "
                   "at C >= 10. The claim 'NB is more accurate and better calibrated than LR on "
                   "short messages' holds only against LR trained on full rows.",
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 92)
    print("NAIVE BAYES vs LOGISTIC REGRESSION -- held-out short messages")
    print("=" * 92)
    print(f"  {'model':<24}{'top-1':>8}{'top-3':>8}{'ECE':>8}{'Brier':>8}{'NLL':>8}{'conf':>8}{'over-conf':>11}")
    for name, m in ev["models"].items():
        print(f"  {name:<24}{m['top1_accuracy']:>8.3f}{m['top3_accuracy']:>8.3f}{m['ece']:>8.3f}"
              f"{m['brier']:>8.3f}{m['nll']:>8.3f}{m['mean_top1_confidence']:>8.3f}{m['overconfidence']:>11.3f}")
    for key in ("nb_vs_lr_trained_on_full_rows", "nb_vs_lr_trained_on_short_subsets"):
        print(f"\n  {key}: {ev[key]}")
    print(f"\n  {ev['finding']}")
    out = out_dir / "eval_nb_vs_lr_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("Naive Bayes vs logistic regression on held-out short messages")
    print_report(run_nb_vs_lr_evaluation(), out_dir)
