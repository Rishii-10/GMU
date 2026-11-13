"""Cross-validated accuracy of the symptom-matching threshold.

The paper's "97.5% accuracy at the chosen setting" is IN-SAMPLE: the threshold
was chosen by calibrate_dataset_threshold() on the 80-item labelled sample and
its accuracy measured on the same 80 items. This harness reports the honest
counterpart: stratified k-fold (k=5, positives/negatives split evenly) and
leave-one-out, where the threshold is chosen on the training items only and
scored on the held-out items.

Needs faiss + sentence-transformers (no Ollama). Deterministic.
Run (from triage-poc/):  PYTHONPATH=. python tests/evaluate_threshold_cv.py [--write]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disambiguation import DEFAULT_DATASET_CONFIDENCE_THRESHOLD, FAISSDisambiguator, calibrate_dataset_threshold
from tests.fixtures.symptom_calibration import CALIBRATION_SAMPLE

K = 5
SEED = 42


def _accuracy_at(raw, threshold):
    ok = 0
    for tok, score, expected in raw:
        pred = tok if (tok is not None and score is not None and score >= threshold) else None
        ok += pred == expected
    return ok / len(raw)


def _cv(disamb, folds):
    raw_all = {i: (*disamb.raw_dataset_top1(t), e) for i, (t, e) in enumerate(CALIBRATION_SAMPLE)}
    correct = total = 0
    chosen = []
    for test_idx in folds:
        train = [CALIBRATION_SAMPLE[i] for i in range(len(CALIBRATION_SAMPLE)) if i not in test_idx]
        thr, _ = calibrate_dataset_threshold(train, disamb)
        chosen.append(thr)
        test_raw = [raw_all[i] for i in test_idx]
        correct += round(_accuracy_at(test_raw, thr) * len(test_raw))
        total += len(test_raw)
    return correct / total, chosen


class _Cached:
    """raw_dataset_top1 memoised -- the only method calibrate_dataset_threshold uses."""

    def __init__(self, inner):
        self.inner, self.cache = inner, {}

    def raw_dataset_top1(self, text):
        if text not in self.cache:
            self.cache[text] = self.inner.raw_dataset_top1(text)
        return self.cache[text]


def run_threshold_cv() -> dict:
    disamb = _Cached(FAISSDisambiguator())
    n = len(CALIBRATION_SAMPLE)
    best, results = calibrate_dataset_threshold(CALIBRATION_SAMPLE, disamb)
    in_sample = next(r.accuracy for r in results if r.threshold == best)

    pos = [i for i, (_, e) in enumerate(CALIBRATION_SAMPLE) if e is not None]
    neg = [i for i, (_, e) in enumerate(CALIBRATION_SAMPLE) if e is None]
    rng = random.Random(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    kfold = [set(pos[k::K]) | set(neg[k::K]) for k in range(K)]
    kfold_acc, kfold_thr = _cv(disamb, kfold)
    loo_acc, loo_thr = _cv(disamb, [{i} for i in range(n)])

    return {
        "sample": "tests/fixtures/symptom_calibration.py (labelled English/Hindi/Hinglish phrases)",
        "sample_size": n,
        "positives": len(pos),
        "negatives": len(neg),
        "shipped_threshold": DEFAULT_DATASET_CONFIDENCE_THRESHOLD,
        "in_sample_best_threshold": best,
        "in_sample_accuracy": round(in_sample, 4),
        "kfold": {"k": K, "accuracy": round(kfold_acc, 4), "chosen_thresholds": kfold_thr},
        "leave_one_out": {"accuracy": round(loo_acc, 4),
                          "distinct_chosen_thresholds": sorted(set(loo_thr))},
        "note": "in_sample_accuracy is the figure previously quoted (threshold chosen and scored "
                "on the same items); kfold/leave_one_out choose it on training items only.",
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print(f"in-sample {ev['in_sample_accuracy']:.4f} at {ev['in_sample_best_threshold']}  |  "
          f"{ev['kfold']['k']}-fold CV {ev['kfold']['accuracy']:.4f} {ev['kfold']['chosen_thresholds']}  |  "
          f"LOO {ev['leave_one_out']['accuracy']:.4f}")
    out = out_dir / "eval_threshold_cv_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"Results -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    print_report(run_threshold_cv(), resolve_out_dir("cross-validated symptom-matching threshold"))
