"""
Balanced 100-case evaluation: 34 EMERGENCY + 33 URGENT + 33 NON-URGENT.

Ground truth comes ONLY from data/disease_severity.csv:
  EMERGENCY  → band EMERGENCY
  SEVERE     → band URGENT
  MODERATE   → band NON_URGENT
  MILD       → band NON_URGENT

Confusion matrix:
  3×3 rows=true band, cols=predicted band
  + binary 2×2 (Emergency / Non-Emergency) with TP/FP/FN/TN

Usage:
    cd triage-poc
    .venv/bin/python tests/evaluate_balanced.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import get_default_classifier, severity_for_disease
from app.disease_kb import normalize_disease_name, normalize_symptom_token
from app.schemas import ClassificationLabel

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"

BANDS = ["EMERGENCY", "URGENT", "NON_URGENT"]

def severity_to_band(sev: ClassificationLabel) -> str:
    if sev == ClassificationLabel.EMERGENCY:
        return "EMERGENCY"
    if sev == ClassificationLabel.SEVERE:
        return "URGENT"
    return "NON_URGENT"   # MODERATE, MILD, INCOMPLETE_ASSESSMENT all map here

# ---------------------------------------------------------------------------
# Load rows from CSV, bucket by band
# ---------------------------------------------------------------------------
def load_rows_by_band() -> dict[str, list[tuple[str, list[str]]]]:
    """Returns {band: [(disease, [tokens]), ...]} using ONLY disease_severity.csv
    as the ground truth source."""
    buckets: dict[str, list[tuple[str, list[str]]]] = {b: [] for b in BANDS}
    with open(DATA_DIR / "disease_symptoms.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            disease = normalize_disease_name(row[0])
            tokens = [normalize_symptom_token(c) for c in row[1:] if c.strip()]
            if not tokens:
                continue
            sev = severity_for_disease(disease)
            band = severity_to_band(sev)
            buckets[band].append((disease, tokens))
    return buckets

# ---------------------------------------------------------------------------
# Sample 34 EMERGENCY + 33 URGENT + 33 NON_URGENT
# ---------------------------------------------------------------------------
def sample_balanced(buckets: dict[str, list[tuple[str, list[str]]]]) -> list[tuple[str, str, list[str]]]:
    """Returns list of (true_disease, true_band, tokens), 100 total."""
    targets = {"EMERGENCY": 34, "URGENT": 33, "NON_URGENT": 33}
    sampled: list[tuple[str, str, list[str]]] = []
    for band, n in targets.items():
        pool = buckets[band]
        chosen = random.sample(pool, min(n, len(pool)))
        for disease, tokens in chosen:
            sampled.append((disease, band, tokens))
    random.shuffle(sampled)
    return sampled

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
def run_evaluation():
    clf = get_default_classifier()
    buckets = load_rows_by_band()
    cases = sample_balanced(buckets)

    # 3×3 matrix: cm[true_band][pred_band]
    cm = {b: {b2: 0 for b2 in BANDS} for b in BANDS}
    log = []

    for true_disease, true_band, tokens in cases:
        cp = clf.classify_with_cp(tokens, alpha=0.05)

        if cp.decision == "ABSTAIN" or not cp.prediction_set:
            pred_disease = "ABSTAIN"
            pred_band = "NON_URGENT"   # conservative: abstain treated as non-emergency
        else:
            pred_disease = cp.prediction_set[0]
            pred_sev = severity_for_disease(pred_disease)
            pred_band = severity_to_band(pred_sev)

        cm[true_band][pred_band] += 1
        log.append({
            "true_disease": true_disease,
            "true_band": true_band,
            "pred_disease": pred_disease,
            "pred_band": pred_band,
            "decision": cp.decision,
            "set_size": cp.set_size,
            "prediction_set": cp.prediction_set,
        })

    # Binary: Emergency vs Non-Emergency
    tp = cm["EMERGENCY"]["EMERGENCY"]
    fp = cm["URGENT"]["EMERGENCY"] + cm["NON_URGENT"]["EMERGENCY"]
    fn = cm["EMERGENCY"]["URGENT"] + cm["EMERGENCY"]["NON_URGENT"]
    tn = (cm["URGENT"]["URGENT"] + cm["URGENT"]["NON_URGENT"] +
          cm["NON_URGENT"]["URGENT"] + cm["NON_URGENT"]["NON_URGENT"])

    return {"cm": cm, "binary": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}, "log": log}

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
def print_report(ev: dict):
    cm = ev["cm"]
    b  = ev["binary"]

    prec = b["TP"] / max(b["TP"] + b["FP"], 1)
    rec  = b["TP"] / max(b["TP"] + b["FN"], 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    print("=" * 65)
    print("BALANCED 100-CASE EVALUATION (34 EMERG / 33 URGENT / 33 NON-URGENT)")
    print("=" * 65)
    print()
    print("3×3 CONFUSION MATRIX (rows=TRUE, cols=PREDICTED)")
    print(f"{'':16} {'EMERGENCY':>12} {'URGENT':>10} {'NON_URGENT':>12}")
    print("-" * 52)
    for tb in BANDS:
        row = "  ".join(f"{cm[tb][pb]:>10}" for pb in BANDS)
        marker = " ←" if tb == "EMERGENCY" else ""
        print(f"  {tb:<14}  {row}{marker}")
    print()
    print("BINARY: EMERGENCY DETECTION")
    print(f"  TP  {b['TP']:>4}  (true EMERGENCY, predicted EMERGENCY)")
    print(f"  FP  {b['FP']:>4}  (non-emergency, predicted EMERGENCY)  ← over-triage")
    print(f"  FN  {b['FN']:>4}  (true EMERGENCY, missed)              ← DANGEROUS MISS")
    print(f"  TN  {b['TN']:>4}  (non-emergency, correctly not flagged)")
    print()
    print(f"  Precision: {prec:.2f}   Recall: {rec:.2f}   F1: {f1:.2f}")
    print("=" * 65)

    # Per-band accuracy
    for band in BANDS:
        total = sum(cm[band].values())
        correct = cm[band][band]
        print(f"  {band} accuracy: {correct}/{total} = {correct/max(total,1):.0%}")

    out = Path(__file__).parent / "eval_balanced_results.json"
    with open(out, "w") as f:
        json.dump({k: v for k, v in ev.items() if k != "log"}, f, indent=2)
    # Also save full log
    out2 = Path(__file__).parent / "eval_balanced_log.json"
    with open(out2, "w") as f:
        json.dump(ev["log"], f, indent=2)
    print(f"\nResults → {out}")
    print(f"Full log → {out2}")

if __name__ == "__main__":
    print("Running balanced evaluation...")
    ev = run_evaluation()
    print_report(ev)
