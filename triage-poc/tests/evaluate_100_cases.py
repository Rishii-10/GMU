"""
100-case evaluation of the rule engine's disease classifier.

Samples real rows from data/disease_symptoms.csv (ground truth: known disease),
runs each through classify_with_cp(), compares predicted severity tier to true
severity tier, and prints a full 4x4 confusion matrix plus binary
EMERGENCY-detection metrics (TP/FP/FN/TN).

Usage:
    cd triage-poc
    .venv/bin/python tests/evaluate_100_cases.py
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
SEVERITY_TIERS = [
    ClassificationLabel.EMERGENCY,
    ClassificationLabel.SEVERE,
    ClassificationLabel.MODERATE,
    ClassificationLabel.MILD,
]
TIER_LABEL = {t: t.value for t in SEVERITY_TIERS}

# ---------------------------------------------------------------------------
# Load raw dataset rows
# ---------------------------------------------------------------------------
def load_rows() -> list[tuple[str, list[str]]]:
    """Returns list of (disease_name, [symptom_tokens]) from disease_symptoms.csv."""
    rows = []
    with open(DATA_DIR / "disease_symptoms.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            disease = normalize_disease_name(row[0])
            tokens = [normalize_symptom_token(c) for c in row[1:] if c.strip()]
            if tokens:
                rows.append((disease, tokens))
    return rows


# ---------------------------------------------------------------------------
# Sample 100 cases stratified by disease
# ---------------------------------------------------------------------------
def sample_100(rows: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    by_disease: dict[str, list[list[str]]] = defaultdict(list)
    for disease, tokens in rows:
        by_disease[disease].append(tokens)

    diseases = sorted(by_disease.keys())
    n_diseases = len(diseases)
    base = 100 // n_diseases
    extra = 100 % n_diseases

    sampled: list[tuple[str, list[str]]] = []
    for i, d in enumerate(diseases):
        n = base + (1 if i < extra else 0)
        pool = by_disease[d]
        chosen = random.sample(pool, min(n, len(pool)))
        for tokens in chosen:
            sampled.append((d, tokens))

    random.shuffle(sampled)
    return sampled[:100]


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def run_evaluation() -> dict:
    clf = get_default_classifier()
    all_rows = load_rows()
    cases = sample_100(all_rows)

    # 4x4 confusion matrix: cm[true_tier][pred_tier] = count
    cm: dict[str, dict[str, int]] = {
        t.value: {t2.value: 0 for t2 in SEVERITY_TIERS} for t in SEVERITY_TIERS
    }
    # Also track: correct disease (top-1), in prediction set, abstained
    correct_disease = 0
    in_pred_set = 0
    abstained = 0
    uncertain = 0
    confident_correct = 0
    confident_total = 0

    results_log = []

    for true_disease, tokens in cases:
        true_sev = severity_for_disease(true_disease).value
        cp = clf.classify_with_cp(tokens, alpha=0.05)

        if cp.decision == "ABSTAIN":
            abstained += 1
            pred_sev = ClassificationLabel.INCOMPLETE_ASSESSMENT.value
            pred_disease = "ABSTAIN"
        else:
            pred_disease = cp.prediction_set[0] if cp.prediction_set else "NONE"
            pred_sev = severity_for_disease(pred_disease).value if pred_disease not in ("ABSTAIN","NONE") else "MILD"
            if cp.decision == "UNCERTAIN":
                uncertain += 1
            else:
                confident_total += 1
                if pred_disease == true_disease:
                    confident_correct += 1

        if true_disease == pred_disease:
            correct_disease += 1
        if true_disease in (cp.prediction_set or []):
            in_pred_set += 1

        if true_sev in cm and pred_sev in cm.get(true_sev, {}):
            cm[true_sev][pred_sev] += 1

        results_log.append({
            "true_disease": true_disease,
            "true_sev": true_sev,
            "pred_disease": pred_disease,
            "pred_sev": pred_sev,
            "decision": cp.decision,
            "set_size": cp.set_size,
            "prediction_set": cp.prediction_set,
        })

    # Binary: EMERGENCY detection metrics
    tp = cm["EMERGENCY"]["EMERGENCY"]
    fp = sum(cm[t]["EMERGENCY"] for t in cm if t != "EMERGENCY")
    fn = sum(cm["EMERGENCY"][t] for t in cm["EMERGENCY"] if t != "EMERGENCY")
    tn = sum(
        cm[tr][pr]
        for tr in cm for pr in cm[tr]
        if tr != "EMERGENCY" and pr != "EMERGENCY"
    )

    return {
        "cases": cases,
        "results_log": results_log,
        "cm": cm,
        "correct_disease": correct_disease,
        "in_pred_set": in_pred_set,
        "abstained": abstained,
        "uncertain": uncertain,
        "confident_correct": confident_correct,
        "confident_total": confident_total,
        "binary": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
def print_report(ev: dict) -> None:
    n = len(ev["cases"])
    cm = ev["cm"]
    b = ev["binary"]

    print("=" * 70)
    print("RULE ENGINE — 100-CASE EVALUATION")
    print("=" * 70)
    print(f"Total cases evaluated: {n}")
    print(f"Correct top-1 disease match: {ev['correct_disease']}/{n}  "
          f"({ev['correct_disease']/n:.0%})")
    print(f"True disease in CP prediction set: {ev['in_pred_set']}/{n}  "
          f"({ev['in_pred_set']/n:.0%})  ← formal coverage")
    print(f"CONFIDENT decisions: {ev['confident_total']}   "
          f"correct of those: {ev['confident_correct']}  "
          f"({ev['confident_correct']/max(ev['confident_total'],1):.0%})")
    print(f"UNCERTAIN (2-3 candidates, needs follow-up): {ev['uncertain']}")
    print(f"ABSTAINED (≥4 candidates, refer): {ev['abstained']}")

    print()
    print("4×4 SEVERITY CONFUSION MATRIX")
    print("(rows = true severity, cols = predicted severity)")
    tiers = [t.value for t in SEVERITY_TIERS]
    col_w = 12
    header = " " * 14 + "".join(f"{t:>{col_w}}" for t in tiers)
    print(header)
    print(" " * 14 + "-" * (col_w * len(tiers)))
    for true_t in tiers:
        row_vals = "".join(
            f"{cm.get(true_t, {}).get(pred_t, 0):>{col_w}}"
            for pred_t in tiers
        )
        marker = " ←" if true_t == "EMERGENCY" else ""
        print(f"  {true_t:<12}{row_vals}{marker}")

    print()
    print("BINARY: EMERGENCY DETECTION")
    print(f"  TP (true EMERGENCY, flagged EMERGENCY):       {b['TP']:>4}")
    print(f"  FP (true non-EMERGENCY, flagged EMERGENCY):   {b['FP']:>4}  ← over-triage")
    print(f"  FN (true EMERGENCY, missed as non-EMERGENCY): {b['FN']:>4}  ← DANGEROUS MISS")
    print(f"  TN (true non-EMERGENCY, not flagged):         {b['TN']:>4}")
    print()
    prec = b['TP'] / max(b['TP'] + b['FP'], 1)
    rec  = b['TP'] / max(b['TP'] + b['FN'], 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  Emergency Precision: {prec:.2f}   Recall: {rec:.2f}   F1: {f1:.2f}")
    print("=" * 70)

    # Save full log
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in ev.items() if k != "cases"}, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    print("Running 100-case evaluation...")
    ev = run_evaluation()
    print_report(ev)
