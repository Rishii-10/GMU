"""
1000-case balanced evaluation: 334 EMERGENCY + 333 URGENT + 333 NON-URGENT.

EMERGENCY band has only 240 unique rows (Heart attack + Paralysis), so that
band is sampled WITH replacement. URGENT (1920 rows) and NON-URGENT (2760 rows)
are sampled without replacement — they have enough headroom.

Outputs:
  - Sample test cases printed to stdout (15 per band so you can read them)
  - 3×3 confusion matrix + binary TP/FP/FN/TN
  - eval_1000_results.json  — numeric summary
  - eval_1000_log.json      — per-case detail

Usage:
    cd triage-poc
    .venv/bin/python tests/evaluate_1000.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import get_default_classifier, severity_for_disease
from app.disease_kb import normalize_disease_name, normalize_symptom_token
from app.rules_engine import classify
from app.schemas import AgeGroup, ClassificationLabel, DangerSigns, ExtractedCase

# Every case is scored through rules_engine.classify() as an adult with no
# danger-sign information, so the full adult path runs: CP gate, worst-case
# severity on UNCERTAIN (fix F3), AHP Stage 2 on CONFIDENT. The earlier
# version of this script called classify_with_cp() directly and scored
# prediction_set[0], which bypassed the rules engine entirely and never
# exercised F3 or the AHP scorer.
EVAL_AGE_MONTHS = 480

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
BANDS = ["EMERGENCY", "URGENT", "NON_URGENT"]
TARGETS = {"EMERGENCY": 334, "URGENT": 333, "NON_URGENT": 333}


def severity_to_band(sev: ClassificationLabel) -> str:
    if sev == ClassificationLabel.EMERGENCY:
        return "EMERGENCY"
    if sev == ClassificationLabel.SEVERE:
        return "URGENT"
    return "NON_URGENT"


# ---------------------------------------------------------------------------
# Load rows
# ---------------------------------------------------------------------------
def load_rows_by_band() -> dict[str, list[tuple[str, list[str]]]]:
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
            band = severity_to_band(severity_for_disease(disease))
            buckets[band].append((disease, tokens))
    return buckets


# ---------------------------------------------------------------------------
# Sample 1000 balanced cases (with replacement where pool is smaller than need)
# ---------------------------------------------------------------------------
def sample_balanced(
    buckets: dict[str, list[tuple[str, list[str]]]]
) -> list[tuple[str, str, list[str]]]:
    sampled: list[tuple[str, str, list[str]]] = []
    for band, n in TARGETS.items():
        pool = buckets[band]
        if len(pool) >= n:
            chosen = random.sample(pool, n)
        else:
            # sample with replacement — EMERGENCY band (240 rows, need 334)
            chosen = random.choices(pool, k=n)
        for disease, tokens in chosen:
            sampled.append((disease, band, tokens))
    random.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Print example cases (first 15 per band before shuffle)
# ---------------------------------------------------------------------------
def print_examples(buckets: dict[str, list[tuple[str, list[str]]]]) -> None:
    print("\n" + "=" * 70)
    print("SAMPLE TEST CASES SHOWN TO THE ENGINE (5 per band)")
    print("=" * 70)
    for band in BANDS:
        pool = buckets[band]
        examples = random.sample(pool, min(5, len(pool)))
        print(f"\n─── {band} ───")
        for i, (disease, tokens) in enumerate(examples, 1):
            print(f"  {i}. Disease : {disease}")
            print(f"     Tokens  : {', '.join(tokens[:8])}{'...' if len(tokens) > 8 else ''}")
            print(f"     (total symptoms in this row: {len(tokens)})")
    print()


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
def run_evaluation():
    clf = get_default_classifier()
    buckets = load_rows_by_band()
    print_examples(buckets)
    cases = sample_balanced(buckets)

    cm = {b: {b2: 0 for b2 in BANDS} for b in BANDS}
    log = []

    for i, (true_disease, true_band, tokens) in enumerate(cases, 1):
        if i % 100 == 0:
            print(f"  ... evaluated {i}/1000", flush=True)
        cp = clf.classify_with_cp(tokens, alpha=0.05)
        case = ExtractedCase(
            raw_symptom_text=" ".join(tokens),
            age_months=EVAL_AGE_MONTHS,
            age_group=AgeGroup.ADULT,
            danger_signs=DangerSigns(),
            symptom_tokens=tokens,
        )
        result = classify(case)

        if result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT:
            # ABSTAIN: scored as NON_URGENT so an abstained emergency counts
            # as a miss, not a free pass.
            pred_disease = "ABSTAIN"
            pred_band = "NON_URGENT"
        else:
            pred_disease = result.condition or "?"
            # Label comes from the rules engine (worst-case over the
            # prediction set when UNCERTAIN), not from the top candidate.
            pred_band = severity_to_band(result.label)

        cm[true_band][pred_band] += 1
        log.append({
            "case_num": i,
            "true_disease": true_disease,
            "true_band": true_band,
            "pred_disease": pred_disease,
            "pred_band": pred_band,
            "label": result.label.value,
            "decision": cp.decision,
            "set_size": cp.set_size,
            "prediction_set": cp.prediction_set,
            "ahp_band": result.emergency_result.band.value if result.emergency_result else None,
            "ahp_score": result.emergency_result.score if result.emergency_result else None,
            "tokens": tokens,
        })

    tp = cm["EMERGENCY"]["EMERGENCY"]
    fp = cm["URGENT"]["EMERGENCY"] + cm["NON_URGENT"]["EMERGENCY"]
    fn = cm["EMERGENCY"]["URGENT"] + cm["EMERGENCY"]["NON_URGENT"]
    tn = (cm["URGENT"]["URGENT"] + cm["URGENT"]["NON_URGENT"] +
          cm["NON_URGENT"]["URGENT"] + cm["NON_URGENT"]["NON_URGENT"])

    return {"cm": cm, "binary": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}, "log": log}


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
def print_report(ev: dict) -> None:
    cm = ev["cm"]
    b  = ev["binary"]
    prec = b["TP"] / max(b["TP"] + b["FP"], 1)
    rec  = b["TP"] / max(b["TP"] + b["FN"], 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    print("=" * 70)
    print("1000-CASE BALANCED EVALUATION (334 EMERG / 333 URGENT / 333 NON-URG)")
    print("=" * 70)
    print()
    print("3×3 CONFUSION MATRIX (rows=TRUE, cols=PREDICTED)")
    print(f"{'':18} {'EMERGENCY':>12} {'URGENT':>10} {'NON_URGENT':>12}")
    print("-" * 56)
    for tb in BANDS:
        row = "  ".join(f"{cm[tb][pb]:>10}" for pb in BANDS)
        marker = " ←" if tb == "EMERGENCY" else ""
        print(f"  {tb:<16}  {row}{marker}")

    print()
    print("BINARY: EMERGENCY DETECTION")
    print(f"  TP  {b['TP']:>5}  (true EMERGENCY, predicted EMERGENCY)")
    print(f"  FP  {b['FP']:>5}  (non-emergency, predicted EMERGENCY)  ← over-triage")
    print(f"  FN  {b['FN']:>5}  (true EMERGENCY, missed)              ← DANGEROUS MISS")
    print(f"  TN  {b['TN']:>5}  (non-emergency, correctly not flagged)")
    print()
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("=" * 70)
    print()

    for band in BANDS:
        total   = sum(cm[band].values())
        correct = cm[band][band]
        print(f"  {band} accuracy: {correct}/{total} = {correct/max(total,1):.1%}")

    # Decision type breakdown
    decisions = [e["decision"] for e in ev["log"]]
    for d in ("CONFIDENT", "UNCERTAIN", "ABSTAIN"):
        cnt = decisions.count(d)
        print(f"  CP decision {d}: {cnt}/1000 = {cnt/10:.1f}%")

    # AHP Stage 2 band vs. disease-tier band on CONFIDENT cases. These are
    # two different scales (AHP is a 6-attribute acuity score; the label is
    # a per-disease lookup), so disagreement is reported, not treated as an
    # error -- but it is the first time the AHP output is measured at all.
    confident = [e for e in ev["log"] if e["decision"] == "CONFIDENT" and e["ahp_band"]]
    agree = sum(1 for e in confident if e["ahp_band"] == e["pred_band"])
    ahp_agreement = agree / max(len(confident), 1)
    print(f"  AHP band == disease-tier band on CONFIDENT cases: {agree}/{len(confident)} = {ahp_agreement:.1%}")
    ahp_cm = {b: {b2: 0 for b2 in BANDS} for b in BANDS}
    for e in confident:
        ahp_cm[e["pred_band"]][e["ahp_band"]] += 1
    print("  AHP band by disease-tier band (rows=disease tier, cols=AHP band):")
    for b in BANDS:
        print(f"    {b:<12}" + "  ".join(f"{ahp_cm[b][b2]:>6}" for b2 in BANDS))

    # Save
    out   = Path(__file__).parent / "eval_1000_results.json"
    out2  = Path(__file__).parent / "eval_1000_log.json"
    summary = {
        "cm": cm,
        "binary": b,
        "metrics": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)},
        "band_accuracy": {
            band: round(cm[band][band] / max(sum(cm[band].values()), 1), 4)
            for band in BANDS
        },
        "decision_counts": {d: decisions.count(d) for d in ("CONFIDENT", "UNCERTAIN", "ABSTAIN")},
        "ahp_vs_disease_tier_on_confident": {
            "n": len(confident),
            "agreement": round(ahp_agreement, 4),
            "matrix": ahp_cm,
        },
        "scored_through": "app.rules_engine.classify (adult, age_months=%d, danger signs unassessed)" % EVAL_AGE_MONTHS,
        "n": 1000,
        "targets": TARGETS,
    }
    with open(out,  "w") as f: json.dump(summary,    f, indent=2)
    with open(out2, "w") as f: json.dump(ev["log"],  f, indent=2)
    print(f"\nResults → {out}")
    print(f"Full log → {out2}")


if __name__ == "__main__":
    print("Running 1000-case balanced evaluation...")
    ev = run_evaluation()
    print_report(ev)
