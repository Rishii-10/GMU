"""Tier 2 sensitivity analysis -- turns the framework's two free constants
into reported curves instead of unexplained magic numbers.

A reviewer's fair question about any hand-set constant is "why that value, and
what happens at other values?". Two constants materially drive the safety
result, so both are swept over the SAME out-of-sample short-message set used by
tests/evaluate_oos.py:

  1. DEFAULT_CP_ALPHA (conformal miscoverage level, production 0.10)
     alpha is the knob the safety bound is stated in: worst-case labelling of a
     conformal set gives P(under-triage) <= P(y not in S) <= alpha. Sweeping it
     does two jobs at once -- it EMPIRICALLY TESTS that bound at every level,
     and it traces the safety/efficiency trade-off (smaller alpha -> wider
     sets -> fewer confident answers -> more over-triage).

  2. EMERGENCY_SAFETY_FACTOR (EMERGENCY class-imbalance boost, production 3.0)
     Swept WITH FULL RE-CALIBRATION: each setting builds a fresh
     DiseaseClassifier(emergency_factor=f), so the conformal calibration is
     refitted under the same boost used at inference. Changing the boost at
     inference only would break exchangeability between calibration and
     inference and make the comparison meaningless. factor = 1.0 disables the
     boost entirely.

Deterministic (seed 42, no Ollama / FAISS), same leave-profiles-out folds as
evaluate_oos.py.

Run:  PYTHONPATH=. python tests/evaluate_sensitivity.py           # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_sensitivity.py --write   # -> tests/eval_sensitivity_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import (
    DEFAULT_CP_ALPHA,
    EMERGENCY_SAFETY_FACTOR,
    DiseaseClassifier,
)
from app.disease_kb import DiseaseKB
from tests.evaluate_oos import run_evaluation

# Spans the usable range either side of production on both knobs.
ALPHA_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
EMERGENCY_FACTOR_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

# The metrics each sweep point reports.
_REPORTED = (
    "cp_coverage_true_in_set",
    "under_triage_rate_emergency_severe",
    "over_triage_rate_mild_moderate",
    "coverage_confident_fraction",
    "abstain_or_ask_rate",
    "mean_set_size",
    "lac_empty_fallback_rate",
)


def _point(ev: dict) -> dict:
    return {k: ev[k] for k in _REPORTED}


def sweep_alpha() -> list[dict]:
    out = []
    for a in ALPHA_GRID:
        ev = run_evaluation(alpha=a)
        row = {"alpha": a, **_point(ev)}
        # The bound the paper states, checked at every alpha.
        row["bound_holds_under_triage_le_alpha"] = (
            row["under_triage_rate_emergency_severe"] <= a + 1e-9
        )
        row["coverage_at_least_1_minus_alpha"] = (
            row["cp_coverage_true_in_set"] >= (1.0 - a) - 1e-9
        )
        out.append(row)
    return out


def sweep_emergency_factor() -> list[dict]:
    kb = DiseaseKB.load()
    out = []
    for f in EMERGENCY_FACTOR_GRID:
        # Fresh classifier => conformal calibration refitted at this boost.
        clf = DiseaseClassifier(kb, emergency_factor=f)
        ev = run_evaluation(alpha=DEFAULT_CP_ALPHA, classifier=clf)
        out.append({"emergency_safety_factor": f, **_point(ev)})
    return out


def run_sensitivity() -> dict:
    alpha_rows = sweep_alpha()
    factor_rows = sweep_emergency_factor()

    # Does the boost earn its place? Compare every factor against no boost.
    base = next(r for r in factor_rows if r["emergency_safety_factor"] == 1.0)
    boost_improves_under_triage = any(
        r["under_triage_rate_emergency_severe"] < base["under_triage_rate_emergency_severe"]
        for r in factor_rows
    )
    boost_worsens_over_triage = any(
        r["over_triage_rate_mild_moderate"] > base["over_triage_rate_mild_moderate"]
        for r in factor_rows
    )

    prod = next((r for r in alpha_rows if r["alpha"] == DEFAULT_CP_ALPHA), None)
    # An operating point is dominated if another alpha is at least as good on
    # every reported axis and strictly better on one. Reported because the
    # production alpha should not be defended if the sweep says it is beaten.
    dominating = []
    if prod is not None:
        for r in alpha_rows:
            if r is prod:
                continue
            at_least_as_good = (
                r["under_triage_rate_emergency_severe"] <= prod["under_triage_rate_emergency_severe"]
                and r["over_triage_rate_mild_moderate"] <= prod["over_triage_rate_mild_moderate"]
                and r["coverage_confident_fraction"] >= prod["coverage_confident_fraction"]
                and r["lac_empty_fallback_rate"] <= prod["lac_empty_fallback_rate"]
            )
            strictly_better = (
                r["over_triage_rate_mild_moderate"] < prod["over_triage_rate_mild_moderate"]
                or r["lac_empty_fallback_rate"] < prod["lac_empty_fallback_rate"]
            )
            if at_least_as_good and strictly_better:
                dominating.append(r["alpha"])

    return {
        "regime": "out-of-sample (leave-profiles-out 5-fold) + short message (2-4 tokens)",
        "production": {"alpha": DEFAULT_CP_ALPHA,
                       "emergency_safety_factor": EMERGENCY_SAFETY_FACTOR},
        "alpha_sweep": alpha_rows,
        "emergency_factor_sweep": factor_rows,
        "bound_holds_at_every_alpha": all(r["bound_holds_under_triage_le_alpha"] for r in alpha_rows),
        "alphas_failing_nominal_coverage": [
            r["alpha"] for r in alpha_rows if not r["coverage_at_least_1_minus_alpha"]
        ],
        "emergency_boost_improves_under_triage": boost_improves_under_triage,
        "emergency_boost_worsens_over_triage": boost_worsens_over_triage,
        "alphas_dominating_production": dominating,
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 78)
    print("SENSITIVITY ANALYSIS (Tier 2) -- free constants as curves")
    print("=" * 78)

    print("\n  alpha sweep (conformal miscoverage level)")
    print(f"  {'alpha':>7}{'coverage':>10}{'under-tri':>11}{'over-tri':>10}"
          f"{'confident':>11}{'set size':>10}{'fallback':>10}{'bound ok':>10}")
    print("  " + "-" * 79)
    for r in ev["alpha_sweep"]:
        print(f"  {r['alpha']:>7.2f}{r['cp_coverage_true_in_set']:>10.4f}"
              f"{r['under_triage_rate_emergency_severe']:>11.4f}"
              f"{r['over_triage_rate_mild_moderate']:>10.4f}"
              f"{r['coverage_confident_fraction']:>11.4f}"
              f"{r['mean_set_size']:>10.3f}"
              f"{r['lac_empty_fallback_rate']:>10.4f}"
              f"{str(r['bound_holds_under_triage_le_alpha']):>10}")

    print("\n  EMERGENCY_SAFETY_FACTOR sweep (conformal gate re-calibrated at each value)")
    print(f"  {'factor':>7}{'coverage':>10}{'under-tri':>11}{'over-tri':>10}"
          f"{'confident':>11}{'set size':>10}")
    print("  " + "-" * 59)
    for r in ev["emergency_factor_sweep"]:
        print(f"  {r['emergency_safety_factor']:>7.1f}{r['cp_coverage_true_in_set']:>10.4f}"
              f"{r['under_triage_rate_emergency_severe']:>11.4f}"
              f"{r['over_triage_rate_mild_moderate']:>10.4f}"
              f"{r['coverage_confident_fraction']:>11.4f}"
              f"{r['mean_set_size']:>10.3f}")

    print(f"\n  under-triage <= alpha at every alpha  : {ev['bound_holds_at_every_alpha']}")
    print(f"  alphas missing nominal coverage       : {ev['alphas_failing_nominal_coverage']}")
    print(f"  boost IMPROVES under-triage anywhere  : {ev['emergency_boost_improves_under_triage']}")
    print(f"  boost WORSENS over-triage somewhere   : {ev['emergency_boost_worsens_over_triage']}")
    print(f"  alphas dominating production alpha    : {ev['alphas_dominating_production']}")

    out = out_dir / "eval_sensitivity_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("sensitivity analysis (Tier 2)")
    print("Running sensitivity sweeps (alpha, emergency boost with re-calibration)...")
    print_report(run_sensitivity(), out_dir)
