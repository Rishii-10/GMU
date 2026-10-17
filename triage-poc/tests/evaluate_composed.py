"""The composed ablation -- the 'money figure' for the framework paper.

Every other harness scores one slice. This one walks a SINGLE metric across the
whole stack on the same held-out short messages, so the reader sees the journey
from a raw model to the full framework:

  C0  LLM-only ............ prompt llama3.2:3b for the tier directly
  C1  + structured NB ..... plain Naive-Bayes argmax (no safety machinery)
  C2  + abstain gate ...... conformal (LAC) prediction set
  C3  + worst-case guard .. label the set with its most severe tier  (full system)
  C4  + active follow-up ... ask up to 3 questions, then decide

The deterministic stages (C1-C4) are computed live by reusing evaluate_oos and
evaluate_followup, so they are the SAME computations the pillar harnesses report
-- no numbers invented here. C0 (LLM-only) is read from the tracked
eval_llm_results.json when present (that harness needs Ollama); it is omitted,
not faked, when unavailable.

Run:  PYTHONPATH=. python tests/evaluate_composed.py            # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_composed.py --write    # -> tests/eval_composed_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import get_default_classifier
from app.disease_kb import DiseaseKB
from tests.evaluate_oos import run_evaluation
from tests.evaluate_followup import _run_one, _summarize

_LLM_JSON = Path(__file__).resolve().parent / "eval_llm_results.json"


def _stage(name: str, label: str, misses_urgent, over_calls, source: str) -> dict:
    return {"stage": name, "label": label,
            "misses_urgent": misses_urgent, "over_calls": over_calls, "source": source}


def run_composed_evaluation() -> dict:
    """Assemble the staged metric from the pillar computations (same cases)."""
    stages: list[dict] = []

    # C0 -- LLM-only (read from the tracked baseline; needs Ollama to regenerate)
    if _LLM_JSON.exists():
        b = json.loads(_LLM_JSON.read_text()).get("triage_baseline", {})
        stages.append(_stage(
            "C0", "LLM-only (prompt the model)",
            b.get("under_triage_rate_emergency_severe"),
            b.get("over_triage_rate_mild_moderate"), "evaluate_llm.py",
        ))
    else:
        stages.append(_stage("C0", "LLM-only (prompt the model)", None, None,
                             "evaluate_llm.py (not generated — run it with Ollama up)"))

    # C1-C3 -- deterministic framework stages, from the OOS comparator grid
    oos = run_evaluation()
    comps = oos.get("comparisons", {})
    c1 = comps.get("baseline_flat_nb_top1", {})
    c2 = comps.get("ablation_no_worstcase", {})
    stages.append(_stage("C1", "+ structured NB (argmax)",
                         c1.get("under_triage_rate_emergency_severe"),
                         c1.get("over_triage_rate_mild_moderate"), "evaluate_oos.py"))
    stages.append(_stage("C2", "+ abstain gate (conformal set)",
                         c2.get("under_triage_rate_emergency_severe"),
                         c2.get("over_triage_rate_mild_moderate"), "evaluate_oos.py"))
    stages.append(_stage("C3", "+ worst-case guard (full system)",
                         oos.get("under_triage_rate_emergency_severe"),
                         oos.get("over_triage_rate_mild_moderate"), "evaluate_oos.py"))

    # C4 -- add the adaptive follow-up loop (selector, absence-aware = shipped)
    kb, full = DiseaseKB.load(), get_default_classifier()
    fu = _summarize(_run_one("selector", True, kb, full))
    stages.append(_stage("C4", "+ active follow-up (shipped)",
                         fu.get("under_triage_after"), fu.get("over_triage_after"),
                         "evaluate_followup.py"))

    return {
        "metric_note": "misses_urgent = EMERGENCY/SEVERE surfaced below true tier; "
                       "over_calls = MILD/MODERATE surfaced as EMERGENCY/SEVERE. "
                       "Same held-out short messages throughout; C0 needs Ollama.",
        "stage_mapping": "C1 adds no safety machinery; the conformal set enters at "
                         "C2; the worst-case guard (C3) is what removes under-triage; "
                         "C4 adds the follow-up loop on top of the full system.",
        "stages": stages,
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 74)
    print("COMPOSED ABLATION -- one metric across the whole stack")
    print("=" * 74)
    print(f"  {'stage':<6}{'configuration':<36}{'misses urgent':>14}{'over-calls':>12}")
    print("  " + "-" * 68)
    for s in ev["stages"]:
        mu = "n/a" if s["misses_urgent"] is None else f"{s['misses_urgent']:.4f}"
        oc = "n/a" if s["over_calls"] is None else f"{s['over_calls']:.4f}"
        print(f"  {s['stage']:<6}{s['label']:<36}{mu:>14}{oc:>12}")
    out = out_dir / "eval_composed_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("composed ablation (LLM-only -> full framework)")
    print("Running composed ablation...")
    print_report(run_composed_evaluation(), out_dir)
