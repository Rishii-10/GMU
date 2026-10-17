"""Systems cost of the pipeline -- the practical numbers a systems reviewer asks
for: how many LLM calls a case costs, and how long each stage takes.

Two parts:
  * CALL MODEL   -- how many LLM calls each stage makes per case (deterministic
                    accounting, no run needed).
  * LATENCY      -- measured wall-clock per stage. The deterministic stages
                    (classify, routing, surveillance) are always measured; the
                    LLM extraction stage is measured only when Ollama is up.

Latency is machine-dependent: read the per-stage BREAKDOWN and the call counts,
not the absolute milliseconds.

Run:  PYTHONPATH=. python tests/evaluate_cost.py            # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_cost.py --write    # -> tests/eval_cost_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app.rules_engine import classify as rules_classify
from app.routing.demo_facilities import build_demo_facility_db
from app.routing.router import route
from app.schemas import ClassificationLabel, ExtractedCase
from app.surveillance.agent import SurveillanceAgent
from app.surveillance.demo_stream import generate_demo_stream

# How many LLM calls each pipeline stage costs per case. The framework is
# deliberately LLM-light: exactly one call is unavoidable (extraction); every
# other call is conditional, and surveillance's is event-driven (only on an
# actual alert), not per case.
CALL_MODEL = {
    "extraction (Agent 1)": "1 (always)",
    "follow-up phrasing (Flow C)": "0-3 (only when the case is uncertain)",
    "doctor report (Agent 3)": "0-1 (only EMERGENCY/SEVERE with a village)",
    "surveillance brief (Agent 2)": "event-driven (only when an alert fires, not per case)",
    "typical MILD case": "1 LLM call",
    "typical EMERGENCY case": "2 LLM calls (extraction + report)",
}

_TOKEN_CASES = [
    ["chest_pain", "sweating", "breathlessness"],
    ["cough", "high_fever"],
    ["stomach_pain", "vomiting"],
    ["headache"],
    ["skin_rash", "itching"],
]


def _ollama_up() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200 and "llama3.2" in r.text
    except requests.RequestException:
        return False


def _time_ms(fn, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return round((time.perf_counter() - t0) / repeats * 1000, 2)


def run_cost_evaluation(include_llm: bool = True) -> dict:
    out = {"call_model": CALL_MODEL, "latency_ms_per_call": {},
           "note": "Latency is machine-dependent; compare stages, not absolutes."}

    # deterministic stage: classify (adult + pediatric mix)
    cases = [ExtractedCase(raw_symptom_text="x", age_months=600, symptom_tokens=t)
             for t in _TOKEN_CASES]
    out["latency_ms_per_call"]["classify (deterministic)"] = _time_ms(
        lambda: [rules_classify(c) for c in cases], repeats=20)

    # deterministic stage: routing
    db = build_demo_facility_db()
    out["latency_ms_per_call"]["routing (deterministic)"] = _time_ms(
        lambda: route(ClassificationLabel.EMERGENCY, "Anchetty", db), repeats=50)

    # deterministic stage: a full surveillance run (fallback brief, no LLM)
    demo = generate_demo_stream(weeks=8)
    out["latency_ms_per_call"]["surveillance run (deterministic)"] = _time_ms(
        lambda: SurveillanceAgent().run(demo), repeats=5)

    # LLM stage: extraction (Ollama-gated)
    if include_llm and _ollama_up():
        from app.agent1_extraction import OllamaBackend, extract_case
        backend = OllamaBackend()
        msgs = ["chest pain and sweating", "my child has fever and cough", "headache"]
        t0 = time.perf_counter()
        for m in msgs:
            try:
                extract_case(m, backend)
            except Exception:
                pass
        out["latency_ms_per_call"]["extraction LLM (Ollama)"] = round(
            (time.perf_counter() - t0) / len(msgs) * 1000, 2)
    else:
        out["latency_ms_per_call"]["extraction LLM (Ollama)"] = None
        out["note"] += " LLM extraction latency omitted (Ollama not reachable)."

    return out


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 64)
    print("SYSTEMS COST")
    print("=" * 64)
    print("  LLM call model (per case):")
    for k, v in ev["call_model"].items():
        print(f"    {k}: {v}")
    print("\n  Latency per call (ms, machine-dependent):")
    for k, v in ev["latency_ms_per_call"].items():
        print(f"    {k}: {v}")
    out = out_dir / "eval_cost_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("systems cost (LLM calls + latency)")
    print("Measuring systems cost...")
    print_report(run_cost_evaluation(), out_dir)
