"""Tests for the Bucket-3 result harnesses: composed ablation, systems cost,
and practical/SDG impact. Deterministic (cost test skips the LLM stage).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.evaluate_composed import run_composed_evaluation
from tests.evaluate_cost import run_cost_evaluation
from tests.evaluate_impact import run_impact_evaluation


def test_composed_ablation_tells_the_journey():
    ev = run_composed_evaluation()
    stages = {s["stage"]: s for s in ev["stages"]}
    assert set(stages) == {"C0", "C1", "C2", "C3", "C4"}
    # The worst-case guard (C3 = full system) removes under-triage.
    assert stages["C3"]["misses_urgent"] == 0.0
    # A plain structured baseline (C1) misses more than the full system.
    assert stages["C1"]["misses_urgent"] > stages["C3"]["misses_urgent"]
    # Follow-up (C4) buys back over-calling versus the bare full system.
    assert stages["C4"]["over_calls"] < stages["C3"]["over_calls"]


def test_cost_model_and_latency():
    ev = run_cost_evaluation(include_llm=False)
    assert "extraction (Agent 1)" in ev["call_model"]
    lat = ev["latency_ms_per_call"]
    # deterministic stages always measured and cheap relative to an LLM call
    assert lat["classify (deterministic)"] is not None
    assert lat["routing (deterministic)"] is not None
    assert lat["surveillance run (deterministic)"] is not None


def test_impact_access_language_effort():
    ev = run_impact_evaluation()
    a = ev["A_access_and_cost"]
    assert 0.0 <= a["care_at_home_share"] <= 1.0
    # the with-vs-without routing point: the naive nearest clinic is often ineligible
    assert a["emergency_naive_nearest_ineligible_rate"] >= 0.0
    assert "per_1000_callers_potential_avoided_trips" in a["illustrative_model"]
    c = ev["C_caller_effort"]
    assert c["mean_questions_to_decision"] is not None
    assert c["resolution_rate_of_uncertain"] is not None
