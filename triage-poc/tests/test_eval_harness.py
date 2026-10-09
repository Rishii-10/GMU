"""Stage 5: the out-of-sample evaluation harness is reproducible and its core
safety claim holds as a build-failing guarantee.

This runs the real OOS short-message harness (deterministic, seed 42, no Ollama/
FAISS) and asserts the numbers the paper leads with: zero under-triage on
held-out EMERGENCY/SEVERE short messages, meaningful uncertainty (the loop has a
role), high paraphrase consistency, and that the framework beats the
flat-threshold baseline on under-triage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.evaluate_oos import run_evaluation

EV = run_evaluation()


def test_harness_runs_over_all_profiles():
    assert EV["n"] == 304
    assert EV["emergency_severe_n"] > 0


def test_zero_under_triage_on_oos_short_emergency_severe():
    # The headline safety number: no held-out EMERGENCY/SEVERE short message is
    # surfaced below its true tier.
    assert EV["under_triage_rate_emergency_severe"] == 0.0


def test_uncertainty_has_a_role_on_short_messages():
    # Unlike the full-profile eval (nearly all CONFIDENT), short messages must
    # produce a real UNCERTAIN/ABSTAIN rate so abstain-and-ask matters.
    dc = EV["decision_counts"]
    assert dc.get("UNCERTAIN", 0) + dc.get("ABSTAIN", 0) >= 20
    assert 0.0 < EV["abstain_or_ask_rate"] < 1.0


def test_selective_risk_and_silent_error_low():
    assert EV["selective_risk_on_confident"] <= 0.05
    assert EV["silent_error_rate"] <= 0.05


def test_consistency_high_across_token_order():
    assert EV["consistency_label_agreement"] >= 0.95


def test_framework_beats_flat_threshold_baseline_on_under_triage():
    base = EV["baseline_flat_threshold"]["under_triage_rate_emergency_severe"]
    assert base > EV["under_triage_rate_emergency_severe"]


def test_over_triage_reported_as_a_bounded_tradeoff():
    # Over-triage is the honest cost of zero under-triage -- present but bounded.
    assert 0.0 < EV["over_triage_rate_mild_moderate"] < 0.5
