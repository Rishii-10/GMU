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
    # Verifies the DESIGN GUARANTEE, not an empirical discovery: worst-case
    # labelling of a conformal set that contains the truth cannot surface a
    # tier below the truth, so under-triage <= miscoverage <= alpha. This test
    # fails if that chain is ever broken (e.g. worst-case labelling removed, or
    # coverage collapsing).
    assert EV["under_triage_rate_emergency_severe"] == 0.0
    assert EV["under_triage_rate_emergency_severe"] <= EV["under_triage_design_bound_alpha"]


def test_coverage_is_the_premise_of_the_under_triage_guarantee():
    # The guarantee is only as good as conformal coverage, so coverage is the
    # quantity that actually has to be measured.
    assert EV["cp_coverage_true_in_set"] >= 1.0 - EV["alpha"]


def test_uncertainty_has_a_role_on_short_messages():
    # Unlike the full-profile eval (nearly all CONFIDENT), short messages must
    # produce a real UNCERTAIN/ABSTAIN rate so abstain-and-ask matters.
    dc = EV["decision_counts"]
    assert dc.get("UNCERTAIN", 0) + dc.get("ABSTAIN", 0) >= 20
    assert 0.0 < EV["abstain_or_ask_rate"] < 1.0


def test_selective_risk_and_silent_error_low():
    assert EV["selective_risk_on_confident"] <= 0.05
    assert EV["silent_error_rate"] <= 0.05


def test_subset_robustness_is_measured_and_not_vacuous():
    # Replaces the old token-order "consistency" check, which could only ever
    # return 1.0 (the model reduces tokens to a sorted set, so it is
    # order-invariant by construction). This measures something real: whether
    # the tier changes when the caller mentions a DIFFERENT subset of the same
    # true profile. It is a finding, not a guarantee -- so assert only that it
    # is computed over a meaningful sample and has not collapsed.
    assert EV["subset_robustness_pairs_compared"] >= 100
    assert 0.5 < EV["subset_robustness_label_agreement"] <= 1.0


def test_framework_beats_independent_baseline_on_under_triage():
    base = EV["comparisons"]["baseline_flat_nb_top1"]
    assert base["under_triage_rate_emergency_severe"] > EV["under_triage_rate_emergency_severe"]


def test_comparators_are_distinct_computations():
    """Regression guard. The baseline and the ablation were once scored by
    identical code and reported the same number twice. Every comparator must
    differ from every other in its mechanism flags AND the comparator set must
    not be degenerate."""
    defs = EV["variant_definitions"]
    assert len(defs) >= 3
    flag_sets = [tuple(sorted(f.items())) for f in defs.values()]
    assert len(set(flag_sets)) == len(flag_sets), "two comparators share a configuration"

    # The mechanisms must demonstrably do something: removing worst-case
    # labelling has to change the under-triage number, otherwise the framework
    # is not responsible for its own headline result.
    no_worst = EV["comparisons"]["ablation_no_worstcase"]
    assert no_worst["under_triage_rate_emergency_severe"] > EV["under_triage_rate_emergency_severe"]


def test_each_comparator_reports_a_full_metric_set():
    for name, m in EV["comparisons"].items():
        assert m["n"] == EV["n"], f"{name} scored a different number of cases"
        for key in ("under_triage_rate_emergency_severe", "over_triage_rate_mild_moderate",
                    "coverage_confident_fraction", "label_distribution"):
            assert key in m, f"{name} missing {key}"


def test_over_triage_reported_as_a_bounded_tradeoff():
    # Over-triage is the honest cost of zero under-triage -- present but bounded.
    assert 0.0 < EV["over_triage_rate_mild_moderate"] < 0.5
