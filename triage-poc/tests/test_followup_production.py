"""Locks tests/evaluate_followup_production.py: the active-follow-up numbers of
the SHIPPED step functions on per-fold, fully out-of-sample classifiers.

These are the numbers the paper should quote for the shipped system. They are
asserted against the design bound (under-call <= alpha) rather than against
the harness copy, and the known gap to the harness copy is locked so it is
reported, not rediscovered."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import DEFAULT_CP_ALPHA
from tests.evaluate_followup_production import run_production_followup_evaluation

EV = run_production_followup_evaluation()
TRACKED = json.loads((Path(__file__).resolve().parent / "eval_followup_production_results.json").read_text())


def test_reproduces_tracked_baseline_exactly():
    assert json.loads(json.dumps(EV)) == TRACKED


def test_same_304_held_out_messages():
    assert EV["n_cases"] == 304


def test_under_call_stays_within_the_design_bound_before_and_after():
    assert EV["under_triage_before"] <= DEFAULT_CP_ALPHA
    assert EV["under_triage_after"] <= DEFAULT_CP_ALPHA


def test_follow_up_resolves_most_uncertain_cases_in_few_questions():
    assert EV["resolution_rate_of_uncertain"] >= 0.7
    assert EV["mean_questions_on_uncertain"] <= 3
    assert EV["over_triage_after"] < EV["over_triage_before"]


def test_unresolved_cases_are_deferred_not_dropped():
    unresolved = 1 - EV["resolution_rate_of_uncertain"]
    assert EV["deferred_rate_of_uncertain"] == pytest.approx(unresolved, abs=1e-3)


def test_answer_source_is_declared_as_simulated():
    assert "simulated" in EV["answer_source"] and "oracle" in EV["answer_source"]


def test_gap_to_the_harness_copy_is_recorded():
    h = EV["harness_copy_for_comparison"]
    # the harness reuses a full-data conformal threshold; out-of-sample calibration
    # yields fewer initially-uncertain cases and a higher post-follow-up under-call
    assert EV["n_started_uncertain"] < h["n_started_uncertain"]
    assert EV["under_triage_after"] > h["under_triage_after"]
