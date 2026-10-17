"""Tier 3/4: the adaptive follow-up loop is evaluated, and the two findings the
paper reports about it are locked in.

Finding A (Tier 4): the information-gain selector is MISALIGNED with a
present-only classifier. It chooses questions by modelling both P(present) and
P(absent), but production Naive Bayes scores present tokens only, so half the
information it optimises for is thrown away. Scored that way it does not beat
asking a random clinically-relevant question.

Finding B: once the classifier can absorb a "no" (Bernoulli absence term), the
selector's objective becomes coherent and it dominates both baselines.

If either finding ever reverses, these tests fail and the paper text must be
revisited.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import DEFAULT_CP_ALPHA
from tests.evaluate_followup import MAX_TURNS, STRATEGIES, run_followup_evaluation

EV = run_followup_evaluation()
_PRESENT = "present_only_production"
_ABSENCE = "with_negative_evidence"


def test_every_policy_is_scored_under_both_evidence_models():
    for s in STRATEGIES:
        for mode in (_PRESENT, _ABSENCE):
            m = EV["strategies"][s][mode]
            assert m["n_started_uncertain"] > 0, f"{s}/{mode} had no uncertain cases to resolve"


def test_answers_come_from_ground_truth_not_simulation():
    assert "ground-truth lookup" in EV["answer_source"]
    assert "no simulation" in EV["answer_source"]


def test_loop_respects_the_production_turn_budget():
    assert EV["max_turns"] == MAX_TURNS
    for s in STRATEGIES:
        for mode in (_PRESENT, _ABSENCE):
            assert EV["strategies"][s][mode]["mean_questions_on_uncertain"] <= MAX_TURNS


def test_followup_actually_reduces_uncertainty():
    for s in STRATEGIES:
        for mode in (_PRESENT, _ABSENCE):
            m = EV["strategies"][s][mode]
            assert m["mean_set_size_after"] < m["mean_set_size_before"]


def test_negative_evidence_substantially_improves_resolution():
    """Finding B, the headline follow-up result."""
    present = EV["strategies"]["selector"][_PRESENT]
    absence = EV["strategies"]["selector"][_ABSENCE]
    assert absence["resolution_rate_of_uncertain"] > 2 * present["resolution_rate_of_uncertain"]
    assert absence["mean_questions_on_uncertain"] < present["mean_questions_on_uncertain"]
    assert absence["mean_set_size_after"] < present["mean_set_size_after"]


def test_selector_only_beats_baselines_when_negatives_count():
    """Finding A: the selector's advantage is conditional on the classifier
    being able to use a 'no'. Documents the misalignment rather than hiding it."""
    present = EV["strategies"]
    assert (present["selector"][_PRESENT]["resolution_rate_of_uncertain"]
            <= present["random"][_PRESENT]["resolution_rate_of_uncertain"])
    # With absence-aware scoring it should dominate BOTH baselines.
    assert (present["selector"][_ABSENCE]["resolution_rate_of_uncertain"]
            > present["random"][_ABSENCE]["resolution_rate_of_uncertain"])
    assert (present["selector"][_ABSENCE]["resolution_rate_of_uncertain"]
            > present["frequency"][_ABSENCE]["resolution_rate_of_uncertain"])


def test_loop_trades_over_triage_down_without_breaking_the_safety_bound():
    m = EV["strategies"]["selector"][_ABSENCE]
    # Resolving uncertainty removes most of the worst-case over-triage ...
    assert m["over_triage_after"] < m["over_triage_before"]
    # ... and the residual under-triage still respects the conformal bound.
    assert m["under_triage_after"] <= DEFAULT_CP_ALPHA


def test_truth_is_rarely_lost_from_the_prediction_set():
    for s in STRATEGIES:
        assert EV["strategies"][s][_ABSENCE]["truth_retained_in_set_rate"] >= 0.90
