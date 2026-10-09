"""Tier 2: the sensitivity sweeps run, and the safety bound the paper states
holds at every alpha -- not just at the production setting.

The important assertion here is `bound_holds_at_every_alpha`: worst-case
labelling over a conformal set gives P(under-triage) <= alpha, so if some alpha
ever produced more under-triage than alpha itself, the stated guarantee would
be false and the paper's central claim would not survive review.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import DEFAULT_CP_ALPHA, EMERGENCY_SAFETY_FACTOR
from tests.evaluate_sensitivity import ALPHA_GRID, EMERGENCY_FACTOR_GRID, run_sensitivity

EV = run_sensitivity()


def test_both_sweeps_cover_their_grids():
    assert [r["alpha"] for r in EV["alpha_sweep"]] == ALPHA_GRID
    assert [r["emergency_safety_factor"] for r in EV["emergency_factor_sweep"]] == EMERGENCY_FACTOR_GRID


def test_under_triage_bound_holds_at_every_alpha():
    # The paper's safety guarantee, checked across the whole grid.
    assert EV["bound_holds_at_every_alpha"]
    for r in EV["alpha_sweep"]:
        assert r["under_triage_rate_emergency_severe"] <= r["alpha"] + 1e-9


def test_production_settings_are_in_the_swept_grids():
    # A swept constant must actually bracket the shipped value, otherwise the
    # sensitivity analysis says nothing about the deployed system.
    assert DEFAULT_CP_ALPHA in ALPHA_GRID
    assert EMERGENCY_SAFETY_FACTOR in EMERGENCY_FACTOR_GRID


def test_emergency_boost_is_not_load_bearing_for_safety():
    """Documents the negative result: the EMERGENCY class-imbalance boost does
    not reduce under-triage at ANY factor (worst-case set labelling already
    does that job), and at some factors it adds over-triage. If a future change
    ever makes the boost genuinely useful, this test fails and the claim in
    docs/PAPER_EVIDENCE.md must be revisited."""
    assert EV["emergency_boost_improves_under_triage"] is False
    no_boost = next(r for r in EV["emergency_factor_sweep"]
                    if r["emergency_safety_factor"] == 1.0)
    assert no_boost["under_triage_rate_emergency_severe"] == 0.0


def test_lac_fallback_rate_is_reported():
    """The empty-LAC fallback returns every candidate instead of a conformal
    set. Its rate must be visible at each alpha so the paper cannot describe
    fallback-driven behaviour as conformal filtering."""
    for r in EV["alpha_sweep"]:
        assert 0.0 <= r["lac_empty_fallback_rate"] <= 1.0
