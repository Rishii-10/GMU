"""Phase 3.4: the AHP and routing stages are measured, not just asserted.

These were the two framework stages with unit tests but no evaluation, so the
paper claimed four contributions while evidencing two. The assertions below
keep the two new measurements honest rather than decorative:

  * AHP must carry information the severity tier does not. If
    within-tier pair discrimination collapsed toward 0, AHP would be a
    relabelling of the tier it sits on and the "urgency prioritization"
    contribution would be empty -- so that is asserted, not just reported.
  * AHP must stay SOUND: monotone in the severity attribute, and a WHO danger
    sign must always pin the output to ESI-1 / score 10 regardless of every
    other attribute.
  * Routing must match an exhaustive brute-force search, and must never
    dispatch a life-threatening case to a facility lacking emergency care or a
    24-hour doctor.

Deterministic, no Ollama/FAISS -- runs in CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.evaluate_prioritization_routing import run_evaluation

EV = run_evaluation()
AHP = EV["ahp_prioritization"]
ROUTING = EV["routing"]


# --- AHP ------------------------------------------------------------------


def test_ahp_sweep_covers_a_meaningful_case_space():
    assert AHP["n_scored"] >= 500
    assert AHP["n_attributes"] == 6


def test_ahp_adds_information_beyond_the_severity_tier():
    """The contribution claim. A tie-everything scorer would be worthless as a
    within-tier prioritizer."""
    assert AHP["within_tier_pair_discrimination"] > 0.25, (
        "AHP barely distinguishes patients inside a tier -- it would be a "
        "relabelling of the tier, not a prioritizer"
    )


def test_every_tier_has_internal_score_spread():
    for tier, spread in AHP["score_spread_by_tier"].items():
        assert spread["distinct_scores"] >= 2, f"{tier} collapses to one score"
        assert spread["range"] >= 1, f"{tier} has no within-tier range"


def test_ahp_is_monotone_in_the_severity_attribute():
    assert AHP["monotone_in_severity_attribute"]


def test_danger_sign_override_dominates_every_other_attribute():
    assert AHP["danger_sign_always_forces_esi_1_and_score_10"]


def test_ahp_makes_no_clinical_correctness_claim():
    # Guards the scope statement the paper relies on.
    assert "none" in AHP["clinical_claim"]
    assert "does NOT set the triage label" in AHP["role"]


# --- Routing --------------------------------------------------------------


def test_router_matches_exhaustive_brute_force_search():
    assert ROUTING["optimality_vs_brute_force"] == 1.0


def test_emergency_dispatch_never_violates_eligibility():
    # A life-threatening case must only go somewhere with emergency care AND a
    # 24-hour doctor. This is a hard safety rule, so 1.0 or it is a bug.
    assert ROUTING["emergency_eligibility_respected"] == 1.0


def test_urgent_dispatch_respects_opening_hours():
    assert ROUTING["urgent_open_hours_respected"] == 1.0


def test_routing_scenarios_cover_all_villages_and_both_urgencies():
    from app.routing.demo_facilities import VILLAGES

    assert ROUTING["n_scenarios"] == len(VILLAGES) * 2
    assert {c["urgency"] for c in ROUTING["scenarios"]} == {"EMERGENCY", "URGENT"}


def test_routing_reports_its_synthetic_scope():
    # The network is a 10-facility demo set with a mock distance provider; the
    # harness must say so rather than imply real-world travel times.
    assert "synthetic" in ROUTING["network"]
    assert "not real-world travel time" in ROUTING["network"]
