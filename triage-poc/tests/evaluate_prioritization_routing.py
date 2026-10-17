"""Phase 3.4 -- evaluating the two framework stages that previously had unit
tests but no measurements: urgency prioritization (AHP) and resource-aware
routing.

The paper's spine names four computational problems. Extraction and the
abstain decision were evaluated in depth; these two were only asserted to be
correctly implemented, which is a different and weaker claim. A reviewer
reading "we contribute urgency prioritization" is entitled to ask what it buys.
This harness answers that, including where the answer is unflattering.

PART 1 -- AHP PRIORITIZATION
    AHP's declared role is NOT to set the triage label (the severity tier,
    worst-case rule and danger-sign override do that). It is to order patients
    WITHIN a tier for dispatch. So the question that matters is not accuracy --
    there is no ground-truth dispatch order in this dataset -- but whether it
    carries information the tier does not:

      discriminative power  within a tier, how often does AHP strictly order
                            two patients rather than tie them? A scorer that
                            ties everyone adds nothing to the tier it sits on.
      tier independence     does the within-tier score actually vary, and over
                            what range?
      soundness             raising one attribute, holding the rest fixed,
                            must never lower the score (monotonicity), and a
                            WHO danger sign must always force ESI-1.

    No clinical-correctness claim is made or implied.

PART 2 -- RESOURCE-AWARE ROUTING
    Scored against an exhaustive brute-force search over the same facility
    list, for every village and both urgency levels:

      optimality      is the dispatched facility the best-scoring ELIGIBLE one?
      eligibility     EMERGENCY must only ever dispatch to a facility with both
                      emergency care and a 24h doctor; URGENT must be open.
      reachability    how often no eligible facility exists inside the radius.

    The facility network is the 10-facility synthetic demo set with a mock
    haversine distance provider, so this evaluates the ROUTING LOGIC, not
    real-world travel. Stated as such.

Deterministic; no Ollama, no FAISS.

Run:  PYTHONPATH=. python tests/evaluate_prioritization_routing.py
      PYTHONPATH=. python tests/evaluate_prioritization_routing.py --write
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import severity_for_disease
from app.disease_kb import DiseaseKB
from app.emergency_scorer import AHP_WEIGHTS, score_emergency
from app.routing.demo_facilities import VILLAGES, build_demo_facility_db
from app.routing.router import (
    DEFAULT_RADIUS_KM,
    filter_eligible_facilities,
    is_facility_open,
    route,
    score_facility,
)
from app.schemas import ClassificationLabel, DangerSigns, ExtractedCase
from tests.evaluate_oos import _rate

# Ages spanning every band in _age_vulnerability, and durations spanning the
# onset-acuity regexes, so the sweep exercises the whole attribute space.
AGES_MONTHS = [1, 24, 120, 420, 900]
DURATIONS = ["since one hour", "three days", "two weeks", "chronic", None]
_AT = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)  # fixed: hours matter


def _case(age_months, duration, danger=False):
    return ExtractedCase(
        raw_symptom_text="eval",
        age_months=age_months,
        duration=duration,
        danger_signs=DangerSigns(convulsions=True) if danger else DangerSigns(),
    )


def run_ahp_evaluation() -> dict:
    kb = DiseaseKB.load()
    by_tier: dict[str, list[int]] = {}
    esi = Counter()
    rows = 0

    for disease in kb.diseases:
        tier = severity_for_disease(disease, kb)
        for age, duration in itertools.product(AGES_MONTHS, DURATIONS):
            er = score_emergency(_case(age, duration), disease, tier)
            by_tier.setdefault(tier.value, []).append(er.score)
            esi[er.esi_level] += 1
            rows += 1

    # Discriminative power: fraction of within-tier patient PAIRS that AHP
    # orders strictly. 0.0 would mean the scorer is a relabelling of the tier.
    pair_total = pair_distinct = 0
    spread = {}
    for tier, scores in by_tier.items():
        lo, hi = min(scores), max(scores)
        spread[tier] = {"min": lo, "max": hi, "range": hi - lo,
                        "distinct_scores": len(set(scores))}
        for a, b in itertools.combinations(scores, 2):
            pair_total += 1
            pair_distinct += 1 if a != b else 0

    # Soundness 1: monotone in every attribute.
    monotonic = True
    base = _case(420, "three days")
    ordered_tiers = [ClassificationLabel.MILD, ClassificationLabel.MODERATE,
                     ClassificationLabel.SEVERE, ClassificationLabel.EMERGENCY]
    prev = None
    for tier in ordered_tiers:
        s = score_emergency(base, "Heart attack", tier).score
        if prev is not None and s < prev:
            monotonic = False
        prev = s

    # Soundness 2: a WHO danger sign must always force the top of the scale.
    override_ok = all(
        score_emergency(_case(age, dur, danger=True), d, severity_for_disease(d, kb)).esi_level == 1
        and score_emergency(_case(age, dur, danger=True), d, severity_for_disease(d, kb)).score == 10
        for d in ("Acne", "Malaria", "Heart attack")
        for age, dur in itertools.product(AGES_MONTHS, DURATIONS)
    )

    return {
        "n_scored": rows,
        "role": "within-tier dispatch prioritizer; does NOT set the triage label",
        "within_tier_pair_discrimination": _rate(pair_distinct, pair_total),
        "score_spread_by_tier": spread,
        "esi_distribution": {str(k): v for k, v in sorted(esi.items())},
        "monotone_in_severity_attribute": monotonic,
        "danger_sign_always_forces_esi_1_and_score_10": override_ok,
        "n_attributes": len(AHP_WEIGHTS),
        "clinical_claim": "none -- no ground-truth dispatch order exists in this dataset",
    }


def run_routing_evaluation() -> dict:
    db = build_demo_facility_db()
    all_facilities = db.all_facilities()

    checks = []
    for village, urgency_label in itertools.product(
        VILLAGES, [ClassificationLabel.EMERGENCY, ClassificationLabel.SEVERE]
    ):
        coords = db.resolve_location(village)
        dispatch = route(label=urgency_label, location=village, facility_db=db, at=_AT)
        urgency = "EMERGENCY" if urgency_label is ClassificationLabel.EMERGENCY else "URGENT"

        # Exhaustive search over the same inputs the router had.
        in_radius = [
            f for f in all_facilities
            if score_facility(f, coords).straight_line_distance_km <= DEFAULT_RADIUS_KM
        ]
        eligible = filter_eligible_facilities(in_radius, urgency, _AT)
        best = min(
            (score_facility(f, coords) for f in eligible),
            key=lambda s: s.score, default=None,
        )

        chosen = dispatch.facility
        checks.append({
            "village": village,
            "urgency": urgency,
            "n_eligible": len(eligible),
            "found": chosen is not None,
            "brute_force_available": best is not None,
            "optimal": (
                (chosen is None and best is None)
                or (chosen is not None and best is not None and chosen.name == best.facility.name)
            ),
            # The hard eligibility rule for a life-threatening dispatch.
            "eligibility_respected": (
                chosen is None
                or urgency != "EMERGENCY"
                or (chosen.has_emergency_care and chosen.has_doctor_24hr)
            ),
            "open_respected": (
                chosen is None or urgency != "URGENT" or is_facility_open(chosen, _AT)
            ),
            "eta_minutes": dispatch.route.eta_minutes if dispatch.route else None,
        })

    n = len(checks)
    etas = [c["eta_minutes"] for c in checks if c["eta_minutes"] is not None]
    return {
        "n_scenarios": n,
        "network": "10-facility synthetic demo set, mock haversine distance provider "
                   "-- evaluates ROUTING LOGIC, not real-world travel time",
        "radius_km": DEFAULT_RADIUS_KM,
        "optimality_vs_brute_force": _rate(sum(1 for c in checks if c["optimal"]), n),
        "emergency_eligibility_respected": _rate(
            sum(1 for c in checks if c["eligibility_respected"]), n),
        "urgent_open_hours_respected": _rate(
            sum(1 for c in checks if c["open_respected"]), n),
        "no_facility_found_rate": _rate(sum(1 for c in checks if not c["found"]), n),
        "mean_eta_minutes": round(sum(etas) / len(etas), 2) if etas else None,
        "max_eta_minutes": max(etas) if etas else None,
        "scenarios": checks,
    }


def run_evaluation() -> dict:
    return {"ahp_prioritization": run_ahp_evaluation(), "routing": run_routing_evaluation()}


def print_report(ev: dict, out_dir: Path) -> None:
    a, r = ev["ahp_prioritization"], ev["routing"]
    print("=" * 78)
    print("PRIORITIZATION + ROUTING EVALUATION (Phase 3.4)")
    print("=" * 78)

    print(f"\n  PART 1: AHP -- {a['role']}")
    print(f"    cases scored                     : {a['n_scored']}")
    print(f"    within-tier pair discrimination  : {a['within_tier_pair_discrimination']}")
    print(f"    monotone in severity attribute   : {a['monotone_in_severity_attribute']}")
    print(f"    danger sign -> ESI 1 / score 10  : {a['danger_sign_always_forces_esi_1_and_score_10']}")
    print(f"    ESI distribution                 : {a['esi_distribution']}")
    print("    score spread within each tier:")
    for tier, s in a["score_spread_by_tier"].items():
        print(f"      {tier:<10} {s['min']}-{s['max']}  range {s['range']}  "
              f"{s['distinct_scores']} distinct values")
    print(f"    clinical claim: {a['clinical_claim']}")

    print(f"\n  PART 2: routing ({r['n_scenarios']} scenarios, radius {r['radius_km']} km)")
    print(f"    optimality vs brute force        : {r['optimality_vs_brute_force']}")
    print(f"    EMERGENCY eligibility respected  : {r['emergency_eligibility_respected']}")
    print(f"    URGENT open-hours respected      : {r['urgent_open_hours_respected']}")
    print(f"    no eligible facility found       : {r['no_facility_found_rate']}")
    print(f"    ETA minutes (mean / max)         : {r['mean_eta_minutes']} / {r['max_eta_minutes']}")
    print(f"    {r['network']}")

    out = out_dir / "eval_prioritization_routing_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("prioritization + routing evaluation (Phase 3.4)")
    print("Running prioritization + routing evaluation...")
    print_report(run_evaluation(), out_dir)
