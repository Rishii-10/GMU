"""Practical / SDG-impact results -- the 'with vs without, for a real person'
numbers, in common-man terms. Aligned to UN SDG-3 (access to health services)
and SDG-10 (reduced inequalities).

These are demonstrated on the sandbox dataset, NOT a field study. Anything about
trips or money saved is an explicitly-labelled illustrative model with stated
assumptions -- never a measured field outcome.

  A  Access & cost    -- how many messages become a clear 'care at home' vs 'go
                         now' decision (potential avoided journeys), and how
                         often the naive 'just go to the nearest clinic' choice
                         would send an emergency somewhere that can't treat it.
  B  Language reach    -- the pipeline serves English/Hindi/Hinglish; an
                         English-only tool would drop the non-English share.
                         (Read from the robustness / end-to-end harness outputs.)
  C  Caller effort     -- how few yes/no questions reach a decision.
                         (Read from the follow-up harness output.)

Run:  PYTHONPATH=. python tests/evaluate_impact.py            # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_impact.py --write    # -> tests/eval_impact_results.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.routing.demo_facilities import FACILITIES, VILLAGES, build_demo_facility_db
from app.routing.router import route
from app.schemas import ClassificationLabel
from tests.evaluate_oos import _rate, run_evaluation

_TESTS = Path(__file__).resolve().parent

# Illustrative cost assumptions (stated, not measured). A rural clinic trip costs
# roughly half a working day plus travel.
_TRIP_HOURS = 4.0
_TRIP_COST_INR = 300.0
_PER_N = 1000


def _haversine_km(a, b) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _emergency_eligible(f) -> bool:
    return f.has_emergency_care and f.has_doctor_24hr


def result_a_access() -> dict:
    """Decision mix (potential avoided trips) + naive-nearest vs eligible routing."""
    oos = run_evaluation()
    dist = oos.get("label_distribution", {})
    total = sum(dist.values()) or 1
    home = dist.get("MILD", 0) + dist.get("MODERATE", 0)
    go = dist.get("EMERGENCY", 0) + dist.get("SEVERE", 0)
    info = dist.get("INCOMPLETE_ASSESSMENT", 0)

    # routing: naive 'nearest clinic' vs the router's nearest *eligible* facility
    db = build_demo_facility_db()
    mismatch = ineligible = n = 0
    for village, coord in VILLAGES.items():
        n += 1
        naive = min(FACILITIES, key=lambda f: _haversine_km(coord, (f.lat, f.lon)))
        if not _emergency_eligible(naive):
            ineligible += 1
        disp = route(ClassificationLabel.EMERGENCY, village, db)
        chosen = disp.facility.name if disp.facility else None
        if chosen != naive.name:
            mismatch += 1

    avoided = home / total
    return {
        "n_messages": total,
        "care_at_home_share": _rate(home, total),
        "go_now_share": _rate(go, total),
        "needs_more_info_share": _rate(info, total),
        "emergency_naive_nearest_ineligible_rate": _rate(ineligible, n),
        "emergency_route_differs_from_naive_nearest_rate": _rate(mismatch, n),
        "illustrative_model": {
            "assumptions": f"1 avoided trip ~= {_TRIP_HOURS} h and ~Rs {_TRIP_COST_INR:g} per caller (stated, not measured)",
            f"per_{_PER_N}_callers_potential_avoided_trips": round(avoided * _PER_N),
            f"per_{_PER_N}_callers_hours_saved": round(avoided * _PER_N * _TRIP_HOURS),
            f"per_{_PER_N}_callers_cost_saved_inr": round(avoided * _PER_N * _TRIP_COST_INR),
        },
        "sdg": "SDG-3.8 access to essential health services; SDG-10 rural equity",
    }


def _read(name: str) -> dict | None:
    p = _TESTS / name
    return json.loads(p.read_text()) if p.exists() else None


def result_b_language() -> dict:
    """Language reach, read from the robustness / end-to-end harnesses."""
    out = {"sdg": "SDG-10 inclusion regardless of language; SDG-3 reaching underserved groups",
           "note": "English/Hindi/Hinglish; an English-only pipeline drops the non-English share."}
    rob = _read("eval_robustness_results.json")
    e2e = _read("eval_end_to_end_results.json")
    if rob:
        dc = rob.get("decision_consistency", {})
        out["repeat_agreement"] = dc.get("repeat_label_agreement")
        out["paraphrase_agreement"] = dc.get("paraphrase_label_agreement")
        rc = rob.get("reply_coverage", {})
        out["reply_not_understood_rate"] = rc.get("reply_not_understood_rate")
        out["hedge_coercion_rate"] = rc.get("hedge_coercion_rate")
    if e2e:
        lay = (e2e.get("modes") or {}).get("lay") or {}
        out["code_mixed_lay_token_recall"] = lay.get("token_recall")
        out["code_mixed_lay_under_triage"] = lay.get("end_to_end_under_triage")
    if not rob and not e2e:
        out["note"] += " (Generate eval_robustness_results.json / eval_end_to_end_results.json for the numbers.)"
    return out


def result_c_effort() -> dict:
    """Caller effort, read from the follow-up harness (or computed if absent)."""
    out = {"sdg": "SDG-3.8 usable by low-literacy, low-connectivity callers",
           "framing": "answer ~2 yes/no questions instead of a full form or being guessed at"}
    fu = _read("eval_followup_results.json")
    if fu:
        m = (fu.get("strategies", {}).get("selector") or {}).get("with_negative_evidence") or {}
        out["mean_questions_to_decision"] = m.get("mean_questions_on_uncertain")
        out["resolution_rate_of_uncertain"] = m.get("resolution_rate_of_uncertain")
    else:
        from app.disease_classifier import get_default_classifier
        from app.disease_kb import DiseaseKB
        from tests.evaluate_followup import _run_one, _summarize
        m = _summarize(_run_one("selector", True, DiseaseKB.load(), get_default_classifier()))
        out["mean_questions_to_decision"] = m["mean_questions_on_uncertain"]
        out["resolution_rate_of_uncertain"] = m["resolution_rate_of_uncertain"]
    return out


def run_impact_evaluation() -> dict:
    return {
        "disclaimer": "Demonstrated on the sandbox dataset; trip/cost figures are an "
                      "illustrative model with stated assumptions, not field measurements.",
        "A_access_and_cost": result_a_access(),
        "B_language_reach": result_b_language(),
        "C_caller_effort": result_c_effort(),
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 72)
    print("PRACTICAL / SDG IMPACT (demonstration)")
    print("=" * 72)
    a = ev["A_access_and_cost"]
    print("  A. Access & cost")
    print(f"     care-at-home share: {a['care_at_home_share']} | go-now: {a['go_now_share']}")
    print(f"     emergency: naive nearest clinic is INELIGIBLE "
          f"{a['emergency_naive_nearest_ineligible_rate']:.0%} of villages")
    print(f"     illustrative: {a['illustrative_model'][f'per_{_PER_N}_callers_potential_avoided_trips']} "
          f"avoided trips / {_PER_N} callers")
    b = ev["B_language_reach"]
    print("  B. Language reach")
    print(f"     code-mixed lay token recall: {b.get('code_mixed_lay_token_recall')} | "
          f"paraphrase agreement: {b.get('paraphrase_agreement')}")
    c = ev["C_caller_effort"]
    print("  C. Caller effort")
    print(f"     ~{c.get('mean_questions_to_decision')} questions to a decision; "
          f"resolves {c.get('resolution_rate_of_uncertain')} of uncertain cases")
    out = out_dir / "eval_impact_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("practical / SDG impact (demonstration)")
    print("Computing practical/SDG impact...")
    print_report(run_impact_evaluation(), out_dir)
