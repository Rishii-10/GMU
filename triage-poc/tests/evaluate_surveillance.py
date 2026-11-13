"""Evaluation of Agent 2 (surveillance) as a detector, not a single demo.

The paper previously showed one injected outbreak (alert) and one flat stream
(silence). This harness measures the detector over a grid:

  * DETECTION vs outbreak size x background noise: each run draws a synthetic
    5-village x 8-week stream in which every (village, disease) pair has a
    Poisson weekly background (rate = noise level) over 8 diseases, including
    the outbreak disease itself, then adds K extra under-5 Gastroenteritis cases
    to one random village in the final week. Detection = the agent SURFACES an
    alert (WATCH or ALERT) for that exact (village, disease).
  * FALSE ALARMS: any surfaced alert for a different (village, disease) pair,
    reported per area-week (5 areas x 1 evaluated week per run); K=0 rows are
    pure-noise streams.
  * TIME TO DETECT: a ramping outbreak (2, 4, 6, 8, 10, 12 cases/week) starts
    in week 8 of a 14-week stream; the agent is run at the end of every week
    with fresh state, and the delay is weeks from onset to the first surfaced
    alert (None = not detected within the ramp).

Synthetic by construction (seeded); it measures the DETECTOR AND ALERT POLICY,
not real-world epidemiology. Deterministic: re-running reproduces identically.

Run (from triage-poc/):  PYTHONPATH=. python tests/evaluate_surveillance.py [--write]
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import severity_for_disease
from app.integrations.messaging import MockProvider
from app.routing.demo_facilities import VILLAGES, build_demo_facility_db
from app.surveillance.agent import SurveillanceAgent
from app.surveillance.schemas import SurveillanceCase
from app.surveillance.state import SurveillanceState
from app.surveillance.tools import SurveillanceTools

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)
OUTBREAK_DISEASE = "Gastroenteritis"
DISEASES = [
    OUTBREAK_DISEASE, "Common Cold", "Fungal infection", "Allergy", "Migraine",
    "GERD", "Bronchial Asthma", "Urinary tract infection",
]
NOISE_LEVELS = {"low": 0.5, "medium": 1.5, "high": 3.0}  # Poisson mean per (village, disease, week)
OUTBREAK_SIZES = [0, 3, 5, 8, 12, 16]
N_SEEDS = 30
WEEKS = 8
RAMP = [2, 4, 6, 8, 10, 12]
RAMP_ONSET_WEEK = 8
RAMP_TOTAL_WEEKS = RAMP_ONSET_WEEK + len(RAMP)

_FACILITY_DB = build_demo_facility_db()
_SEVERITY = {d: severity_for_disease(d) for d in DISEASES}


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _case(area, disease, when, age):
    return SurveillanceCase(area=area, disease=disease, symptom_tokens=[], severity=_SEVERITY[disease],
                            age_months=age, recorded_at=when, language="en")


def _stream(rng, weeks, end, noise, outbreak_area=None, extra_by_week=None):
    cases = []
    for w in range(weeks):  # week 0 oldest ... weeks-1 current (ends at `end`)
        anchor = end - timedelta(days=7 * (weeks - 1 - w))

        def when():
            return anchor - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 23))

        for area in VILLAGES:
            for d in DISEASES:
                for _ in range(_poisson(rng, noise)):
                    cases.append(_case(area, d, when(), rng.randint(1, 900)))
        for _ in range((extra_by_week or {}).get(w, 0)):
            cases.append(_case(outbreak_area, OUTBREAK_DISEASE, when(), rng.randint(1, 59)))
    return cases


def _agent():
    state = SurveillanceState(":memory:")
    return SurveillanceAgent(state=state, tools=SurveillanceTools(state, _FACILITY_DB, MockProvider()))


def detection_grid() -> list[dict]:
    rows = []
    for noise_name, lam in NOISE_LEVELS.items():
        for size in OUTBREAK_SIZES:
            detected, false_alarms, alerted_at_alert_level = 0, 0, 0
            for seed in range(N_SEEDS):
                rng = random.Random(1000 * seed + size * 7 + int(lam * 10))
                area = rng.choice(sorted(VILLAGES))
                stream = _stream(rng, WEEKS, NOW, lam, area, {WEEKS - 1: size})
                rep = _agent().run(stream, now=NOW)
                hit = [a for a in rep.alerts if a.area == area and a.disease == OUTBREAK_DISEASE]
                if size > 0 and hit:
                    detected += 1
                    alerted_at_alert_level += hit[0].level.value == "ALERT"
                false_alarms += sum(
                    1 for a in rep.alerts
                    if size == 0 or not (a.area == area and a.disease == OUTBREAK_DISEASE)
                )
            rows.append({
                "noise": noise_name,
                "background_rate_per_pair_week": lam,
                "outbreak_extra_cases": size,
                "runs": N_SEEDS,
                "detection_rate": None if size == 0 else round(detected / N_SEEDS, 4),
                "alert_level_share_of_detections": None if not detected else round(alerted_at_alert_level / detected, 4),
                "false_alarms_per_area_week": round(false_alarms / (N_SEEDS * len(VILLAGES)), 4),
            })
    return rows


def time_to_detect() -> list[dict]:
    rows = []
    for noise_name, lam in NOISE_LEVELS.items():
        delays = []
        for seed in range(N_SEEDS):
            rng = random.Random(77 + seed)
            area = rng.choice(sorted(VILLAGES))
            extra = {RAMP_ONSET_WEEK + i: n for i, n in enumerate(RAMP)}
            end = NOW
            stream = _stream(rng, RAMP_TOTAL_WEEKS, end, lam, area, extra)
            delay = None
            for i in range(len(RAMP)):
                at = end - timedelta(days=7 * (len(RAMP) - 1 - i))
                visible = [c for c in stream if c.recorded_at <= at]
                rep = _agent().run(visible, now=at)
                if any(a.area == area and a.disease == OUTBREAK_DISEASE for a in rep.alerts):
                    delay = i  # 0 = detected in the onset week
                    break
            delays.append(delay)
        found = [d for d in delays if d is not None]
        rows.append({
            "noise": noise_name,
            "background_rate_per_pair_week": lam,
            "ramp_cases_per_week": RAMP,
            "runs": N_SEEDS,
            "detected_within_ramp": round(len(found) / N_SEEDS, 4),
            "mean_weeks_to_detect": round(mean(found), 3) if found else None,
            "median_weeks_to_detect": median(found) if found else None,
            "cases_in_week_of_detection_median": RAMP[int(median(found))] if found else None,
        })
    return rows


def run_surveillance_evaluation() -> dict:
    return {
        "regime": "synthetic, seeded: 5 demo villages x 8 diseases, Poisson weekly background per "
                  "(village, disease); outbreak = extra under-5 Gastroenteritis cases in one random "
                  "village. Measures the detector + alert policy, NOT real epidemiology.",
        "detector": "baseline deviation (current week vs prior 4 weeks), min 5 cases, z >= 2, "
                    "severity-weighted; WATCH >= 3, ALERT >= 6 (app.surveillance.detect / alert_policy)",
        "detected_means": "a WATCH or ALERT surfaced for the injected (village, disease)",
        "n_seeds": N_SEEDS,
        "detection": detection_grid(),
        "time_to_detect": time_to_detect(),
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 78)
    print("SURVEILLANCE DETECTOR EVALUATION")
    print("=" * 78)
    print(f"  {'noise':<8}{'extra':>6}{'detect':>9}{'ALERT share':>13}{'false alarms/area-wk':>23}")
    for r in ev["detection"]:
        d = "  -  " if r["detection_rate"] is None else f"{r['detection_rate']:.2f}"
        a = "  -  " if r["alert_level_share_of_detections"] is None else f"{r['alert_level_share_of_detections']:.2f}"
        print(f"  {r['noise']:<8}{r['outbreak_extra_cases']:>6}{d:>9}{a:>13}{r['false_alarms_per_area_week']:>23.4f}")
    print("\n  time to detect (ramp " + str(RAMP) + ")")
    for r in ev["time_to_detect"]:
        print(f"  {r['noise']:<8} detected {r['detected_within_ramp']:.2f}  median weeks {r['median_weeks_to_detect']}"
              f"  mean {r['mean_weeks_to_detect']}")
    out = out_dir / "eval_surveillance_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("surveillance detector evaluation (synthetic grid)")
    print_report(run_surveillance_evaluation(), out_dir)
