"""Locks the surveillance detector evaluation (tests/evaluate_surveillance.py):
the harness reproduces its tracked baseline exactly, and the headline
properties the paper may state are true of it -- including the unflattering
one (false alarms grow with background noise), which is asserted so it cannot
silently disappear from the results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.evaluate_surveillance import OUTBREAK_SIZES, run_surveillance_evaluation

_TRACKED = Path(__file__).resolve().parent / "eval_surveillance_results.json"
EV = run_surveillance_evaluation()


def test_reproduces_tracked_baseline_exactly():
    assert json.loads(json.dumps(EV, default=str)) == json.loads(_TRACKED.read_text())


def _rows(noise):
    return [r for r in EV["detection"] if r["noise"] == noise]


def test_full_grid_reported():
    assert len(EV["detection"]) == 3 * len(OUTBREAK_SIZES)
    assert all(r["runs"] == EV["n_seeds"] for r in EV["detection"])


def test_detection_rises_with_outbreak_size_and_is_complete_at_8_plus():
    for noise in ("low", "medium", "high"):
        rates = [r["detection_rate"] for r in _rows(noise) if r["outbreak_extra_cases"] >= 5]
        assert rates == sorted(rates)
        assert all(r["detection_rate"] == 1.0 for r in _rows(noise) if r["outbreak_extra_cases"] >= 8)


def test_low_noise_false_alarms_are_rare():
    assert all(r["false_alarms_per_area_week"] <= 0.05 for r in _rows("low"))


def test_false_alarms_grow_with_background_noise():
    pure = {r["noise"]: r["false_alarms_per_area_week"] for r in EV["detection"] if r["outbreak_extra_cases"] == 0}
    assert pure["low"] < pure["medium"] < pure["high"]


def test_ramping_outbreak_is_detected_within_the_ramp():
    for r in EV["time_to_detect"]:
        assert r["detected_within_ramp"] >= 0.9
        assert r["median_weeks_to_detect"] is not None and r["median_weeks_to_detect"] <= 2
