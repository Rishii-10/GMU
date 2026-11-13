"""Locks tests/evaluate_nb_vs_lr.py: the comparison reproduces its tracked
baseline, and the finding the paper must state is asserted in BOTH directions,
so neither half can silently disappear (NB beats full-row LR; short-subset LR
beats NB on accuracy)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("sklearn")
from tests.evaluate_nb_vs_lr import C_GRID, run_nb_vs_lr_evaluation

EV = run_nb_vs_lr_evaluation()
TRACKED = json.loads((Path(__file__).resolve().parent / "eval_nb_vs_lr_results.json").read_text())


def test_same_test_set_as_the_oos_harness():
    oos = json.loads((Path(__file__).resolve().parent / "eval_oos_results.json").read_text())
    assert EV["n_test_cases"] == oos["n"] == 304


def test_reproduces_tracked_baseline():
    for name, m in TRACKED["models"].items():
        for k, v in m.items():
            assert EV["models"][name][k] == pytest.approx(v, abs=2e-3), (name, k)


def test_every_comparator_reported():
    assert len(EV["models"]) == 1 + 2 * len(C_GRID)


def test_nb_beats_lr_trained_on_full_rows():
    v = EV["nb_vs_lr_trained_on_full_rows"]
    assert v["nb_more_accurate"] and v["nb_lower_ece"] and v["nb_lower_brier"]
    assert v["lr_direction_of_miscalibration"] == "underconfident"


def test_lr_trained_on_short_subsets_is_more_accurate_than_nb():
    v = EV["nb_vs_lr_trained_on_short_subsets"]
    assert not v["nb_more_accurate"]
    nb = EV["models"]["naive_bayes"]["top1_accuracy"]
    assert EV["models"]["logreg_short_C=1"]["top1_accuracy"] > nb
