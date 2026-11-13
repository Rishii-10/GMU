"""Locks tests/evaluate_threshold_cv.py: the symptom-matching accuracy the paper
quotes must be the cross-validated one (threshold chosen on training items
only), with the in-sample figure kept alongside for transparency."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRACKED = json.loads((Path(__file__).resolve().parent / "eval_threshold_cv_results.json").read_text())


@pytest.fixture(scope="module")
def ev():
    try:
        from tests.evaluate_threshold_cv import run_threshold_cv
        return run_threshold_cv()
    except ImportError as e:
        pytest.skip(f"faiss-cpu / sentence-transformers not installed: {e}")


def test_reproduces_tracked_baseline(ev):
    assert ev == TRACKED


def test_cross_validated_accuracy_is_reported_and_not_above_in_sample(ev):
    assert ev["kfold"]["accuracy"] <= ev["in_sample_accuracy"]
    assert ev["leave_one_out"]["accuracy"] <= ev["in_sample_accuracy"]
    assert ev["kfold"]["accuracy"] >= 0.9


def test_shipped_threshold_is_the_in_sample_choice(ev):
    assert ev["shipped_threshold"] == ev["in_sample_best_threshold"]
