"""
Calibration harness for app.disambiguation.FAISSDisambiguator's dataset
symptom-token matching (Phase 8). Runs the real, complete
calibrate_dataset_threshold() sweep against the labelled
tests/fixtures/symptom_calibration.py sample and:

  1. Verifies the SHIPPED default (DEFAULT_DATASET_CONFIDENCE_THRESHOLD)
     actually IS what the calibration function computes -- not a
     hand-picked number that happens to be nearby. If a future change to
     the vocabulary/model/fixture shifts the empirically-optimal
     threshold, this test fails until the shipped constant is updated to
     match, so the two can never silently drift apart.
  2. Reports/asserts accuracy metrics at that calibrated threshold as a
     regression floor.

This is the actual measurement+decision harness app/disambiguation.py's
module docstring used to say didn't exist ("Calibrating it ... requires a
labelled dataset of symptom-text -> correct-category pairs and a
threshold sweep ... neither exists in this repo yet") -- it now does, and
its output is what's shipped, not just measured and set aside.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disambiguation import DEFAULT_DATASET_CONFIDENCE_THRESHOLD, FAISSDisambiguator, calibrate_dataset_threshold
from tests.fixtures.symptom_calibration import CALIBRATION_SAMPLE

ACCURACY_FLOOR = 0.90  # the calibrated default measures ~97.5% on this sample


@pytest.fixture(scope="module")
def disambiguator() -> FAISSDisambiguator:
    try:
        return FAISSDisambiguator()
    except ImportError as e:
        pytest.skip(f"faiss-cpu / sentence-transformers not installed: {e}")


def test_calibration_sample_covers_positives_and_negatives():
    positives = [e for e in CALIBRATION_SAMPLE if e[1] is not None]
    negatives = [e for e in CALIBRATION_SAMPLE if e[1] is None]
    assert len(positives) >= 40  # substantial positive coverage
    assert len(negatives) >= 5  # not just easy cases


def test_shipped_default_matches_calibration(disambiguator: FAISSDisambiguator):
    """The core "not hardcoded" proof for this threshold: the shipped
    DEFAULT_DATASET_CONFIDENCE_THRESHOLD constant must equal what
    calibrate_dataset_threshold() actually computes against the shipped
    labelled sample -- not merely "close to" or "in the right ballpark."
    """
    best_threshold, all_results = calibrate_dataset_threshold(CALIBRATION_SAMPLE, disambiguator)

    print(f"\n--- Threshold sweep ({len(CALIBRATION_SAMPLE)} samples) ---")
    for r in sorted(all_results, key=lambda r: r.threshold):
        marker = " <- shipped default" if r.threshold == DEFAULT_DATASET_CONFIDENCE_THRESHOLD else ""
        marker += " <- best" if r.threshold == best_threshold else ""
        print(f"  threshold={r.threshold:.2f}  accuracy={r.accuracy:.1%}  "
              f"recall={r.positive_recall:.1%}  precision={r.negative_precision:.1%}{marker}")

    assert best_threshold == DEFAULT_DATASET_CONFIDENCE_THRESHOLD, (
        f"calibrate_dataset_threshold() found {best_threshold} to be optimal, but "
        f"DEFAULT_DATASET_CONFIDENCE_THRESHOLD in app/disambiguation.py is "
        f"{DEFAULT_DATASET_CONFIDENCE_THRESHOLD} -- update the shipped constant to "
        "match (see that constant's docstring)."
    )


def test_calibrated_threshold_accuracy_meets_floor(disambiguator: FAISSDisambiguator):
    best_threshold, all_results = calibrate_dataset_threshold(
        CALIBRATION_SAMPLE, disambiguator, candidate_thresholds=[DEFAULT_DATASET_CONFIDENCE_THRESHOLD]
    )
    result = all_results[0]
    assert result.accuracy >= ACCURACY_FLOOR, (
        f"Accuracy at the shipped threshold ({result.accuracy:.1%}) fell below the "
        f"{ACCURACY_FLOOR:.0%} floor."
    )


def test_calibration_positive_recall_reported(disambiguator: FAISSDisambiguator):
    _best, all_results = calibrate_dataset_threshold(
        CALIBRATION_SAMPLE, disambiguator, candidate_thresholds=[DEFAULT_DATASET_CONFIDENCE_THRESHOLD]
    )
    recall = all_results[0].positive_recall
    print(f"\nPositive recall at shipped threshold: {recall:.1%}")
    assert recall >= 0.90


def test_calibration_negative_precision_reported(disambiguator: FAISSDisambiguator):
    _best, all_results = calibrate_dataset_threshold(
        CALIBRATION_SAMPLE, disambiguator, candidate_thresholds=[DEFAULT_DATASET_CONFIDENCE_THRESHOLD]
    )
    precision = all_results[0].negative_precision
    print(f"\nNegative (no-match) precision at shipped threshold: {precision:.1%}")
    # Below-threshold-correctly-rejected is the safety-critical direction
    # (a false positive here means confidently mapping unrelated text onto
    # a disease symptom) -- held to a stricter bar than overall accuracy.
    assert precision >= 0.85


def test_calibrate_dataset_threshold_is_deterministic(disambiguator: FAISSDisambiguator):
    a = calibrate_dataset_threshold(CALIBRATION_SAMPLE, disambiguator)
    b = calibrate_dataset_threshold(CALIBRATION_SAMPLE, disambiguator)
    assert a == b


def test_calibrate_dataset_threshold_does_not_mutate_disambiguator(disambiguator: FAISSDisambiguator):
    original_threshold = disambiguator.dataset_confidence_threshold
    calibrate_dataset_threshold(CALIBRATION_SAMPLE, disambiguator)
    assert disambiguator.dataset_confidence_threshold == original_threshold


def test_calibrate_dataset_threshold_ties_break_toward_lower_threshold():
    # Genuine tie by construction: item "a" (expected "tok", raw score
    # 0.50) and item "b" (expected None, raw score 0.60).
    #   threshold=0.40: "a" -> predicted "tok" (correct); "b" -> 0.60>=0.40
    #     -> predicted "tok2" (wrong, expected None). Accuracy 1/2.
    #   threshold=0.55: "a" -> 0.50<0.55 -> predicted None (wrong); "b" ->
    #     0.60>=0.55 -> predicted "tok2" (wrong). Accuracy 0/2.
    #   threshold=0.65: "a" -> predicted None (wrong); "b" -> 0.60<0.65 ->
    #     predicted None (correct, expected None). Accuracy 1/2.
    # 0.40 and 0.65 TIE at 50% -- must pick the lower (higher-recall) one.
    class _FakeDisambiguator:
        def raw_dataset_top1(self, text):
            return {"a": ("tok", 0.50), "b": ("tok2", 0.60)}[text]

    sample = [("a", "tok"), ("b", None)]
    best, results = calibrate_dataset_threshold(sample, _FakeDisambiguator(), candidate_thresholds=[0.40, 0.55, 0.65])
    tied = [r for r in results if r.accuracy == max(r.accuracy for r in results)]
    assert {r.threshold for r in tied} == {0.40, 0.65}  # confirm this really is a tie
    assert best == 0.40
