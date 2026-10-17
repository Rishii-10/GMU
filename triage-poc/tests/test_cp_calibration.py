"""Stage 3 regression tests: the conformal gate is non-vacuous.

Before Stage 3 the isotonic calibrator was fit on top-1 correctness but applied
per class, collapsing calibrated posteriors to {0, 1}; that pinned q̂ to 1.0 at
α=0.05 (the gate admitted every candidate ≥1% -- its coverage claim was
trivially true, CP-1) and left displayed posteriors degenerate at 0.000 (CP-3).
The gate now scores on the (emergency-boosted) NB posterior directly (split
conformal / LAC) at α = DEFAULT_CP_ALPHA.

Classifier-only (no Ollama, no FAISS) -- runs in CI.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import DEFAULT_CP_ALPHA, get_default_classifier

clf = get_default_classifier()


def test_q_hat_is_non_vacuous():
    # q̂ = 1.0 would admit every candidate (the old bug). It must be < 1.
    cp = clf.classify_with_cp(["chest_pain"])
    assert cp.q_hat < 1.0
    assert cp.alpha == DEFAULT_CP_ALPHA


def test_prediction_set_shrinks_as_discriminating_tokens_are_added():
    # Heart-attack tokens, added one at a time: the set must not grow, and must
    # be strictly smaller with the full picture than with a single lay token.
    seq = ["chest_pain", "vomiting", "breathlessness", "sweating"]
    sizes = [clf.classify_with_cp(seq[:k]).set_size for k in range(1, len(seq) + 1)]
    assert all(b <= a for a, b in zip(sizes, sizes[1:])), sizes  # non-increasing
    assert sizes[-1] < sizes[0], sizes


def test_a_discriminating_token_can_resolve_to_confident():
    # Adding altered_sensorium (a Paralysis red-flag) collapses a wide set to 1.
    sizes = [
        clf.classify_with_cp(["headache", "vomiting", "altered_sensorium"][:k]).set_size
        for k in (1, 3)
    ]
    assert sizes[0] >= 4 and sizes[1] == 1, sizes


def test_displayed_posteriors_are_not_degenerate_zero():
    # CP-3: the posteriors shown for a wide set must be real probabilities, not
    # all 0.000 (the isotonic-collapse symptom).
    cp = clf.classify_with_cp(["chest_pain"])
    assert cp.posteriors
    assert any(v > 0.0 for v in cp.posteriors.values())
    assert max(cp.posteriors.values()) <= 1.0


def test_ece_and_brier_are_reported():
    assert isinstance(clf.ece_, float) and 0.0 <= clf.ece_ <= 1.0
    assert isinstance(clf.brier_, float) and 0.0 <= clf.brier_ <= 1.0


def test_no_dead_tau_delta_reject_option():
    # CP-2: the unused τ/δ classify_with_abstention path was removed.
    assert not hasattr(clf, "classify_with_abstention")
    assert not hasattr(clf, "_tau")
    assert not hasattr(clf, "_delta")


def test_calibration_nonconformity_scores_in_range():
    scores = clf._cp_cal_nonconformity_scores
    assert len(scores) == 304
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Not all identical (the old degenerate {0,1} spike dominated); there is a
    # real spread now.
    assert len(set(round(s, 3) for s in scores)) > 5
