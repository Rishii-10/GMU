"""Within-level prioritisation of cases for the health worker's queue.

The urgency LEVEL comes from the decision module (rules_engine). This module
only orders cases that share a level, using the AHP emergency score
(app.emergency_scorer) -- it never moves a case to another level.

Two gaps this closes:
  * The AHP score used to exist only on CONFIDENT adult decisions, so an
    UNCERTAIN case (labelled by its worst-case condition) or a paediatric case
    had nothing to be ordered by. `priority_for` scores every case against the
    condition that actually set its label.
  * Nothing ordered several cases. `queue_key` is the single ordering rule the
    case store and the health-worker view use.

Queue tiers, most urgent first:
    EMERGENCY, SEVERE, NEEDS_REVIEW (deferred and still INCOMPLETE), MODERATE, MILD
A case the system could not assess is NOT placed below MODERATE: "we could not
tell" must never be read as "routine" (same principle as the three-state record).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.emergency_scorer import score_emergency
from app.schemas import (
    SEVERITY_RANK,
    ClassificationLabel,
    ClassificationResult,
    EmergencyResult,
    ExtractedCase,
)

QUEUE_TIERS: tuple[str, ...] = ("EMERGENCY", "SEVERE", "NEEDS_REVIEW", "MODERATE", "MILD")
_TIER_RANK = {t: i for i, t in enumerate(QUEUE_TIERS)}


def queue_tier(label: str, deferred: bool = False) -> str:
    """Queue tier for a stored label. INCOMPLETE_ASSESSMENT -> NEEDS_REVIEW
    (whether or not it has been formally deferred yet)."""
    if label == ClassificationLabel.INCOMPLETE_ASSESSMENT.value:
        return "NEEDS_REVIEW"
    return label if label in _TIER_RANK else "NEEDS_REVIEW"


def _label_setting_condition(result: ClassificationResult) -> Optional[str]:
    """The condition whose severity set the label: the CP-confident disease,
    else the most severe member of the prediction set (ties -> highest score),
    else the top candidate. None on the paediatric/IMNCI and incomplete paths."""
    from app.disease_classifier import severity_for_disease

    if result.prediction_set:
        scores = {c.name: c.score for c in result.candidates}
        return min(
            result.prediction_set,
            key=lambda d: (SEVERITY_RANK.get(severity_for_disease(d), 9), -scores.get(d, 0.0)),
        )
    return result.probable_disease


def priority_for(case: ExtractedCase, result: ClassificationResult) -> EmergencyResult:
    """AHP emergency score used to order this case within its level.

    Reuses `result.emergency_result` when the decision module already produced
    one; otherwise scores the label-setting condition with the case's own
    label. Never alters `result`.
    """
    if result.emergency_result is not None:
        return result.emergency_result
    label = result.label
    if label == ClassificationLabel.INCOMPLETE_ASSESSMENT:
        # severity unknown: score as SEVERE-tier so an unassessed case is not
        # ranked as if it were mild within NEEDS_REVIEW
        label = ClassificationLabel.SEVERE
    return score_emergency(case, _label_setting_condition(result) or "", label)


def queue_key(label: str, deferred: bool, emergency_score: Optional[int], recorded_at: str):
    """Sort key: tier first (never crossed), then AHP score high->low, then
    oldest first. A missing score sorts after scored cases in the same tier."""
    try:
        when = datetime.fromisoformat(recorded_at).timestamp()
    except (TypeError, ValueError):
        when = 0.0
    score = emergency_score if emergency_score is not None else -1
    return (_TIER_RANK[queue_tier(label, deferred)], -score, when)
