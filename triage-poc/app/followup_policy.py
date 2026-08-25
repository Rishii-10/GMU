"""
Minimum-viable-info follow-up gate (user-flagged requirement): defines the
BARE MINIMUM a case needs before it can be triaged, so
app.agent1_extraction.extract_case_with_followup() asks only for genuinely
missing, genuinely relevant information -- never more.

WHY THIS EXISTS AS ITS OWN MODULE
------------------------------------
A caregiver messaging in an emergency has low patience and often low
health literacy; every extra follow-up question costs a round-trip (SMS
latency, LLM tokens, and real risk of losing the caregiver entirely before
the case is ever triaged). This module is the single place that answers
"is this enough to act on?" -- kept separate from HOW to ask (the fixed
question templates stay in agent1_extraction.py) and separate from WHAT
the answer means clinically (that stays in rules_engine.py /
disease_classifier.py). No LLM here: like rules_engine.py, this is plain
deterministic Python so the gate itself is auditable.

Route-aware, deliberately mirroring app.rules_engine.classify()'s own age
routing (see MODULE_AGE_MAX_MONTHS there) so the follow-up gate never asks
a pediatric-only question of an adult case, or vice versa:
  - Pediatric IMNCI route (age < 60mo, not explicitly adult/elderly): the
    bare minimum is the four WHO IMCI general danger signs all explicitly
    assessed (True/False, never left None) -- UNLESS one is already
    confirmed True, in which case rules_engine.classify() short-circuits
    to EMERGENCY immediately and asking about the other three would be
    pure waste (mirrors classify_danger_signs()'s own short-circuit, see
    that function's docstring). This is exactly the check the pre-existing
    follow-up loop already performed; this module names and centralizes
    it rather than changing its behavior.
  - Adult/out-of-band dataset route: the bare minimum is having *some*
    reported symptom text for the LLM to work with -- classify_via_dataset()
    honestly returns INCOMPLETE_ASSESSMENT otherwise. There is no fixed
    per-field checklist here (unlike the pediatric danger-sign four)
    because the dataset vocabulary is broad (131 tokens) and case-specific;
    the gate is simply "do we have any signal to classify from."

GUARDRAIL: this module only ever returns field names for which
app.agent1_extraction has a deterministic question template
(DANGER_SIGN_FOLLOWUP_QUESTIONS, or the adult symptom-clarifier template).
It never names an exam-only/unknowable field (breaths_per_minute, chest
indrawing, skin pinch, etc.) -- those cannot be asked over SMS/IVR at all
(see app/schemas.py's docstring), so they are simply not part of either
minimum-viable-info set below.
"""
from __future__ import annotations

from app.schemas import AgeGroup, ExtractedCase

# Mirrors app.rules_engine.MODULE_AGE_MAX_MONTHS -- duplicated as a literal
# here rather than imported, to avoid this lightweight policy module
# pulling in rules_engine.py's disease_classifier dependency chain (CSV
# load, Naive-Bayes model construction) just to make a routing decision
# that only needs the boundary number itself.
_PEDIATRIC_UPPER_BOUND_MONTHS = 60


def missing_required_pediatric(case: ExtractedCase) -> list[str]:
    """Pediatric IMNCI route's bare-minimum check: the four general danger
    signs, unless one is already confirmed True (already actionable, no
    more questions needed)."""
    if case.danger_signs.any_true():
        return []
    return case.danger_signs.missing_fields()


def missing_required_adult(case: ExtractedCase) -> list[str]:
    """Adult/out-of-band dataset route's bare-minimum check: at least one
    reported symptom to classify from. Checks `case.symptom` (populated
    directly by extraction), not `case.symptom_tokens` -- symptom_tokens is
    only populated by FAISS disambiguation at the pipeline's single shared
    exit point (see agent1_extraction._apply_disambiguation), which runs
    AFTER the follow-up loop exits, not per-turn. Checking symptom_tokens
    here would make this gate see "still missing" on every single turn of
    the loop, since disambiguation never runs mid-loop -- that would be
    exactly the "ask too much" failure this module exists to prevent. A
    follow-up answer that gives the LLM enough to fill in `symptom` is
    genuine progress; disambiguation converts that to symptom_tokens once
    the loop exits.
    """
    if case.symptom:
        return []
    return ["symptom_tokens"]


def missing_required(case: ExtractedCase) -> list[str]:
    """Route-aware entry point: picks the pediatric or adult minimum-
    viable-info check based on the same age boundary
    app.rules_engine.classify() routes on, so this gate and the
    classifier always agree on which route a case is taking."""
    routes_to_dataset = (
        case.age_months is not None and case.age_months >= _PEDIATRIC_UPPER_BOUND_MONTHS
    ) or (case.age_months is None and case.age_group in (AgeGroup.ADULT, AgeGroup.ELDERLY))
    if routes_to_dataset:
        return missing_required_adult(case)
    return missing_required_pediatric(case)
