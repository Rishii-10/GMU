"""
Shared schema between Agent 1 (LLM extraction) and the IMNCI Rules Engine.

ARCHITECTURAL NOTE (flagged, not silently resolved):
The extraction prompt in Implementation Plan v2 Section 3.4 only collects
{symptom, duration, severity, age_group, location, notes}. Full WHO IMCI
classification also needs (a) the four general danger signs and (b)
exam-only clinical signs (respiratory rate count, chest indrawing, skin
pinch, etc.). Danger signs ARE things a caregiver can plausibly report over
SMS/IVR ("is the child able to drink?", "has the child had fits?"), so this
schema extends Agent 1's collected fields to include them as an explicit,
separately-populated `DangerSigns` block. Exam-only signs (breathing rate,
chest indrawing, skin pinch) cannot be reliably self-reported via text at
all -- they require a health worker's physical exam. This schema still
models them (as Optional, defaulting to None/"not assessed") so the Rules
Engine can honestly report INCOMPLETE_ASSESSMENT and route to a facility
for the physical exam, rather than guessing. This is a deliberate clinical
safety decision, not a placeholder to "fix" later.

HARD SAFETY REQUIREMENT: every clinical boolean field in this module is
Optional[bool] with default None, and None/False/True are never conflated:
  - None  -> "not assessed" (caregiver was never asked, or field is
             exam-only and unreachable via this channel)
  - False -> "assessed, sign is absent"
  - True  -> "assessed, sign is present"
Do not replace `Optional[bool] = None` with a plain `bool = False` anywhere
in this file -- that would silently turn "we never asked" into "we checked
and it's fine," which is exactly the failure mode this schema exists to
prevent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    """Lay age bucket, as extracted verbatim per the Section 3.4 prompt.

    This is NOT sufficient on its own for IMCI classification, which splits
    at exactly 2 months (young infant vs. child) and at exactly 12 months
    (fast-breathing cutoff). See `age_months` below.
    """

    INFANT = "infant"
    CHILD = "child"
    ADULT = "adult"
    ELDERLY = "elderly"
    UNKNOWN = "unknown"


class DangerSigns(BaseModel):
    """WHO IMCI general danger signs (child 2 months - 5 years).

    Caregiver-reportable via SMS/IVR in principle -- these are the fields
    Agent 1's follow-up questions should target when severity looks high
    but this block is incomplete. Any single True here is normally
    sufficient for a severe classification regardless of other findings.
    """

    not_able_to_drink_or_breastfeed: Optional[bool] = None
    vomits_everything: Optional[bool] = None
    convulsions: Optional[bool] = None
    lethargic_or_unconscious: Optional[bool] = None

    def any_true(self) -> bool:
        return any(
            v is True
            for v in (
                self.not_able_to_drink_or_breastfeed,
                self.vomits_everything,
                self.convulsions,
                self.lethargic_or_unconscious,
            )
        )

    def all_assessed(self) -> bool:
        """True only if every field is explicitly True/False (no None)."""
        return all(
            v is not None
            for v in (
                self.not_able_to_drink_or_breastfeed,
                self.vomits_everything,
                self.convulsions,
                self.lethargic_or_unconscious,
            )
        )

    def missing_fields(self) -> list[str]:
        return [
            name
            for name, v in (
                ("not_able_to_drink_or_breastfeed", self.not_able_to_drink_or_breastfeed),
                ("vomits_everything", self.vomits_everything),
                ("convulsions", self.convulsions),
                ("lethargic_or_unconscious", self.lethargic_or_unconscious),
            )
            if v is None
        ]


class CoughDifficultBreathing(BaseModel):
    """Cough / difficult breathing assessment block.

    `breaths_per_minute` and `chest_indrawing` and `stridor_when_calm` are
    exam-only signs -- expect these to be None from an SMS-only channel.
    """

    present: Optional[bool] = None
    duration_days: Optional[int] = None
    breaths_per_minute: Optional[int] = None  # exam-only
    chest_indrawing: Optional[bool] = None  # exam-only
    stridor_when_calm: Optional[bool] = None  # exam-only


class DiarrheaAssessment(BaseModel):
    """Diarrhea / dehydration assessment block.

    Dehydration classification per WHO IMCI needs at least two of:
    restlessness/irritability, sunken eyes, drinks eagerly/thirsty or
    drinks poorly, skin pinch behavior. `skin_pinch_*` is exam-only.
    """

    present: Optional[bool] = None
    duration_days: Optional[int] = None
    blood_in_stool: Optional[bool] = None
    restless_or_irritable: Optional[bool] = None
    sunken_eyes: Optional[bool] = None
    drinks_eagerly_thirsty: Optional[bool] = None
    drinks_poorly_or_not_able: Optional[bool] = None
    skin_pinch_goes_back_slowly: Optional[bool] = None  # exam-only
    skin_pinch_goes_back_very_slowly: Optional[bool] = None  # exam-only


class ExtractedCase(BaseModel):
    """Agent 1's output. Validated instances of this are the Rules Engine's
    only input -- the Rules Engine never reads raw LLM text."""

    case_id: Optional[str] = None
    raw_symptom_text: str
    symptom: Optional[str] = None
    duration: Optional[str] = None
    severity: Severity = Severity.UNKNOWN
    age_group: AgeGroup = AgeGroup.UNKNOWN
    age_months: Optional[int] = Field(
        default=None,
        description=(
            "Exact age in months if stated or inferable from the message "
            "(e.g. '3 month old baby'). None if not stated -- the Rules "
            "Engine must not guess an age band from age_group alone when "
            "this is None and the case is near the 2-month or 12-month "
            "clinical cutoffs."
        ),
    )
    location: Optional[str] = None
    notes: Optional[str] = None

    danger_signs: DangerSigns = Field(default_factory=DangerSigns)
    cough: Optional[CoughDifficultBreathing] = None
    diarrhea: Optional[DiarrheaAssessment] = None

    language: Optional[str] = None  # e.g. "hi", "en", "hi-en"
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_backend: Optional[str] = None  # which backend produced this (audit trail)
    disambiguation_confidence: Optional[float] = None  # set once FAISS (Milestone 3) is wired

    symptom_tokens: list[str] = Field(
        default_factory=list,
        description=(
            "Normalized disease_symptoms.csv vocabulary tokens for this case "
            "(e.g. 'high_fever', 'chest_pain'), populated by FAISS "
            "disambiguation against app.disease_kb.DiseaseKB.vocabulary. "
            "Empty list means 'none matched yet', not 'no symptoms' -- do "
            "not treat an empty list as a negative finding."
        ),
    )

    @field_validator("age_months")
    @classmethod
    def _age_months_sane(cls, v: Optional[int]) -> Optional[int]:
        # Bound raised from an earlier 600 (50yr) to 1500 (125yr) so this
        # schema can validate elderly/adult cases -- the dataset-driven
        # disease classifier (app/disease_classifier.py) now routes cases
        # outside the pediatric 2-60mo band, so a 50-year cap silently
        # rejected exactly the population that route was built for.
        if v is not None and (v < 0 or v > 1500):
            raise ValueError("age_months out of plausible human range")
        return v


class ClassificationLabel(str, Enum):
    """IMCI-style severity classification, most-severe-first ordering.

    Ordering matters: `most_severe_wins` conflict resolution (Step 2)
    depends on this being ranked worst-to-best.
    """

    # `condition` on ClassificationResult distinguishes WHY assessment is
    # incomplete (missing danger-sign fields, out-of-module age, an
    # abstained dataset diagnosis, or -- see app.rules_engine.classify()'s
    # young-infant branch -- a real clinical gap awaiting a validated
    # ruleset, condition="YOUNG_INFANT_NO_VALIDATED_RULESET", deliberately
    # a distinct string from the generic out-of-scope guard's
    # "AGE_OUT_OF_MODULE_SCOPE": one is "we don't have inputs", the other is
    # "we have inputs but no validated rules exist yet" -- different
    # failure modes, kept distinct rather than conflated into one label.
    INCOMPLETE_ASSESSMENT = "INCOMPLETE_ASSESSMENT"
    EMERGENCY = "EMERGENCY"  # danger sign present -> refer urgently
    SEVERE = "SEVERE"  # e.g. severe pneumonia, severe dehydration
    MODERATE = "MODERATE"  # e.g. pneumonia (non-severe), some dehydration
    MILD = "MILD"  # e.g. cough/cold, no dehydration


# Worst-to-best ranking used by most-severe-wins conflict resolution.
# INCOMPLETE_ASSESSMENT is intentionally NOT in this ranking: it is a
# distinct state (Step 3), not a point on the severity scale, and must be
# handled before classification ranking is even applied.
SEVERITY_RANK: dict[ClassificationLabel, int] = {
    ClassificationLabel.EMERGENCY: 0,
    ClassificationLabel.SEVERE: 1,
    ClassificationLabel.MODERATE: 2,
    ClassificationLabel.MILD: 3,
}


class EmergencyBand(str, Enum):
    """Three-band emergency classification aligned to ESI levels."""
    NON_URGENT = "NON_URGENT"   # score 1-3 → ESI 4-5
    URGENT = "URGENT"           # score 4-7 → ESI 3
    EMERGENCY = "EMERGENCY"     # score 8-10 → ESI 1-2


class EmergencyResult(BaseModel):
    """Stage 2 output: AHP-weighted six-attribute emergency classification.

    All attribute scores are 0-1 normalised. Weights are the AHP eigenvector
    derived from the pairwise comparison matrix documented in
    app/emergency_scorer.py (CR = 0.0205, well within Saaty's 0.10 limit).
    """
    score: int                           # 1-10 composite emergency score
    band: EmergencyBand                  # NON_URGENT / URGENT / EMERGENCY
    esi_level: int                       # 1-5 ESI-aligned level
    override_triggered: bool             # True = WHO IMCI danger sign forced Emergency
    attribute_scores: dict[str, float]   # per-attribute 0-1 scores (audit trail)
    attribute_weights: dict[str, float]  # AHP weights used (audit trail)
    reasoning: list[str]                 # human-readable decision trail


class DiseaseCandidate(BaseModel):
    """One ranked candidate from app.disease_classifier.classify_diseases().

    `score` is a posterior probability estimate (Naive-Bayes-style, see
    app/disease_classifier.py) -- NOT a calibrated clinical probability.
    Surfaced for audit ("why this disease and not another"), not as a
    number to quote as ground truth.
    """

    name: str
    score: float
    matched_symptoms: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    label: ClassificationLabel
    condition: Optional[str] = None  # e.g. "SEVERE_PNEUMONIA", "SOME_DEHYDRATION"
    reasoning: list[str] = Field(default_factory=list)  # audit trail, one entry per rule fired
    missing_fields: list[str] = Field(default_factory=list)  # populated when INCOMPLETE_ASSESSMENT
    case_id: Optional[str] = None

    candidates: list[DiseaseCandidate] = Field(
        default_factory=list,
        description=(
            "Ranked disease candidates from the dataset classifier, most "
            "probable first. For in-band pediatric cases (Step 1's IMNCI "
            "path) this is supplementary context alongside `label`, never "
            "overriding it. For out-of-band/adult cases this is the "
            "primary signal `probable_disease` is drawn from."
        ),
    )
    probable_disease: Optional[str] = Field(
        default=None,
        description=(
            "Top candidate's name, mirrored here for convenience when the "
            "dataset classifier is the primary route (see `candidates`). "
            "None when the classifier didn't run or found no candidate "
            "above its own confidence floor."
        ),
    )

    # Stage 1 calibration fields (populated on the adult/dataset path only)
    calibrated_confidence: Optional[float] = Field(
        default=None,
        description="Isotonic-calibrated posterior P(top disease is correct). "
                    "None on the pediatric IMCI path where NB is supplementary.",
    )
    raw_confidence: Optional[float] = Field(
        default=None,
        description="Raw NB posterior for top disease before calibration.",
    )
    gap_to_second: Optional[float] = Field(
        default=None,
        description="Difference between top and second calibrated posteriors.",
    )
    abstention_triggered: bool = Field(
        default=False,
        description="True when the reject-option check fired and forced INCOMPLETE_ASSESSMENT.",
    )

    # Stage 2 emergency scoring (populated when a confident diagnosis is reached)
    emergency_result: Optional[EmergencyResult] = Field(
        default=None,
        description="AHP-weighted Stage 2 emergency classification. None when "
                    "diagnosis was absent or abstained.",
    )
