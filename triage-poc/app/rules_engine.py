"""
IMNCI Rules Engine -- deterministic, auditable, no LLM anywhere in this file.

Two routes, selected by age (see classify()):
  - Pediatric IMNCI (age 2mo-5yr): the original fixed clinical rules below.
    Young infant 0-<2mo and malnutrition/anaemia charts are explicitly NOT
    yet implemented -- see Step 2 of the build order.
  - Adult/elderly/out-of-band (age >=60mo, or age_group ADULT/ELDERLY with
    no exact age): routed to app.disease_classifier.classify_via_dataset(),
    the dataset-driven disease classifier built from the user-supplied
    disease/symptom CSVs. See classify_via_dataset()'s docstring below.

Implements (for the 2-month-to-5-year child age band; young infant 0-<2mo
and malnutrition/anaemia charts are explicitly NOT yet implemented -- see
Step 2 of the build order):
  - Danger-sign completeness check (Step 3): a hard gate that runs before
    any clinical classification and produces INCOMPLETE_ASSESSMENT as a
    distinct state, never silently defaulting missing signs to "absent".
  - General danger-sign classification -> EMERGENCY.
  - Cough / difficult breathing classification (fast-breathing cutoffs,
    chest indrawing, stridor).
  - Diarrhea / dehydration classification (two-of-the-following logic).
  - most_severe_wins: explicit, testable multi-condition conflict
    resolution (Step 2) -- this is a real function, not classification
    order/whichever-check-happens-to-run-last.

Every classification function returns a ClassificationResult carrying a
`reasoning` list (one entry per rule that fired) so the decision is
traceable end to end, per the "no black-box scoring outside the two LLM
agents" constraint.
"""
from __future__ import annotations

from typing import Optional

from app.disease_classifier import get_default_classifier, severity_for_disease
from app.emergency_scorer import score_emergency
from app.schemas import (
    AgeGroup,
    ClassificationLabel,
    ClassificationResult,
    ExtractedCase,
    SEVERITY_RANK,
)

# WHO IMCI fast-breathing cutoffs (breaths/minute), child 2mo-<5yr band.
FAST_BREATHING_CUTOFF_2_TO_12_MONTHS = 50
FAST_BREATHING_CUTOFF_12_MONTHS_TO_5_YEARS = 40

# Age band this module implements. Cases outside it are out of scope and
# must be flagged, not silently misclassified with adult/young-infant logic.
MODULE_AGE_MIN_MONTHS = 2
MODULE_AGE_MAX_MONTHS = 60


def check_danger_sign_completeness(case: ExtractedCase) -> Optional[ClassificationResult]:
    """Hard gate (Step 3). Returns an INCOMPLETE_ASSESSMENT result if any of
    the four general danger-sign fields is None ("not assessed"). Returns
    None (i.e. "proceed") only if all four are explicitly True/False.

    IMPORTANT ordering note: this gate exists to protect the case where we
    might be about to conclude "no danger signs" without having actually
    checked all four -- it is NOT meant to delay an EMERGENCY call when one
    danger sign is already confirmed True. classify() below calls
    classify_danger_signs() first for exactly that reason: one confirmed
    danger sign is sufficient to act on immediately, regardless of whether
    the other three were ever asked about. Only when no danger sign has
    been confirmed True does completeness become the safety-relevant
    question ("are we sure there really isn't one?").
    """
    missing = case.danger_signs.missing_fields()
    if missing:
        return ClassificationResult(
            label=ClassificationLabel.INCOMPLETE_ASSESSMENT,
            condition="DANGER_SIGNS_NOT_FULLY_ASSESSED",
            reasoning=[
                f"danger sign field(s) not assessed (None, not False): {missing}",
                "IMCI requires all four general danger signs to be explicitly "
                "checked before any classification is considered safe; "
                "returning INCOMPLETE_ASSESSMENT rather than assuming absence.",
            ],
            missing_fields=missing,
            case_id=case.case_id,
        )
    return None


def classify_danger_signs(case: ExtractedCase) -> Optional[ClassificationResult]:
    """Assumes completeness already checked. Returns EMERGENCY if any
    danger sign is present, else None (this axis contributes nothing)."""
    ds = case.danger_signs
    if ds.any_true():
        fired = [
            name
            for name, v in (
                ("not_able_to_drink_or_breastfeed", ds.not_able_to_drink_or_breastfeed),
                ("vomits_everything", ds.vomits_everything),
                ("convulsions", ds.convulsions),
                ("lethargic_or_unconscious", ds.lethargic_or_unconscious),
            )
            if v is True
        ]
        return ClassificationResult(
            label=ClassificationLabel.EMERGENCY,
            condition="GENERAL_DANGER_SIGN",
            reasoning=[f"danger sign(s) present: {fired}", "any true danger sign -> EMERGENCY"],
            case_id=case.case_id,
        )
    return None


def _fast_breathing_cutoff_for_age(age_months: int) -> int:
    if age_months < 12:
        return FAST_BREATHING_CUTOFF_2_TO_12_MONTHS
    return FAST_BREATHING_CUTOFF_12_MONTHS_TO_5_YEARS


def classify_cough_or_difficult_breathing(case: ExtractedCase) -> Optional[ClassificationResult]:
    """Returns None if cough/difficult breathing was not reported/assessed
    at all (present is None or False) -- this axis is not applicable."""
    cough = case.cough
    if cough is None or cough.present is not True:
        return None

    reasoning: list[str] = ["cough/difficult breathing present"]

    if cough.chest_indrawing is True or cough.stridor_when_calm is True:
        signs = [
            n
            for n, v in (("chest_indrawing", cough.chest_indrawing), ("stridor_when_calm", cough.stridor_when_calm))
            if v is True
        ]
        reasoning.append(f"exam-only severe sign(s) present: {signs} -> SEVERE")
        return ClassificationResult(
            label=ClassificationLabel.SEVERE,
            condition="SEVERE_PNEUMONIA_OR_SEVERE_DISEASE",
            reasoning=reasoning,
            case_id=case.case_id,
        )

    if cough.breaths_per_minute is not None and case.age_months is not None:
        if MODULE_AGE_MIN_MONTHS <= case.age_months < MODULE_AGE_MAX_MONTHS:
            cutoff = _fast_breathing_cutoff_for_age(case.age_months)
            reasoning.append(
                f"breaths_per_minute={cough.breaths_per_minute}, age_months={case.age_months}, cutoff={cutoff}"
            )
            if cough.breaths_per_minute >= cutoff:
                reasoning.append("breathing rate >= age-appropriate cutoff -> MODERATE (pneumonia)")
                return ClassificationResult(
                    label=ClassificationLabel.MODERATE,
                    condition="PNEUMONIA",
                    reasoning=reasoning,
                    case_id=case.case_id,
                )

    reasoning.append(
        "no chest indrawing/stridor, and breathing rate either not fast or not assessable -> MILD"
    )
    return ClassificationResult(
        label=ClassificationLabel.MILD,
        condition="COUGH_OR_COLD",
        reasoning=reasoning,
        case_id=case.case_id,
    )


def classify_diarrhea_dehydration(case: ExtractedCase) -> Optional[ClassificationResult]:
    """Two-of-the-following logic per WHO IMCI. Returns None if diarrhea
    was not reported/assessed at all."""
    d = case.diarrhea
    if d is None or d.present is not True:
        return None

    lethargic = case.danger_signs.lethargic_or_unconscious  # shared sign, not duplicated

    severe_signs = {
        "lethargic_or_unconscious": lethargic,
        "sunken_eyes": d.sunken_eyes,
        "drinks_poorly_or_not_able": d.drinks_poorly_or_not_able,
        "skin_pinch_goes_back_very_slowly": d.skin_pinch_goes_back_very_slowly,
    }
    severe_present = [k for k, v in severe_signs.items() if v is True]
    if len(severe_present) >= 2:
        return ClassificationResult(
            label=ClassificationLabel.SEVERE,
            condition="SEVERE_DEHYDRATION",
            reasoning=[f">=2 severe dehydration signs present: {severe_present}"],
            case_id=case.case_id,
        )

    some_signs = {
        "restless_or_irritable": d.restless_or_irritable,
        "sunken_eyes": d.sunken_eyes,
        "drinks_eagerly_thirsty": d.drinks_eagerly_thirsty,
        "skin_pinch_goes_back_slowly": d.skin_pinch_goes_back_slowly,
    }
    some_present = [k for k, v in some_signs.items() if v is True]
    if len(some_present) >= 2:
        return ClassificationResult(
            label=ClassificationLabel.MODERATE,
            condition="SOME_DEHYDRATION",
            reasoning=[f">=2 'some dehydration' signs present: {some_present}"],
            case_id=case.case_id,
        )

    return ClassificationResult(
        label=ClassificationLabel.MILD,
        condition="NO_DEHYDRATION",
        reasoning=["fewer than 2 signs in either dehydration tier -> NO_DEHYDRATION"],
        case_id=case.case_id,
    )


def most_severe_wins(results: list[ClassificationResult]) -> ClassificationResult:
    """Explicit, testable conflict resolution (Step 2). Given classification
    results from multiple axes (danger signs, cough, diarrhea, ...), returns
    the single most severe one, ranked by SEVERITY_RANK. Ties keep the
    first-seen result. Reasoning trails from ALL inputs are preserved on the
    winner so the audit trail shows what else was considered, not just the
    winning axis.

    Raises ValueError on an empty list -- callers must supply at least a
    default MILD/no-finding result rather than relying on this function to
    invent one.
    """
    if not results:
        raise ValueError("most_severe_wins requires at least one ClassificationResult")

    winner = min(results, key=lambda r: SEVERITY_RANK[r.label])
    if len(results) > 1:
        other_summaries = [
            f"{r.condition or r.label.value} ({r.label.value})" for r in results if r is not winner
        ]
        winner = winner.model_copy(
            update={
                "reasoning": winner.reasoning
                + [f"most_severe_wins: also considered and overridden: {other_summaries}"]
            }
        )
    return winner


def _attach_dataset_context(result: ClassificationResult, case: ExtractedCase) -> ClassificationResult:
    """Attaches dataset-classifier candidate diseases + precautions onto an
    IMNCI result as SUPPLEMENTARY context -- never overrides `label`. Used
    only on the in-band pediatric path (age 2-60mo), where IMNCI stays
    authoritative per the reconciliation decision: severity tier comes from
    IMNCI's deterministic rules, `probable_disease` is just "which disease
    the reported symptom tokens most resemble," attached for the doctor
    report / audit trail, not as a competing classification.

    No-op (returns result unchanged) when the case has no recognized
    symptom_tokens yet -- e.g. FAISS disambiguation hasn't populated them,
    or the reported symptoms are all pediatric-IMNCI-specific
    (cough/diarrhea/danger-signs) with nothing in the disease-CSV
    vocabulary. An empty `candidates` list on the result honestly reflects
    "no supplementary signal," not a forced guess.
    """
    if not case.symptom_tokens:
        return result
    candidates = get_default_classifier().classify_diseases(case.symptom_tokens)
    if not candidates:
        return result
    return result.model_copy(
        update={"candidates": candidates, "probable_disease": candidates[0].name}
    )


def classify_via_dataset(case: ExtractedCase) -> ClassificationResult:
    """Primary classification path for adult/elderly/out-of-band-age cases.

    Implements the two-stage rule engine (Rule engine final.md):

    Stage 1 — Probabilistic diagnosis with calibration + abstention:
      - Bernoulli Naive Bayes posteriors over 41 diseases.
      - Isotonic calibration (Niculescu-Mizil & Caruana, ICML 2005).
      - Reject-option abstention: answer only when calibrated_confidence >= τ
        (Youden's J threshold) AND gap_to_second >= δ (risk-coverage threshold).
        Chow (1970); Geifman & El-Yaniv (2017).
      - If abstain → INCOMPLETE_ASSESSMENT; coordination layer triggers follow-up.

    Stage 2 — AHP-weighted emergency scoring (only when Stage 1 answers):
      - Six-attribute acuity score (Saaty 1980 AHP, CR=0.0205).
      - WHO IMCI danger-sign hard override.
      - ESI-aligned band mapping (AHRQ ESI Handbook v5).

    No LLM call happens in this function.
    """
    clf = get_default_classifier()
    diag = clf.classify_with_abstention(case.symptom_tokens)

    # --- Stage 1: abstention path ---
    if diag.abstain:
        missing = ["symptom_tokens"] if not diag.candidates else []
        return ClassificationResult(
            label=ClassificationLabel.INCOMPLETE_ASSESSMENT,
            condition="UNCERTAIN_DIAGNOSIS",
            reasoning=[
                "Stage 1 reject-option fired — engine abstains rather than guessing.",
                f"Reason: {diag.abstain_reason}",
                f"Thresholds used: τ={diag.tau:.3f} (Youden's J), δ={diag.delta:.3f} (risk-coverage).",
                "Abstention is a safety property: the coordination layer will "
                "trigger a follow-up question to disambiguate before re-classifying.",
            ],
            missing_fields=missing,
            abstention_triggered=True,
            calibrated_confidence=diag.calibrated_confidence,
            raw_confidence=diag.raw_confidence,
            gap_to_second=diag.gap_to_second,
            candidates=diag.candidates[:5] if diag.candidates else [],
            case_id=case.case_id,
        )

    # --- Stage 1: confident diagnosis ---
    top = diag.candidates[0]
    label = severity_for_disease(top.name)

    reasoning = [
        "Stage 1 — dataset-driven disease classifier (adult/out-of-band path).",
        f"Top diagnosis: {top.name!r}  calibrated_confidence={diag.calibrated_confidence:.3f}  "
        f"raw_posterior={diag.raw_confidence:.3f}  gap_to_second={diag.gap_to_second:.3f}.",
        f"Abstention thresholds: τ={diag.tau:.3f} (Youden's J), δ={diag.delta:.3f} (risk-coverage) — both passed ✓.",
        f"Matched symptoms: {top.matched_symptoms}.",
        f"Severity lookup (disease_severity.csv): {top.name!r} → {label.value}.",
    ]
    if len(diag.candidates) > 1:
        others = [f"{c.name} ({c.score:.3f})" for c in diag.candidates[1:4]]
        reasoning.append(f"Other candidates considered: {others}.")

    # --- Stage 2: AHP emergency scoring ---
    emergency = score_emergency(case, top.name, label)
    reasoning.append(
        f"Stage 2 — AHP emergency score: {emergency.score}/10  "
        f"band: {emergency.band.value}  ESI: {emergency.esi_level}  "
        f"override: {emergency.override_triggered}."
    )

    return ClassificationResult(
        label=label,
        condition=top.name,
        reasoning=reasoning,
        candidates=diag.candidates,
        probable_disease=top.name,
        calibrated_confidence=diag.calibrated_confidence,
        raw_confidence=diag.raw_confidence,
        gap_to_second=diag.gap_to_second,
        abstention_triggered=False,
        emergency_result=emergency,
        case_id=case.case_id,
    )


def classify(case: ExtractedCase) -> ClassificationResult:
    """Top-level orchestrator: age routing -> (pediatric: completeness gate
    -> per-axis classification -> most-severe-wins) OR (adult/out-of-band:
    dataset-driven disease classification). This is the only function
    Agent 1's pipeline should call.

    Age routing:
      - age_months < 2 (young infant): IMNCI's young-infant chart is not
        yet built (Step 2 territory) -- returns INCOMPLETE_ASSESSMENT
        unchanged from prior behavior; NOT rerouted to the dataset
        classifier, since that path is validated against adult-disease
        data, not neonatal presentations.
      - age_months >= 60, OR age_months is None with age_group explicitly
        ADULT/ELDERLY: routed to classify_via_dataset() as the PRIMARY path
        (see that function's docstring for what "primary" means here).
      - Otherwise (age_months in [2, 60), or age unknown entirely with no
        adult/elderly signal): the existing pediatric IMNCI path, with
        dataset-classifier candidates attached as supplementary context
        (_attach_dataset_context) -- never overriding the IMNCI label.
    """

    if case.age_months is not None and case.age_months < MODULE_AGE_MIN_MONTHS:
        return ClassificationResult(
            label=ClassificationLabel.INCOMPLETE_ASSESSMENT,
            condition="AGE_OUT_OF_MODULE_SCOPE",
            reasoning=[
                f"age_months={case.age_months} is below the 2-month lower bound "
                "this rules engine implements (young infant <2mo chart is not "
                "yet built -- see Step 2 of the build order)."
            ],
            case_id=case.case_id,
        )

    routes_to_dataset = (
        case.age_months is not None and case.age_months >= MODULE_AGE_MAX_MONTHS
    ) or (case.age_months is None and case.age_group in (AgeGroup.ADULT, AgeGroup.ELDERLY))
    if routes_to_dataset:
        return classify_via_dataset(case)

    # Check for a confirmed danger sign FIRST: one True danger sign is
    # sufficient to act on immediately and must not be delayed by asking
    # whether the other three were ever assessed (see docstring above).
    danger_sign_hit = classify_danger_signs(case)
    if danger_sign_hit is not None:
        return _attach_dataset_context(danger_sign_hit, case)

    # No danger sign confirmed True yet -- now completeness matters, because
    # we're about to reason as if there are none, and an unassessed field
    # could still be hiding one.
    incomplete = check_danger_sign_completeness(case)
    if incomplete is not None:
        return _attach_dataset_context(incomplete, case)

    axis_results: list[ClassificationResult] = []
    for fn in (classify_cough_or_difficult_breathing, classify_diarrhea_dehydration):
        r = fn(case)
        if r is not None:
            axis_results.append(r)

    if not axis_results:
        result = ClassificationResult(
            label=ClassificationLabel.MILD,
            condition="NO_DANGER_SIGNS_NO_SPECIFIC_ILLNESS_CLASSIFIED",
            reasoning=[
                "no danger signs present, and no cough/diarrhea axis was "
                "applicable (not reported or not present) -- default to MILD.",
                "NOTE: this is an absence-of-evidence result, not a clean "
                "bill of health; it means no implemented classification "
                "axis matched, not that no illness exists.",
            ],
            case_id=case.case_id,
        )
        return _attach_dataset_context(result, case)

    return _attach_dataset_context(most_severe_wins(axis_results), case)
