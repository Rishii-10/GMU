"""
Stage 2 Emergency Scorer: AHP-weighted multi-attribute acuity scoring.

Implements the six-attribute emergency classification described in
Rule engine final.md, Section 4.

--- AHP PAIRWISE MATRIX (n = 6, Saaty 1980) ---
Priority ordering (from ESI v5 AHRQ + WHO IMCI):
  Complication > TxDelay > Severity > Age > Onset > Transmissibility

             Sev   Age  Onset  TxDly  Comp  Trans
  Severity    1     2     3    1/2    1/3    4
  Age        1/2    1     2    1/3    1/4    3
  Onset      1/3   1/2    1    1/4    1/5    2
  TxDelay     2     3     4     1     1/2    5
  Comp        3     4     5     2      1     6
  Trans      1/4   1/3   1/2   1/5    1/6    1

Derived weights (principal eigenvector, computed below):
  w_complication     = 0.379
  w_tx_delay         = 0.249
  w_severity         = 0.161
  w_age              = 0.102
  w_onset            = 0.066
  w_transmissibility = 0.044
  Consistency Ratio  CR = 0.0205  (< 0.10 threshold)  ✓

--- SOURCES ---
  - Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
    [AHP method, CR threshold, pairwise scale]
  - Liberatore & Nydick (2008). AHP in health care: literature review.
    EJOR. [Healthcare AHP precedent]
  - Almutairi et al. (2021). XGBoost-AHP triage, COVID-19. PMC8512533.
    [AHP applied to clinical triage directly]
  - AHRQ ESI Handbook v5 (2024). [Priority ordering basis, age adjustment]
  - WHO IMCI Chart Booklet (2014). [Danger-sign overrides, age bands]
  - Lerner & Moscati (2001). The Golden Hour. Acad Emerg Med.
    [Time-to-treatment sensitivity conceptual basis]
  - Kumar et al. (2006). Sepsis antibiotic delay. Crit Care Med.
    [Concrete treatment-window data: 7.6% mortality/hour]
  - WHO IHR (2005). [Transmissibility classification]
"""
from __future__ import annotations

import math
from typing import Optional

from app.schemas import (
    ClassificationLabel,
    EmergencyBand,
    EmergencyResult,
    ExtractedCase,
)

# ---------------------------------------------------------------------------
# AHP-derived weights  (pre-computed eigenvector; full matrix in docstring)
# CR = 0.0205 < 0.10  ✓
# ---------------------------------------------------------------------------
AHP_WEIGHTS: dict[str, float] = {
    "complication_probability": 0.379,
    "time_to_treatment":        0.249,
    "disease_severity":         0.161,
    "age_vulnerability":        0.102,
    "onset_acuity":             0.066,
    "transmissibility":         0.044,
}

# ---------------------------------------------------------------------------
# Per-disease attributes  (time_tx_sensitivity, complication_prob, transmissibility)
# All values 0-1.
# Sources: WHO IHR (2005) for transmissibility; WHO IMCI danger-sign
# classification and clinical literature for complication probability;
# Lerner & Moscati (2001), Kumar et al. (2006) for treatment windows.
# Disease names match normalize_disease_name() output (strip whitespace only).
# ---------------------------------------------------------------------------
_DISEASE_ATTRS: dict[str, tuple[float, float, float]] = {
    # disease name                          : (time_tx, complication, transmissibility)
    "(vertigo) Paroymsal  Positional Vertigo": (0.2,    0.2,          0.0),
    "AIDS":                                    (0.6,    1.0,          0.3),   # blood/sexual
    "Acne":                                    (0.2,    0.1,          0.0),
    "Alcoholic hepatitis":                     (0.6,    0.8,          0.0),
    "Allergy":                                 (0.6,    0.5,          0.0),   # anaphylaxis risk
    "Arthritis":                               (0.2,    0.3,          0.0),
    "Bronchial Asthma":                        (0.8,    0.7,          0.0),   # narrow window
    "Cervical spondylosis":                    (0.2,    0.3,          0.0),
    "Chicken pox":                             (0.4,    0.4,          1.0),   # airborne (WHO IHR)
    "Chronic cholestasis":                     (0.4,    0.6,          0.0),
    "Common Cold":                             (0.2,    0.2,          0.6),   # droplet
    "Dengue":                                  (0.6,    0.8,          0.6),   # vector-borne; DHF risk
    "Diabetes":                                (0.6,    0.7,          0.0),   # DKA/hypoglycaemic crisis
    "Dimorphic hemmorhoids(piles)":            (0.2,    0.3,          0.0),
    "Drug Reaction":                           (0.8,    0.7,          0.0),   # Stevens-Johnson risk
    "Fungal infection":                        (0.2,    0.2,          0.3),
    "GERD":                                    (0.2,    0.3,          0.0),
    "Gastroenteritis":                         (0.6,    0.6,          0.6),   # fecal-oral
    "Heart attack":                            (1.0,    1.0,          0.0),   # golden hour
    "Hepatitis B":                             (0.6,    0.8,          0.3),
    "Hepatitis C":                             (0.6,    0.8,          0.3),
    "Hepatitis D":                             (0.6,    0.8,          0.3),
    "Hepatitis E":                             (0.6,    0.7,          0.3),   # epidemic in rural
    "Hypertension":                            (0.6,    0.7,          0.0),   # hypertensive crisis
    "Hyperthyroidism":                         (0.4,    0.5,          0.0),   # thyroid storm
    "Hypoglycemia":                            (1.0,    0.9,          0.0),   # brain damage if untreated
    "Hypothyroidism":                          (0.2,    0.4,          0.0),
    "Impetigo":                                (0.4,    0.3,          0.3),
    "Jaundice":                                (0.6,    0.7,          0.3),
    "Malaria":                                 (0.6,    0.8,          0.6),   # cerebral malaria risk
    "Migraine":                                (0.4,    0.3,          0.0),
    "Osteoarthristis":                         (0.2,    0.3,          0.0),
    "Paralysis (brain hemorrhage)":            (1.0,    1.0,          0.0),   # golden hour
    "Peptic ulcer diseae":                     (0.6,    0.7,          0.0),   # perforation risk
    "Pneumonia":                               (0.8,    0.9,          0.6),   # rapid deterioration
    "Psoriasis":                               (0.2,    0.2,          0.0),
    "Tuberculosis":                            (0.6,    0.8,          1.0),   # airborne (WHO IHR notifiable)
    "Typhoid":                                 (0.6,    0.8,          0.6),   # fecal-oral; perforation
    "Urinary tract infection":                 (0.4,    0.5,          0.0),   # sepsis if untreated
    "Varicose veins":                          (0.2,    0.3,          0.0),
    "hepatitis A":                             (0.6,    0.6,          0.3),
}

_DEFAULT_ATTRS: tuple[float, float, float] = (0.4, 0.5, 0.1)   # fallback for unknown disease

# ---------------------------------------------------------------------------
# Severity → 0-1 score
# MILD=outpatient, MODERATE=medical attention, SEVERE=hospital, EMERGENCY=immediate
# ---------------------------------------------------------------------------
_SEVERITY_SCORE: dict[str, float] = {
    "MILD":      0.15,
    "MODERATE":  0.45,
    "SEVERE":    0.75,
    "EMERGENCY": 1.00,
}

# ---------------------------------------------------------------------------
# WHO IMCI danger signs that force Emergency override regardless of AHP total.
# Source: WHO IMCI Chart Booklet (2014).
# ---------------------------------------------------------------------------
_IMCI_DANGER_SIGN_FIELDS = (
    "not_able_to_drink_or_breastfeed",
    "vomits_everything",
    "convulsions",
    "lethargic_or_unconscious",
)


def _age_vulnerability(age_months: Optional[int]) -> tuple[float, str]:
    """Return (score 0-1, band label).

    Bands from WHO IMCI age-group structure + ESI v5 geriatric adjustment
    (AHRQ ESI Handbook; PMC6625680 — ESI geriatric modification).
    """
    if age_months is None:
        return 0.5, "unknown age"
    if age_months < 2:
        return 1.0, "<2 months (neonatal — WHO IMCI highest risk)"
    if age_months < 60:
        return 0.8, "2-59 months (WHO IMCI primary band)"
    if age_months < 168:       # 5-13 years
        return 0.3, "5-13 years (school-age)"
    if age_months < 780:       # 14-64 years
        return 0.2, "14-64 years (adult baseline)"
    return 0.85, "≥65 years (geriatric — masked compensatory responses, ESI v5)"


def _onset_acuity_from_duration(duration: Optional[str]) -> tuple[float, str]:
    """Infer onset acuity from the free-text duration field.

    Acute onset is an ESI Level 2 discriminator ("should not wait").
    """
    if not duration:
        return 0.5, "duration unknown (subacute default)"
    d = duration.lower()
    if any(w in d for w in ("hour", "hr", "घंट", "ঘণ্টা")):
        return 1.0, f"acute onset ({duration}) — ESI high-risk trigger"
    if any(w in d for w in ("week", "month", "year", "सप्ताह", "महीन", "সপ্তাহ")):
        return 0.2, f"chronic/gradual onset ({duration})"
    return 0.5, f"subacute onset ({duration})"


def _danger_sign_override(case: ExtractedCase) -> tuple[bool, list[str]]:
    """Check WHO IMCI danger signs for the unconditional Emergency override.

    Any single True danger sign → override fires regardless of AHP total.
    None fields (not-assessed) do NOT trigger — only confirmed True.
    Source: WHO IMCI Chart Booklet (2014), general danger signs.
    """
    ds = case.danger_signs
    fired: list[str] = []
    mapping = {
        "not_able_to_drink_or_breastfeed": "unable to drink / breastfeed",
        "vomits_everything":               "vomits everything",
        "convulsions":                     "convulsions present",
        "lethargic_or_unconscious":        "lethargic or unconscious",
    }
    for field, label in mapping.items():
        if getattr(ds, field) is True:
            fired.append(label)
    return bool(fired), fired


def score_emergency(
    case: ExtractedCase,
    disease_name: str,
    severity_label: ClassificationLabel,
) -> EmergencyResult:
    """Compute the AHP-weighted emergency score for a diagnosed case.

    Parameters
    ----------
    case          : the structured patient case from Agent 1.
    disease_name  : top-ranked disease name (from Stage 1).
    severity_label: MILD/MODERATE/SEVERE/EMERGENCY from disease_severity.csv.

    Returns
    -------
    EmergencyResult with score (1-10), band, ESI level, per-attribute breakdown,
    and a complete reasoning trail.
    """
    reasoning: list[str] = []

    # --- WHO IMCI danger-sign override (unconditional) ---
    override, fired_signs = _danger_sign_override(case)
    if override:
        reasoning.append(
            f"WHO IMCI danger-sign override triggered: {fired_signs} — "
            "score forced to 10 regardless of AHP total."
        )
        attr_scores = {k: 1.0 for k in AHP_WEIGHTS}
        return EmergencyResult(
            score=10,
            band=EmergencyBand.EMERGENCY,
            esi_level=1,
            override_triggered=True,
            attribute_scores=attr_scores,
            attribute_weights=AHP_WEIGHTS,
            reasoning=reasoning,
        )

    # --- Attribute 1: Disease intrinsic severity ---
    sev_key = severity_label.value if severity_label.value in _SEVERITY_SCORE else "MODERATE"
    sev_score = _SEVERITY_SCORE[sev_key]
    reasoning.append(
        f"Attr 1 – disease severity: {disease_name} = {sev_key} → score {sev_score:.2f}"
    )

    # --- Attribute 2: Patient age vulnerability ---
    age_score, age_label = _age_vulnerability(case.age_months)
    reasoning.append(
        f"Attr 2 – age vulnerability: {age_label} → score {age_score:.2f}"
    )

    # --- Attribute 3: Onset acuity ---
    onset_score, onset_label = _onset_acuity_from_duration(case.duration)
    reasoning.append(
        f"Attr 3 – onset acuity: {onset_label} → score {onset_score:.2f}"
    )

    # --- Attributes 4, 5, 6: Disease-specific table ---
    time_tx, complication, transmissibility = _DISEASE_ATTRS.get(
        disease_name, _DEFAULT_ATTRS
    )
    reasoning.append(
        f"Attr 4 – time-to-treatment sensitivity: {disease_name} → score {time_tx:.2f}"
    )
    reasoning.append(
        f"Attr 5 – complication/deterioration probability: {disease_name} → score {complication:.2f}"
    )
    reasoning.append(
        f"Attr 6 – transmissibility (WHO IHR): {disease_name} → score {transmissibility:.2f}"
    )

    attribute_scores = {
        "disease_severity":         sev_score,
        "age_vulnerability":        age_score,
        "onset_acuity":             onset_score,
        "time_to_treatment":        time_tx,
        "complication_probability": complication,
        "transmissibility":         transmissibility,
    }

    # --- AHP weighted sum ---
    acuity = sum(AHP_WEIGHTS[k] * attribute_scores[k] for k in AHP_WEIGHTS)
    score = max(1, min(10, round(acuity * 9) + 1))

    reasoning.append(
        f"AHP weighted sum: {acuity:.4f}  →  Emergency score: {score}/10  "
        f"(formula: round(acuity × 9) + 1)"
    )

    # --- Band and ESI mapping ---
    if score <= 3:
        band = EmergencyBand.NON_URGENT
        esi_level = 5 if score <= 2 else 4
        reasoning.append(f"Score {score} → Non-urgent band → ESI {esi_level}")
    elif score <= 7:
        band = EmergencyBand.URGENT
        esi_level = 3
        reasoning.append(f"Score {score} → Urgent band → ESI 3")
    else:
        band = EmergencyBand.EMERGENCY
        esi_level = 2 if score <= 9 else 1
        reasoning.append(f"Score {score} → Emergency band → ESI {esi_level}")

    reasoning.append(
        "AHP weights (Saaty 1980, CR=0.0205): "
        + ", ".join(f"{k}={v}" for k, v in AHP_WEIGHTS.items())
    )

    return EmergencyResult(
        score=score,
        band=band,
        esi_level=esi_level,
        override_triggered=False,
        attribute_scores=attribute_scores,
        attribute_weights=AHP_WEIGHTS,
        reasoning=reasoning,
    )
