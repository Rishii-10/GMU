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

The weights, lambda_max and the Consistency Ratio are DERIVED FROM THAT MATRIX
at import time by _derive_ahp_weights() (power iteration on the principal
eigenvector) -- they are not transcribed constants. An earlier version hardcoded
w = 0.379/0.249/0.161/0.102/0.066/0.044 and quoted CR = 0.0205, neither of which
could be reproduced from the matrix above (recomputing it gives CR = 0.0198);
deriving them removes six unverifiable numbers and makes the matrix the single
source of truth. Read the current values from AHP_WEIGHTS /
AHP_CONSISTENCY_RATIO rather than restating them here.

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
import re
from typing import Optional

from app.schemas import (
    ClassificationLabel,
    EmergencyBand,
    EmergencyResult,
    ExtractedCase,
)

# ---------------------------------------------------------------------------
# AHP weights, DERIVED from the pairwise matrix (not transcribed).
# ---------------------------------------------------------------------------

# Saaty's Random Consistency Index by matrix order n (Saaty 1980, Table 1.2).
_SAATY_RANDOM_INDEX: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

# Row/column order of _AHP_PAIRWISE below.
_AHP_CRITERIA: tuple[str, ...] = (
    "disease_severity",
    "age_vulnerability",
    "onset_acuity",
    "time_to_treatment",
    "complication_probability",
    "transmissibility",
)

# The expert pairwise judgements (module docstring documents the ESI v5 / WHO
# IMCI priority ordering they encode). THIS IS THE ONLY PLACE THE JUDGEMENTS
# LIVE -- everything downstream is computed from it.
_AHP_PAIRWISE: tuple[tuple[float, ...], ...] = (
    #  Sev      Age    Onset   TxDly   Comp    Trans
    (1.0,     2.0,    3.0,   1 / 2,  1 / 3,   4.0),  # Severity
    (1 / 2,   1.0,    2.0,   1 / 3,  1 / 4,   3.0),  # Age
    (1 / 3,  1 / 2,   1.0,   1 / 4,  1 / 5,   2.0),  # Onset
    (2.0,     3.0,    4.0,    1.0,   1 / 2,   5.0),  # TxDelay
    (3.0,     4.0,    5.0,    2.0,    1.0,    6.0),  # Complication
    (1 / 4,  1 / 3,  1 / 2,  1 / 5,  1 / 6,   1.0),  # Transmissibility
)


def _principal_eigenvector(
    matrix: tuple[tuple[float, ...], ...],
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> tuple[list[float], float]:
    """Power iteration -> (principal eigenvector normalized to sum 1, lambda_max).

    Pure stdlib and deterministic (fixed uniform start, fixed tolerance), so
    the derived weights are byte-identical on every machine and every run.
    """
    n = len(matrix)
    vec = [1.0 / n] * n
    for _ in range(max_iter):
        nxt = [sum(matrix[i][j] * vec[j] for j in range(n)) for i in range(n)]
        total = sum(nxt)
        if total <= 0:
            break
        nxt = [v / total for v in nxt]
        converged = max(abs(a - b) for a, b in zip(nxt, vec)) < tol
        vec = nxt
        if converged:
            break
    av = [sum(matrix[i][j] * vec[j] for j in range(n)) for i in range(n)]
    lambda_max = sum(a / v for a, v in zip(av, vec) if v > 0) / n
    return vec, lambda_max


def _derive_ahp_weights() -> tuple[dict[str, float], float, float, float]:
    """Return (weights, lambda_max, consistency_index, consistency_ratio).

    CR = CI / RI with CI = (lambda_max - n)/(n - 1). CR < 0.10 is Saaty's
    acceptability threshold; _validate_ahp_matrix() enforces it at import.
    """
    vec, lambda_max = _principal_eigenvector(_AHP_PAIRWISE)
    n = len(_AHP_CRITERIA)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _SAATY_RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    weights = dict(zip(_AHP_CRITERIA, vec))
    # Present in descending weight order (audit trails and the Rule Engine tab
    # iterate this dict directly).
    ordered = dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True))
    return ordered, lambda_max, ci, cr


def _validate_ahp_matrix() -> None:
    """Fail loudly at import if the judgement matrix is malformed or
    inconsistent, rather than silently scoring patients with bad weights."""
    n = len(_AHP_CRITERIA)
    if len(_AHP_PAIRWISE) != n or any(len(r) != n for r in _AHP_PAIRWISE):
        raise ValueError(f"AHP matrix must be {n}x{n}")
    for i in range(n):
        if abs(_AHP_PAIRWISE[i][i] - 1.0) > 1e-9:
            raise ValueError("AHP matrix diagonal must be 1.0")
        for j in range(n):
            if abs(_AHP_PAIRWISE[i][j] * _AHP_PAIRWISE[j][i] - 1.0) > 1e-6:
                raise ValueError(
                    f"AHP matrix not reciprocal at ({i},{j}): "
                    f"a_ij * a_ji = {_AHP_PAIRWISE[i][j] * _AHP_PAIRWISE[j][i]}"
                )


_validate_ahp_matrix()

(
    AHP_WEIGHTS,
    AHP_LAMBDA_MAX,
    AHP_CONSISTENCY_INDEX,
    AHP_CONSISTENCY_RATIO,
) = _derive_ahp_weights()

# Saaty's acceptability threshold for the consistency ratio.
AHP_CR_THRESHOLD = 0.10
if AHP_CONSISTENCY_RATIO >= AHP_CR_THRESHOLD:
    raise ValueError(
        f"AHP pairwise judgements are inconsistent: CR={AHP_CONSISTENCY_RATIO:.4f} "
        f">= {AHP_CR_THRESHOLD}"
    )

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


# Word-boundary matching -- a plain `"hr" in d` substring test (the pre-Stage-4
# bug, AHP-2) fired "acute" for "tHRee days" and "cHRonic", exactly backwards.
# Devanagari/Bengali markers have no ASCII word boundary so they are matched as
# bare substrings (OR'd in).
_ACUTE_ONSET_RE = re.compile(
    r"\b(?:hour|hours|hr|hrs|minute|minutes|min|mins|sudden|acute|today|now|just)\b"
    r"|घंट|ঘণ্টা",
    re.IGNORECASE,
)
_GRADUAL_ONSET_RE = re.compile(
    r"\b(?:week|weeks|month|months|year|years|chronic|long[- ]?standing)\b"
    r"|सप्ताह|महीन|সপ্তাহ",
    re.IGNORECASE,
)


def _onset_acuity_from_duration(duration: Optional[str]) -> tuple[float, str]:
    """Infer onset acuity from the free-text duration field.

    Acute onset is an ESI Level 2 discriminator ("should not wait").
    """
    if not duration:
        return 0.5, "duration unknown (subacute default)"
    if _ACUTE_ONSET_RE.search(duration):
        return 1.0, f"acute onset ({duration}) — ESI high-risk trigger"
    if _GRADUAL_ONSET_RE.search(duration):
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
        f"AHP weights (Saaty 1980, derived from the pairwise matrix; "
        f"lambda_max={AHP_LAMBDA_MAX:.4f}, CI={AHP_CONSISTENCY_INDEX:.4f}, "
        f"CR={AHP_CONSISTENCY_RATIO:.4f} < {AHP_CR_THRESHOLD}): "
        + ", ".join(f"{k}={v:.4f}" for k, v in AHP_WEIGHTS.items())
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
