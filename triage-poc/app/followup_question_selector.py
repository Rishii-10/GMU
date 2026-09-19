"""
Follow-up question selector for the adaptive post-classification loop.

Called by agent1_extraction.classify_with_adaptive_followup() when the
Conformal Prediction gate returns a prediction set of size > 1 (the engine
knows the answer is in a set but can't commit to one disease yet).

Two selection strategies, applied based on the severity profile of the set:

  emergency_first — prediction set contains diseases from DIFFERENT severity
    tiers (e.g. Heart attack EMERGENCY + Gastritis MODERATE). Safety policy:
    always pick the symptom that best discriminates the highest-severity
    disease from lower-severity ones, regardless of overall information gain.
    Ruling out an EMERGENCY candidate takes precedence over optimally
    shrinking the set.

  info_gain — prediction set contains diseases from the SAME severity tier
    (e.g. Dengue SEVERE + Malaria SEVERE + Typhoid SEVERE). No safety-first
    override; pick the symptom with the highest Information Gain over the
    posterior distribution within the set.

No LLM anywhere in this file. The selected `symptom_token` (a normalized
dataset-vocabulary token like "radiating_pain") is handed to
agent1_extraction.generate_followup_question_text(), which is the one place
the Groq LLM does work in this pipeline: converting the raw token into a
natural-language question appropriate to the patient's language.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from app.schemas import ClassificationLabel


# Map ClassificationLabel values to numeric rank for severity comparison.
_SEVERITY_RANK: dict[str, int] = {
    ClassificationLabel.EMERGENCY.value: 0,
    ClassificationLabel.SEVERE.value: 1,
    ClassificationLabel.MODERATE.value: 2,
    ClassificationLabel.MILD.value: 3,
}

# Min frequency a symptom must appear in the emergency disease's rows to be
# worth asking about. Below this it's a noise association in the dataset.
_MIN_EMERGENCY_FREQ = 0.20

# Fallback question templates for when the Groq backend is unavailable.
# Keys are normalized dataset vocabulary tokens.
_QUESTION_TEMPLATES: dict[str, str] = {
    "high_fever":          "Does the patient have a high fever (feels very hot)?",
    "mild_fever":          "Does the patient have a mild or low-grade fever?",
    "chills":              "Is the patient having chills or shivering?",
    "sweating":            "Is the patient sweating heavily or having cold/clammy skin?",
    "chest_pain":          "Is the patient having pain or tightness in the chest?",
    "radiating_pain":      "Does the chest or body pain spread to the left arm, jaw, or back?",
    "breathlessness":      "Is the patient having difficulty breathing or shortness of breath?",
    "cough":               "Does the patient have a cough?",
    "phlegm":              "Is the patient coughing up phlegm or mucus?",
    "vomiting":            "Is the patient vomiting?",
    "nausea":              "Does the patient feel like vomiting (nausea)?",
    "abdominal_pain":      "Does the patient have stomach or abdominal pain?",
    "diarrhoea":           "Is the patient having loose or watery stools?",
    "headache":            "Does the patient have a headache?",
    "skin_rash":           "Is there a rash or eruption on the skin?",
    "itching":             "Is the patient having itching or skin irritation?",
    "joint_pain":          "Is the patient having pain in the joints?",
    "fatigue":             "Is the patient extremely tired or unable to do normal activities?",
    "loss_of_appetite":    "Has the patient stopped eating or lost interest in food?",
    "yellowish_skin":      "Has the patient's skin or the whites of their eyes turned yellow?",
    "stiff_neck":          "Does the patient have a stiff or painful neck?",
    "seizures":            "Has the patient had any fits, seizures, or shaking episodes?",
    "altered_sensorium":   "Is the patient confused, unusually drowsy, or difficult to wake?",
    "burning_micturition": "Is the patient having pain or burning while urinating?",
    "nodal_skin_eruptions":"Does the patient have raised bumps or nodules on the skin?",
    "dischromic_patches":  "Are there any discolored patches on the patient's skin?",
    "muscle_wasting":      "Have the patient's muscles become visibly thinner or weaker recently?",
    "weight_loss":         "Has the patient lost significant weight recently without trying?",
    "swelled_lymph_nodes": "Are there any swollen lumps in the neck, armpits, or groin?",
    "malaise":             "Is the patient feeling generally unwell and weak overall?",
    "dehydration":         "Does the patient have dry mouth, sunken eyes, or little urination?",
    "sunken_eyes":         "Do the patient's eyes look sunken or hollow?",
}


@dataclass
class FollowupQuestion:
    """Output of select_followup_question(). Passed to generate_followup_question_text()
    for LLM-based natural language generation, or used directly via
    template_question if the LLM backend is unavailable."""

    symptom_token: str            # normalized dataset vocab token (e.g. "radiating_pain")
    template_question: str        # fallback natural-language question, no LLM needed
    prediction_set: list[str]     # diseases still in contention
    severity_map: dict[str, str]  # disease -> severity label value
    severity_urgency: str         # "EMERGENCY_PRIORITY" | "ROUTINE"
    score: float                  # discriminative score or IG value
    rationale: str                # human-readable explanation for audit trail
    loop_count: int = 0


def _rank(label_str: str) -> int:
    return _SEVERITY_RANK.get(label_str, 3)


def _symptom_freq(disease: str, symptom: str, rows_by_disease: dict[str, list[list[str]]]) -> float:
    rows = rows_by_disease.get(disease, [])
    if not rows:
        return 0.0
    return sum(1 for r in rows if symptom in r) / len(rows)


def _entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total == 0.0:
        return 0.0
    h = 0.0
    for w in weights:
        if w > 0:
            p = w / total
            h -= p * math.log2(p)
    return h


def _emergency_first_token(
    prediction_set: list[str],
    severity_map: dict[str, str],
    known_tokens: set[str],
    rows_by_disease: dict[str, list[list[str]]],
    vocabulary: list[str],
) -> Optional[tuple[str, float, str]]:
    """Find the unanswered symptom token that best discriminates the
    highest-severity disease in the prediction set from the others."""
    sorted_by_sev = sorted(prediction_set, key=lambda d: _rank(severity_map.get(d, "MILD")))
    if len(sorted_by_sev) < 2:
        return None

    emergency_disease = sorted_by_sev[0]
    others = sorted_by_sev[1:]

    best_token: Optional[str] = None
    best_score = -1.0
    best_rationale = ""

    for token in vocabulary:
        if token in known_tokens:
            continue
        freq_em = _symptom_freq(emergency_disease, token, rows_by_disease)
        if freq_em < _MIN_EMERGENCY_FREQ:
            continue
        freq_other_avg = (
            sum(_symptom_freq(d, token, rows_by_disease) for d in others) / len(others)
        )
        score = freq_em - freq_other_avg
        if score > best_score:
            best_score = score
            best_token = token
            best_rationale = (
                f"Emergency-first: '{token}' present in {freq_em:.0%} of "
                f"'{emergency_disease}' rows vs {freq_other_avg:.0%} avg in others. "
                f"Discriminative delta = {score:.2f}."
            )

    if best_token is None:
        return None
    return best_token, best_score, best_rationale


def _info_gain_token(
    prediction_set: list[str],
    known_tokens: set[str],
    rows_by_disease: dict[str, list[list[str]]],
    vocabulary: list[str],
    posteriors: Optional[dict[str, float]] = None,
) -> Optional[tuple[str, float, str]]:
    """Find the unanswered token with the highest Information Gain over the
    posterior distribution within the prediction set."""
    n = len(prediction_set)
    if n == 0:
        return None

    if posteriors is None:
        posteriors = {d: 1.0 / n for d in prediction_set}

    prior_entropy = _entropy([posteriors.get(d, 1.0 / n) for d in prediction_set])

    best_token: Optional[str] = None
    best_ig = -1.0
    best_rationale = ""

    for token in vocabulary:
        if token in known_tokens:
            continue
        freqs = {d: _symptom_freq(d, token, rows_by_disease) for d in prediction_set}
        p_present = sum(posteriors.get(d, 1.0 / n) * freqs[d] for d in prediction_set)
        p_absent = 1.0 - p_present
        if p_present < 0.01 or p_absent < 0.01:
            continue  # token gives no useful information split

        probs_given_present = [
            posteriors.get(d, 1.0 / n) * freqs[d] / p_present for d in prediction_set
        ]
        probs_given_absent = [
            posteriors.get(d, 1.0 / n) * (1.0 - freqs[d]) / p_absent for d in prediction_set
        ]
        ig = prior_entropy - (p_present * _entropy(probs_given_present) + p_absent * _entropy(probs_given_absent))
        if ig > best_ig:
            best_ig = ig
            best_token = token
            best_rationale = (
                f"Info Gain = {ig:.3f} bits (prior entropy = {prior_entropy:.3f} bits). "
                f"P(present) = {p_present:.2f}, P(absent) = {p_absent:.2f}."
            )

    if best_token is None:
        return None
    return best_token, best_ig, best_rationale


def template_for(token: str) -> str:
    """Fallback natural-language question for a symptom token when the LLM
    backend is unavailable. Used by agent1_extraction.generate_followup_question_text()."""
    return _QUESTION_TEMPLATES.get(
        token,
        f"Is the patient experiencing {token.replace('_', ' ')}?",
    )


def select_followup_question(
    prediction_set: list[str],
    severity_map: dict[str, str],
    known_tokens: list[str],
    rows_by_disease: dict[str, list[list[str]]],
    vocabulary: list[str],
    posteriors: Optional[dict[str, float]] = None,
    loop_count: int = 0,
) -> Optional[FollowupQuestion]:
    """Main entry point. Selects the single best follow-up question given
    the current prediction set.

    Applies emergency_first when the set spans different severity tiers;
    falls back to info_gain for same-tier sets or when emergency_first finds
    nothing useful.

    Returns None when no unanswered discriminating question can be found —
    caller should then abstain and refer to a higher facility.
    """
    if not prediction_set:
        return None

    known_set = set(known_tokens)
    tiers = {_rank(severity_map.get(d, "MILD")) for d in prediction_set}
    has_tier_conflict = len(tiers) > 1

    result = None
    urgency = "ROUTINE"

    if has_tier_conflict:
        result = _emergency_first_token(
            prediction_set, severity_map, known_set, rows_by_disease, vocabulary
        )
        urgency = "EMERGENCY_PRIORITY"

    if result is None:
        result = _info_gain_token(
            prediction_set, known_set, rows_by_disease, vocabulary, posteriors
        )
        if not has_tier_conflict:
            urgency = "ROUTINE"

    if result is None:
        return None

    token, score, rationale = result
    return FollowupQuestion(
        symptom_token=token,
        template_question=template_for(token),
        prediction_set=list(prediction_set),
        severity_map=dict(severity_map),
        severity_urgency=urgency,
        score=score,
        rationale=rationale,
        loop_count=loop_count,
    )
