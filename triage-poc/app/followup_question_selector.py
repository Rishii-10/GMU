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

# Stage 4 (FU-1): a follow-up question is only CLINICALLY RELEVANT if it asks
# about a symptom that is characteristic of a disease actually in the
# differential. A token must appear in at least this fraction of at least one
# prediction-set disease's rows to be eligible -- otherwise the selector could
# pick a token that statistically splits the set in the DATASET but is
# incoherent for the diseases in contention (the live cough+fever case asked an
# exertional-heart-rate question about a Common-Cold/Pneumonia/TB/Asthma set).
_MIN_RELEVANCE_FREQ = 0.30


# Dataset tokens that are NOT answerable by a patient or caregiver over SMS,
# and so must never be chosen as a follow-up question however discriminating
# they are statistically. Measured need: before this filter, 53.6% of selected
# questions fell back to the generic "Is the patient experiencing {token}?"
# phrasing, and the loop was asking things like "Is the patient experiencing
# acute liver failure?", "...coma?" and "...receiving blood transfusion?" --
# incoherent as questions to a layperson regardless of how the LLM rewords
# them, because the underlying token is not a reportable symptom.
#
# Four reasons for exclusion:
#   DIAGNOSIS      a clinical conclusion, not an observation
#   HISTORY        a past exposure, not a current symptom
#   INSTRUMENTED   needs a measurement or physical exam
#   DUPLICATE      already covered by the WHO danger-sign block
# These tokens remain perfectly valid as POSITIVE evidence when a caller
# volunteers them or a health worker records them -- this list only governs
# what the adaptive loop is allowed to ASK about.
_NOT_CALLER_ASKABLE: frozenset[str] = frozenset({
    "acute_liver_failure",             # DIAGNOSIS
    "fluid_overload",                  # DIAGNOSIS
    "irregular_sugar_level",           # INSTRUMENTED (blood glucose)
    "toxic_look_(typhos)",             # DIAGNOSIS (clinician gestalt)
    "coma",                            # DUPLICATE of lethargic_or_unconscious
    "family_history",                  # HISTORY
    "history_of_alcohol_consumption",  # HISTORY
    "extra_marital_contacts",          # HISTORY (and not askable this way)
    "receiving_blood_transfusion",     # HISTORY
    "receiving_unsterile_injections",  # HISTORY
    "enlarged_thyroid",                # INSTRUMENTED (palpation)
    "fast_heart_rate",                 # INSTRUMENTED -- ask "palpitations" instead
})


def _relevant_tokens(
    prediction_set: list[str],
    known_tokens: set[str],
    rows_by_disease: dict[str, list[list[str]]],
    vocabulary: list[str],
    min_relevance: float = _MIN_RELEVANCE_FREQ,
) -> list[str]:
    """Candidate tokens that are characteristic of at least one disease in the
    prediction set (freq >= _MIN_RELEVANCE_FREQ in some set disease's rows),
    excluding already-known tokens. This is the clinically-plausible
    discriminator pool both strategies draw from."""
    pool = []
    for token in vocabulary:
        if token in known_tokens or token in _NOT_CALLER_ASKABLE:
            continue
        if any(
            _symptom_freq(d, token, rows_by_disease) >= min_relevance
            for d in prediction_set
        ):
            pool.append(token)
    return pool

# Fallback question templates for when the Groq backend is unavailable.
# Keys are normalized dataset vocabulary tokens.
_QUESTION_TEMPLATES: dict[str, str] = {
    "high_fever":          "Does the patient have a high fever (feels very hot)?",
    "mild_fever":          "Does the patient have a mild or low-grade fever?",
    "chills":              "Is the patient having chills or shivering?",
    "sweating":            "Is the patient sweating heavily or having cold/clammy skin?",
    "chest_pain":          "Is the patient having pain or tightness in the chest?",
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

    # --- Added after measuring the generic-fallback rate at 53.6% ----------
    # Every entry below is phrased as something a caregiver can actually see,
    # feel or be told, in plain words -- no clinical terminology, because the
    # whole point is that the person answering is not a clinician. These cover
    # the tokens the selector asks most often on this dataset.
    "blood_in_sputum":     "Is there any blood in what the patient coughs up?",
    "rusty_sputum":        "Is what the patient coughs up brown or rust-coloured?",
    "mucoid_sputum":       "Is the patient coughing up thick or slimy mucus?",
    "back_pain":           "Does the patient have pain in the back?",
    "neck_pain":           "Does the patient have pain in the neck?",
    "knee_pain":           "Does the patient have pain in the knees?",
    "hip_joint_pain":      "Does the patient have pain in the hip?",
    "muscle_pain":         "Does the patient have aching muscles?",
    "muscle_weakness":     "Have the patient's muscles become weak?",
    "stomach_pain":        "Does the patient have pain in the stomach?",
    "belly_pain":          "Does the patient have pain in the lower belly?",
    "dark_urine":          "Is the patient's urine dark in colour?",
    "yellow_urine":        "Is the patient's urine deep yellow?",
    "foul_smell_of_urine": "Does the patient's urine smell unusually bad?",
    "polyuria":            "Is the patient passing urine much more often than usual?",
    "yellowing_of_eyes":   "Have the whites of the patient's eyes turned yellow?",
    "palpitations":        "Can the patient feel their heart racing or pounding?",
    "drying_and_tingling_lips": "Are the patient's lips dry, cracked or tingling?",
    "dizziness":           "Does the patient feel dizzy or light-headed?",
    "loss_of_balance":     "Is the patient unsteady or losing their balance?",
    "spinning_movements":  "Does the patient feel like the room is spinning?",
    "constipation":        "Is the patient unable to pass stool normally?",
    "bloody_stool":        "Is there blood in the patient's stool?",
    "indigestion":         "Does the patient have indigestion or a burning stomach?",
    "acidity":            "Does the patient have acid reflux or heartburn?",
    "excessive_hunger":    "Is the patient unusually hungry?",
    "increased_appetite":  "Is the patient eating much more than usual?",
    "weight_gain":         "Has the patient gained weight recently without trying?",
    "restlessness":        "Is the patient restless or unable to settle?",
    "irritability":        "Is the patient unusually irritable?",
    "lethargy":            "Is the patient unusually tired or sluggish?",
    "anxiety":             "Does the patient feel anxious or panicky?",
    "runny_nose":          "Does the patient have a runny nose?",
    "continuous_sneezing": "Is the patient sneezing continuously?",
    "throat_irritation":   "Does the patient have a sore or scratchy throat?",
    "patches_in_throat":   "Are there white patches in the patient's throat?",
    "congestion":          "Is the patient's nose blocked?",
    "sinus_pressure":      "Does the patient feel pressure around the nose and forehead?",
    "redness_of_eyes":     "Are the patient's eyes red?",
    "watering_from_eyes":  "Are the patient's eyes watering?",
    "blurred_and_distorted_vision": "Is the patient's vision blurred?",
    "visual_disturbances": "Is the patient seeing spots, flashes or double?",
    "pain_behind_the_eyes": "Does the patient have pain behind the eyes?",
    "swollen_legs":        "Are the patient's legs swollen?",
    "swollen_extremeties": "Are the patient's hands or feet swollen?",
    "swelling_joints":     "Are any of the patient's joints swollen?",
    "movement_stiffness":  "Does the patient feel stiff when moving?",
    "painful_walking":     "Is it painful for the patient to walk?",
    "cold_hands_and_feets": "Are the patient's hands and feet cold?",
    "shivering":           "Is the patient shivering?",
    "sweating":            "Is the patient sweating heavily?",
    "bruising":            "Does the patient bruise easily?",
    "blister":             "Does the patient have blisters on the skin?",
    "red_spots_over_body": "Are there red spots over the patient's body?",
    "skin_peeling":        "Is the patient's skin peeling?",
    "pus_filled_pimples":  "Does the patient have pus-filled pimples?",
    "blackheads":          "Does the patient have blackheads?",
    "brittle_nails":       "Are the patient's nails brittle or breaking?",
    "puffy_face_and_eyes": "Is the patient's face or are their eyes puffy?",
    "swollen_blood_vessels": "Are there swollen veins visible on the patient's skin?",
    "prominent_veins_on_calf": "Are there bulging veins on the patient's calves?",
    "ulcers_on_tongue":    "Does the patient have sores on the tongue?",
    "slurred_speech":      "Is the patient's speech slurred?",
    "weakness_of_one_body_side": "Is one side of the patient's body weak?",
    "weakness_in_limbs":   "Are the patient's arms or legs weak?",
    "unsteadiness":        "Is the patient unsteady on their feet?",
    "lack_of_concentration": "Is the patient unable to concentrate?",
    "depression":          "Has the patient been feeling very low or hopeless?",
    "mood_swings":         "Has the patient's mood been swinging?",
    "bladder_discomfort":  "Does the patient have discomfort in the bladder?",
    "continuous_feel_of_urine": "Does the patient constantly feel the need to urinate?",
    "spotting_urination":  "Is there spotting when the patient urinates?",
    "irritation_in_anus":  "Does the patient have irritation around the anus?",
    "pain_in_anal_region": "Does the patient have pain around the anus?",
    "pain_during_bowel_movements": "Is it painful when the patient passes stool?",
    "passage_of_gases":    "Is the patient passing a lot of gas?",
    "distention_of_abdomen": "Is the patient's abdomen swollen or bloated?",
    "swelling_of_stomach": "Is the patient's stomach swollen?",
    "stomach_bleeding":    "Is there any bleeding from the patient's stomach?",
    "abnormal_menstruation": "Have the patient's periods been abnormal?",
    "obesity":             "Is the patient significantly overweight?",
    "internal_itching":    "Does the patient feel itching inside the body?",
    "yellow_crust_ooze":   "Is there yellow crust or oozing on the skin?",
    "red_sore_around_nose": "Is there a red sore around the patient's nose?",
    "silver_like_dusting": "Does the patient's skin have silvery scales?",
    "small_dents_in_nails": "Are there small dents or pits in the patient's nails?",
    "inflammatory_nails":  "Are the patient's nails red or inflamed?",
    "scurring":            "Does the patient have scarring on the skin?",
    "loss_of_smell":       "Has the patient lost their sense of smell?",
    "cramps":              "Does the patient have cramps?",
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
    min_relevance: float = _MIN_RELEVANCE_FREQ,
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

    for token in _relevant_tokens(prediction_set, known_tokens, rows_by_disease, vocabulary, min_relevance):
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
    min_relevance: float = _MIN_RELEVANCE_FREQ,
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

    for token in _relevant_tokens(prediction_set, known_tokens, rows_by_disease, vocabulary, min_relevance):
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
    min_relevance: float = _MIN_RELEVANCE_FREQ,
) -> Optional[FollowupQuestion]:
    """Main entry point. Selects the single best follow-up question given
    the current prediction set.

    Applies emergency_first when the set spans different severity tiers;
    falls back to info_gain for same-tier sets or when emergency_first finds
    nothing useful.

    Returns None when no unanswered discriminating question can be found —
    caller should then abstain and refer to a higher facility.

    `min_relevance` is injectable only so the sensitivity analysis can sweep
    it (tests/evaluate_followup.py::sweep_relevance_threshold); production
    always uses the module default.
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
            prediction_set, severity_map, known_set, rows_by_disease, vocabulary,
            min_relevance,
        )
        urgency = "EMERGENCY_PRIORITY"

    if result is None:
        result = _info_gain_token(
            prediction_set, known_set, rows_by_disease, vocabulary, posteriors,
            min_relevance,
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
