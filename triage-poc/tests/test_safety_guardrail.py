"""Stage 2 safety guardrail: the engine must never SILENTLY under-triage an
emergency.

The core safety claim of the paper is "does not miss emergencies." This file
turns that into a build-failing guardrail: for short (2-3 token) messages drawn
from every EMERGENCY/SEVERE disease, the surfaced label must not be less severe
than the disease's true tier, and must never be a silent INCOMPLETE_ASSESSMENT.

Mechanism under test (app/rules_engine.py):
  - classify_via_dataset ABSTAIN branch: a wide prediction set that still
    contains an EMERGENCY/SEVERE disease surfaces the worst-case label instead
    of a silent INCOMPLETE.
  - UNCERTAIN branch: worst-case severity across the set (pre-existing).
  - _apply_danger_sign_override: a confirmed general danger sign is a hard
    EMERGENCY across all adult CP decisions.

Classifier-only: no Ollama, no FAISS -- tokens are fed directly, so this runs
in CI.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import get_default_classifier, severity_for_disease
from app.rules_engine import classify
from app.schemas import (
    AgeGroup,
    ClassificationLabel,
    DangerSigns,
    ExtractedCase,
    SEVERITY_RANK,
)

_clf = get_default_classifier()
_EMERGENCY_SEVERE = sorted(
    d for d in _clf._rows_by_disease
    if SEVERITY_RANK[severity_for_disease(d)] <= SEVERITY_RANK[ClassificationLabel.SEVERE]
)


def _short_tokens(disease: str, k: int) -> list[str]:
    """The k most frequent tokens for a disease -- a realistic terse message."""
    counts = Counter(t for row in _clf._rows_by_disease[disease] for t in row)
    return [t for t, _ in counts.most_common(k)]


def _adult(tokens, danger_signs=None) -> ExtractedCase:
    return ExtractedCase(
        raw_symptom_text="x",
        age_group=AgeGroup.ADULT,
        symptom_tokens=tokens,
        danger_signs=danger_signs or DangerSigns(),
    )


def _surfaced_rank(result) -> int:
    # INCOMPLETE_ASSESSMENT has no severity rank -- for an emergency/severe true
    # case it is a silent under-triage, so it ranks worse than any real tier.
    return SEVERITY_RANK.get(result.label, 99)


@pytest.mark.parametrize("disease", _EMERGENCY_SEVERE)
@pytest.mark.parametrize("k", [2, 3])
def test_no_undertriage_on_short_emergency_severe_message(disease, k):
    true_rank = SEVERITY_RANK[severity_for_disease(disease)]
    tokens = _short_tokens(disease, k)
    result = classify(_adult(tokens))
    assert _surfaced_rank(result) <= true_rank, (
        f"UNDER-TRIAGE: {disease} ({severity_for_disease(disease).value}) short message "
        f"{tokens} surfaced {result.label.value} (condition={result.condition})"
    )
    assert result.label != ClassificationLabel.INCOMPLETE_ASSESSMENT, (
        f"SILENT INCOMPLETE on an emergency/severe case: {disease} {tokens}"
    )


def test_chest_pain_alone_escalates_not_silent_incomplete():
    # The signature heart-attack failure: one ambiguous token used to produce a
    # 6-way ABSTAIN set -> silent INCOMPLETE. Now it escalates to EMERGENCY.
    result = classify(_adult(["chest_pain"]))
    assert result.label == ClassificationLabel.EMERGENCY
    assert "Heart attack" in result.prediction_set
    assert result.abstention_triggered is True  # still asks to narrow
    assert result.probable_disease is None       # not committed to one disease


def test_full_mi_message_is_emergency():
    result = classify(_adult(["chest_pain", "sweating", "breathlessness", "vomiting"]))
    assert result.label == ClassificationLabel.EMERGENCY


def test_no_symptom_tokens_still_incomplete():
    # Guardrail must not globally escalate: with NO evidence the engine still
    # honestly abstains (INCOMPLETE / INSUFFICIENT_SYMPTOM_DATA).
    result = classify(_adult([]))
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.condition == "INSUFFICIENT_SYMPTOM_DATA"


@pytest.mark.parametrize("disease", ["Common Cold", "Acne"])
def test_mild_confident_message_stays_mild(disease):
    # The emergency escalation must not over-triage a confident MILD case.
    result = classify(_adult(_short_tokens(disease, 3)))
    assert result.label == ClassificationLabel.MILD


def test_danger_sign_overrides_adult_to_emergency_even_when_set_is_mild():
    # A confirmed general danger sign is a hard EMERGENCY across every adult CP
    # decision -- even when the disease differential is all mild/moderate.
    result = classify(_adult(["itching"], DangerSigns(convulsions=True)))
    assert result.label == ClassificationLabel.EMERGENCY
    assert result.emergency_result is not None
    assert result.emergency_result.override_triggered is True


def test_no_danger_sign_assessed_does_not_fire_override():
    # None (not assessed) must never be treated as True.
    result = classify(_adult(_short_tokens("Acne", 3), DangerSigns()))
    assert result.label == ClassificationLabel.MILD
