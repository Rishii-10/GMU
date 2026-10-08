"""classify_with_adaptive_followup(): never re-asks, keeps all context, keeps
asking on a wide (ABSTAIN) set, and routes questions through the gateway."""
from __future__ import annotations

from app.agent1_extraction import classify_with_adaptive_followup
from app.schemas import AgeGroup, ExtractedCase


class _NoLLM:
    """generate_text always fails -> the template fallback phrases the question."""

    def generate_text(self, *_a, **_k):
        raise RuntimeError("no llm")


class _MarkGateway:
    def from_english(self, text):
        return f"[hi] {text}"


def _case(tokens):
    return ExtractedCase(
        raw_symptom_text="chest pain", symptom="chest pain",
        age_group=AgeGroup.ADULT, symptom_tokens=list(tokens),
    )


def test_abstain_with_real_symptoms_still_asks_and_never_repeats():
    asked = []
    result, trail = classify_with_adaptive_followup(
        _case(["cough"]), _NoLLM(),
        answer_provider=lambda q: asked.append(q) or "no",
    )
    tokens = [t["token"] for t in trail]
    assert len(tokens) == 3  # 'cough' alone abstains -> loop keeps asking up to the cap
    assert len(set(tokens)) == len(tokens)  # "no" answers are never re-asked


def test_yes_answer_is_kept_with_original_symptoms():
    seen = {}

    def provider(q):
        seen.setdefault("n", 0)
        seen["n"] += 1
        return "yes" if seen["n"] == 1 else "no"

    result, trail = classify_with_adaptive_followup(
        _case(["chest_pain"]), _NoLLM(), answer_provider=provider
    )
    assert trail[0]["answer"] == "yes"
    # the classifier ran on chest_pain + the confirmed token, not on the answer alone
    assert result.candidates


def test_questions_go_through_gateway_but_trail_keeps_english():
    sent = []
    _, trail = classify_with_adaptive_followup(
        _case(["cough"]), _NoLLM(),
        answer_provider=lambda q: sent.append(q) or "no",
        gateway=_MarkGateway(),
    )
    assert sent and all(q.startswith("[hi] ") for q in sent)
    assert not trail[0]["question"].startswith("[hi] ")


def test_no_answer_provider_skips_loop():
    _, trail = classify_with_adaptive_followup(_case(["cough"]), _NoLLM())
    assert trail == []
