"""The agent's autonomous situational brief -- its one LLM use, fired ONLY when
an alert is raised (so model cost is tied to real events, not every refresh).

Reuses app.agent1_extraction.LLMBackend.generate_text(system, user), exactly
like app/routing/report.py, and degrades to a deterministic template when the
LLM tier is unreachable (BackendUnavailable) so headless/CI runs never break.
The LLM only rewords facts the agent already computed -- it invents nothing and
does not change the detection or the recommendation.
"""
from __future__ import annotations

from typing import Optional

from app.agent1_extraction import BackendUnavailable, LLMBackend
from app.surveillance.schemas import AlertLevel, OutbreakSignal

SURVEILLANCE_BRIEF_SYSTEM_PROMPT = (
    "You are a public-health surveillance assistant writing for a district "
    "health officer. You are given a detected signal: a village, a disease, how "
    "many cases this week versus the recent baseline, the affected age group, "
    "and a recommended action already decided by the system. Write a short "
    "situational brief (at most 80 words): what is happening, how unusual it is, "
    "and the recommended next step. Plain text, no markdown, measured tone (no "
    "alarmism), and invent nothing beyond the facts given."
)


def _build_prompt(signal: OutbreakSignal, level: AlertLevel, action: str) -> str:
    age = signal.dominant_age_band or "all ages"
    return "\n".join([
        f"Alert level: {level.value}",
        f"Village: {signal.area}",
        f"Disease: {signal.disease} (triage severity {signal.severity.value})",
        f"Cases this week: {signal.observed}",
        f"Recent weekly baseline: mean {signal.baseline_mean}, spread {signal.baseline_std}",
        f"Deviation score: {signal.score} (severity-weighted {signal.weighted_score})",
        f"Most affected age group: {age}",
        f"Recommended action: {action}",
        "Write the brief now.",
    ])


def _fallback_brief(signal: OutbreakSignal, level: AlertLevel, action: str) -> str:
    """Deterministic brief when the LLM tier is unreachable -- same facts, plain
    wording, nothing invented."""
    age = signal.dominant_age_band or "all ages"
    return (
        f"{level.value}: {signal.area} shows {signal.observed} {signal.disease} "
        f"case(s) this week versus a recent baseline of {signal.baseline_mean} "
        f"(deviation score {signal.score}). Most affected: {age}. "
        f"Recommended action: {action}"
    )


def generate_brief(
    signal: OutbreakSignal,
    level: AlertLevel,
    action: str,
    backend: Optional[LLMBackend] = None,
) -> str:
    """Natural-language brief for one alert. With no backend, or an unreachable
    one, returns the deterministic fallback."""
    if backend is None:
        return _fallback_brief(signal, level, action)
    try:
        return backend.generate_text(SURVEILLANCE_BRIEF_SYSTEM_PROMPT,
                                     _build_prompt(signal, level, action))
    except BackendUnavailable:
        return _fallback_brief(signal, level, action)
