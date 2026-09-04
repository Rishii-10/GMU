"""
Agent 3's LLM report layer -- the user's explicit override of plan diagram
2 (which labels routing "No LLM"): on top of the deterministic routing core
in app/routing/router.py, generate an AI-reasoned, doctor-facing report
that pulls together everything computed upstream (Agent 1's extraction,
the Rules-Engine/dataset-classifier result, and the routing/dispatch
result) and hands it to the receiving doctor.

Two layers, kept structurally separate (per the plan's own reconciliation):
  (a) app/routing/router.py -- deterministic routing core, exactly as the
      diagram specifies (auditable, reproducible, no LLM).
  (b) THIS FILE -- LLM reasoning + report generation on top of (a)'s
      output + full case data. This is Agent 3's only LLM use, consistent
      with the project-wide rule "LLM only at exactly two [now three,
      Agent-3-report-inclusive] pipeline points" -- see
      app/agent1_extraction.py's module docstring for the original rule
      this extends.

Reuses app.agent1_extraction.LLMBackend (Ollama/Groq/Regex) rather than
inventing a second backend abstraction, via LLMBackend.generate_text() --
a single polymorphic method every backend goes through, no isinstance
checks anywhere in this file. RegexBackend doesn't override
generate_text() (its ABC default raises BackendUnavailable), so it
naturally falls through to the non-LLM fallback below, same as any live
backend whose HTTP call fails -- one code path handles both cases.
"""
from __future__ import annotations

from app.agent1_extraction import BackendUnavailable, LLMBackend
from app.routing.schemas import DispatchResult
from app.schemas import ClassificationResult, ExtractedCase

REPORT_SYSTEM_PROMPT = """You are a clinical-handoff assistant preparing a concise report for a \
doctor who is about to receive a patient. You are given: the patient's reported symptoms and \
history, an automated triage classification (produced by a deterministic, non-AI rules engine -- \
you must not contradict or re-diagnose it), and the facility/route the patient is being sent to. \
Write a short, clear report a busy doctor can read in under 30 seconds before the patient arrives. \
Do not invent facts not present in the data you are given. Do not give a different diagnosis than \
the one provided. If information is missing, say so plainly rather than guessing. Plain text, no \
markdown formatting, no more than 150 words."""


def _build_report_prompt(
    case: ExtractedCase, classification: ClassificationResult, dispatch: DispatchResult
) -> str:
    lines = [
        f"Patient-reported symptom text: {case.raw_symptom_text!r}",
        f"Age: {case.age_months} months"
        if case.age_months is not None
        else (f"Age group: {case.age_group.value}" if case.age_group is not None else "Age: not stated"),
        f"Duration: {case.duration or 'not stated'}",
    ]
    if case.danger_signs.any_true():
        fired = [
            name
            for name, v in (
                ("not able to drink/breastfeed", case.danger_signs.not_able_to_drink_or_breastfeed),
                ("vomits everything", case.danger_signs.vomits_everything),
                ("convulsions", case.danger_signs.convulsions),
                ("lethargic/unconscious", case.danger_signs.lethargic_or_unconscious),
            )
            if v is True
        ]
        lines.append(f"Danger signs present: {', '.join(fired)}")
    lines.append(f"Triage classification: {classification.label.value} ({classification.condition or 'no specific condition'})")
    if classification.probable_disease:
        lines.append(
            f"Most probable disease (dataset classifier, score={classification.candidates[0].score if classification.candidates else 'n/a'}): "
            f"{classification.probable_disease}"
        )
        if classification.candidates and classification.candidates[0].precautions:
            lines.append(f"Suggested precautions on file: {', '.join(classification.candidates[0].precautions)}")
    if dispatch.facility:
        lines.append(
            f"Being routed to: {dispatch.facility.name} ({dispatch.facility.facility_type.value}), "
            f"ETA {dispatch.route.eta_minutes if dispatch.route else 'unknown'} minutes"
        )
    lines.append("Write the handoff report now.")
    return "\n".join(lines)


def generate_doctor_report(
    case: ExtractedCase,
    classification: ClassificationResult,
    dispatch: DispatchResult,
    backend: LLMBackend,
) -> str:
    """Generates the doctor-facing report text via backend.generate_text()
    -- one polymorphic call, no isinstance dispatch on backend type.
    Returns a plain deterministic fallback summary (NOT an LLM call) when
    `backend` can't hold up its end (RegexBackend's ABC default raises
    BackendUnavailable, or a live LLMBackend's HTTP call fails) -- an
    honest, useful-if-plain report is better than no report at all when
    the LLM tier is unreachable, matching this codebase's degrade-
    gracefully stance elsewhere (e.g. RegexBackend itself as the SMS/USSD
    tier)."""
    prompt = _build_report_prompt(case, classification, dispatch)
    try:
        return backend.generate_text(REPORT_SYSTEM_PROMPT, prompt)
    except BackendUnavailable:
        return _fallback_report(case, classification, dispatch)


def _fallback_report(case: ExtractedCase, classification: ClassificationResult, dispatch: DispatchResult) -> str:
    """Deterministic, non-LLM report used when the LLM tier is unreachable
    (or the backend can't hold a conversation at all, e.g. RegexBackend).
    Plain string concatenation of exactly the same facts the LLM prompt
    would have received -- no invented content, same "honest degrade"
    posture as the rest of this codebase."""
    parts = [
        f"TRIAGE: {classification.label.value}"
        + (f" ({classification.condition})" if classification.condition else ""),
        f"Reported: {case.raw_symptom_text}",
    ]
    if classification.probable_disease:
        parts.append(f"Probable disease (dataset classifier): {classification.probable_disease}")
    if dispatch.facility:
        parts.append(
            f"Routed to: {dispatch.facility.name}"
            + (f", ETA {dispatch.route.eta_minutes} min" if dispatch.route else "")
        )
    return " | ".join(parts)
