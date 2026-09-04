"""
Agent 1 -- Field Extraction & Triage Agent.

Pluggable LLM backend + one callable pipeline function that goes:
    raw SMS/IVR text  ->  LLM backend  ->  schemas.ExtractedCase (validated)
    ->  rules_engine.classify()  ->  schemas.ClassificationResult

Backend selection here is manual (pass a backend instance in); the
Degradation Controller (Step 4 / Milestone 6.5) is what will make this
choice automatically per connectivity tier. Building three backends now
(Ollama, Groq, Regex) is not scope creep -- Section 5.2 of the Implementation
Plan assigns exactly these three to the WiFi/2G, and SMS/USSD tiers, so the
Degradation Controller has real things to select between when it's built.

FAISS-based disambiguation (Milestone 3 in the plan's own build order) IS
wired in, via extract_case_with_followup()'s optional `disambiguator`
param (see app/disambiguation.py and _apply_disambiguation() below). It
sets `symptom`/`disambiguation_confidence`, and can additionally backfill
case.cough.present/case.diarrhea.present via apply_disambiguation_fallback()
-- but only for a field a backend never assessed at all; see that
function's docstring for the exact rule. `disambiguation_confidence` stays
None whenever no disambiguator is passed (the default is a StubDisambiguator
no-op) or the backend never populated `symptom` in the first place.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

import requests

from app.schemas import (
    AgeGroup,
    ClassificationLabel,
    ClassificationResult,
    CoughDifficultBreathing,
    DiarrheaAssessment,
    ExtractedCase,
    Severity,
)
from app.rules_engine import classify as rules_classify
from app.disambiguation import DisambiguationResult, Disambiguator, StubDisambiguator
from app.case_store import CaseStore
from app import followup_policy

EXTRACTION_SYSTEM_PROMPT = """You are a medical field-extraction assistant for a rural SMS/IVR triage \
system. Extract ONLY the fields below from the patient/caregiver message. Support Hindi, Tamil, \
English, and mixed lay terms. Do not diagnose. Do not invent values -- if a field is genuinely not \
present in the message, set it to null (for booleans, null means "not stated", NOT false). Return \
strict JSON only, matching exactly this shape:

{
  "symptom": string or null,
  "duration": string or null,
  "severity": "mild" | "moderate" | "severe" | "unknown",
  "age_group": "infant" | "child" | "adult" | "elderly" | null,
  "age_months": integer or null,
  "location": string or null,
  "notes": string or null,
  "danger_signs": {
    "not_able_to_drink_or_breastfeed": true | false | null,
    "vomits_everything": true | false | null,
    "convulsions": true | false | null,
    "lethargic_or_unconscious": true | false | null
  },
  "cough": {"present": true | false | null, "duration_days": integer or null},
  "diarrhea": {
    "present": true | false | null,
    "duration_days": integer or null,
    "blood_in_stool": true | false | null,
    "restless_or_irritable": true | false | null,
    "sunken_eyes": true | false | null,
    "drinks_eagerly_thirsty": true | false | null,
    "drinks_poorly_or_not_able": true | false | null
  }
}

Only set a danger_signs/cough/diarrhea field to true or false if the message actually states or \
clearly implies it; otherwise leave it null. Exam-only signs (breathing rate count, chest indrawing, \
skin pinch) are never askable from a text message -- do not attempt to fill those; they are not in \
this schema for that reason.

AGE NORMALIZATION -- age_months must always be a whole number of MONTHS. Watch the unit; never \
copy the bare number when the unit is not months:
  - "<N> years old" / "<N> saal"        -> N * 12   (e.g. "2 years" -> 24, "5 years" -> 60, "70 years" -> 840)
  - "<N> months old" / "<N> mahine"     -> N        (e.g. "6 months" -> 6)
  - "<N> weeks old" / "<N> hafte"       -> about N / 4, rounded down (e.g. "6 weeks" -> 1, "3 weeks" -> 0)
  - "<N> days old" / "<N> din"          -> about N / 30, rounded down (e.g. "10 days" -> 0)
  - "newborn" / "just born" / "abhi paida hua" / "a few days old"  -> 0
  - "one and a half years" / "1.5 years" -> 18   (convert the fraction too)
If the message gives a phrase like "turned 5 last month" or "5th birthday", that means 5 YEARS -> 60, \
not 5 months -- read the whole phrase, not just the nearest number to a unit word. \
If age is given only vaguely with no number ("a baby", "a toddler", "an old man", "elderly"), leave \
age_months null and set age_group instead. If age is not mentioned at all, set BOTH age_months and \
age_group to null."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot serve this request at all (missing
    credentials, connection refused, non-2xx after retries). The
    Degradation Controller (Step 4) catches this to fall back a tier."""


class LLMBackend(ABC):
    name: str = "abstract"

    # Whether this backend can meaningfully hold up its end of a follow-up
    # round-trip (send a question, get a reply, re-extract). False by
    # default -- a backend has to opt in. RegexBackend deliberately does
    # not override this: a keyword matcher can't ask or interpret a
    # clarifying question, so pretending it can would be dishonest. See
    # extract_case_with_followup().
    supports_followup: bool = False

    @abstractmethod
    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        """Return the raw parsed JSON dict described in EXTRACTION_SYSTEM_PROMPT."""

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Freeform text generation (arbitrary system + user prompt -> raw
        text back), for pipeline points that need prose rather than
        extract()'s structured-JSON contract -- currently
        app.routing.report's doctor-facing report generation, Agent 3's
        only LLM use.

        Default here raises BackendUnavailable -- same explicit-opt-in
        pattern as `supports_followup` above: a backend has to override
        this to claim it can generate freeform text. RegexBackend
        deliberately does not override it (a keyword matcher cannot
        generate prose), so calling generate_text() on it correctly raises
        rather than silently returning something meaningless. Callers
        (e.g. generate_doctor_report()) catch BackendUnavailable and
        degrade to a non-LLM fallback rather than needing to know in
        advance which concrete backend type they were given -- this is
        the single polymorphic call site every backend goes through, with
        no isinstance-based dispatch anywhere in the caller.
        """
        raise BackendUnavailable(f"{self.name} does not support generate_text()")


def _extract_json_object(text: str) -> dict:
    """LLMs sometimes wrap JSON in prose or code fences even when asked not
    to. Pull out the first balanced {...} block and parse that."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


class OllamaBackend(LLMBackend):
    """Local Llama 3.2 3B via Ollama's HTTP API. This is the 2G/offline
    fallback per Section 5.2 -- no external network dependency once the
    model is pulled locally."""

    name = "ollama_llama3.2:3b"
    supports_followup = True

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434", timeout: int = 30):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        context = context or []
        user_msg = patient_text
        if context:
            user_msg = "PRIOR TURNS:\n" + "\n".join(context) + f"\n\nLATEST MESSAGE:\n{patient_text}"
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": EXTRACTION_SYSTEM_PROMPT,
                    "prompt": user_msg,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BackendUnavailable(f"Ollama unreachable at {self.host}: {e}") from e

        body = resp.json()
        raw = body.get("response", "")
        try:
            return _extract_json_object(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise BackendUnavailable(f"Ollama returned non-JSON output: {raw[:300]!r}") from e

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BackendUnavailable(f"Ollama unreachable at {self.host}: {e}") from e
        return resp.json().get("response", "").strip()


class GroqBackend(LLMBackend):
    """Cloud primary backend (WiFi/2G tiers per Section 5.2). Requires
    GROQ_API_KEY. Not exercised by the Step 1 integration test in this repo
    -- no API key is configured in this environment -- but the interface is
    real and ready for the Degradation Controller to call."""

    name = "groq"
    supports_followup = True

    def __init__(self, model: str = "llama-3.1-8b-instant", timeout: int = 15):
        self.model = model
        self.timeout = timeout
        self.api_key = os.environ.get("GROQ_API_KEY")

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        if not self.api_key:
            raise BackendUnavailable("GROQ_API_KEY not set")
        context = context or []
        user_msg = patient_text
        if context:
            user_msg = "PRIOR TURNS:\n" + "\n".join(context) + f"\n\nLATEST MESSAGE:\n{patient_text}"
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BackendUnavailable(f"Groq API call failed: {e}") from e

        body = resp.json()
        raw = body["choices"][0]["message"]["content"]
        try:
            return _extract_json_object(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise BackendUnavailable(f"Groq returned non-JSON output: {raw[:300]!r}") from e

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise BackendUnavailable("GROQ_API_KEY not set")
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BackendUnavailable(f"Groq API call failed: {e}") from e
        return resp.json()["choices"][0]["message"]["content"].strip()


# Minimal bilingual keyword table for the SMS/USSD (Tier 4) fallback. This is
# a deliberately small starting point, not a claim of NLP coverage -- see
# module docstring. Extend as Milestone 6.5 benchmark data reveals gaps.
_KEYWORDS = {
    "symptom": [
        (re.compile(r"बुखार|fever"), "fever"),
        (re.compile(r"खांसी|खाँसी|cough"), "cough"),
        (re.compile(r"दस्त|diarrhea|diarrhoea|loose motion"), "diarrhea"),
        (re.compile(r"साँस|sans|breath"), "difficulty breathing"),
    ],
    "danger_true": [
        (re.compile(r"पी नहीं|not drinking|can'?t drink|unable to drink"), "not_able_to_drink_or_breastfeed"),
        (re.compile(r"बेहोश|unconscious|unresponsive"), "lethargic_or_unconscious"),
        (re.compile(r"दौरा|convuls|fit\b|seizure"), "convulsions"),
        (re.compile(r"vomit.*everything|उल्टी.*सब"), "vomits_everything"),
    ],
}


class RegexBackend(LLMBackend):
    """No-LLM deterministic keyword extractor. This is the SMS/USSD tier
    (Tier 4) backend per Section 5.2 -- 'Regex/keyword extraction, no LLM
    call'. Only ever produces true (never false) for danger signs it
    matches a keyword for; everything else stays null/not-assessed, which
    is the honest, safe behavior for a method this coarse."""

    name = "regex_keyword"

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        text = patient_text.lower()
        symptom = None
        for pattern, label in _KEYWORDS["symptom"]:
            if pattern.search(text):
                symptom = label
                break

        danger_signs = {
            "not_able_to_drink_or_breastfeed": None,
            "vomits_everything": None,
            "convulsions": None,
            "lethargic_or_unconscious": None,
        }
        for pattern, field in _KEYWORDS["danger_true"]:
            if pattern.search(text):
                danger_signs[field] = True

        return {
            "symptom": symptom,
            "duration": None,
            "severity": "unknown",
            # No age keywords in this backend -- leave age genuinely unset
            # (None), same honest-absence handling as age_months just below.
            # Coercing to a literal "unknown" here was a hardcoded default:
            # it made "this coarse backend never looks at age" indistinguishable
            # from "age was assessed and is unknown". rules_engine.classify()
            # now routes a None age to AGE_UNKNOWN_CANNOT_ROUTE.
            "age_group": None,
            "age_months": None,
            "location": None,
            "notes": patient_text,
            "danger_signs": danger_signs,
            "cough": {"present": symptom == "cough" or None, "duration_days": None} if symptom == "cough" else None,
            "diarrhea": {"present": True, "duration_days": None, "blood_in_stool": None,
                         "restless_or_irritable": None, "sunken_eyes": None,
                         "drinks_eagerly_thirsty": None, "drinks_poorly_or_not_able": None}
            if symptom == "diarrhea" else None,
        }


def _sanitize_enum_fields(parsed: dict) -> dict:
    """Small local models occasionally emit a near-miss enum value (stray
    characters, wrong case, a translated word) instead of one of the exact
    literals the prompt specifies. Sanitization's only job is "don't let
    garbage through as if it were data" -- not to guess what the model
    meant.

      - severity: an unrecognized string -> "unknown". "unknown" IS a
        first-class, honest value on the severity scale (there is no
        "severity not assessed" state distinct from it), so this is a safe
        floor, not an invented value.
      - age_group: an unrecognized string -> None ("not stated"). Unlike
        severity, age_group now has a real absent state (Optional, default
        None), and None/UNKNOWN are NOT the same: None = "we don't have a
        usable age", UNKNOWN = "a caller explicitly set it to unknown".
        Coercing junk to the literal "unknown" would manufacture the
        second from the first. An unrecognized value carries no usable age
        information, so it becomes None and rules_engine.classify() routes
        it to AGE_UNKNOWN_CANNOT_ROUTE rather than assuming a pediatric
        age band.

    This does NOT try to recover a trivially-fixable near-miss (case-fold
    "ADULT" -> "adult", map the lay term "baby" -> "infant"). That is
    extraction's job, and doing it here would blur the line between
    "reject malformed output" and "re-do the model's work". (If a specific
    recoverable pattern turns out to matter, that's a deliberate extraction
    -prompt or post-processing change to propose, not something to smuggle
    into sanitization.)

    Intentionally does NOT touch the danger_signs/cough/diarrhea booleans
    -- those must stay exactly True/False/None, with no coercion, since
    that distinction is the hard safety requirement this schema exists to
    enforce."""
    parsed = dict(parsed)
    sev = parsed.get("severity")
    if sev is not None and sev not in {e.value for e in Severity}:
        parsed["severity"] = Severity.UNKNOWN.value
    ag = parsed.get("age_group")
    if ag is not None and ag not in {e.value for e in AgeGroup}:
        parsed["age_group"] = None
    return parsed


# --- Deterministic <2-month infant-age safety net -----------------------------
#
# Phase 1 Part B found that llama3.2:3b (and small models generally) copy the
# bare number when the age unit is "weeks"/"days" ("3 weeks old" -> age_months=3,
# "6 week old" -> 6) and return None for "newborn". Every one of those failures
# pushes a genuine <2-month infant OUT of the young-infant escalation range
# (rules_engine: age_months < 2), silently routing a neonate through the
# pediatric IMNCI path -- the single highest-risk failure mode in the pipeline.
#
# Prompt instructions alone were judged insufficient for a boundary this
# dangerous, so this regex runs on the RAW patient text, independently of the
# LLM, and overrides age_months when it finds explicit evidence of a <2-month
# infant that the LLM's value contradicts or misses.
#
# COVERAGE (deliberately narrow -- this is a safety net, not a general age
# parser):
#   - "<N> day(s) old",  "<N>-day-old"      -> N // ~30.44 months
#   - "<N> week(s) old",  "<N>-week-old"     -> N*7 // ~30.44 months
#   - "newborn" / "new-born" / "just born" / "just delivered" / "neonate" /
#     "neonatal"                              -> 0
# and only when the derived value is < 2 (i.e. actually inside the young-infant
# band). A "10 week old" -> 2 months is left entirely to the LLM; this net does
# not touch it.
#
# KNOWN GAPS (not covered, by design -- flagged, not silently half-handled):
#   - "born three weeks ago", "delivered last month", spelled-out numbers
#     ("three weeks old"), and any age in a non-English script. Extend only
#     with real benchmark evidence, same policy as the _KEYWORDS table.
#   - It never RAISES an age (e.g. "turned 5 last month" -> LLM's wrong "5"
#     is not an infant pattern, so this net stays out of it -- see the Part B
#     report note on why number/unit-separated phrasing is not reliably
#     fixable at this layer).
_MONTHS_PER_DAY = 1.0 / 30.44
_YOUNG_INFANT_BOUNDARY_MONTHS = 2  # mirrors rules_engine.MODULE_AGE_MIN_MONTHS

_INFANT_AGE_TEXT_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], int]]] = [
    (re.compile(r"\b(?:new[\s-]?born|just\s+born|just\s+delivered|neonate|neonatal)\b", re.IGNORECASE),
     lambda m: 0),
    (re.compile(r"\b(\d{1,3})\s*[-\s]?\s*days?\s*[-\s]?\s*old\b", re.IGNORECASE),
     lambda m: int(int(m.group(1)) * _MONTHS_PER_DAY)),
    (re.compile(r"\b(\d{1,2})\s*[-\s]?\s*weeks?\s*[-\s]?\s*old\b", re.IGNORECASE),
     lambda m: int(int(m.group(1)) * 7 * _MONTHS_PER_DAY)),
]


def _infant_age_months_from_text(raw_text: str) -> Optional[int]:
    """Return an explicit <2-month age in whole months if `raw_text` contains
    day/week/newborn infant-age phrasing that lands inside the young-infant
    band; None otherwise. Never returns a value >= 2 -- patterns that would
    compute higher are out of this net's scope and are left to the LLM."""
    best: Optional[int] = None
    for pattern, to_months in _INFANT_AGE_TEXT_PATTERNS:
        m = pattern.search(raw_text)
        if m is None:
            continue
        months = to_months(m)
        if months < _YOUNG_INFANT_BOUNDARY_MONTHS:
            best = months if best is None else min(best, months)
    return best


def _apply_infant_age_floor(raw_text: str, parsed: dict) -> dict:
    """If the raw text carries explicit <2-month infant-age evidence and the
    backend's age_months does NOT already sit in that band (it's None, or it's
    >= 2 -- the exact Part B failure shape), override age_months with the
    deterministic value and leave an audit note. A backend value that is
    already < 2 is left untouched (it and this net agree on the thing that
    matters -- the young-infant boundary)."""
    floor = _infant_age_months_from_text(raw_text)
    if floor is None:
        return parsed
    llm_age = parsed.get("age_months")
    if isinstance(llm_age, int) and 0 <= llm_age < _YOUNG_INFANT_BOUNDARY_MONTHS:
        return parsed
    parsed = dict(parsed)
    parsed["age_months"] = floor
    note = (
        f"[age_months set to {floor} by deterministic infant-age rule: raw text "
        f"matched day/week/newborn phrasing; backend had age_months={llm_age!r}]"
    )
    existing = parsed.get("notes")
    parsed["notes"] = f"{existing} {note}" if existing else note
    return parsed


class ExtractionValidationError(RuntimeError):
    """Raised when the backend's output can't be made to fit ExtractedCase
    even after enum sanitization -- e.g. a required field has the wrong
    type. Distinct from BackendUnavailable: the backend responded, but its
    content doesn't validate. Callers (the future Degradation Controller)
    should treat this the same as a failed attempt for fallback purposes."""


def extract_case(
    raw_text: str,
    backend: LLMBackend,
    context: Optional[list[str]] = None,
    case_id: Optional[str] = None,
    language: Optional[str] = None,
) -> ExtractedCase:
    """LLM/backend call -> schema validation. Raises pydantic.ValidationError
    if the backend's output doesn't fit the schema -- this is intentional:
    a malformed extraction must not silently become a classified case."""
    parsed = backend.extract(raw_text, context=context)
    parsed = _sanitize_enum_fields(parsed)
    parsed = _apply_infant_age_floor(raw_text, parsed)
    try:
        return ExtractedCase(
            raw_symptom_text=raw_text,
            case_id=case_id,
            language=language,
            llm_backend=backend.name,
            **parsed,
        )
    except Exception as e:  # pydantic.ValidationError or a malformed-shape TypeError
        raise ExtractionValidationError(
            f"{backend.name} output did not validate against ExtractedCase: {e}"
        ) from e


def extract_and_classify(
    raw_text: str,
    backend: LLMBackend,
    context: Optional[list[str]] = None,
    case_id: Optional[str] = None,
    language: Optional[str] = None,
) -> tuple[ExtractedCase, ClassificationResult]:
    """The one callable pipeline function Step 1 asks for: raw text -> validated
    schema -> IMNCI classification, as a single call."""
    case = extract_case(raw_text, backend, context=context, case_id=case_id, language=language)
    result = rules_classify(case)
    return case, result


# ---------------------------------------------------------------------------
# Multi-turn follow-up loop (additive to the single-shot flow above; nothing
# in extract_case()/extract_and_classify() changes).
# ---------------------------------------------------------------------------

# Deterministic, auditable question templates -- one per DangerSigns field.
# Question generation is plain Python, not an LLM call: this keeps the
# question itself part of the auditable rules layer, consistent with the
# project's "LLM only at exactly two pipeline points" design (Agent 1
# extraction and Agent 2 reasoning). If a case ever needs a
# clinically-tailored/LLM-generated follow-up question instead of a fixed
# template, that's a real design change and should be asked about, not
# assumed -- see the caller's task instructions.
DANGER_SIGN_FOLLOWUP_QUESTIONS: dict[str, str] = {
    "not_able_to_drink_or_breastfeed": "Is the patient able to drink or breastfeed normally?",
    "vomits_everything": "Does the patient vomit up everything they eat or drink?",
    "convulsions": "Has the patient had any convulsions or fits during this illness?",
    "lethargic_or_unconscious": "Is the patient unusually sleepy, hard to wake, or unconscious?",
}

# One generic clarifier for app.followup_policy.missing_required_adult()'s
# single sentinel field ("symptom_tokens" -- see that function's docstring
# for why it's checked via case.symptom rather than a fixed per-field
# checklist like the pediatric danger signs above). Deliberately one broad
# question, not 131 field-specific ones -- the dataset vocabulary is far
# too large for a per-token template, and one open question is exactly the
# "ask minimal" behavior the follow-up policy exists to enforce.
ADULT_SYMPTOM_FOLLOWUP_QUESTION = (
    "What is the main problem? Please describe it in a few words "
    "(for example: fever, chest pain, vomiting, loose motions)."
)

# Clarifier for rules_engine.classify()'s AGE_UNKNOWN_CANNOT_ROUTE
# (missing_fields=["age_months"]): every route -- pediatric IMNCI, adult
# dataset, young-infant escalation -- turns on age, so a case with no age
# at all cannot be routed without assuming one. Asks for the actual age
# rather than just a band; the unit examples deliberately include "weeks"
# so a caregiver of a young infant answers in a form the <2-month boundary
# can use.
AGE_CLARIFIER_FOLLOWUP_QUESTION = (
    "How old is the patient? Please give an age -- for example "
    "\"3 weeks\", \"6 months\", \"4 years\", or \"70 years\"."
)

# Merged lookup _followup_question_for() searches -- keeps
# DANGER_SIGN_FOLLOWUP_QUESTIONS itself unchanged (existing callers/tests
# import it directly and expect exactly its 4 entries).
_FOLLOWUP_QUESTIONS: dict[str, str] = {
    **DANGER_SIGN_FOLLOWUP_QUESTIONS,
    "symptom_tokens": ADULT_SYMPTOM_FOLLOWUP_QUESTION,
    "age_months": AGE_CLARIFIER_FOLLOWUP_QUESTION,
}


def _followup_question_for(missing_fields: list[str]) -> Optional[str]:
    """One targeted question per turn (Section 2.3 step 5 of the
    Implementation Plan), for the first missing field we have a template
    for. Returns None if nothing missing has a template -- the loop stops
    rather than asking a question it can't generate deterministically."""
    for field in missing_fields:
        if field in _FOLLOWUP_QUESTIONS:
            return _FOLLOWUP_QUESTIONS[field]
    return None


def question_for_incomplete_result(result: ClassificationResult) -> Optional[str]:
    """The SECOND entry point into the follow-up module, alongside
    extract_case_with_followup()'s pre-classification loop below.

    Two valid flows into the follow-up module, both real:
      (A) Agent 1 -> follow-up module directly, PRE-classification:
          extract_case_with_followup()'s loop below, gated by
          app.followup_policy.missing_required() -- asks before ever
          calling rules_classify(), so a case that's already resolvable
          doesn't waste a classify() call.
      (B) Rules Engine -> follow-up module, POST-classification: THIS
          function. Takes rules_classify()'s own ClassificationResult --
          specifically one with label=INCOMPLETE_ASSESSMENT and a real
          `missing_fields` list (from
          rules_engine.check_danger_sign_completeness() on the pediatric
          route, or classify_via_dataset()'s INSUFFICIENT_SYMPTOM_DATA on
          the adult route) -- and returns the next question to ask,
          reusing the exact same deterministic templates (_FOLLOWUP_
          QUESTIONS) flow (A) uses. This is the real, standalone shape a
          caller using the single-shot extract_and_classify() (no loop)
          needs: extract -> classify -> got INCOMPLETE_ASSESSMENT? -> ask
          this question -> (later, on the next inbound message) extract
          again with the answer folded into context -> classify again.
          It is also the more production-realistic shape for an async
          SMS/IVR channel, where "classify, then decide whether to ask"
          naturally happens as a separate step per inbound message rather
          than a single blocking loop.

    Returns None if `result.label` isn't INCOMPLETE_ASSESSMENT (nothing to
    ask -- the case was already classified) or if none of
    `result.missing_fields` has a template (same "don't generate a
    question we can't ask deterministically" rule as
    _followup_question_for()).
    """
    if result.label != ClassificationLabel.INCOMPLETE_ASSESSMENT:
        return None
    return _followup_question_for(result.missing_fields)


# Maps a disambiguated symptom category onto the ExtractedCase field it can
# fall back into. Deliberately mirrors app.disambiguation.VOCABULARY's two
# categories exactly -- there is no third entry here for the same reason
# there is no "fever" entry in VOCABULARY: rules_engine.py has no axis for
# anything else, so there is nothing else a fallback could usefully target.
_FALLBACK_CATEGORY_TO_FIELD: dict[str, str] = {
    "cough_or_difficult_breathing": "cough",
    "diarrhea": "diarrhea",
}


def apply_disambiguation_fallback(
    case: ExtractedCase, disambiguation: DisambiguationResult
) -> ExtractedCase:
    """Backfills case.cough.present / case.diarrhea.present from an already-
    computed disambiguation result -- ONLY when a backend never assessed
    that field at all. This is a fallback for what structured extraction
    missed, never a correction to what it found:

      - `disambiguation.matched_category` is None (no match, or below the
        disambiguator's own confidence threshold -- both StubDisambiguator
        and FAISSDisambiguator only ever return a non-None category when
        confident; see their docstrings) -> no-op.
      - The matched category isn't one this rules engine can act on (today
        that's impossible given app.disambiguation.VOCABULARY, but this is
        a defensive no-op, not an assumption) -> no-op.
      - The relevant block (case.cough / case.diarrhea) is None entirely,
        i.e. the backend never populated it -> create a fresh block with
        present=True and every other field left at its default None
        (duration_days, breaths_per_minute, chest_indrawing, etc. were
        never assessed by anything and must stay honestly None -- this
        function must never invent a value for them).
      - The block exists but its `present` field is specifically None
        (the backend built the block -- e.g. it captured duration_days
        from context -- but never resolved whether the symptom is present)
        -> set `present=True` on a copy of that SAME block, preserving
        every other field already on it untouched. Discarding a field the
        backend legitimately extracted (e.g. duration_days=3) just because
        this function is also filling in `present` would throw away real
        information for no clinical-safety reason -- the invariant this
        function protects is about `present` specifically, not about
        whether the block object already exists.
      - The block exists and `present` is already True or False -- the
        backend already explicitly assessed this axis from the patient's
        actual words -- -> untouched, no matter what the symptom string
        says. The LLM backend reading the real message gets first say;
        disambiguation on the coarser `symptom` string never overrides it.

    Never touches danger_signs, symptom, disambiguation_confidence, or any
    field on an axis the matched category doesn't correspond to (e.g. a
    "diarrhea" match never touches case.cough).

    Non-mutating: returns a new ExtractedCase (and, when it changes, a new
    nested CoughDifficultBreathing/DiarrheaAssessment) via model_copy(
    update=...), same pattern as _apply_disambiguation() and
    rules_engine.most_severe_wins().

    Kept separate from _apply_disambiguation() (which only ever touches
    `symptom`/`disambiguation_confidence`) so this classification-affecting
    step is independently testable/auditable on its own, without needing a
    live backend or LLM call -- just a case and a DisambiguationResult.
    """
    category = disambiguation.matched_category
    field_name = _FALLBACK_CATEGORY_TO_FIELD.get(category) if category is not None else None
    if field_name is None:
        return case

    block = getattr(case, field_name)

    if block is None:
        block_cls = CoughDifficultBreathing if field_name == "cough" else DiarrheaAssessment
        return case.model_copy(update={field_name: block_cls(present=True)})

    if block.present is None:
        return case.model_copy(update={field_name: block.model_copy(update={"present": True})})

    # block.present is already True or False -- already explicitly assessed
    # by the backend from the patient's own words; disambiguation must
    # never override that, regardless of what the symptom string says.
    return case


# Splits a free-text symptom string on common list separators ("and",
# commas, semicolons, ampersands) so a message reporting several symptoms
# at once (e.g. "chest pain and breathlessness and sweating" -- a very
# common real caregiver message shape) can have EACH symptom matched
# against the dataset vocabulary independently, rather than the whole
# multi-symptom sentence being embedded as one query and matching only
# whichever single symptom dominates that combined embedding. Each
# resulting clause is still matched via a single top-1 nearest-neighbor
# call (match_dataset_symptom() itself is unchanged) -- this only changes
# WHAT text gets matched, not how matching works.
_SYMPTOM_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|&|\band\b)\s*", re.IGNORECASE)


def _split_symptom_clauses(text: str) -> list[str]:
    clauses = [c.strip() for c in _SYMPTOM_CLAUSE_SPLIT.split(text) if c.strip()]
    return clauses or [text]


def _apply_disambiguation(case: ExtractedCase, disambiguator: Disambiguator) -> ExtractedCase:
    """Runs disambiguation on the free-text `symptom` field once, applies
    its result to `symptom`/`disambiguation_confidence`, and then hands the
    same result to apply_disambiguation_fallback() to (maybe) backfill
    case.cough/case.diarrhea. Running the disambiguator exactly once here
    -- rather than a second time inside the fallback step -- matters for
    two reasons: it avoids a redundant embedding-model call, and it avoids
    a subtly wrong second call, since by the time the fallback step would
    run, `case.symptom` may already have been overwritten to the matched
    category string below (disambiguating that string a second time is not
    the same question as disambiguating the original free text).

    If `case.symptom` is None, disambiguation is skipped entirely -- there
    is nothing to disambiguate, and running it on None would just be
    manufacturing a result out of nothing. (case.cough/case.diarrhea are
    left exactly as the backend produced them in this case, same as
    before.)

    Non-mutating: returns a new ExtractedCase via model_copy(update=...),
    same pattern already used in rules_engine.most_severe_wins(). Because
    model_copy's default update is shallow, danger_signs remains the exact
    same nested object on the copy -- not merely equal, but literally
    unchanged; cough/diarrhea are also left as the same object unless
    apply_disambiguation_fallback() decides to replace one.

    Also calls disambiguator.match_dataset_symptom() on each clause of the
    ORIGINAL symptom text (split via _split_symptom_clauses(), before
    `symptom` may be overwritten to the pediatric matched category above)
    to populate `symptom_tokens` for app.disease_classifier's dataset-
    driven path -- a separate axis from the pediatric symptom/cough/
    diarrhea fallback above (see app.disambiguation module docstring for
    why the two are independent indexes). Matching per-clause rather than
    the whole string once means a message reporting several symptoms (e.g.
    "chest pain and breathlessness and sweating") gets multiple tokens
    instead of just whichever symptom dominates the combined embedding.
    Safe to call unconditionally: the Disambiguator ABC's default
    match_dataset_symptom() is an honest no-op, so StubDisambiguator and
    any future implementation that doesn't support it simply leave
    symptom_tokens untouched.
    """
    if case.symptom is None:
        return case
    original_symptom_text = case.symptom
    result = disambiguator.disambiguate(case.symptom)
    updates: dict = {"disambiguation_confidence": result.confidence}
    if result.matched_category is not None:
        updates["symptom"] = result.matched_category

    new_tokens: list[str] = []
    for clause in _split_symptom_clauses(original_symptom_text):
        dataset_result = disambiguator.match_dataset_symptom(clause)
        token = dataset_result.matched_category
        if token is not None and token not in case.symptom_tokens and token not in new_tokens:
            new_tokens.append(token)
    if new_tokens:
        updates["symptom_tokens"] = case.symptom_tokens + new_tokens

    case = case.model_copy(update=updates)
    return apply_disambiguation_fallback(case, result)


def extract_case_with_followup(
    raw_text: str,
    backend: LLMBackend,
    answer_provider: Optional[Callable[[str], str]] = None,
    case_id: Optional[str] = None,
    language: Optional[str] = None,
    max_followup_turns: int = 2,
    disambiguator: Optional[Disambiguator] = None,
) -> tuple[ExtractedCase, list[dict[str, str]]]:
    """Wraps extract_case() with a bounded multi-turn follow-up loop gated
    by app.followup_policy.missing_required() -- the minimum-viable-info
    gate (user-flagged requirement): ask only for what's genuinely missing
    and genuinely needed to triage, nothing more, since a caregiver
    messaging in an emergency has low patience for extra questions. For
    the pediatric route that's the four DangerSigns fields (a single
    confirmed True danger sign overrides every other finding in
    rules_engine.classify, so an unassessed danger sign is the one gap
    worth a round-trip on before falling through to INCOMPLETE_ASSESSMENT);
    for the adult/out-of-band dataset route it's having any reported
    symptom text at all. See app/followup_policy.py for the exact rules.

    Backend scope: only backends with `supports_followup = True` (Ollama,
    Groq) participate in the loop. RegexBackend (supports_followup=False by
    default) always falls through after exactly one pass -- a keyword
    matcher cannot hold up its end of a clarifying-question exchange, so
    this function does not pretend otherwise.

    `answer_provider`: given the generated follow-up question text, returns
    the caregiver's reply. This callback is the seam between this
    synchronous loop and the real system's future async webhook/session-
    store layer (Milestone 1/6 -- not built yet): in production, "ask a
    question and get an answer" means "send an SMS/IVR prompt and wait for
    the next inbound message on this phone number's session," which needs
    infrastructure this repo doesn't have yet. If `answer_provider` is None,
    the loop has no way to get an answer and does not attempt one -- this
    degrades to exactly one extract_case() call, matching prior behavior.

    Context handling (the specific failure mode to guard against): the
    backend context passed on every re-extraction call always starts with
    the original message and only ever grows -- it is never rebuilt from
    just the latest turn, so the original message cannot silently drop out
    after turn 1. `raw_text` itself is also passed unchanged as the primary
    argument on every call (so the returned case's `raw_symptom_text` always
    reflects what the patient originally said, never a follow-up answer).

    Turn cap: `max_followup_turns` (default 2) bounds cost/latency on the
    Ollama/Groq round-trip. If the cap is hit and DangerSigns is still
    incomplete, this function returns the case as-is -- it does NOT fill
    the remaining fields with a guessed False. The caller (rules_engine.
    classify, via extract_and_classify_with_followup or directly) will then
    correctly produce INCOMPLETE_ASSESSMENT. Defaulting unresolved fields to
    False after the cap was considered and rejected: that would silently
    convert "we tried and still don't know" into "assessed as absent,"
    exactly the None/False conflation this schema exists to prevent.

    `disambiguator`: applied exactly once, on the FINAL case, right before
    this function returns -- regardless of which of the three exit paths
    above produced that final case (RegexBackend/no-followup-support,
    natural completion once DangerSigns resolves, or the turn cap being
    hit). Concretely this means the loop above now has a single shared exit
    point that every path funnels through, with disambiguation applied
    there once, rather than wastefully re-disambiguating on every
    intermediate re-extraction. Defaults to StubDisambiguator() (a no-op
    that leaves `symptom` untouched and `disambiguation_confidence` at
    None) so every caller that predates this parameter -- including the
    existing follow-up test suite -- sees byte-for-byte the same behavior
    as before. Disambiguation updates `symptom`/`disambiguation_confidence`
    directly, and may ALSO backfill case.cough.present or
    case.diarrhea.present via apply_disambiguation_fallback() -- but only
    when a backend never assessed that field at all (block missing, or
    block present but its `present` field is None); it never overrides a
    field a backend already explicitly set to True or False, and it never
    touches danger_signs. See _apply_disambiguation()'s and
    apply_disambiguation_fallback()'s docstrings for the exact rules.

    Returns (final_case, followup_audit_trail). `followup_audit_trail` is a
    list of {"question": ..., "answer": ...} dicts, one entry per follow-up
    turn actually taken -- it never includes the original raw_text. This is
    a separate structure from the backend context on purpose: the context
    list is an internal detail assembled to keep the backend grounded, the
    audit trail is the caller-facing record of what was actually asked.
    """
    backend_context: list[str] = [f"ORIGINAL MESSAGE: {raw_text}"]
    followup_trail: list[dict[str, str]] = []

    case = extract_case(raw_text, backend, context=None, case_id=case_id, language=language)

    if backend.supports_followup:
        turns_used = 0
        while (
            answer_provider is not None
            and turns_used < max_followup_turns
            and followup_policy.missing_required(case)  # bare-minimum-info gate, see app/followup_policy.py
        ):
            question = _followup_question_for(followup_policy.missing_required(case))
            if question is None:
                break  # nothing missing has a template -- can't generate a deterministic question

            answer = answer_provider(question)
            followup_trail.append({"question": question, "answer": answer})
            turns_used += 1

            backend_context.append(f"FOLLOW-UP Q: {question}")
            backend_context.append(f"FOLLOW-UP A: {answer}")

            case = extract_case(
                raw_text, backend, context=list(backend_context), case_id=case_id, language=language
            )

    # Single shared exit point (see docstring above): disambiguation always
    # runs here, once, on whatever case this function is about to return.
    case = _apply_disambiguation(case, disambiguator or StubDisambiguator())
    return case, followup_trail


def extract_and_classify_with_followup(
    raw_text: str,
    backend: LLMBackend,
    answer_provider: Optional[Callable[[str], str]] = None,
    case_id: Optional[str] = None,
    language: Optional[str] = None,
    max_followup_turns: int = 2,
    disambiguator: Optional[Disambiguator] = None,
    case_store: Optional[CaseStore] = None,
) -> tuple[ExtractedCase, list[dict[str, str]], ClassificationResult]:
    """Convenience wrapper mirroring extract_and_classify(), but through the
    follow-up loop. Thin by design -- all the actual logic lives in
    extract_case_with_followup(); this just adds the rules_engine.classify()
    call so callers don't have to remember to do it themselves, matching the
    existing extract_case()/extract_and_classify() pairing.

    `case_store`: optional app.case_store.CaseStore. When provided, this is
    the pipeline's single point where a finished case gets logged (area +
    classification, see CaseStore.record() for exactly what's persisted --
    never the raw patient text). Defaults to None (no recording) so every
    existing caller -- including the full pre-this-task test suite -- keeps
    working with zero I/O side effects unless a caller opts in."""
    case, followup_trail = extract_case_with_followup(
        raw_text,
        backend,
        answer_provider=answer_provider,
        case_id=case_id,
        language=language,
        max_followup_turns=max_followup_turns,
        disambiguator=disambiguator,
    )
    result = rules_classify(case)
    if case_store is not None:
        case_store.record(case, result)
    return case, followup_trail, result
