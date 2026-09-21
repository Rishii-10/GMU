"""
Translator: i18n for Agent 1 (plan diagram 1, "Patient input layer" --
Google Translate between languages).

SCOPE -- decision, stated explicitly
------------------------------------------------
This is a real, usable, independently-testable adapter (GoogleTranslateProvider,
via the Google Cloud Translation API's plain REST v2 endpoint over
`requests` with an API key -- no `google-cloud-translate` SDK dependency,
same "avoid an SDK where a direct HTTP call suffices" bias as
app.integrations.messaging.TwilioProvider).

This adapter is NOT wired into app.agent1_extraction.extract_case()/
extract_case_with_followup() itself -- those stay language-agnostic, and
their backends (Ollama/Groq) are already prompted to read
Hindi/Tamil/English/mixed lay terms directly (see EXTRACTION_SYSTEM_PROMPT
in agent1_extraction.py). The input/output translation wrapper lives one
layer out, in app.integrations.language_gateway.LanguageGateway, which the
Streamlit frontend now uses to detect the caregiver's language, translate
the inbound message to English before the pipeline, and translate every
user-facing string back afterward. A caller who wants to compose this
adapter directly instead does:

    translator = GoogleTranslateProvider()
    detected = translator.detect_language(patient_text)
    translated = translator.translate(patient_text, target_language="en")
    case = extract_case(translated.translated_text, backend, language=detected)
    # case.raw_symptom_text will be `translated.translated_text`, not the
    # caregiver's original text, if a caller does this -- preserve the
    # original alongside it (e.g. in `notes`, or by keeping the untranslated
    # string on the caller's own record) if audit needs the original too.

This keeps the seam real (both providers actually work end-to-end against
their real APIs) and testable, without silently changing what
extract_case() does for every existing caller.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class TranslationResult(BaseModel):
    original_text: str
    translated_text: str
    source_language: Optional[str] = None  # None if undetected/unknown
    target_language: str


class Translator(ABC):
    """Two-method interface: translate() for the actual conversion,
    detect_language() as a separate call since a caller may want to know
    the source language without necessarily translating (e.g. to decide
    which language to reply in)."""

    @abstractmethod
    def translate(self, text: str, target_language: str = "en") -> TranslationResult:
        """Must never raise on ordinary text input -- an honest passthrough
        (translated_text == original_text, source_language=None) is
        preferable to a crash, matching this codebase's "no forced guess,
        no crash on ordinary input" convention (e.g.
        app.disambiguation.Disambiguator)."""

    @abstractmethod
    def detect_language(self, text: str) -> Optional[str]:
        """Returns an ISO 639-1 code (e.g. "hi", "en") or None if
        undetected/unsupported."""

    def translate_batch(self, texts: list[str], target_language: str = "en") -> list[TranslationResult]:
        """Translate several strings at once, preserving order. Default
        implementation just loops translate() -- concrete providers override
        this when their API can do it in a single round trip (see
        GoogleTranslateProvider). Same no-raise contract as translate():
        an unreachable backend degrades to per-item passthrough."""
        return [self.translate(t, target_language=target_language) for t in texts]


class IdentityTranslator(Translator):
    """Offline default: no-op passthrough. translated_text is always the
    original text unchanged; source_language/detect_language() always
    None ("never actually run"), matching
    app.disambiguation.StubDisambiguator's honest-no-op philosophy."""

    def translate(self, text: str, target_language: str = "en") -> TranslationResult:
        return TranslationResult(
            original_text=text, translated_text=text, source_language=None, target_language=target_language
        )

    def detect_language(self, text: str) -> Optional[str]:
        return None


class GoogleTranslateProvider(Translator):
    """Real implementation: Google Cloud Translation API v2
    (POST https://translation.googleapis.com/language/translate/v2),
    authenticated via a plain API key query parameter -- no SDK, no OAuth
    service-account flow, so this stays a single env var
    (GOOGLE_TRANSLATE_API_KEY) to configure.

    Raises ValueError immediately at construction if the API key is
    missing -- fail fast, not silently at first call.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("GoogleTranslateProvider requires GOOGLE_TRANSLATE_API_KEY")

    def translate(self, text: str, target_language: str = "en") -> TranslationResult:
        import requests

        try:
            resp = requests.post(
                "https://translation.googleapis.com/language/translate/v2",
                params={"key": self.api_key},
                data={"q": text, "target": target_language, "format": "text"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            # Honest degrade: an unreachable translation API must not break
            # the pipeline -- fall through to the original text, same
            # posture as app.routing.report's LLM-unreachable fallback.
            return TranslationResult(
                original_text=text, translated_text=text, source_language=None, target_language=target_language
            )

        body = resp.json()
        translation = body["data"]["translations"][0]
        return TranslationResult(
            original_text=text,
            translated_text=translation["translatedText"],
            source_language=translation.get("detectedSourceLanguage"),
            target_language=target_language,
        )

    def translate_batch(self, texts: list[str], target_language: str = "en") -> list[TranslationResult]:
        """Single POST with repeated `q` params -- Google Translate v2
        returns a parallel `translations` array in the same order. `requests`
        serializes data={"q": [...]} as repeated params automatically. On
        network failure, degrades to element-wise passthrough (same posture
        as translate())."""
        if not texts:
            return []
        import requests

        try:
            resp = requests.post(
                "https://translation.googleapis.com/language/translate/v2",
                params={"key": self.api_key},
                data={"q": list(texts), "target": target_language, "format": "text"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            return [
                TranslationResult(
                    original_text=t, translated_text=t, source_language=None, target_language=target_language
                )
                for t in texts
            ]

        translations = resp.json()["data"]["translations"]
        return [
            TranslationResult(
                original_text=orig,
                translated_text=tr["translatedText"],
                source_language=tr.get("detectedSourceLanguage"),
                target_language=target_language,
            )
            for orig, tr in zip(texts, translations)
        ]

    def detect_language(self, text: str) -> Optional[str]:
        import requests

        try:
            resp = requests.post(
                "https://translation.googleapis.com/language/translate/v2/detect",
                params={"key": self.api_key},
                data={"q": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            return None
        body = resp.json()
        detections = body.get("data", {}).get("detections", [[]])[0]
        return detections[0]["language"] if detections else None
