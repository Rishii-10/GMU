"""
LanguageGateway -- the single input/output wrapper that makes the whole
triage pipeline monolingual internally.

Contract:
  - On input: detect the caregiver's language, remember it, and translate
    the message to English before anything else runs. English (or
    undetected) input passes straight through.
  - On output: translate every user-facing string back from English into
    the remembered language, right before it is shown. Applies to the
    doctor report, follow-up questions, the reasoning trail, home-care
    guidance, and the caller/ASHA phrasing -- the full clinical decision,
    per project decision, not a curated subset.

Backend: app.integrations.translation.Translator. get_translator() picks
GoogleTranslateProvider when GOOGLE_TRANSLATE_API_KEY is set, otherwise the
no-op IdentityTranslator. Every failure mode degrades to English
passthrough (no key -> IdentityTranslator; API unreachable -> the provider
itself already falls back to the original text), so the pipeline never
breaks and never blocks on translation -- same posture as the rest of
app/integrations/.

This wrapper stores nothing on the schema: the detected language is held
on the gateway instance and handed to the existing
extract_and_classify(..., language=...) parameter, which already persists
it on ExtractedCase.language.
"""
from __future__ import annotations

from typing import Optional

from app.integrations.translation import (
    GoogleTranslateProvider,
    IdentityTranslator,
    Translator,
)


def get_translator() -> Translator:
    """GoogleTranslateProvider if GOOGLE_TRANSLATE_API_KEY is configured,
    else IdentityTranslator. GoogleTranslateProvider.__init__ raises
    ValueError when the key is missing -- caught here so a missing key is a
    silent degrade to no-op, not a crash."""
    try:
        return GoogleTranslateProvider()
    except ValueError:
        return IdentityTranslator()


def _is_english(lang: Optional[str]) -> bool:
    """True for English or an unknown/undetected language -- both mean
    'nothing to translate'."""
    return not lang or lang.split("-")[0].lower() == "en"


class LanguageGateway:
    """One instance per triage interaction. `to_english()` is called once on
    the inbound message (and sets `.language`); `from_english()` /
    `from_english_batch()` are called on every outbound string."""

    def __init__(self, translator: Optional[Translator] = None) -> None:
        self.translator = translator or get_translator()
        self.language: Optional[str] = None  # detected caregiver language, ISO 639-1

    def to_english(self, text: str) -> str:
        """Detect and remember the source language, then return the English
        text. No-op (returns `text` unchanged) when the detected language is
        English or undetected."""
        self.language = self.translator.detect_language(text)
        if _is_english(self.language):
            return text
        return self.translator.translate(text, target_language="en").translated_text

    def from_english(self, text: Optional[str]) -> Optional[str]:
        """Translate one English string into the remembered language.
        Passes `None`/empty and English-target cases straight through."""
        if not text or _is_english(self.language):
            return text
        return self.translator.translate(text, target_language=self.language).translated_text

    def from_english_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of English strings in one round trip (order
        preserved). Passes through unchanged when the target is English."""
        if not texts or _is_english(self.language):
            return list(texts)
        results = self.translator.translate_batch(texts, target_language=self.language)
        return [r.translated_text for r in results]
