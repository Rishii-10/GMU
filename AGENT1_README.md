# Agent 1 — Multilingual Symptom Parsing & Extraction

## Status: Follow-up loop, FAISS disambiguation, and fallback wiring all implemented and tested

Agent 1 handles unstructured, multilingual patient-reported symptom text and
converts it into the structured schema consumed by the WHO IMCI Rules Engine.
It is one of only two components in the system that use an LLM (the other is
Agent 2's outbreak reasoning) — everything else in the pipeline is
deterministic Python, by design, for auditability.

This document reflects the actual state of the code as of the most recent
session, verified directly against the repo rather than written ahead of it.
A previous version of this README described work that had not actually been
built yet — that mismatch was caught and corrected. If anything below stops
matching the code, fix this file rather than let it drift again.

## Architecture

- **Primary backend:** Llama 3.2 3B via Ollama (local inference)
- **Cloud fallback:** Groq API (key obtained, not yet tested live against
  the full pipeline)
- **Tier-4 path:** `RegexBackend`, used for low-connectivity SMS/USSD input
  where LLM inference isn't viable. Structurally excluded from the
  follow-up loop and disambiguation fallback (see below) — it has no
  conversational capability and needn't be looped into either.

## What's implemented

### 1. Core extraction (`app/schemas.py`, `app/agent1_extraction.py`)

- Three backends (`OllamaBackend`, `GroqBackend`, `RegexBackend`) behind a
  common `LLMBackend` interface.
- `extract_case()` / `extract_and_classify()`: the original single-shot
  pipeline — raw text in, one `ExtractedCase` out, straight into
  `rules_engine.classify()`.
- **Hard safety invariant, enforced throughout:** every clinical boolean
  field is `Optional[bool]` with default `None`.
  - `None` → not assessed
  - `False` → assessed, sign absent
  - `True` → assessed, sign present
  These are never conflated anywhere in the pipeline. This is the single
  most load-bearing design rule in the codebase.

### 2. Multi-turn follow-up loop (`extract_case_with_followup`)

When a single extraction pass leaves `DangerSigns` fields unassessed, this
wraps `extract_case()` to ask up to 2 targeted follow-up questions before
falling through to `INCOMPLETE_ASSESSMENT`.

- **`answer_provider: Callable[[str], str]`** is the seam between this
  loop and a real conversation: today it's a scripted/test callback;
  later it becomes "send the question, block on the session store for the
  next inbound message" (Milestone 1/6 infrastructure, not yet built).
- Two separate structures are tracked per turn: `backend_context` (grows
  every turn, always opens with the original patient message, is what the
  backend sees) and `followup_trail` (caller-facing audit trail —
  question/answer pairs only, never contains the original message).
  Keeping these separate — rather than one overloaded list — was
  deliberate: an earlier version of this loop had a real bug where the
  original message silently dropped from context after the first
  follow-up turn, causing the system to re-ask about symptoms already
  reported. This split is the fix.
- Questions are generated from a small fixed template
  (`DANGER_SIGN_FOLLOWUP_QUESTIONS`) keyed off
  `DangerSigns.missing_fields()`, not by asking the LLM to generate them —
  consistent with the project's "LLM only at exactly two pipeline points"
  design rule.
- The loop stops early the moment a follow-up answer resolves a confirmed
  `True` danger sign, rather than spending remaining turns — this mirrors
  `rules_engine.classify()`'s own short-circuit (one confirmed danger sign
  is already actionable).
- On cap-out with fields still missing: the case is returned as-is. Unresolved
  fields are **never** defaulted to `False` — that would silently convert
  "we tried and still don't know" into "assessed as absent," exactly the
  conflation the schema exists to prevent. `classify()` then correctly
  reports `INCOMPLETE_ASSESSMENT`.
- `RegexBackend` is excluded structurally, not via a branch someone has to
  remember: `LLMBackend.supports_followup` defaults to `False`; only
  `OllamaBackend`/`GroqBackend` override it to `True`.

### 3. Symptom disambiguation (`app/disambiguation.py`)

Maps the LLM's free-text `symptom` string (e.g. "tummy pain", "पेट दर्द")
to a small controlled vocabulary via FAISS + embedding similarity.

- **Interface:** `Disambiguator` (ABC), `StubDisambiguator` (honest
  no-op placeholder), `FAISSDisambiguator` (real implementation) — kept
  swappable for a future non-FAISS approach.
- **Vocabulary:** exactly `{cough_or_difficult_breathing, diarrhea}` —
  derived directly from what `rules_engine.py` actually branches on
  (`case.cough.present`, `case.diarrhea.present`), not from an external
  dataset. **Fever is deliberately excluded**: it appears in nearly every
  test message, but the rules engine has no fever axis yet (Step 2
  territory), so mapping to it would produce a confident-looking match the
  engine can't act on.
- **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (~470MB) —
  free, fully local, multilingual (handles Hindi/Hinglish), chosen over
  larger (1–2GB) alternatives to stay within a lightweight footprint.
- **Confidence threshold:** `DEFAULT_CONFIDENCE_THRESHOLD = 0.65`, named
  and overridable at construction. Explicitly commented as
  **uncalibrated** — this is a placeholder, not a validated value.
  Calibration needs a labelled symptom→category sample and a threshold
  sweep, which doesn't exist yet.
- **Deterministic:** same input + same index always produces the same
  output.
- Below threshold or no match: returns the original text with no forced
  guess. A wrong confident match is worse than an honest non-match.

### 4. Disambiguation fallback into structured fields
   (`apply_disambiguation_fallback`)

Disambiguation on its own only set `symptom` and `disambiguation_confidence`
on `ExtractedCase` — it had no effect on classification, since
`rules_engine.py` never reads `symptom` at all. This function closes that
gap by feeding a confident disambiguation match into the structured
`cough`/`diarrhea` fields the rules engine actually branches on, under a
strict fallback-only rule:

- **Never overrides a field a backend already explicitly set to `True` or
  `False`.** The LLM backend reading the actual patient message always
  gets first say; disambiguation on the coarser `symptom` string is a
  fallback for what structured extraction missed, not a correction to
  what it found.
- Behavior by case shape (field-level, not block-level):
  - `case.cough is None` (block never populated) + confident match →
    creates `CoughDifficultBreathing(present=True)`, every other field
    (`duration_days`, `breaths_per_minute`, etc.) left `None` — nothing
    actually assessed those.
  - `case.cough` exists but `case.cough.present is None` (backend
    partially populated the block, e.g. captured `duration_days` but never
    resolved presence) + confident match → updates `present` to `True`,
    **preserving every other field already on the block.**
  - `case.cough.present` already `True` or `False` → untouched,
    unconditionally, regardless of match confidence.
  - Below threshold / no match / unrecognized category → no-op.
  - Same mirrored logic for diarrhea. `danger_signs` is never touched by
    this function under any circumstance.
- Takes an already-computed `DisambiguationResult`, not a live
  `disambiguator` — avoids a redundant embedding-model call and avoids
  disambiguating a `symptom` string that may have already been
  overwritten by the preceding step.
- Wired into `extract_case_with_followup`'s single shared exit point,
  immediately after `symptom`/`disambiguation_confidence` are set.
- `rules_engine.py` itself required **no changes** — it already correctly
  branches on `cough.present`/`diarrhea.present`; this closes the gap in
  what populates that field, not how it's read.

## Known limitations (documented, not hidden)

Two real, reproducible model limitations are captured as `xfail` tests
rather than worked around or silently tuned away — both are candidate data
points for the paper's field-extraction / disambiguation accuracy results:

1. **LLM negation gap:** Llama 3.2 3B sometimes leaves a symptom as `None`
   rather than correctly inferring `False` when a patient explicitly
   denies it inside a multi-fact sentence (denying one symptom while
   confirming others in the same message).
2. **Embedding-model Hinglish gap:** the disambiguation embedding model
   correctly handles native-script Hindi and Hinglish-cough phrasing, but
   a specific class of romanized-Hindi (Hinglish) diarrhea phrasing gets
   misrouted to the cough category.

## Since this doc was written: dataset-driven extension (all-ages workflow)

The following were built on top of everything above, in a later session,
extending Agent 1 and the Rules Engine to cover adult/elderly ages via the
user-supplied disease/symptom CSVs (see `triage-poc/README.md` and
`triage-poc/ARCHITECTURE.md` for the full picture — this section only
covers what changed in files this doc already describes):

- **A second, independent FAISS index** (`app/disambiguation.py`,
  `FAISSDisambiguator.match_dataset_symptom()`) built from
  `app.disease_kb.DiseaseKB`'s 131-token vocabulary + curated Hindi/Hinglish
  aliases, populating a new `ExtractedCase.symptom_tokens` field for
  `app.disease_classifier`'s dataset-driven disease ranking. Kept as a
  *separate* index from the original two-category vocabulary above (not
  merged into it) because some surface forms — "cough" is both a pediatric
  category surface form and a literal dataset token — would otherwise
  collide and make `disambiguate()`'s top-1 result depend on FAISS
  tie-breaking instead of staying pinned to the pediatric category the ~20
  existing tests in this doc's scope depend on.
- **The follow-up loop's stopping condition** is now generalized behind
  `app/followup_policy.py` (route-aware minimum-viable-info gate) instead
  of hardcoding `DangerSigns.missing_fields()` inline — behavior for the
  pediatric route is byte-for-byte unchanged (verified: every test in this
  file's original scope still passes untouched), and an adult/out-of-band
  route now asks one generic symptom-clarifier question
  (`ADULT_SYMPTOM_FOLLOWUP_QUESTION`) when `case.symptom` is still `None`.
- **`extract_and_classify_with_followup`** gained an optional `case_store`
  parameter (`app/case_store.py`) for opt-in previous-calls/area logging —
  default `None`, zero I/O side effects unless a caller passes one.

## Open work

- [x] Calibrate the FAISS confidence threshold — the DATASET-vocabulary
      threshold (`DEFAULT_DATASET_CONFIDENCE_THRESHOLD`) is now genuinely
      calibrated: `app.disambiguation.calibrate_dataset_threshold()` is a
      real, reusable sweep function, run against the labelled 80-item
      `tests/fixtures/symptom_calibration.py` sample. Its output (0.45,
      97.5% accuracy) IS the shipped default —
      `tests/test_calibration.py::test_shipped_default_matches_calibration`
      asserts they can't silently drift apart. The original PEDIATRIC
      2-category threshold (`DEFAULT_CONFIDENCE_THRESHOLD`, 0.65) is still
      uncalibrated — no labelled sample exists for that vocabulary yet.
- [ ] Test Groq API live against the full pipeline (key obtained, untested;
      shares the same `extract_case_with_followup` path as Ollama, so this
      should require no additional wiring once a key is configured)
- [ ] Real session-store/webhook layer for `answer_provider` (Milestone 1/6)
      so the follow-up loop can run against actual SMS/IVR replies instead
      of a scripted callback. `app.integrations.messaging.TwilioProvider`
      now provides a real, working `send_message()` half of this — the
      "block until the next inbound reply" half is still not built (see
      that module's docstring for the honest boundary).
- [ ] Make PII anonymization an explicit, named pipeline step
- [ ] Young infant (0–2 month) and malnutrition/anaemia IMCI charts
      (Rules Engine gap, not Agent 1 — Step 2 territory; adult/elderly ages
      are now covered via the dataset classifier, but young infant is
      deliberately NOT rerouted there — see `app/rules_engine.py::classify`)
- [x] Routing Agent — built (`app/routing/`), see `triage-poc/README.md`.
- [ ] Degradation Controller, Agent 2, facility ingestion pipeline,
      benchmark harness — not started
