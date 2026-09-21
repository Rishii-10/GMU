# Architecture Overview

## Purpose

This repository is a proof-of-concept triage pipeline for rural health messaging. It takes raw patient or caregiver text, extracts structured clinical information, applies an auditable rules engine, and produces a triage classification.

The design is intentionally modular so the extraction layer, the classification logic, and the shared schema can evolve independently.

---

## High-Level Flow

1. A caller provides raw text such as an SMS or IVR message.
2. Agent 1 extracts structured fields into a validated case object.
3. The rules engine evaluates the case against IMCI-style logic.
4. The system returns a classification result with reasoning for auditability.

```text
Raw message
  -> Agent 1 extraction
  -> Validated ExtractedCase
  -> Rules Engine classification
  -> ClassificationResult
```

---

## Repository Structure

```text
./                                  # project root
├── README.md                       # Project overview and test instructions
├── ARCHITECTURE.md                 # This file
├── AGENT1_README.md                # Agent 1 deep-dive
└── triage-poc/
    ├── requirements.txt            # Python dependencies
    ├── streamlit_app.py            # Frontend (caller/ASHA/doctor tabs)
    ├── app/                        # Core application logic
    │   ├── __init__.py
    │   ├── schemas.py              # Shared Pydantic data models (Agent 1 <-> Rules Engine)
    │   ├── agent1_extraction.py    # Extraction pipeline, backends, follow-up loop
    │   ├── disambiguation.py       # FAISS symptom matching (two independent indexes)
    │   ├── disease_kb.py           # Disease/symptom/precaution knowledge base (CSV-backed)
    │   ├── disease_classifier.py   # Naive-Bayes-style disease ranking + severity table
    │   ├── followup_policy.py      # Minimum-viable-info follow-up gate
    │   ├── case_store.py           # SQLite previous-calls/area log (opt-in)
    │   ├── rules_engine.py         # Deterministic pediatric IMNCI + adult-route dispatch
    │   ├── routing/                # Routing Agent (Agent 3)
    │   │   ├── schemas.py          # Facility/RouteInfo/DispatchResult models
    │   │   ├── facility_db.py      # SQLite facility master data + village coords
    │   │   ├── demo_facilities.py  # Synthetic facility network for the frontend
    │   │   ├── router.py           # 6-step deterministic routing core
    │   │   └── report.py           # LLM doctor-report layer (+ non-LLM fallback)
    │   └── integrations/           # External adapters, offline-mock default
    │       ├── messaging.py        # Twilio (plain REST, no SDK)
    │       ├── translation.py      # Google Translate (plain REST, no SDK)
    │       └── geo.py              # OpenRouteService + Google Maps links
    ├── data/                       # disease_symptoms.csv, disease_precautions.csv
    ├── tests/                      # Automated tests (one module per concern)
    │   └── fixtures/               # Fabricated facility/village/calibration/script fixtures
    └── .venv/                      # Local Python environment
```

---

## Core Components

### 1. Application Layer: app/

#### app/schemas.py

This file defines the shared schema used across the pipeline.

Key responsibilities:
- Defines the data model for extracted patient information.
- Represents clinical findings as explicit optional booleans to distinguish:
  - `True` = present
  - `False` = assessed and absent
  - `None` = not assessed
- Provides the output model for classification results.

#### app/agent1_extraction.py

This module is the extraction pipeline for Agent 1.

Key responsibilities:
- Accepts raw text and converts it into a structured case.
- Supports multiple backends:
  - Ollama backend for local LLM extraction
  - Groq backend for cloud-based extraction
  - Regex keyword fallback for simple deterministic extraction
- Validates extracted data against the shared schema.
- Hands the validated case to the rules engine.

#### app/rules_engine.py

This module contains the deterministic classification logic.

Key responsibilities:
- Implements IMCI-style logic for the pediatric age band (2-60 months).
- Checks danger signs and completeness safeguards.
- Classifies cough/difficult breathing and diarrhea/dehydration patterns.
- Resolves conflicting classifications with explicit severity ordering.
- Routes adult/elderly/out-of-band-age cases to `app/disease_classifier.py`'s
  dataset-driven classifier instead (see `classify_via_dataset`), and
  attaches its candidates as supplementary context on in-band pediatric
  results without overriding the IMNCI label.

---

## Data Flow

### Extraction Path

- Input: raw free-text message
- Processing: backend-specific extraction logic
- Output: `ExtractedCase`

### Classification Path

- Input: `ExtractedCase`
- Processing: completeness checks, clinical rule evaluation, severity resolution
- Output: `ClassificationResult`

This separation ensures that raw text and model output are not directly mixed with decision logic.

---

## Testing Structure

One test module per concern, deterministic/fast tests separated from
live-model tests that auto-skip when Ollama isn't reachable:

- `tests/test_rules_engine.py` / `tests/test_rules_engine_dataset_routing.py`
  -- pediatric IMNCI rules and the age-based pediatric/adult routing decision.
- `tests/test_disease_classifier.py` / `tests/test_disease_coverage.py`
  -- the probabilistic classifier and an exhaustive all-41-disease sweep.
- `tests/test_dataset_symptom_matching.py` / `tests/test_disambiguation.py`
  -- the two independent FAISS indexes.
- `tests/test_followup_policy.py` / `tests/test_agent1_followup.py` /
  `tests/test_followup_scripts.py` -- the minimum-viable-info gate,
  mechanism tests, and fabricated conversation-script sweeps.
- `tests/test_case_store.py`, `tests/test_routing.py` /
  `tests/test_routing_fixtures.py`, `tests/test_integrations.py`,
  `tests/test_calibration.py` -- case store, Routing Agent, external
  adapters (all HTTP mocked), and the FAISS threshold calibration harness.
- `tests/test_integration_agent1_pipeline.py` -- end-to-end extraction
  pipeline against a real local Ollama model.
- `tests/fixtures/` -- fabricated facility/village, symptom-calibration,
  and follow-up-script data, balanced across classes/outcomes rather than
  skewed to easy cases.

---

## Current Implementation Scope

The main workflow (Rules Engine + Agent 1 + Routing Agent) is built and
tested end to end, covering both the original pediatric IMNCI band and
(via a dataset-driven classifier) adult/elderly/out-of-band ages. It
includes:
- shared schemas, deterministic rules evaluation, pluggable extraction
  backends
- FAISS-based symptom disambiguation (two independent indexes)
- a probabilistic disease classifier for the broader age range
- a minimum-viable-info follow-up gate
- a deterministic Routing Agent with an LLM report layer
- pluggable external adapters (Twilio, Google Translate, OpenRouteService)
  with offline-mock defaults
- broad, balanced test coverage (fabricated fixtures where real data
  doesn't exist)

Not yet included:
- young infant (<2mo) and malnutrition/anaemia IMCI charts
- a degradation controller for automatic backend selection
- Agent 2 (DBSCAN outbreak surveillance) -- separate side-track, out of
  scope for the main workflow
- a dashboard, or the real SMS/IVR session-store layer behind
  `answer_provider`/`MessagingProvider`

---

## Design Notes

- The architecture favors clarity and auditability over hidden heuristics.
- Clinical safety is represented explicitly through optional booleans and completeness checks.
- The rules engine is deterministic, which makes it easier to test and reason about.
- The extraction layer is intentionally backend-agnostic so newer models or fallback methods can be added without changing the rest of the pipeline.
