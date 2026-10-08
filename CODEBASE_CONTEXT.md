<!-- AGENT-ONLY CONTEXT FILE. Optimized for machine reading, not humans. Dense by design. -->
# CODEBASE_CONTEXT — triage-poc (repo: Rishii-10/GMU)

## §0 SYNC CONTRACT — READ FIRST — MANDATORY FOR EVERY AGENT / HUMAN / MACHINE
- PURPOSE: single source of truth for: structure, architecture, pipeline, engines, I/O, rubrics, UI, keys, current state. Read this INSTEAD of re-analyzing the repo, then open only the files your task needs (§17 TASK INDEX).
- AUTHORITY ORDER: code > this file > every other *.md in repo (others are partly outdated, see §15). If this file disagrees with code, code wins AND this file is stale → fix this file in the same change.
- UPDATE RULE (NON-OPTIONAL): any change to tracked files under `triage-poc/` (code, data, tests, config, requirements) or repo-root config/docs MUST update every affected section here IN THE SAME COMMIT, then update §0.SYNC_STAMP and append one line to §18 SYNC_LOG. A change without the matching update here is incomplete.
- GROUNDING RULE: write only facts verified by reading code or running commands. Unverified → write `UNVERIFIED`. No plans/wishes except inside explicit `NOT_BUILT` lists. Derived (not directly executed) facts are tagged `(derived)`.
- REFERENCE STYLE: `path::symbol` (grep-able). Line numbers deliberately avoided (they drift). Constants are listed with values — re-grep before editing.
- STALENESS CHECK (run when starting work):
  - `git log -1 --format=%h -- CODEBASE_CONTEXT.md` vs `git log -1 --format=%h -- triage-poc .gitignore .claude`
  - if code commit is newer → `git diff <context-commit>..HEAD --stat -- triage-poc` → update affected sections before relying on them.
  - also check `git status --short` — uncommitted code edits not reflected here mean this file is stale for the current tree.
  - note: the first command prints nothing until this file is committed.
- ENFORCEMENT HOOK: repo-root `CLAUDE.md` (auto-loaded by Claude Code on any machine that clones the repo) points here and restates the update rule.
- SYNC_STAMP:
  - date: 2026-10-08
  - branch: `fix`; base commit: `a192ded` (== `main` == `origin/main`)
  - tree state described: base + UNCOMMITTED working-tree changes listed in §16.2 (if those are now committed, update this line + §16)
  - test status at sync: 386 collected → 383 passed, 1 failed (stale test, §13), 2 xfailed (Ollama llama3.2:3b + faiss + sentence-transformers installed)
  - verification: 4 parallel read-only fact-extraction agents wrote the facts; 1 independent verifier agent checked every section against code (21 findings, all applied). Not re-verifiable read-only: full test pass count (from the session's own run) and live-UI observations (§14.1).

## §1 SNAPSHOT
- WHAT: "Beyond Triage / Rural Health Triage POC". Input = free-text patient/caregiver symptom message (English, Hindi, Hinglish). Output = triage label + per-role views; for EMERGENCY/SEVERE a nearest eligible facility + LLM doctor handoff report; when adult diagnosis is uncertain, adaptive yes/no follow-up questions to the caller.
- LANGUAGE/RUNTIME: Python 3.11.9 in use. `streamlit_app.py` uses `X | None` annotations without `from __future__ import annotations` → requires Python ≥ 3.10.
- PROCESS MODEL: one Python process (Streamlit). No web API, no background workers, no persistent DB in the app path (facility DB = SQLite `:memory:` rebuilt on demand). Session state = Streamlit `st.session_state` (per browser session).
- LLM: local Ollama `llama3.2:3b` at `http://localhost:11434` — the only LLM backend the app uses. 3 LLM call sites (§5.3).
- LABELS: `EMERGENCY` > `SEVERE` > `MODERATE` > `MILD`; plus `INCOMPLETE_ASSESSMENT` (needs more info).
- HEALTH: working tree runs end-to-end (UI verified manually 2026-10-08). Committed HEAD `a192ded` is BROKEN: `app/disease_classifier.py::_fit_calibration_and_thresholds` references undefined `held_out_rows` → `NameError` on every `DiseaseClassifier()` construction; fixed only in the uncommitted tree (§16.2).

## §2 GLOSSARY
- case = `ExtractedCase`; result = `ClassificationResult` (both `app/schemas.py`).
- token = dataset symptom token (snake_case), vocabulary = 131 tokens from `data/disease_symptoms.csv`.
- CP = split conformal prediction gate; set = `ClassificationResult.prediction_set`.
- Stage 1 = dataset disease classifier (NB + isotonic + CP). Stage 2 = AHP emergency scorer.
- IMNCI/IMCI = WHO pediatric rules (children 2–59 months).
- Flow A = pre-classification follow-up loop; Flow B = INCOMPLETE template question; Flow C = adaptive CP follow-up to caller (§8).
- gateway = `app/integrations/language_gateway.py::LanguageGateway`.
- ASHA = community health worker (one UI tab). Village = routing location chosen in sidebar.

## §3 REPO MAP (tracked files: 72 at base commit; + 1 untracked test + this file + CLAUDE.md)
```
/ (repo root)
├─ CODEBASE_CONTEXT.md            THIS FILE (agent context, source of truth)
├─ CLAUDE.md                      auto-loaded pointer + update rule (enforcement)
├─ EVALUATION_METRICS.md          proposed paper eval metrics table (definitions only, pending verification; nothing implemented)
├─ .gitignore                     __pycache__/ *.pyc .venv/ venv/ .pytest_cache/ *.sqlite *.index .DS_Store .env .claude/settings.local.json triage-poc/venv_audit/
├─ .env                           UNTRACKED+gitignored, machine-local (see §4.3)
├─ .claude/launch.json            config "triage-poc": python -m streamlit run triage-poc/streamlit_app.py --server.headless true --server.port 8501
├─ README.md                      project overview (partly outdated, §15)
├─ ARCHITECTURE.md                older architecture (outdated, §15)
├─ AGENT1_README.md               Agent 1 deep-dive (partly outdated)
├─ CHANGES_AGE_HANDLING.md        historical changelog of young-infant branch
├─ RULE_ENGINE_DESIGN.md          superseded proposal (nothing in it was built)
├─ Rule engine final.md           paper-style spec; Stage 2 matches code, Stage 1 does not
└─ triage-poc/
   ├─ streamlit_app.py            ONLY UI (frontend + orchestration), 4 role tabs + caller follow-up chat
   ├─ demo.py                     CLI: 3 English samples → extract_and_classify (Ollama); no FAISS/translation/routing/follow-up
   ├─ requirements.txt            deps (scikit-learn commented out but REQUIRED, §4.1)
   ├─ .env.example                only GROQ_API_KEY=your_groq_api_key_here
   ├─ run_streamlit.sh            cd to script dir; exec $DIR/.venv/bin/streamlit ... port ${PORT:-8501} (.venv absent in checkout)
   ├─ .claude/launch.json         config "triage-streamlit": sh run_streamlit.sh, port 8501
   ├─ .streamlit/config.toml      dark theme (base dark, primary #5AA9FF, bg #0E1117, secondary #161B22, text #E6EDF3)
   ├─ .gitignore                  .venv/ __pycache__/ *.pyc .env .DS_Store .pytest_cache/
   ├─ rule_engine_design_final.md "implemented design" doc (several confirmed mismatches, §15)
   ├─ app/
   │  ├─ schemas.py               shared pydantic v2 contract (ExtractedCase, ClassificationResult, enums)
   │  ├─ agent1_extraction.py     Agent 1: LLM backends, extraction, disambiguation wiring, follow-up flows A/B/C
   │  ├─ disambiguation.py        FAISS + multilingual MiniLM symptom matching (2 indexes) + threshold calibration
   │  ├─ followup_policy.py       Flow A gate: which fields are still required
   │  ├─ followup_question_selector.py  Flow C: pick most discriminating symptom token (emergency-first / info-gain)
   │  ├─ rules_engine.py          classify(): age router + IMNCI rules + classify_via_dataset (CP + AHP)
   │  ├─ disease_classifier.py    Stage 1: Naive Bayes, k-fold isotonic calibration, CP gate, severity lookup
   │  ├─ emergency_scorer.py      Stage 2: AHP 6-attribute emergency score, ESI band, danger-sign override
   │  ├─ disease_kb.py            loads 3 CSVs; vocabulary/symptoms/precautions/severity lookups
   │  ├─ case_store.py            opt-in SQLite case log (hash of text only); NOT used by UI
   │  ├─ routing/                 "Agent 3" (per routing/__init__ docstring)
   │  │  ├─ schemas.py            Facility, FacilityType, ScoredFacility, RouteInfo, DispatchResult
   │  │  ├─ facility_db.py        SQLite facilities + villages, haversine radius query
   │  │  ├─ demo_facilities.py    10 synthetic facilities, 5 villages (Denkanikottai/Hosur area, KA/TN)
   │  │  ├─ router.py             6-step route(); MockRoutingProvider (default)
   │  │  └─ report.py             LLM doctor handoff report + deterministic fallback
   │  └─ integrations/
   │     ├─ language_gateway.py   detect + translate in/out (USED by UI)
   │     ├─ translation.py        Translator ABC, IdentityTranslator, GoogleTranslateProvider
   │     ├─ messaging.py          MockProvider, TwilioProvider (NOT used by app)
   │     └─ geo.py                ORSProvider (OpenRouteService) (NOT used by app)
   ├─ data/
   │  ├─ disease_symptoms.csv     4920 rows, 41 diseases × 120 rows, 131 tokens
   │  ├─ disease_precautions.csv  41 rows, ≤4 precautions each
   │  └─ disease_severity.csv     41 rows, disease → EMERGENCY/SEVERE/MODERATE/MILD (no clinical provenance documented)
   └─ tests/                      18 test_*.py (17 tracked + test_adaptive_followup_loop.py untracked), fixtures/, 3 eval scripts, 5 eval JSONs
```
- NOT present: pyproject.toml, setup.cfg, pytest.ini, tox.ini, conftest.py, Makefile, Dockerfile, CI workflows (.github/), triage-poc/.venv, data/*.sqlite.

## §4 RUNTIME, DEPENDENCIES, ENV VARS / API KEYS, HOW TO RUN
### §4.1 Dependencies
- requirements.txt active: pydantic>=2.6, requests>=2.31, pytest>=8.0, faiss-cpu>=1.8,<2.0, sentence-transformers>=2.6,<3.0, streamlit>=1.32, python-dotenv>=1.0.
- requirements.txt commented: `# scikit-learn>=1.4` ("Milestone 7") — BUT `disease_classifier.py::_fit_calibration_and_thresholds` imports `sklearn.isotonic.IsotonicRegression` at classifier construction → scikit-learn is REQUIRED in practice (defect D-REQ, §14).
- Installed (this machine, system Python 3.11.9 user site): scikit-learn 1.6.1, faiss 1.15.1, sentence-transformers 2.7.0, numpy 2.1.3, pydantic 2.11.9, requests 2.32.3, streamlit 1.40.1, pytest 8.4.2, python-dotenv.
- Model downloads: `FAISSDisambiguator()` loads SentenceTransformer `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace cache, ~470MB per docstring). First Streamlit Assess took ~1 min (model load, observed); later calls fast (`@st.cache_resource`).
- Ollama: `ollama pull llama3.2:3b` required; server at localhost:11434. Verified working 2026-10-08.

### §4.2 Env vars / API keys (complete; repo-wide grep)
| env var | read in | consumer | required | when missing | used by running app |
|---|---|---|---|---|---|
| GOOGLE_TRANSLATE_API_KEY | integrations/translation.py::GoogleTranslateProvider.__init__ | language_gateway.get_translator() | optional | ValueError caught → IdentityTranslator (no translation; gw.language=None) | YES |
| GROQ_API_KEY | agent1_extraction.py::GroqBackend.__init__ | GroqBackend.extract/generate_text | optional | BackendUnavailable("GROQ_API_KEY not set") at call time | NO (GroqBackend never instantiated) |
| ORS_API_KEY | integrations/geo.py::ORSProvider.__init__ | ORSProvider | optional | ValueError at construction | NO |
| TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER | integrations/messaging.py::TwilioProvider.__init__ | TwilioProvider | optional | ValueError listing missing | NO |
| PORT | run_streamlit.sh | streamlit port | optional | 8501 | only via triage-poc/.claude/launch.json |
- Distinct external credential services: 4 (Google Translate, Groq, OpenRouteService, Twilio). Only Google Translate is wired into the app.
- Hardcoded (not env): Ollama model `llama3.2:3b`, host `http://localhost:11434`, timeout 30s; Groq model `llama-3.1-8b-instant`, timeout 15s; FAISS model name.
- `.env` loading: only `streamlit_app.py` (guarded `from dotenv import find_dotenv, load_dotenv; load_dotenv(find_dotenv(usecwd=True) or None)`; ImportError ignored). `demo.py` and tests do NOT load .env.

### §4.3 Machine-local state observed at sync (NOT in git; other machines differ)
- Repo-root `.env` exists (gitignored) and defines `GOOGLE_TRANSLATE_API_KEY` with a non-empty value (value never read/printed). With cwd = repo root, `find_dotenv(usecwd=True)` finds it and `get_translator()` returns `GoogleTranslateProvider` (verified locally, no network call). Key validity: UNVERIFIED. Owner believed the key was not set — confirm before relying on translation.

### §4.4 How to run (from repo root unless noted)
- App: `python -m streamlit run triage-poc/streamlit_app.py --server.headless true --server.port 8501` (= root .claude/launch.json). Needs Ollama running.
- Tests (from triage-poc/): `python -m pytest tests -q -p no:cacheprovider` (~2.5–3 min with Ollama + FAISS). Collection probes Ollama 4× (`GET http://localhost:11434/api/tags`, 2 s timeout each, needs a model named llama3.2*) via `test_integration_agent1_pipeline.py::_ollama_available`.
- Evals (from triage-poc/): `PYTHONPATH=. PYTHONIOENCODING=utf-8 python tests/evaluate_1000.py` (also evaluate_balanced.py, evaluate_100_cases.py). WARNING: they OVERWRITE tracked tests/eval_*.json. On Windows cp1252 console they crash without PYTHONIOENCODING=utf-8.
- Demo CLI (from triage-poc/): `python demo.py`.
- Docstrings/run_streamlit.sh reference `./.venv/bin/...` — no .venv exists in this checkout.

## §5 ARCHITECTURE
### §5.1 Component inventory (engines / processing units = 13; numbering used throughout)
| id | name | file::entry | kind | in app path? |
|---|---|---|---|---|
| E0 | Language gateway | integrations/language_gateway.py::LanguageGateway | external API (Google) or no-op | YES |
| E1 | Agent 1 extraction | agent1_extraction.py::extract_case | LLM | YES |
| E2 | Symptom disambiguation | agent1_extraction.py::_apply_disambiguation + disambiguation.py::FAISSDisambiguator | embeddings (FAISS) | YES |
| E3 | Flow A pre-classification follow-up | agent1_extraction.py::extract_case_with_followup + followup_policy.py | deterministic templates + LLM re-extract | NO (tests only) |
| E4 | Rules router | rules_engine.py::classify | deterministic | YES |
| E5 | IMNCI pediatric rules | rules_engine.py::classify_danger_signs/…_cough…/…_diarrhea… | deterministic | YES (2–59 mo) |
| E6 | Stage 1 dataset classifier | rules_engine.py::classify_via_dataset → disease_classifier.py::classify_with_cp | statistical (NB+isotonic+CP) | YES (≥60 mo/adult) |
| E7 | Stage 2 AHP emergency scorer | emergency_scorer.py::score_emergency | deterministic weighted sum | YES (CONFIDENT only) |
| E8 | Flow C adaptive follow-up | agent1_extraction.py::next_followup_question/record_followup_answer/parse_yes_no + followup_question_selector.py | info-theoretic selection + LLM phrasing | YES |
| E9 | Flow B incomplete question | agent1_extraction.py::question_for_incomplete_result | deterministic templates | YES (display only) |
| E10 | Routing (Agent 3) | routing/router.py::route | deterministic | YES (EMERGENCY/SEVERE + village) |
| E11 | Doctor handoff report | routing/report.py::generate_doctor_report | LLM + fallback | YES |
| E12 | Case store | case_store.py::CaseStore | SQLite | NO (opt-in, tests only) |

### §5.2 "Agents" (project naming)
- Agent 1 = extraction layer (E1–E3, E8 LLM phrasing lives in same module). BUILT.
- Agent 2 = DBSCAN outbreak surveillance / RAG. NOT_BUILT (only mentioned in docs/comments).
- Agent 3 = Routing agent (E10 + E11). BUILT.
- No autonomous/looping LLM agents; all orchestration is plain Python.

### §5.3 LLM call sites (all via `LLMBackend`; app always passes `OllamaBackend()`)
1. E1 `OllamaBackend.extract` — POST /api/generate, system=EXTRACTION_SYSTEM_PROMPT, format=json, temperature 0.0. On Assess (and re-extract per turn in Flow A, unused by app).
2. E8 `generate_followup_question_text` → `backend.generate_text(_ADAPTIVE_FOLLOWUP_SYSTEM_PROMPT, …)`, temperature 0.2. Per adaptive question (≤3). Any exception → template fallback.
3. E11 `generate_doctor_report` → `backend.generate_text(REPORT_SYSTEM_PROMPT, …)`, temperature 0.2. Only EMERGENCY/SEVERE + village + facility found, on cache miss. BackendUnavailable → deterministic fallback.
- Backends: `OllamaBackend` (name "ollama_llama3.2:3b", supports_followup=True), `GroqBackend` (name "groq", supports_followup=True, unused), `RegexBackend` (name "regex_keyword", supports_followup=False, generate_text raises → always fallback; unused by app). No degradation/auto-failover controller exists.

## §6 PIPELINE (as executed by streamlit_app.py)
```
[User types message in text_area + optional village in sidebar] --click Assess-->
 E0 gw = LanguageGateway(); english = gw.to_english(text)        (detect lang; translate→en if not en; no-op w/o key)
 E1 case = extract_case(english, OllamaBackend(), language=gw.language)   (LLM JSON → sanitize → infant-age floor → ExtractedCase)
 E2 case = _apply_disambiguation(case, cached FAISSDisambiguator)  (symptom text → symptom_tokens per clause; cough/diarrhea backfill)
 E4 result = rules_classify(case)
     ├─ age_months<2            → INCOMPLETE (YOUNG_INFANT_NO_VALIDATED_RULESET)            [stop]
     ├─ no age & no age_group   → INCOMPLETE (AGE_UNKNOWN_CANNOT_ROUTE, missing age_months) [stop]
     ├─ ≥60 mo / ADULT / ELDERLY→ E6 classify_via_dataset: NB→boost→CP
     │        ├─ ABSTAIN (0 or ≥4 in set) → INCOMPLETE (INSUFFICIENT_SYMPTOM_DATA | UNCERTAIN_DIAGNOSIS), prediction_set kept
     │        ├─ UNCERTAIN (2–3)          → worst-case severity label in set
     │        └─ CONFIDENT (1)            → CSV severity label + E7 AHP emergency_result
     └─ else (2–59 mo / INFANT / CHILD) → E5 IMNCI (danger signs → completeness → cough → diarrhea → most_severe_wins) + NB context
 restore raw_symptom_text to original language if translated
 st.session_state.current = {case, result, gw}; history += initial label
 E8 _start_adaptive: next_followup_question(case, result, trail=[]) → (token, English question) or None
      question → _t(gw, q) → appended to Caller-tab chat
 ┌──────────── per caller reply (one message at a time, SMS-like) ────────────┐
 │ text_input in Caller tab → _answer_adaptive(reply):                          │
 │   parse_yes_no(reply) None → bot "Sorry, please reply with yes or no." (pending question unchanged, no turn used) │
 │   else record_followup_answer (yes → token appended) → rules_classify(case)  │
 │        → current.update(case, result) → next_followup_question → loop        │
 │   stop when: set size ≤1, 3 questions asked, or selector finds nothing       │
 └──────────────────────────────────────────────────────────────────────────────┘
 every rerun: ensure_dispatch_and_report(current):
   label EMERGENCY/SEVERE AND village chosen → E10 route() → if facility → E11 report (English)
   cache key = (village, label, condition)
 render 4 tabs: Caller (translated) | ASHA | Doctor | Rule Engine (English)
```
### §6.1 Stage I/O table
| stage | input | output | failure behavior |
|---|---|---|---|
| E0 to_english | raw text | English text; gw.language (ISO 639-1 or None) | Google error → passthrough original text |
| E1 extract_case | English text, backend, language | ExtractedCase | BackendUnavailable (propagates; UI shows "start ollama serve…"); ExtractionValidationError (UI "Could not process this message") |
| E2 _apply_disambiguation | case (uses case.symptom) | case with symptom_tokens, disambiguation_confidence, maybe symptom category + cough/diarrhea present=True | symptom None → unchanged; ImportError from FAISS uncaught in UI |
| E4 classify | case | ClassificationResult | — |
| E8 next_followup_question | case, result, trail, backend | (token, English question) or None | LLM failure → template text |
| E10 route | label, village, FacilityDB, now(UTC) | DispatchResult | no facility → no_facility_found=True |
| E11 report | case, result, dispatch, backend | str | BackendUnavailable → fallback string |

## §7 ENGINE SPECS (input / output / rubric = decision rules + constants + invariants)

### E0 LanguageGateway (`integrations/language_gateway.py`)
- `get_translator()`: try `GoogleTranslateProvider()` except ValueError → `IdentityTranslator()`.
- `LanguageGateway(translator=None)`: attrs `.translator`, `.language` (None initially). No caching.
- `to_english(text)`: `language = translator.detect_language(text)` (1 API call); if English/None → return text; else translate to "en" (2nd call).
- `from_english(text)`: None/empty or English target → passthrough; else translate to `.language`. `from_english_batch(texts)`: one batch call.
- `_is_english(lang)`: None/empty or base code "en".
- GoogleTranslateProvider: POST `https://translation.googleapis.com/language/translate/v2` (params key; data q,target,format=text); detect POST `.../v2/detect`. Only `requests.RequestException` caught (→ passthrough / None); malformed JSON (KeyError) would raise.
- Docstring claims doctor report + reasoning are translated; actual UI translates ONLY the Caller tab (§11).

### E1 Agent 1 extraction (`agent1_extraction.py`)
- `extract_case(raw_text, backend, context=None, case_id=None, language=None) -> ExtractedCase`: `backend.extract` → `_sanitize_enum_fields` → `_apply_infant_age_floor(raw_text, …)` → `ExtractedCase(raw_symptom_text=raw_text, case_id, language, llm_backend=backend.name, **parsed)`; construction error → `ExtractionValidationError`; BackendUnavailable propagates. Extra keys ignored (pydantic default).
- `extract_and_classify(raw_text, backend, …) -> (case, result)`: extract_case + rules_classify. Used by demo.py + tests; NOT by UI (UI inlines extract_case + disambiguation + classify).
- EXTRACTION_SYSTEM_PROMPT rubric: extraction only, no diagnosis, no invented values; Hindi/Tamil/English/mixed; null = not stated (NOT false); strict JSON; fields symptom, duration, severity(mild|moderate|severe|unknown), age_group(infant|child|adult|elderly|null), age_months(int|null), location, notes, danger_signs{4 bools|null}, cough{present,duration_days}, diarrhea{present,duration_days,blood_in_stool,restless_or_irritable,sunken_eyes,drinks_eagerly_thirsty,drinks_poorly_or_not_able}; booleans only when stated/clearly implied; exam-only signs never filled; age normalization: years×12, months, weeks/4 floor, days/30 floor, newborn→0, 1.5y→18, "5th birthday"→60, vague → age_months null + age_group.
- `_sanitize_enum_fields`: severity not in {mild,moderate,severe,unknown} (and not None) → "unknown"; age_group not in {infant,child,adult,elderly,unknown} (and not None) → None. Exact case-sensitive. Explicit `severity: null` NOT fixed → validation error (defect D-SEV).
- Infant floor: `_INFANT_AGE_TEXT_PATTERNS` (IGNORECASE): newborn/just born/just delivered/neonate(al) → 0; "N day(s) old" → int(N/30.44); "N week(s) old" → int(N·7/30.44); floor = min of matches < 2; applied when LLM age_months is None, ≥2, non-int or negative; appends audit note to `notes`. Never raises an age. Hindi/spelled numbers not covered.
- `_extract_json_object`: fenced ```json block or first `{`…last `}` → json.loads.
- OllamaBackend: `__init__(model="llama3.2:3b", host="http://localhost:11434", timeout=30)`; context format "PRIOR TURNS:\n…\n\nLATEST MESSAGE:\n…". `resp.json()` outside try (defect D-HTTP).
- GroqBackend: `https://api.groq.com/openai/v1/chat/completions`, response_format json_object; unused.
- RegexBackend (`_KEYWORDS`): first matching symptom of fever→cough→diarrhea→breath; danger signs True only on match else None; unused by app.

### E2 Disambiguation (`agent1_extraction.py::_apply_disambiguation`, `disambiguation.py`)
- `_apply_disambiguation(case, disambiguator)`: if case.symptom None → unchanged. `r = disambiguate(case.symptom)` → `disambiguation_confidence=r.confidence`; `symptom = r.matched_category` if not None. Then split ORIGINAL symptom text by `_SYMPTOM_CLAUSE_SPLIT = r"\s*(?:,|;|&|\band\b)\s*"` (IGNORECASE; only English "and") → `match_dataset_symptom(clause)` each → append new unique tokens to `symptom_tokens`. Then `apply_disambiguation_fallback`.
- `apply_disambiguation_fallback`: matched category `cough_or_difficult_breathing`→cough, `diarrhea`→diarrhea; block None → new block present=True; present None → True; explicit True/False never changed; never touches danger_signs.
- `FAISSDisambiguator(vocabulary=None, model_name="paraphrase-multilingual-MiniLM-L12-v2", confidence_threshold=0.65, enable_dataset_matching=True, dataset_vocabulary=None, dataset_confidence_threshold=0.45)`; lazy imports faiss/numpy/sentence_transformers (ImportError message names faiss-cpu + sentence-transformers). IndexFlatIP on normalized embeddings (exact cosine).
  - Index 1 pediatric: `VOCABULARY` 27 surface forms → 2 categories (15 cough_or_difficult_breathing, 12 diarrhea; fever excluded). threshold `DEFAULT_CONFIDENCE_THRESHOLD = 0.65` (flagged UNCALIBRATED in code). score ≥ threshold accepted.
  - Index 2 dataset: `_build_dataset_vocabulary()` = 131 tokens (both "a b" and "a_b" forms) + `_DATASET_LAY_TERM_ALIASES` (45 Hindi/Hinglish/English aliases → 22 tokens) = 273 entries. threshold `DEFAULT_DATASET_CONFIDENCE_THRESHOLD = 0.45` (calibrated on tests/fixtures/symptom_calibration.py, 80 items: 97.5% acc at 0.45 vs 93.8% at 0.65).
- `StubDisambiguator`: no-op (default in Flow A). `calibrate_dataset_threshold(...)`: grid 0.30–0.95 step 0.05, best = max(accuracy, -threshold).
- Observed weakness (2026-10-08 live run): multi-symptom messages often yield 1 token because LLM `symptom` field holds one phrase; "pet dard aur ulti" → `yellowing_of_eyes` (wrong); xfail test: Hinglish diarrhea phrase → cough category.

### E3 Flow A — pre-classification follow-up (NOT used by app)
- `followup_policy.missing_required(case)`: no age_months AND age_group in (None, UNKNOWN) → ["age_months"]; dataset route (age_months ≥ 60, or None with ADULT/ELDERLY) → [] if case.symptom truthy else ["symptom_tokens"]; else pediatric → [] if any danger sign True else names of None danger signs. `_PEDIATRIC_UPPER_BOUND_MONTHS = 60` (literal copy).
- `extract_case_with_followup(raw_text, backend, answer_provider=None, case_id=None, language=None, max_followup_turns=2, disambiguator=None)`: loop only if backend.supports_followup AND answer_provider AND turns < max AND missing non-empty; question = template for first missing field; context grows "ORIGINAL MESSAGE:", "FOLLOW-UP Q:", "FOLLOW-UP A:"; re-extract each turn; never defaults None→False; exits through `_apply_disambiguation(case, disambiguator or StubDisambiguator())`. Trail entries {question, answer} (no "token" key → incompatible with Flow C functions).
- `extract_and_classify_with_followup(…, case_store=None)` → (case, trail, result); records to case_store if given.
- Templates `_FOLLOWUP_QUESTIONS` (6): 4 danger signs (DANGER_SIGN_FOLLOWUP_QUESTIONS), "symptom_tokens" → ADULT_SYMPTOM_FOLLOWUP_QUESTION, "age_months" → AGE_CLARIFIER_FOLLOWUP_QUESTION.
- Gate/classifier disagreement: for age_months < 2 gate returns pediatric danger-sign list while classify returns YOUNG_INFANT INCOMPLETE.

### E4 Rules router (`rules_engine.py::classify`) — exact branch order
1. `age_months is not None and age_months < 2` → INCOMPLETE_ASSESSMENT, condition `YOUNG_INFANT_NO_VALIDATED_RULESET` (no dataset context, no missing_fields).
2. `age_months is None and age_group in (None, UNKNOWN)` → INCOMPLETE_ASSESSMENT, `AGE_UNKNOWN_CANNOT_ROUTE`, missing_fields ["age_months"].
3. `age_months >= 60` OR (`age_months is None` and age_group in (ADULT, ELDERLY)) → `classify_via_dataset(case)` (E6).
4. else pediatric E5 (age 2–59 months, or no age with INFANT/CHILD).
- Constants: FAST_BREATHING_CUTOFF_2_TO_12_MONTHS=50, FAST_BREATHING_CUTOFF_12_MONTHS_TO_5_YEARS=40, MODULE_AGE_MIN_MONTHS=2, MODULE_AGE_MAX_MONTHS=60.
- `AGE_OUT_OF_MODULE_SCOPE` is never produced (only comments/docs/tests asserting absence).

### E5 IMNCI pediatric engine (`rules_engine.py`)
- Order: `classify_danger_signs` → `check_danger_sign_completeness` → `classify_cough_or_difficult_breathing` + `classify_diarrhea_dehydration` → `most_severe_wins`; every pediatric return passes `_attach_dataset_context`.
- Danger signs (4: not_able_to_drink_or_breastfeed, vomits_everything, convulsions, lethargic_or_unconscious): any True → EMERGENCY `GENERAL_DANGER_SIGN` (short-circuit, before completeness).
- Completeness: any of the 4 None → INCOMPLETE `DANGER_SIGNS_NOT_FULLY_ASSESSED`, missing_fields = None ones.
- Cough (only if cough.present is True): chest_indrawing True or stridor_when_calm True → SEVERE `SEVERE_PNEUMONIA_OR_SEVERE_DISEASE`; else breaths_per_minute ≥ cutoff (50 if <12 mo else 40, needs both values and 2≤age<60) → MODERATE `PNEUMONIA`; else MILD `COUGH_OR_COLD`. duration ignored.
- Diarrhea (only if present True): ≥2 of {lethargic_or_unconscious, sunken_eyes, drinks_poorly_or_not_able, skin_pinch_goes_back_very_slowly} → SEVERE `SEVERE_DEHYDRATION`; else ≥2 of {restless_or_irritable, sunken_eyes, drinks_eagerly_thirsty, skin_pinch_goes_back_slowly} → MODERATE `SOME_DEHYDRATION`; else MILD `NO_DEHYDRATION`. blood_in_stool/duration ignored. (derived) lethargic is always False when reached → severe needs 2 of the other 3.
- No axis applies → MILD `NO_DANGER_SIGNS_NO_SPECIFIC_ILLNESS_CLASSIFIED`.
- `most_severe_wins`: min by SEVERITY_RANK (ties → first); appends "also considered and overridden" line; empty → ValueError.
- `_attach_dataset_context`: if symptom_tokens non-empty and classify_diseases non-empty → set candidates + probable_disease; NEVER changes label/condition/reasoning (plain NB, no CP/boost).
- Note: exam-only fields (breaths_per_minute, chest_indrawing, stridor, skin pinch) are never extracted from text → in practice cough → MILD unless a health worker fills them (no UI to fill them).

### E6 Stage 1 dataset classifier (`rules_engine.py::classify_via_dataset`, `disease_classifier.py`)
- `get_default_classifier()`: lazy module singleton `DiseaseClassifier(DiseaseKB.load())` (no lock). Construction: load rows (4920) → `_fit_naive_bayes` on all rows → `_fit_calibration_and_thresholds()` (needs sklearn).
- Naive Bayes (`_fit_naive_bayes`, `_nb_posteriors`): log_prior = log(rows_d/total) (uniform: 120 rows each); log_lik[t] = log((count_rows_with_t + 1)/(n_rows + |V|)), `_LAPLACE_ALPHA = 1.0`; ONLY present tokens scored (no absence term; code comments call it Bernoulli — it is not); softmax.
- `classify_diseases(tokens, top_n=None)`: recognized = tokens ∩ vocab; empty → []; keep p ≥ `MIN_CANDIDATE_SCORE = 0.01`; renormalize; round 4; DiseaseCandidate(name, score, matched_symptoms, precautions).
- Calibration `_fit_calibration_and_thresholds` (5-fold, `_CALIBRATION_N_FOLDS = 5`, `_CALIBRATION_SEED = 42`): unique (disease, profile) pairs = 304; stratified round-robin folds; per fold refit NB on rows not held out; per held-out profile record raw top posterior, is_correct, gap(top−second), TRUE-class raw posterior (out-of-fold; fixed in working tree). IsotonicRegression(y 0..1, clip) on (raw_top, is_correct). τ (Youden J) = 1.0 (computed). δ (risk-coverage; `_TARGET_SELECTIVE_ERROR = 0.03`, `_MIN_COVERAGE = 0.60`) = 0.1090 (computed). CP nonconformity = 1 − iso(true-class raw posterior), n = 304. Returns 4-tuple (docstring says 3).
- `classify_with_cp(tokens, alpha=0.05) -> ConformalResult(prediction_set, set_size, decision, alpha, q_hat, candidates, posteriors)`:
  1. candidates = classify_diseases(tokens).
  2. Emergency boost: if any EMERGENCY candidate → its score × `EMERGENCY_SAFETY_FACTOR = 3.0`, renormalize, re-sort.
  3. no candidates → ABSTAIN (set [], q_hat 0.0).
  4. q_idx = min(ceil((n+1)(1−α)), n) − 1; q̂ = sorted(scores)[q_idx]. COMPUTED: α=0.05 → q̂ = 1.0 (α=0.10 → 0.0769; α=0.20 → 0.0).
  5. include candidate if 1 − iso(candidate score) ≤ q̂ → with q̂ = 1.0 EVERY candidate is included ⇒ prediction_set == all candidates with score ≥ 0.01 (verified on all 304 profiles). Empty → fallback [top].
  6. decision: size 1 CONFIDENT; 2–3 UNCERTAIN; ≥4 ABSTAIN.
  7. Red-flag override (decision ≠ CONFIDENT): raw tokens ∩ `_RED_FLAG_TOKENS` {radiating_pain, altered_sensorium, loss_of_consciousness, sudden_severe_headache, weakness_in_limbs} AND an EMERGENCY disease in set → set = [highest-posterior EMERGENCY disease], CONFIDENT. radiating_pain, loss_of_consciousness, sudden_severe_headache are NOT in the 131 vocab. altered_sensorium (95% of Paralysis rows) behaves as intended. weakness_in_limbs is a Cervical-spondylosis marker (90% of its rows, 0% of Paralysis rows) yet forces Paralysis/EMERGENCY whenever Paralysis is in the set (e.g. weakness_in_limbs + abdominal_pain + weakness_of_one_body_side → CONFIDENT Paralysis while NB top = Hepatitis E); lay alias "kamzori"/"कमज़ोरी" maps to weakness_in_limbs.
  - EFFECTIVE BEHAVIOR (derived from q̂=1.0): the CP "gate" reduces to counting NB candidates ≥ 1% (+ red flag). The ≥95% coverage claim holds trivially.
  - Isotonic map: scores ≲ 0.612 calibrate to 0.0 → `cp.posteriors` of set members and the UNCERTAIN reasoning line ("calibrated posterior = 0.000") are often 0.000. `calibrated_confidence` (CONFIDENT only) is 1.0 for every normal CONFIDENT case (single candidate, renormalized score 1.0 → iso 1.0; verified 282/282 in-sample) — so Rule Engine tab shows 100.0%; after a red-flag override it is the chosen emergency disease's calibrated posterior, often 0.0.
  - Single-token behavior (computed over all 131 tokens): 80 CONFIDENT, 26 UNCERTAIN, 25 ABSTAIN. Common lay symptoms (chest_pain, cough) ABSTAIN.
- `classify_via_dataset(case)` (alpha hardcoded 0.05):
  - ABSTAIN → INCOMPLETE_ASSESSMENT; condition `INSUFFICIENT_SYMPTOM_DATA` (no candidates, missing_fields ["symptom_tokens"]) else `UNCERTAIN_DIAGNOSIS` (missing_fields []); abstention_triggered True; candidates[:5]; prediction_set = cp.prediction_set (working-tree change; enables Flow C on short messages); probable_disease None.
  - UNCERTAIN → label = most severe `severity_for_disease` over set (worst case, "F3"); condition = top candidate name; probable_disease = top; prediction_set = set; no AHP; calibrated_confidence None.
  - CONFIDENT → top = the candidate named `cp.prediction_set[0]` (the disease CP chose; equals cp.candidates[0] unless the red-flag override fired); label = `severity_for_disease(top)` (CSV); condition/probable_disease = top; calibrated_confidence = cp.posteriors.get(top); "Other candidates considered" = up to 3 candidates excluding top; prediction_set []; emergency_result = E7(top). AHP does NOT change label. (FIXED 2026-10-08, was D-REDFLAG-TOP: code used cp.candidates[0], so altered_sensorium + burning_micturition + foul_smell_of_urine gave MODERATE "Urinary tract infection"; now EMERGENCY "Paralysis (brain hemorrhage)", AHP 9.)
  - Adult path ignores danger signs except inside E7 (CONFIDENT only).
  - `raw_confidence`, `gap_to_second` never set (always None).
- `severity_for_disease(name, kb=None)`: CSV value; missing/invalid/INCOMPLETE → MODERATE (`DEFAULT_UNKNOWN_DISEASE_SEVERITY`).
- `classify_with_abstention` (τ/δ reject option): exists, NO callers anywhere.
- Example outputs (computed): ["chest_pain"] → ABSTAIN size 6; ["chest_pain","vomiting","breathlessness","sweating"] → UNCERTAIN [Heart attack, Tuberculosis] → EMERGENCY; ["cough"] → ABSTAIN size 5 (Common Cold, GERD, Pneumonia, Tuberculosis, Bronchial Asthma); ["altered_sensorium"] → CONFIDENT Paralysis.

### E7 Stage 2 AHP emergency scorer (`emergency_scorer.py::score_emergency(case, disease_name, severity_label) -> EmergencyResult`)
- `AHP_WEIGHTS`: complication_probability 0.379, time_to_treatment 0.249, disease_severity 0.161, age_vulnerability 0.102, onset_acuity 0.066, transmissibility 0.044 (sum 1.001; hardcoded — docstring says "computed below" but nothing is computed). Recomputed principal eigenvector of the docstring 6×6 matrix: 0.382/0.250/0.160/0.101/0.064/0.043, λmax 6.1225, CR 0.0198 (RI 1.24) vs stated CR 0.0205 — UNVERIFIED how 0.0205 was derived.
- `_DISEASE_ATTRS`: 41 diseases (exact dataset names) → (time_tx, complication, transmissibility); `_DEFAULT_ATTRS = (0.4, 0.5, 0.1)`.
- `_SEVERITY_SCORE`: MILD .15, MODERATE .45, SEVERE .75, EMERGENCY 1.0.
- `_age_vulnerability`: None .5; <2 1.0; <60 .8; <168 .3; <780 .2; else .85.
- `_onset_acuity_from_duration(duration str)`: None → .5; contains hour/hr/घंट/ঘণ্টা → 1.0; else week/month/year/सप्ताह/महीन/সপ্তাহ → .2; else .5. QUIRK: "three days", "three weeks", "chronic" contain "hr" → 1.0 (defect D-ONSET).
- Override: any of 4 danger signs `is True` → score 10, EMERGENCY, ESI 1, override_triggered True.
- Score = clamp(round(Σ w·attr × 9) + 1, 1, 10). Bands: ≤3 NON_URGENT (ESI 5 if ≤2 else 4); 4–7 URGENT (ESI 3); 8–9 EMERGENCY (ESI 2); 10 EMERGENCY (ESI 1).
- Display: full AHP breakdown in Rule Engine tab; a one-line "Stage 2 — AHP emergency score …" summary is appended to `result.reasoning` (shown in ASHA/Doctor/Rule Engine reasoning trails) and emergency_result appears in the Doctor raw JSON. Never alters label or routing.

### E8 Flow C — adaptive follow-up to caller (`agent1_extraction.py`, `followup_question_selector.py`) — USED BY APP
- `next_followup_question(case, result, trail, backend, max_turns=3, language=None) -> (token, english_text) | None`:
  - None if len(trail) ≥ max_turns or len(result.prediction_set) ≤ 1 (does not inspect label).
  - severity_map over set; `fqs.select_followup_question(prediction_set, severity_map, known_tokens = case.symptom_tokens + [t["token"] for t in trail], rows_by_disease = clf._rows_by_disease, vocabulary = clf.kb.vocabulary, posteriors=None, loop_count=len(trail))`.
  - text = `generate_followup_question_text(token, set, severity_map, backend)`.
- `record_followup_answer(case, trail, token, question, answer) -> case`: appends {question, answer, token} to trail (mutates); parse_yes_no(answer) True and token new → token appended to symptom_tokens. "No" only marks as asked (classifier has no absence evidence).
- `parse_yes_no(text) -> True|False|None`: first word of `re.findall(r"[\wऀ-ॿ]+", lower)`; `_YES_WORDS` {yes,y,yeah,yep,1,true,haan,han,haa,ha,ji,हाँ,हां,हा}; `_NO_WORDS` {no,n,nope,0,false,nahi,nahin,nai,na,नहीं,नही,ना}; else None.
- `classify_with_adaptive_followup(case, backend, answer_provider=None, max_turns=3, language=None, case_store=None, gateway=None) -> (result, trail)`: blocking loop using the 2 functions above; outgoing question = gateway.from_english(q) if gateway; trail keeps English; records final to case_store if given; does NOT return updated case. answer_provider None → no loop. Callers: only tests/test_adaptive_followup_loop.py (UI uses the step functions directly).
- `generate_followup_question_text`: `_ADAPTIVE_FOLLOWUP_SYSTEM_PROMPT` rules: single yes/no question, plain language, no jargon, discriminates listed diseases, simple English ≤12 words (for machine translation), output only question. Any exception → `fqs.template_for(token)`. `language` param unused. Observed: 3B model sometimes exceeds 12 words / awkward wording (accepted by owner).
- Selector (`followup_question_selector.py`):
  - `select_followup_question`: tiers = set of severity ranks in set; tier conflict → `_emergency_first_token` (urgency "EMERGENCY_PRIORITY"); if None → `_info_gain_token` (urgency reset to ROUTINE only if no conflict).
  - `_emergency_first_token`: most severe disease vs others; for each unknown vocab token with freq_in_emergency ≥ `_MIN_EMERGENCY_FREQ = 0.20`: score = freq_em − mean(freq others); argmax (strict >, init −1.0).
  - `_info_gain_token`: uniform prior over set (posteriors None); p_present = Σ prior·freq; skip if p_present or p_absent < 0.01; IG = H_prior − expected H; argmax.
  - `_QUESTION_TEMPLATES`: 32 tokens; `radiating_pain`, `seizures` not in vocab → unreachable. `template_for` default "Is the patient experiencing {token with spaces}?".
  - `FollowupQuestion` dataclass: symptom_token, template_question, prediction_set, severity_map, severity_urgency, score, rationale, loop_count.
- RUBRIC / INVARIANTS (test coverage in test_adaptive_followup_loop.py: 1 tested with "no" answers only; 2 only weakly asserted (result.candidates non-empty); 3 and 4 tested; 5–6 untested):
  1. never re-ask a token (yes OR no) within a case;
  2. re-classification uses original tokens + all yes tokens (full context; raw_symptom_text kept);
  3. outgoing question passes through gateway; stored trail stays English;
  4. no answer_provider → no questions;
  5. ≤3 questions; stops when set size ≤ 1;
  6. unparseable reply → bot says "Sorry, please reply with yes or no."; pending question unchanged (not re-sent); no turn consumed (UI only).

### E9 Flow B — INCOMPLETE template question (`question_for_incomplete_result(result)`)
- Returns template for first `result.missing_fields` entry that has one (keys: 4 danger signs, symptom_tokens, age_months), only when label INCOMPLETE_ASSESSMENT; else None. UNCERTAIN_DIAGNOSIS abstain has missing_fields [] → None.
- UI shows it in Caller tab ("Please answer this") and reasoning trail; user must append the answer to the original message and click Assess again (no automatic loop; full re-extraction).

### E10 Routing — Agent 3 (`routing/router.py::route`)
- `route(label, location, facility_db, at=None, radius_km=60.0, routing_provider=None, case_id=None) -> DispatchResult`; defaults at = now(UTC), provider = MockRoutingProvider.
- Steps: (1a) `urgency_for_label`: EMERGENCY→"EMERGENCY", SEVERE→"URGENT", else None → return NON_URGENT no_facility_found; (1b) `resolve_location` (village name exact match or (lat,lon)); (2) `facilities_within_radius` (haversine ≤ 60 km, `DEFAULT_RADIUS_KM`); (3) `filter_eligible_facilities`: EMERGENCY needs has_emergency_care AND has_doctor_24hr (hours ignored); URGENT needs `is_facility_open(at)`; (4) `score_facility` lower-better = haversine − 5·doctor_24hr − 3·emergency_care − 2·blood_bank; (5) provider.get_route for top only; (6) DispatchResult(facility, route, urgency, reasoning[], case_id, no_facility_found).
- `is_facility_open`: is_24hr → True; open_days excludes weekday → False; missing times → False; overnight spans supported. DEFECT D-TZ: UI passes UTC `now`, facility hours are local → URGENT open check compares local hours vs UTC.
- MockRoutingProvider: AVERAGE_SPEED_KMH 40; road_distance = haversine (2dp); eta = dist/40·60 (1dp); maps_link `https://www.google.com/maps/dir/?api=1&origin=lat,lon&destination=lat,lon`; provider "mock_haversine".
- FacilityDB: SQLite tables facilities(…) and villages(name PK, lat, lon); app uses `build_demo_facility_db()` → `:memory:` (rebuilt on each call). DEFAULT_DB_PATH data/facility_db.sqlite unused.
- Demo data: villages Denkanikottai (12.5333,77.7667), Anchetty (12.4667,77.6333), Thally (12.6167,77.8333), Kelamangalam (12.4833,77.7333), Bannerghatta (12.8000,77.5667). Facilities (10): Denkanikottai PHC; Thally CHC (ER, 24h doc, 24hr); Anchetty PHC; Kelamangalam PHC; Hosur District Hospital (ER, 24h doc, blood bank, 24hr); Krishnagiri SDH (ER, blood bank, 00:00–23:59); Bannerghatta CHC (ER, 08–20); St. John's Medical College (all, 24hr); Unregistered Rural Clinic (no hours — never selectable); Bangalore City Medical College (all, 24hr). EMERGENCY-eligible: Thally CHC, Hosur DH, St. John's, Bangalore City MC. (derived) EMERGENCY winner: Thally CHC for Denkanikottai/Anchetty/Thally/Kelamangalam; St. John's for Bannerghatta.

### E11 Doctor handoff report (`routing/report.py::generate_doctor_report(case, classification, dispatch, backend) -> str`)
- REPORT_SYSTEM_PROMPT: clinical handoff; do not contradict/re-diagnose triage; readable < 30 s; no invented facts; plain text, no markdown, ≤150 words.
- Prompt includes raw text (original language), age, duration, true danger signs, label+condition, probable disease + score + precautions, facility + ETA.
- Catches only BackendUnavailable → `_fallback_report` = "TRIAGE: … | Reported: … | Probable disease … | Routed to: …, ETA … min".
- Always English. Rendered in Doctor tab via unsafe_allow_html without escaping (defect D-HTML).

### E12 Case store (`case_store.py::CaseStore`) — opt-in, NOT used by UI
- Default path data/case_store.sqlite (gitignored *.sqlite). Table `cases(case_id, area NOT NULL, recorded_at, symptom_tokens, label, probable_disease, language, raw_text_hash)`, index on area, no PK.
- `record(case, result)`: area = case.location or "unknown"; stores sha256 of raw text (never raw text); `recent_by_area(area, limit=20)` ORDER BY recorded_at DESC, rowid DESC (tiebreak added in working tree).
- Used by: extract_and_classify_with_followup(case_store=), classify_with_adaptive_followup(case_store=), tests.

## §8 FOLLOW-UP FLOWS (3 distinct mechanisms — do not confuse)
| flow | when | question source | delivery | answer handling | used by UI |
|---|---|---|---|---|---|
| A pre-classification | missing age / adult symptom / pediatric danger signs BEFORE classify | 6 fixed templates | answer_provider callback | appended to LLM context, full re-extraction | NO |
| B incomplete template | result INCOMPLETE with missing_fields | same 6 templates | shown in Caller tab + reasoning | user edits message + presses Assess (manual) | YES (display) |
| C adaptive CP | adult/dataset result with prediction_set size ≥ 2 (UNCERTAIN or ABSTAIN-with-candidates) | selector token + LLM phrasing (template fallback) | Caller tab chat, translated via gateway | free-text reply → parse_yes_no → yes adds token → reclassify | YES |
- Pediatric IMNCI results never trigger Flow C (prediction_set empty on pediatric path).

## §9 DATA CONTRACTS (`app/schemas.py`, pydantic v2, extra keys ignored)
- HARD INVARIANT: clinical booleans are `Optional[bool] = None`; None = not assessed, False = assessed absent, True = present. Never replace None with False.
- Enums: `Severity` (mild, moderate, severe, unknown); `AgeGroup` (infant, child, adult, elderly, unknown; None = not stated); `ClassificationLabel` (INCOMPLETE_ASSESSMENT, EMERGENCY, SEVERE, MODERATE, MILD); `EmergencyBand` (NON_URGENT, URGENT, EMERGENCY). `SEVERITY_RANK` {EMERGENCY 0, SEVERE 1, MODERATE 2, MILD 3} (INCOMPLETE absent on purpose).
- `DangerSigns`: not_able_to_drink_or_breastfeed, vomits_everything, convulsions, lethargic_or_unconscious; methods any_true(), all_assessed(), missing_fields().
- `CoughDifficultBreathing`: present, duration_days, breaths_per_minute*, chest_indrawing*, stridor_when_calm* (*exam-only).
- `DiarrheaAssessment`: present, duration_days, blood_in_stool, restless_or_irritable, sunken_eyes, drinks_eagerly_thirsty, drinks_poorly_or_not_able, skin_pinch_goes_back_slowly*, skin_pinch_goes_back_very_slowly*.
- `ExtractedCase`: case_id?, raw_symptom_text (required), symptom?, duration?, severity (default UNKNOWN, non-Optional → null fails), age_group?, age_months? (validator 0..1500), location?, notes?, danger_signs (default factory), cough?, diarrhea?, language?, extracted_at (now UTC), llm_backend?, disambiguation_confidence?, symptom_tokens (list, default []; empty = none matched yet, not negative).
- `ClassificationResult`: label, condition?, reasoning[], missing_fields[], case_id?, candidates[DiseaseCandidate], probable_disease?, calibrated_confidence?, raw_confidence? (never set), gap_to_second? (never set), abstention_triggered=False, prediction_set[] (stale comment says empty on abstain — no longer true), emergency_result?.
- `DiseaseCandidate`: name, score (renormalized NB share ≥ 0.01, not calibrated; on the adult path EMERGENCY candidates are ×3-boosted and renormalized by classify_with_cp; pediatric context uses unboosted classify_diseases; UI shows it as likelihood %), matched_symptoms[], precautions[].
- `EmergencyResult`: score int 1–10, band, esi_level int 1–5, override_triggered, attribute_scores{}, attribute_weights{}, reasoning[].
- Routing schemas: `Facility` (facility_id?, name, facility_type PHC|CHC|DH|SDH|Medical College, lat, lon, has_emergency_care, has_doctor_24hr, has_blood_bank, is_24hr, open_time "HH:MM" local, close_time, open_days 0=Mon..6=Sun|None=all, phone?), `ScoredFacility`, `RouteInfo` (road_distance_km, eta_minutes, maps_link, provider), `DispatchResult` (facility?, route?, urgency, reasoning[], case_id?, no_facility_found).

## §10 DATA FILES (`triage-poc/data/`, loaded by `disease_kb.py::DiseaseKB.load`)
- disease_symptoms.csv: columns Disease, Symptom_1..Symptom_17; 4920 rows; 41 diseases × 120 rows; 131 unique tokens after `normalize_symptom_token` (strip→lower→whitespace runs to "_"→collapse "_"); 3–17 symptoms/row; 304 unique (disease, profile) pairs (5–10 per disease, no profile shared across diseases).
- disease_precautions.csv: Disease, Precaution_1..4; 41 rows; Allergy and Heart attack have 3.
- disease_severity.csv: Disease, Severity; 41 rows. EMERGENCY (2): Heart attack, Paralysis (brain hemorrhage). SEVERE (16): AIDS, Alcoholic hepatitis, Chronic cholestasis, Dengue, Gastroenteritis, Hepatitis B, Hepatitis C, Hepatitis D, Hepatitis E, Hypoglycemia, Jaundice, Peptic ulcer diseae, Pneumonia, Tuberculosis, Typhoid, hepatitis A. MODERATE (18): (vertigo) Paroymsal  Positional Vertigo, Arthritis, Bronchial Asthma, Cervical spondylosis, Chicken pox, Diabetes, Dimorphic hemmorhoids(piles), Drug Reaction, GERD, Hypertension, Hyperthyroidism, Hypothyroidism, Impetigo, Malaria, Migraine, Osteoarthristis, Urinary tract infection, Varicose veins. MILD (5): Acne, Allergy, Common Cold, Fungal infection, Psoriasis.
- Name quirks are authoritative keys (do not "fix" spelling without updating all 3 CSVs + emergency_scorer._DISEASE_ATTRS + tests): double space in vertigo name, "diseae", "Osteoarthristis", lowercase "hepatitis A".
- `normalize_disease_name`: strip only (case-sensitive).

## §11 UI — FRONTEND (`triage-poc/streamlit_app.py`; backend = same process)
- Page: title "Rural Health Triage", icon 🩺, wide layout, custom dark CSS (palette constants, classes .banner, .card, .action-card, .facility-card, .doctor-report, .reasoning-line, .checklist-item, pills).
- Sidebar: info box "Extraction backend: Local LLM (Ollama)…" (no backend selector); `village_choice` selectbox ["Not specified"] + 5 villages (module global); "Demo facility network" expander; "Session history" expander (last 15: time, INITIAL label, text[:40]).
- Main: `st.text_area("Patient / caregiver message")` + `st.button("Assess")`. Spinner "Detecting language, extracting and classifying...".
- session_state: `history` list[{time,label,text}] (initial label only); `current` {case, result, gw} + dispatch, report, dispatch_key; `adaptive` {case, trail[{question(English), answer(raw), token}], pending_token, pending_question(English), chat[(role "bot"|"user", text)]} (comment in file lists stale keys).
- Tabs (st.tabs): "📞 Caller / Caregiver", "🏥 ASHA Worker", "🩺 Doctor", "🔬 Rule Engine Output".
- Caller tab (`render_caller_view`) — ONLY translated tab: CALLER_MESSAGE banner; "Where to go" + facility card (card itself not translated); village hint; MILD/MODERATE "What you should do" precautions (translated batch); INCOMPLETE "Please answer this" (Flow B) + caption "Add the answer to your message above and press Assess again."; Flow C chat "A few more questions": st.chat_message history, `st.form("followup_reply_form", clear_on_submit=True)` with `st.text_input("Your reply")` + `st.form_submit_button("Send")` (Enter submits) while a question is pending.
- ASHA tab (`render_asha_view`, English): banner, ASHA_ACTION card, "Physical checks to do in person" (exam-only fields still None), danger-sign pills, referral facility + distance/ETA metrics, top-3 candidates, collapsed reasoning.
- Doctor tab (`render_doctor_view`, English): banner; for EMERGENCY/SEVERE "🚑 Doctor handoff" (facility card, 3 metrics, routing audit trail expander, LLM report); home care; top-5 candidates with matched symptoms + precautions; full reasoning; case details; raw JSON.
- Rule Engine tab (`render_rule_engine_view`, English): Stage 1 metrics (calibrated confidence, raw posterior "—", gap "—"), candidates with bars, Stage 2 AHP gauge/table/ESI/override (CR caption hardcoded 0.0205), reasoning trails. Stale/misleading texts: (a) "τ threshold (Youden's J)" caption prints the confidence value; (b) every CONFIDENT case shows success box "Abstention check passed — both τ (confidence) and δ (gap) conditions met" (CP is used, not τ/δ); (c) candidates caption "Scores are calibrated isotonic posteriors … cleared τ and δ" (they are NB/boosted shares); (d) "pediatric IMCI path" Stage 1 info shown for adult UNCERTAIN and adult ABSTAIN too (calibrated_confidence None); (e) Stage 2 info "pediatric IMCI path or Stage 1 abstained" also shown for UNCERTAIN.
- Display tables: LABEL_DISPLAY, CALLER_MESSAGE (EMERGENCY "This is an emergency / Go to the hospital…", SEVERE "Please see a doctor today", MODERATE "Please visit a health worker soon", MILD "You can care for this at home", INCOMPLETE "We need a little more information"), ASHA_ACTION, MISSING_FIELD_LABELS, EXAM_ONLY_CHECKS; SEVERITY_TAG unused.
- Translation cost note: Caller tab `_t`/`_tb` re-call the translator on every rerun when language ≠ English (bot chat messages stored pre-translated). User replies are NOT translated (parse_yes_no word lists only).
- No authentication; any viewer sees all tabs; follow-up chat assumes the caller is at this screen (no SMS/IVR yet).

## §12 OUTPUTS → RECIPIENTS
| recipient (tab) | receives | language |
|---|---|---|
| Caller/caregiver | plain banner, where-to-go facility card, home-care precautions, missing-info question, adaptive follow-up questions + reply box | caller's language via gateway (needs valid Google key; else English) |
| ASHA worker | action instruction, exam checklist, danger signs, referral facility, top-3 candidates, reasoning | English |
| Doctor | handoff (facility, metrics, routing audit trail, LLM report), candidates, full reasoning, case details, JSON | English |
| Rule-engine/auditor | Stage 1 + CP numbers, candidates, AHP breakdown, reasoning | English |
- NOT produced anywhere: SMS/IVR/WhatsApp messages, ASHA alerts, 108 ambulance dispatch, outbreak dashboards, persisted case logs (UI), exports.

## §13 TESTS & EVALS
### §13.1 Test suite (`triage-poc/tests/`, no conftest; 17/18 files insert triage-poc into sys.path; test_adaptive_followup_loop.py relies on pytest rootdir prepend (tests/ is a package) or PYTHONPATH=.)
| file | n | covers | gating |
|---|---|---|---|
| test_adaptive_followup_loop.py (UNTRACKED new) | 4 | Flow C invariants 1–4 | none |
| test_age_handling.py | 68 | age None defaults, AGE_UNKNOWN, age clarifier, infant floor, young-infant escalation | 12 need Ollama |
| test_agent1_followup.py | 9 | Flow A loop | 2 need Ollama |
| test_calibration.py | 8 | dataset threshold calibration = 0.45 | 6 need faiss |
| test_case_store.py | 10 | CaseStore | none |
| test_dataset_symptom_matching.py | 19 | dataset index, aliases, clause split, end-to-end | 16 need faiss; 1 FAILS (stale) |
| test_disambiguation.py | 22 | pediatric index, fallback rules | 10 need faiss (1 xfail); test_fuzzy_match_above_threshold unguarded (FAILS with ImportError w/o faiss) |
| test_disease_classifier.py | 27 | KB, classify_diseases, severity lookup | none |
| test_disease_coverage.py | 84 | each disease top-3 / top-1 for its mode row | none |
| test_followup_policy.py | 17 | missing_required routes | none |
| test_followup_scripts.py | 13 | 6 scripted Flow A conversations | none |
| test_integration_agent1_pipeline.py | 5 | live Ollama end-to-end | all need Ollama (1 xfail) |
| test_integrations.py | 31 | messaging, translation, gateway, ORS (HTTP mocked) | none |
| test_routing.py | 25 | haversine, open hours, scoring, route, report fallback | none |
| test_routing_fixtures.py | 9 | route over fixtures (10 facilities, 5 villages) | none |
| test_rules_engine.py | 16 | IMNCI rules | none |
| test_rules_engine_dataset_routing.py | 13 | age routing, INSUFFICIENT, dataset context | none |
| test_two_followup_flows.py | 6 | Flow A vs Flow B same first question | none |
- Total 386. Last run (2026-10-08, working tree): 383 passed, 1 failed, 2 xfailed.
- FAILING (stale, pre-existing, also failed at 650bbca^): test_dataset_symptom_matching.py::test_pipeline_dataset_tokens_feed_disease_classifier_end_to_end — asserts probable_disease is not None for single "chest pain"; engine now abstains (CP set of 6, result.candidates capped at 5, probable_disease None). Decision pending: update test vs keep.
- XFAIL (documented known limitations): test_disambiguation.py::test_hinglish_diarrhea_known_limitation; test_integration_agent1_pipeline.py::test_schema_boundary_false_survives_llm_roundtrip_on_simple_negation.
- No direct tests: emergency_scorer.py, followup_question_selector.py, classify_with_cp, parse_yes_no, demo_facilities.py, streamlit_app.py.
- Fixtures: facilities.py (10 facilities, 5 villages), followup_scripts.py (6 scripts), symptom_calibration.py (80 labelled phrases; its docstring still says 0.65 uncalibrated).

### §13.2 Eval scripts (classifier-only; NOT end-to-end)
- All: seed 42; cases = complete rows of disease_symptoms.csv (IN-SAMPLE, the data the NB model is fit on; 3–17 symptoms each); call `classify_with_cp(tokens, 0.05)` only (no Agent 1, FAISS, rules_engine worst-case, AHP); predicted = prediction_set[0].
- All 3 eval JSON sets regenerated on the current tree 2026-10-08 and match current code. Evals do not call rules_engine, so rules_engine changes cannot affect them; disease_classifier/data changes can → re-run all 3 after such changes and keep the regenerated JSONs.
- evaluate_100_cases.py (stratified 4×4) → tests/eval_results.json (regenerated 2026-10-08; previous committed version was stale: 96/100, EMERGENCY TP 2 / FN 2). Current: correct 98/100, EMERGENCY 4/4, FN 0, P/R/F1 1.00, CONFIDENT 95 (all correct), UNCERTAIN 5, ABSTAIN 0.
- evaluate_balanced.py (34/33/33) → eval_balanced_results.json + _log.json (re-run identical to committed): EMERGENCY 34/34, URGENT 32/33, NON_URGENT 33/33.
- evaluate_1000.py (334/333/333, EMERGENCY sampled with replacement) → eval_1000_results.json + _log.json (re-run identical): EMERGENCY 334/334, URGENT 332/333, NON_URGENT 333/333; CONFIDENT 793, UNCERTAIN 206, ABSTAIN 1 (Chronic cholestasis).
- INTERPRETATION RULE: near-perfect eval scores measure internal consistency on full in-sample symptom rows, NOT real caller performance. Real short messages carry few tokens: single tokens give 80/131 CONFIDENT, 26 UNCERTAIN, 25 ABSTAIN, and common lay symptoms (chest_pain, cough) ABSTAIN; first-message extraction/matching quality (E1/E2) is unmeasured by any eval.

## §14 CURRENT STATE
### §14.1 Works (verified 2026-10-08)
- Ollama llama3.2:3b extraction validated end-to-end; FAISS matching; adult CP path; IMNCI path; AHP; routing + report; Flow C in Caller tab (live UI test: "I am 30 years old and I have cough" → question → "maybe" re-asked → "nahi" → next question → "haan" → SEVERE "Please see a doctor today", box closed); test suite 383/386.
### §14.2 Known defects (confirmed in code; not fixed)
- D-HEAD: committed HEAD a192ded crashes (NameError held_out_rows) — fixed in uncommitted tree only.
- D-STALE-TEST: 1 failing stale test (§13.1).
- D-REQ: scikit-learn required but commented out in requirements.txt.
- D-CP-TRIVIAL: q̂ = 1.0 at α = 0.05 → CP set = all NB candidates ≥ 1%; τ = 1.0 (unused path).
- D-REDFLAG: 3 of 5 red-flag tokens not in vocabulary; weakness_in_limbs (Cervical-spondylosis marker, 0% of Paralysis rows; alias "kamzori") forces Paralysis/EMERGENCY whenever Paralysis is in the set (over-triage); frequency comments don't match CSV. OWNER DECISION 2026-10-08: keep weakness_in_limbs as a red flag (accepted over-triage) — do not remove without asking.
- (FIXED 2026-10-08) D-REDFLAG-TOP — CONFIDENT path now labels from the CP-chosen disease (§7 E6).
- D-ONSET: onset acuity substring "hr" matches "three", "chronic".
- D-TZ: routing open-hours check uses UTC vs local facility hours.
- D-HTML: LLM report inserted unescaped into HTML (Doctor tab).
- D-HTTP: Ollama/Groq `resp.json()` / body parsing outside try → non-JSON HTTP body raises uncaught; Google translate KeyError uncaught.
- D-SEV: LLM explicit `severity: null` → ExtractionValidationError.
- D-UI-STALE: Rule Engine tab stale texts (a)–(e) in §11, raw_confidence/gap "—"; adaptive-state comment lists stale keys; history keeps only initial label.
- D-TEMPLATES: selector templates radiating_pain, seizures unreachable; urgency stays EMERGENCY_PRIORITY on info-gain fallback (not displayed anymore).
- D-FLOWA-C: Flow A trail lacks "token" → incompatible with next_followup_question; classify_with_adaptive_followup doesn't return updated case.
- D-GATE: followup_policy vs classify disagree for age < 2 months.
- D-EXTRACT: weak first-message symptom capture with 3B model (1 token from multi-symptom text; Hinglish mis-mapping observed).
- D-DOCSTRINGS: stale docstrings (extract_case raises wrapper not ValidationError; disambiguation "no CSV exists"; schemas prediction_set comment; selector "called by classify_with_adaptive_followup"; "Groq" mentions; language_gateway "doctor report translated"; disease_classifier "stdlib csv only"; streamlit_app docstring "extract_and_classify"; fixtures 0.65).
### §14.3 Design limitations (by design / documented)
- "No" answers add no negative evidence (NB scores present tokens only).
- Pediatric exam-only signs cannot come from text → IMNCI cough/diarrhea mostly MILD without on-site exam; no UI to enter exam findings.
- Young infants (<2 months) always INCOMPLETE (no validated ruleset).
- disease_severity.csv has no documented clinical provenance.
- Dataset (Kaggle-style 41 diseases) not representative of real rural presentations.
### §14.4 NOT_BUILT
- Agent 2 (DBSCAN outbreak surveillance, RAG); inbound SMS/IVR; outbound SMS (TwilioProvider exists, never called); ASHA alerts; 108 ambulance dispatch; real road routing in app (ORSProvider exists, unused); backend selector / auto-failover; persistence in UI; auth; CI.

## §15 DOCS INDEX + CONFIRMED DOC-vs-CODE MISMATCHES (code is truth)
| doc | status |
|---|---|
| EVALUATION_METRICS.md | proposal for paper evaluation (8 metric rows); no eval code/test set built for it yet |
| README.md | partly outdated (pre-CP/AHP/adaptive pipeline; "only two AI points"; AGE_OUT_OF_MODULE_SCOPE; Streamlit described as single-shot extract_and_classify with no loop — false; RegexBackend selectable in UI — false; Google Translate "not used" — false; "DISEASE_SEVERITY table" — is CSV). Its top documentation table links triage-poc/rule_engine_design_final.md as canonical |
| ARCHITECTURE.md | outdated (omits emergency_scorer, followup_question_selector, language_gateway, disease_severity.csv, several tests, evals; says no dashboard) |
| AGENT1_README.md | partly outdated (no adaptive Flow C, no LanguageGateway; says the other LLM use is "Agent 2's outbreak reasoning" — never built; omits report.py and Flow C phrasing LLM calls) |
| CHANGES_AGE_HANDLING.md | historical (test counts/branch state outdated; τ/δ described as active mechanism) |
| RULE_ENGINE_DESIGN.md | superseded proposal; nothing implemented |
| Rule engine final.md | Stage 2 formula/bands/ESI/severity+onset scores/weight order match code. Diffs: override 6 danger signs (doc) vs 4 (code, CONFIDENT path only); age bands 5–14/15–64 (doc) vs 5–13/14–64 (code); attributes 4–6 use per-disease values in code. Stage 1 not as written: presence-only NB (no absence term), no epidemiological prior, no LR/Brier/ECE/κ tooling; τ/δ reject option exists (classify_with_abstention) but is uncalled — CP runs instead |
| triage-poc/rule_engine_design_final.md | presented as current but mismatched: AHP weights/attributes (doc 0.412/0.263/0.142/0.091/0.058/0.034 vs code 0.379/0.249/0.161/0.102/0.066/0.044), bands (doc 5–7 URGENT, 0–4 NON_URGENT vs code 4–7, ≤3), EmergencyResult types (doc float/str vs int/int), calibration (doc 80/20 vs code 5-fold), Groq phrasing (code: whichever backend; app uses Ollama), ABSTAIN stops loop (no longer), Yes/No/Skip UI (now chat), L7 duplicate logic (resolved), fix numbering F1/F2 vs code comments "Fix 2/Fix 3", class names CoughInfo/DiarrheaInfo vs CoughDifficultBreathing/DiarrheaAssessment |
- Remote branch origin/22sep (95a874c, unmerged, docs-only): moves AGENT1_README.md + ARCHITECTURE.md to docs/; moves README body to docs/README.md and replaces root README with an index; copies (not moves) rule_engine_design_final.md to docs/; deletes RULE_ENGINE_DESIGN.md and "Rule engine final.md"; leaves CHANGES_AGE_HANDLING.md at root.

## §16 GIT STATE
### §16.1 Repo
- remote origin https://github.com/Rishii-10/GMU.git; origin/HEAD → origin/main. Current branch `fix` at a192ded (= main = origin/main), no upstream set. Other refs: translate + origin/translate (f37fa08), origin/22sep (95a874c, the ONLY branch not merged into main), origin/rule-engine-design (02624f3), origin/young-infant-escalation (a38b18c). stash@{0} "WIP on translate: be28b4b".
- Last commits: a192ded Merge PR #3 rule-engine-design; 02624f3 Merge origin/main into rule-engine-design; e9aec23 "y"; 650bbca Add CP gate, AHP Stage 2, adaptive follow-up, design doc; ed1111b Merge PR #2 translate.
- Root cause of D-HEAD: merge 02624f3 combined 650bbca's 80/20 calibration code with 109e966's k-fold code, leaving `held_out_rows` undefined.
### §16.2 Uncommitted working-tree changes at sync (2026-10-08)
- app/disease_classifier.py: true-class posterior collected inside k-fold loop (out-of-fold); broken `held_out_rows` block removed → classifier builds.
- app/case_store.py: ORDER BY recorded_at DESC, rowid DESC.
- app/rules_engine.py: ABSTAIN result carries prediction_set; CONFIDENT path uses the CP-chosen disease (cp.prediction_set[0]) instead of cp.candidates[0], and "Other candidates considered" excludes it (fixes D-REDFLAG-TOP).
- tests/eval_results.json: regenerated on current code (98/100, EMERGENCY 4/4).
- app/agent1_extraction.py: Flow C refactor → parse_yes_no, next_followup_question, record_followup_answer; classify_with_adaptive_followup rebuilt on them + `gateway` param; never re-asks asked tokens; continues on ABSTAIN; prompt English-only ≤12 words; `clf._kb` → `clf.kb`.
- streamlit_app.py: FAISS disambiguation added to Assess flow (cached); old Yes/No/Skip panel above tabs removed; Flow C chat + reply box in Caller tab (translated); in-place `current.update` keeps gw/dispatch; dispatch cache key adds condition; `clf._kb` fix.
- tests/test_adaptive_followup_loop.py: new (untracked).
- CODEBASE_CONTEXT.md, CLAUDE.md: new. FOLLOWUP_PLAN.md: deleted (was untracked, never committed).

## §17 TASK → FILES INDEX (open only these)
| task | files |
|---|---|
| change extraction / prompt / LLM backend | app/agent1_extraction.py (backends, EXTRACTION_SYSTEM_PROMPT, extract_case), app/schemas.py, tests/test_age_handling.py, tests/test_integration_agent1_pipeline.py |
| symptom matching / Hindi aliases | app/disambiguation.py, app/agent1_extraction.py::_apply_disambiguation, tests/test_disambiguation.py, tests/test_dataset_symptom_matching.py, tests/test_calibration.py, tests/fixtures/symptom_calibration.py |
| pediatric rules | app/rules_engine.py (IMNCI functions), tests/test_rules_engine.py |
| age routing | app/rules_engine.py::classify, app/followup_policy.py, tests/test_rules_engine_dataset_routing.py, tests/test_age_handling.py |
| adult disease classification / CP / calibration | app/disease_classifier.py, app/rules_engine.py::classify_via_dataset, tests/test_disease_classifier.py, tests/test_disease_coverage.py, tests/evaluate_*.py |
| emergency score / AHP | app/emergency_scorer.py (no direct tests), streamlit_app.py::render_rule_engine_view |
| adaptive follow-up (Flow C) | app/agent1_extraction.py (parse_yes_no, next_followup_question, record_followup_answer, generate_followup_question_text, classify_with_adaptive_followup), app/followup_question_selector.py, streamlit_app.py (_ask_next, _start_adaptive, _answer_adaptive, render_caller_view), tests/test_adaptive_followup_loop.py |
| Flow A / B questions | app/agent1_extraction.py (templates, extract_case_with_followup, question_for_incomplete_result), app/followup_policy.py, tests/test_agent1_followup.py, tests/test_followup_policy.py, tests/test_followup_scripts.py, tests/test_two_followup_flows.py |
| routing / facilities | app/routing/router.py, facility_db.py, demo_facilities.py, schemas.py, tests/test_routing.py, tests/test_routing_fixtures.py, tests/fixtures/facilities.py |
| doctor report | app/routing/report.py, tests/test_routing.py |
| translation | app/integrations/language_gateway.py, translation.py, streamlit_app.py (_t, _tb), tests/test_integrations.py |
| SMS / messaging channel (future) | app/integrations/messaging.py; plug into Flow C step functions (next_followup_question / record_followup_answer) or classify_with_adaptive_followup(answer_provider=…) |
| UI | triage-poc/streamlit_app.py, triage-poc/.streamlit/config.toml |
| disease data / severities | triage-poc/data/*.csv, app/disease_kb.py, app/emergency_scorer.py::_DISEASE_ATTRS |
| case logging | app/case_store.py, tests/test_case_store.py |

## §18 SYNC LOG (append one line per update: date | commit/tree | sections changed | by)
- 2026-10-08 | fix @ a192ded + uncommitted tree (§16.2) | initial creation, all sections | Claude Code session (4 fact agents)
- 2026-10-08 | same tree | verifier pass: 21 corrections (§0, §4.4, §6, §7 E6/E7/E8, §9, §11, §13, §14.2, §15, §16.1); added D-REDFLAG-TOP | Claude Code session (verifier agent)
- 2026-10-08 | same tree + rules_engine CONFIDENT fix + regenerated eval_results.json | §7 E6, §13.2, §14.2 (D-REDFLAG-TOP fixed; weakness_in_limbs kept by owner decision), §16.2; tests re-run 383 pass / 1 fail (stale) / 2 xfail; all 3 evals re-run | Claude Code session
- 2026-10-08 | fix tree | §3, §15: added EVALUATION_METRICS.md (paper metrics proposal, docs only) | Claude Code session
