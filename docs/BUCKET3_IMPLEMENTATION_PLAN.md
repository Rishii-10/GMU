# Bucket 3 — Implementation Plan

> **Purpose.** The full, buildable plan for Bucket 3: the **Surveillance Agent
> (Agent 2)** plus the new evaluation/measurement work. Written to be worked
> from directly. Grounded against the current code (`case_store.py`,
> `agent1_extraction.LLMBackend.generate_text(system, user)`, the
> `report.py` `BackendUnavailable → fallback` pattern, and the `st.tabs` strip at
> `streamlit_app.py:1426`).
>
> Companion docs: [`PAPER_FRAMING.md`](PAPER_FRAMING.md),
> [`EVALUATION_REDESIGN.md`](EVALUATION_REDESIGN.md),
> [`../CODEBASE_CONTEXT.md`](../CODEBASE_CONTEXT.md).

---

## 0. Scope & principles

- **Build, don't rewrite.** Everything here is **new, additive code**. Existing
  triage code is untouched except two tiny, back-compatible additions (a case
  store field; one optional, flag-gated feedback hook).
- **Reuse the seams.** Agent 2 consumes the **existing shared case contract**
  (`CaseStore`) and the **existing `LLMBackend`** — this reuse is itself the
  evidence for the reusability/extensibility claim (Pillar 4).
- **Light models, agentic orchestration.** Detection stays simple statistics;
  the "intelligence" is the agent's autonomous decide→brief→recommend loop.
- **Degrade gracefully.** The LLM brief falls back to a deterministic template
  when Ollama is down, so headless/CI runs never break.
- **No new dependencies.** stdlib + `numpy` (present) + optional `scikit-learn`
  (present) only.

---

## 1. Surveillance Agent (Agent 2) — detailed

### 1.1 Package layout (mirrors `app/routing/`, the existing "Agent 3")
```
app/surveillance/
├─ __init__.py          # "Agent 2 — syndromic surveillance" docstring
├─ schemas.py           # SurveillanceCase, VillageStat, OutbreakSignal, Alert, SurveillanceReport
├─ ingest.py            # CaseStore / list[SurveillanceCase] -> normalized records
├─ analytics.py         # counts & trends by area/disease/age-band/severity/week
├─ detect.py            # baseline-deviation detector (+ optional DBSCAN)
├─ alert_policy.py      # decide alert level ("only when out of hand")
├─ recommend.py         # pattern -> recommended public-health action (rules)
├─ brief.py             # LLM situational brief (reuse LLMBackend) + fallback
├─ agent.py             # SurveillanceAgent orchestrator + watchlist state
├─ feedback.py          # OPTIONAL, flag-gated: outbreak -> triage prior boosts
└─ demo_stream.py       # synthetic multi-village stream with an injected outbreak
```

### 1.2 Data contract & ingest
- `SurveillanceCase` (what the agent reasons over): `area`, `disease`
  (`probable_disease`, may be None), `symptom_tokens`, `severity`
  (`ClassificationLabel`), `age_months` (Optional), `recorded_at` (datetime),
  `language`.
- `ingest.from_case_store(store, since=None) -> list[SurveillanceCase]` reads the
  existing `cases` table; `ingest.from_stream(list) -> list[SurveillanceCase]`
  for the demo/synthetic path.
- **Small additive change to `case_store.py`:** add a nullable `age_months`
  column (so live age-band analytics is possible). Back-compatible:
  `CREATE TABLE IF NOT EXISTS` plus a guarded `ALTER TABLE … ADD COLUMN` for
  existing DBs; `record()` writes `case.age_months`. Old rows read as NULL.
  *(If we want zero schema change, the demo stream still carries age directly and
  live age-band charts simply show "unknown" — but adding the column is cleaner.)*

### 1.3 Analytics layer (`analytics.py`) — the "data analysis"
Pure-Python/`numpy` aggregation (no pandas):
- `counts_by(records, keys=("area","disease","week"))` → nested counts.
- `age_band(age_months)` → `<5y / 5–14y / 15–64y / 65+`.
- `village_table(records, now)` → per-village: total cases, severe+emergency
  share, top disease, week-over-week growth.
- `trend(records, area, disease)` → weekly series for charting.

### 1.4 Detection layer (`detect.py`) — light ML/stats
- **Primary — baseline-deviation.** For each (area, disease) in the current
  window: `observed` vs `baseline` = mean over the prior `W` windows; flag when
  `observed ≥ baseline + k·σ` **and** `observed ≥ MIN_COUNT` (so 1→2 never
  alerts). Severity-weighted: emergency/severe diseases use a lower threshold.
  Returns `OutbreakSignal(area, disease, window, observed, baseline, score,
  severity_weight, age_skew)`.
- **Optional — spatial-temporal clustering.** Guarded `sklearn` DBSCAN over
  `(lat, lon, week)` of same-disease cases, as a secondary view. Skipped
  cleanly if too few points / sklearn absent.
- Deterministic; no LLM.

### 1.5 Alert policy (`alert_policy.py`) — "alert only when things get out of hand"
This is the behaviour that justifies "agent": the module **decides when a signal
is worth a human's attention**, and stays silent otherwise.
- `classify_alert(signal) -> Alert | None`: map `score` (× severity weight) to
  `INFO / WATCH / ALERT`; **emit only `WATCH`+ to the official**, log `INFO`
  silently.
- Thresholds are named constants (`ALERT_SCORE`, `WATCH_SCORE`, `MIN_COUNT`,
  `BASELINE_WINDOWS`), swept-able later if needed.

### 1.6 Action recommendation (`recommend.py`)
Small rules table mapping a disease/category to a public-health action:
- diarrheal (Gastroenteritis, Typhoid, Hepatitis A/E, …) → "inspect water/sanitation; distribute ORS"
- respiratory (Pneumonia, Tuberculosis, Bronchial Asthma, Common Cold) → "active screening for pneumonia/TB"
- vector-borne (Dengue, Malaria) → "vector-control / source reduction"
- (default) → "monitor; confirm with field visit"
`recommend_action(disease) -> str`. Covers all 41 dataset diseases via category map.

### 1.7 LLM situational brief (`brief.py`) — the agentic "tool use"
- `generate_brief(signal, stats, action, backend) -> str`, mirroring
  `routing/report.py`: `try backend.generate_text(SURVEILLANCE_BRIEF_SYSTEM_PROMPT,
  prompt) except BackendUnavailable: return _fallback_brief(...)`.
- `SURVEILLANCE_BRIEF_SYSTEM_PROMPT`: "You are a public-health surveillance
  assistant for a district officer. Given a detected signal (village, disease,
  observed vs baseline, affected age group) write a ≤80-word situational brief:
  what is happening, how unusual it is, and the recommended action. Plain text,
  no alarmism, invent nothing."
- **Called only when an alert fires** (ties LLM cost to real events — a genuine
  agent behaviour and a clean cost story for the paper).
- Deterministic `_fallback_brief` template so CI/headless always works.

### 1.8 Agent orchestrator (`agent.py`) + state
`SurveillanceAgent(backend=None, store=None)`:
- `run(now=None, window="week") -> SurveillanceReport`:
  **perceive** (ingest) → **analyze** (analytics) → **detect** (signals) →
  **decide** (alert_policy) → for each alert: **brief** (LLM) + **recommend** →
  **update watchlist** (persistent state across runs) → optionally **emit triage
  priors** (§1.9).
- `SurveillanceReport`: `generated_at`, `village_table`, `signals`, `alerts`
  (with brief + action), `watchlist`.
- Watchlist = standing goal/state ("surface emerging outbreaks early"), updated
  each run — gives the agent memory.

### 1.9 Triage feedback hook (`feedback.py`) — OPTIONAL, flag-gated (default OFF)
- `active_outbreak_priors(area) -> dict[disease, float]`: for active alerts in
  `area`, a small prior-boost factor.
- Consumed by the triage classifier **only if** `SURVEILLANCE_FEEDBACK_ENABLED`
  (default `False`): one guarded multiply on candidate scores at a single, clean
  call site. Off by default so the triage path and its 605 tests are untouched;
  documented as the **bidirectional-coordination demonstration** (Agent 2 → Agent
  1) for the paper's USP. We decide at the end whether to enable it.

### 1.10 Synthetic demo stream (`demo_stream.py`)
- `generate_demo_stream(weeks=8, seed=42) -> list[SurveillanceCase]`: endemic
  baseline across the 5 existing villages (reuse `routing/demo_facilities` coords)
  + **one injected outbreak** (e.g. a Gastroenteritis spike in one village among
  under-5s in the last 2 weeks) so detection visibly fires.
- Clearly labelled **demonstration data** in the UI and docstring (consistent
  with the sandbox-dataset framing).

### 1.11 UI — "🏛️ Official / Surveillance" tab (`streamlit_app.py`)
- New `render_surveillance_view()`:
  - a "Load demonstration stream" control (live store is usually empty in a POC);
  - **village risk table** (`st.dataframe`);
  - **trends** (`st.bar_chart` / `st.line_chart`, fed plain dicts — no pandas);
  - **Active alerts**: per alert, a card with level, village, disease, observed
    vs baseline, the **LLM brief**, and the **recommended action**;
  - watchlist; a "demonstration data" caption.
- Add `tab_official` to the `st.tabs([...])` call at `streamlit_app.py:1426` and
  render it (always visible, no case required — like the Results tab).

### 1.12 Tests (`tests/test_surveillance.py`)
- analytics counts/age-band correct;
- detector **fires on the injected outbreak**, **silent on a flat stream**;
- alert policy bands (INFO stays silent, ALERT surfaces);
- `recommend_action` covers all 41 diseases;
- brief **fallback** when backend raises `BackendUnavailable` (stub backend);
- `agent.run()` end-to-end on the demo stream → ≥1 alert on the outbreak, 0 on
  flat data, watchlist updated;
- feedback priors empty when flag off; present for active alerts when on;
- **structural/extensibility test:** `app.surveillance` imports nothing from the
  triage decision internals except `schemas` and `case_store` (enforces low
  coupling — doubles as Pillar-4 evidence).

---

## 2. Composed ablation harness (`tests/evaluate_composed.py`)
The "money figure": one metric across the whole stack on the same cases.
- Stages: `C0 LLM-only` → `C1 +structured extraction (NB top-1)` →
  `C2 +abstain gate` → `C3 +worst-case guard (full)` → `C4 +follow-up loop`.
- Reuse: `evaluate_oos._FoldModel/_score_variant/_VARIANTS`, the
  `evaluate_llm` LLM path (C0), the `evaluate_followup` loop (C4).
- Metric(s) tracked across stages: **misses-urgent**, **over-calling**,
  **confidently-wrong** (choose one as the headline line).
- Output `eval_composed_results.json` + a staged table; C0 is Ollama-gated, the
  rest deterministic.
- Test `tests/test_composed_eval.py` (monotonic improvement where expected;
  stages are distinct configs).

## 3. Systems-cost measurement (`tests/evaluate_cost.py`)
- Per case: **# LLM calls** (extraction 1; follow-up phrasing ≤3; doctor report
  conditional; surveillance brief event-driven) and **per-stage latency** (wrap
  with timers) over a few representative messages (Ollama-gated), plus the
  deterministic stages' timings.
- Output `eval_cost_results.json`; honest "latency is machine-dependent, read the
  per-stage breakdown and call counts" note.

## 4. Practical / SDG impact results (`tests/evaluate_impact.py`) — A, B, C
- **A — avoided trips + right facility.** Run the OOS short-message set through
  `classify`; label distribution → "stay home" (MILD/MODERATE) vs "go now"
  (EMERGENCY/SEVERE) = potential avoided-trip share; then over villages×urgencies
  (reuse routing) the **naive-nearest vs nearest-eligible mismatch rate** + ETA.
  An **illustrative cost box** (trip-cost/distance constants, clearly labelled a
  model). SDG 3.8 / 10.
- **B — language reach/inclusion.** From the end-to-end lay renderings grouped by
  English/Hindi/Hinglish (reuse `evaluate_end_to_end` building blocks): share
  correctly handled per language; "an English-only pipeline drops the non-English
  share." SDG 10. (Ollama+FAISS-gated; a deterministic fixture-coverage count as
  a fallback number.)
- **C — caller effort.** Read `eval_followup_results.json`: ~2 questions to a
  decision, ~75% resolved → "answer two yes/no questions vs a full form." SDG 3.8.
  Deterministic.
- Output `eval_impact_results.json`; test `tests/test_impact_eval.py`.

## 5. Extensibility evidence
- `tests/test_extensibility.py`: structural assertions (Agent 2 low coupling;
  backends/translators/routing providers are swappable behind interfaces) +
  a **touch-count** ("to add Agent 2: new package + 1 UI tab, 0 triage-core
  edits with feedback off").
- **Architecture diagram** (mermaid) in `EVALUATION_REDESIGN.md` showing the
  seams and Agent 2 plugging into the case stream.

## 6. Docs & CODEBASE_CONTEXT sync (repo rule — at commit)
- `EVALUATION_REDESIGN.md`: add **Pillar 5 (practical/SDG: A/B/C)**, the composed
  figure, cost, and extensibility rows.
- `CODEBASE_CONTEXT.md`: new engine **E13 surveillance** (§5 inventory), §3 repo
  map, §6 pipeline note, §13 tests, §14/§16.2 state, **§0 SYNC_STAMP + §18
  SYNC_LOG** — in the same commit as the code.
- `PAPER_EVIDENCE.md`: add the Agent-2 + Pillar-5 claims (when we touch the paper).

---

## 7. Build order & milestones
- **M1 — Agent core (headless):** §1.1–§1.8 + §1.10 → `agent.run()` produces
  alerts+briefs on the demo stream, verified by test. *(+ §1.2 case_store field.)*
- **M2 — Agent UI:** §1.11 Official tab; Streamlit boots HTTP 200, alert cards
  render.
- **M3 — New results:** §2 composed, §3 cost, §4 impact (A/B/C).
- **M4 — Extensibility + feedback:** §5 evidence; §1.9 feedback hook (flag off).
- **M5 — Tests + docs:** full suite green, §6 doc/context sync, Streamlit boot
  check.

Items in M3/M4 are independent of the agent, so nothing blocks M1.

## 8. Dependencies
stdlib + `numpy` (present) + optional `scikit-learn` (present, DBSCAN only). **No
new dependency.** Charts use Streamlit's native chart calls with plain dicts (no
pandas).

## 9. Acceptance criteria
- Agent: on the demo stream, exactly the injected outbreak surfaces as an alert
  with a brief + action; a flat stream produces none; brief works with Ollama up
  **and** falls back with it down.
- Composed/cost/impact harnesses each `--write` a tracked JSON and are covered by
  a test; deterministic ones reproduce byte-for-byte.
- Triage path unchanged: all existing tests still pass (feedback flag off).
- Streamlit boots headless (HTTP 200) with the new tab.
- `CODEBASE_CONTEXT.md` updated in the same commit.

## 10. Risks & guardrails
- **Feedback loop** is **off by default** behind one flag; decide at M4 whether
  to enable or leave as a documented hook.
- **LLM brief** only on alerts, with a deterministic fallback → no CI breakage,
  controlled cost.
- **Synthetic stream** is labelled demonstration data everywhere it appears.
- **case_store age column** is additive/nullable → old DBs keep working.
- Keep detection simple (stats) — resist over-modelling (consistent with the
  "models are standard, the framework is the contribution" stance).

---

**One-line summary:** build a self-contained `app/surveillance/` agent that reads
the existing case stream, detects outbreaks with simple statistics, and — only
when something is genuinely out of hand — autonomously briefs and advises a
government official via the LLM; then add the composed-ablation, systems-cost,
and practical/SDG(A/B/C) result harnesses, plus the extensibility evidence. All
additive; the triage core is untouched.
