# Age-handling fixes — `young-infant-escalation` branch

_For teammates reviewing this branch before it merges. Full commit-by-commit history:
`git log --oneline origin/main..young-infant-escalation` and `git show <sha>` for any commit below._

Branch state: 4 commits ahead of `origin/main` (this doc is a 5th). Not pushed.

## Summary

This branch closes the "any age" gap in `rules_engine.classify()` safely. It adds an
explicit young-infant (<2 months) escalation path instead of quietly handing neonates to
adult/child logic; it fixes a real bug where a patient of **unknown** age was silently given
a *pediatric* classification; it fixes **broken age-phrasing normalization** that was letting
genuine young infants ("3 weeks old", "6 week old") skip the young-infant safety gate
entirely because the extractor read "3 weeks" as "3 months"; and it fixes a
**data-leakage bug** in the adult-path probability calibration that made the abstention
threshold meaningless. No new clinical rules were invented — where the project has no
validated ruleset (young infants) or unverified data (`disease_severity.csv`), the code now
says so honestly rather than guessing.

## What changed, and why

### `806b2c6` — Route young infants (<2mo) to an explicit escalation

**Gap.** `classify()` handled `age_months < 2` by returning
`condition="AGE_OUT_OF_MODULE_SCOPE"` — the same generic string used for any out-of-range
age. That conflated "we have no usable inputs" with "we have a young infant in front of us
but no validated WHO IMCI young-infant ruleset exists to classify them." A reviewer reading
the output couldn't tell a neonate apart from a data error.

**Change.** `app/rules_engine.py`, `app/schemas.py` (+ tests). New distinct
`condition="YOUNG_INFANT_NO_VALIDATED_RULESET"` under the existing
`INCOMPLETE_ASSESSMENT` label (no new enum value — `streamlit_app.py` indexes three dicts
by label with no default, so a new label would crash the dashboard). The branch runs
**first** in `classify()`, before dataset-routing is even computed, so a young infant can
never reach the adult disease classifier (which is built from a CSV with zero neonatal
conditions). A monkeypatch spy test asserts that directly.

**Before / after.**
- `age_months=1`, febrile → **before:** `AGE_OUT_OF_MODULE_SCOPE` (indistinguishable from junk input).
  **after:** `YOUNG_INFANT_NO_VALIDATED_RULESET` — a clear "escalate to human / neonatal-capable facility" signal.
- No clinical thresholds were added — that's still out of scope pending validated inputs.

### `79c78c4` — Distinguish `INSUFFICIENT_SYMPTOM_DATA` from `UNCERTAIN_DIAGNOSIS`

**Bug.** The adult/dataset path's abstention branch hardcoded
`condition="UNCERTAIN_DIAGNOSIS"` for **every** abstention — even when it had already
computed that the real problem was "no recognizable symptoms were provided at all." The
follow-up layer needs that distinction: "ask the caller for symptoms" (fixable) vs. "the
symptoms are genuinely ambiguous" (not fixable by asking).

**Change.** `app/rules_engine.py`, one branch in `classify_via_dataset()`. When there are
zero candidate diseases → `INSUFFICIENT_SYMPTOM_DATA` with `missing_fields=["symptom_tokens"]`;
otherwise → `UNCERTAIN_DIAGNOSIS`. Mechanical; no statistical change.

**Before / after.**
- Adult case, message contained no parseable symptom → **before:** `UNCERTAIN_DIAGNOSIS`
  (looks like a hard clinical call). **after:** `INSUFFICIENT_SYMPTOM_DATA` → the pipeline
  asks a symptom-clarifier follow-up.

### `109e966` — Leak-free k-fold calibration for the adult-path classifier

**Bug.** The adult classifier's abstention thresholds (τ = confidence floor, δ = top-two
gap floor) were fit from a single 80/20 split over **raw CSV rows**, scored by a model
trained on **all** rows including the held-out ones. Two compounding leaks:
1. `data/disease_symptoms.csv` has 4,920 rows but only **304 unique** (disease, symptom-set)
   profiles — 94% of rows are exact duplicates (one profile repeats up to 90×). A row-level
   split puts near-identical copies on both sides.
2. Held-out rows were scored by a model that had trained on those same rows. Measured
   directly: 930/984 held-out rows scored ~1.0 confidence by memorization; the ~54 genuinely
   novel rows were wrong 94% of the time. Youden's J, fit against that artificial split,
   collapsed τ to exactly 1.0.

**Change.** `app/disease_classifier.py` only. Stratified **5-fold cross-validation over the
304 unique profiles**: each fold refits the Naive-Bayes model on its training profiles only
and scores its held-out profiles with that fold's model. The 304 out-of-fold triples — each
scored by a model that never saw that profile — are what isotonic calibration and τ/δ are
now derived from. The **production** model is unchanged (still fit on all rows); only the
calibration curve and thresholds moved.

**Before / after.**
- τ still lands at **exactly 1.0** — but now legitimately. On the 304 leak-free profiles:
  true accuracy 93.4%, and Youden's J narrowly (0.873 vs 0.850) prefers the zero-false-
  positive operating point. δ moved 0.077 → 0.109. At τ=1.0 / δ=0.109: 81.6% of profiles
  answer with **zero wrong answers**, 18.4% honestly abstain.
- `chest_pain + breathlessness + sweating`, which ties Pneumonia/Tuberculosis at ~0.35 each
  → **before:** confident guess. **after:** abstains.

### `52f0dd1` — Eliminate implicit age defaults; fix age-phrasing normalization + a <2mo safety net

**Gaps (four of them).**
1. `ExtractedCase.age_group` defaulted to `AgeGroup.UNKNOWN`. "Never asked" was
   indistinguishable from "asked, unknown."
2. `RegexBackend` emitted a literal `"age_group": "unknown"` despite having **no age logic
   at all**.
3. `_sanitize_enum_fields` coerced any unrecognized age_group string (`"teenager"`,
   `"baby"`) to the literal `"unknown"` — manufacturing a value out of garbage.
4. When age was fully unknown, `classify()` **silently fell through to the pediatric IMCI
   path** — an 80-year-old with a cough could be classified `MILD / COUGH_OR_COLD` on a
   child algorithm.

   Plus a normalization bug: the LLM extractor read **"3 weeks old" as `age_months = 3`**
   (copied the number, ignored the unit). Every week-phrased neonate landed ≥ 2 months and
   **bypassed the young-infant gate from commit `806b2c6` entirely** — the single
   highest-risk failure in the pipeline. "newborn" extracted as `age_months = null`.

**Change.** `app/schemas.py` (`age_group` → `Optional[AgeGroup] = None`),
`app/agent1_extraction.py` (prompt gains explicit year/week/day→months conversion rules and
"newborn"→0; `RegexBackend` and the sanitizer stop inventing values; **new deterministic
`_apply_infant_age_floor()`** that runs a regex on the raw text, independent of the LLM, and
forces `age_months` into the correct <2 range for "N week(s)/day(s) old" / "newborn" /
"just born" phrasing), `app/rules_engine.py` (new
`condition="AGE_UNKNOWN_CANNOT_ROUTE"` when age is fully unresolved — runs before any
route), `app/followup_policy.py` + a new age-clarifier follow-up question,
`app/routing/report.py` + `streamlit_app.py` (guard the now-nullable `age_group`). New
`tests/test_age_handling.py` (68 cases incl. a full replay of the audited phrasing table
and Ollama-gated live checks).

**Before / after** (real `llama3.2:3b` extraction → `classify()`):

| message | before | after |
|---|---|---|
| `6 week old infant, lethargic` | `age_months=6` → pediatric IMCI path | `age_months=1` → **`YOUNG_INFANT_NO_VALIDATED_RULESET`** |
| `the baby is 3 weeks old and not feeding` | `age_months=3` → pediatric IMCI path | `age_months=0` → **`YOUNG_INFANT_NO_VALIDATED_RULESET`** |
| `newborn with yellow eyes` | `age_months=null` → pediatric IMCI path | `age_months=0` → **`YOUNG_INFANT_NO_VALIDATED_RULESET`** |
| unknown-age patient, febrile cough | `MILD / COUGH_OR_COLD` (child algorithm, silently) | **`AGE_UNKNOWN_CANNOT_ROUTE`** → age-clarifier follow-up |
| `2 month old baby coughing` | `age_months=2` (correct) | `age_months=2` — still correctly **outside** the young-infant range |

## Known remaining gaps

**Be aware of these before relying on this branch for anything past a prototype demo.**

- **`data/disease_severity.csv` has no verified clinical provenance.** The disease→severity
  mapping the adult path uses to turn a diagnosis into EMERGENCY/SEVERE/MODERATE/MILD is a
  curated default, not a sourced classification. `Hypoglycemia` is rated **SEVERE, not
  EMERGENCY**, and `Malaria` is rated **MODERATE** — both look under-rated and could
  under-triage a real patient. This file needs clinical review before it is trusted.
  **Untouched by this branch** (explicitly out of scope).
- **4 pre-existing test failures remain**, confirmed unrelated to this branch (same 4 names
  before and after every commit here):
  - `tests/test_dataset_symptom_matching.py::test_pipeline_dataset_tokens_feed_disease_classifier_end_to_end`
  - `tests/test_rules_engine_dataset_routing.py::test_age_months_60_plus_routes_to_dataset`
  - `tests/test_rules_engine_dataset_routing.py::test_age_group_adult_with_no_age_months_routes_to_dataset`
  - `tests/test_rules_engine_dataset_routing.py::test_age_group_elderly_with_no_age_months_routes_to_dataset`

  Root cause: these assert a *confident* diagnosis for symptom sets that are genuinely tied
  between multiple diseases under honest, leak-free scoring (e.g. Malaria vs Tuberculosis on
  fever + chills + sweating + vomiting). After `109e966` the calibration **correctly
  abstains** on those ties. This is the classifier behaving well; the **tests still encode
  the old (over-confident) expectation** and should be rewritten separately — that's a
  product decision about desired tie-breaking, not a bug to patch here.
- **The deterministic <2-month safety net is not exhaustive.** It covers digit + unit
  phrasing (`"3 weeks old"`, `"10 days old"`, `"6-week-old"`) and the words
  `newborn / new-born / just born / just delivered / neonate`. It does **not** cover:
  spelled-out numbers (`"three weeks old"`), relative phrasing (`"born 3 weeks ago"`,
  `"delivered last month"`), or ages written in non-English scripts. Do not assume full
  coverage — the LLM prompt is the first line for those; the regex is only a backstop for
  the digit+unit cases the model was observed to get wrong.
- **AgeGroup case-folding / lay-term recovery was identified but deliberately not done.**
  `"ADULT"` → `"adult"` (case), `"baby"` → `"infant"` (lay term) are recoverable near-misses
  that currently become `None`. Left as a possible follow-up; recovering them belongs in
  extraction, not in the sanitizer.

## Test suite state

| | before this branch | after `52f0dd1` |
|---|---|---|
| passed | 298 | **368** (+70: `test_age_handling.py` + updated `test_followup_policy.py`) |
| failed | 4 | **4** (identical names — see gaps above) |
| xfailed | 2 | 2 |

The 4 failures are the adult-path classifier correctly abstaining on genuinely ambiguous
symptom overlaps, against tests that still assert the pre-calibration-fix confident answer.
Not a regression; not introduced by this branch.
