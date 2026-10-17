# Beyond Triage — Paper Evidence Map (Stage 7)

> Assembles the implementation into paper-ready form: every claim the paper makes
> is mapped to the code mechanism that implements it and the evaluation that
> proves it, with a final logicality pass confirming no claim outruns the code.
> Grounded against the `9th-october` tree after Stages 0–6. Code is authoritative;
> see `CODEBASE_CONTEXT.md` for the verified internals and `triage-poc/rule_engine_design_final.md`
> for the primary design reference.

## 1. USP and scope-of-claims contract

**USP:** the *end-to-end agentic workflow* and the *coordination within and between
modules* for one use case — CHW-assisted triage in low-resource settings, framed
under UN SDG-3. The contribution is a software/CS framework: how free-text
extraction, a safety-calibrated abstain decision, urgency prioritization, and
resource-aware routing are **coordinated**, with *active follow-up* as the
mechanism that links uncertainty to action. It is **not** a claim about the
rule-engine model internals or the specific models (which are standard,
interpretable, defensible choices — confirmed by the out-of-sample bake-off).

**We claim:** (1) framework/coordination correctness; (2) selective-prediction
safety properties (abstain-and-ask, near-zero under-triage); (3) robustness to
language variation. **We do not claim** clinical diagnostic accuracy — stated as a
limitation and future work. Every mechanism below is made true in code (or
reframed honestly) before the paper claims it.

### The four computational problems (paper spine)
1. robust free-text → structured extraction under language variation;
2. classification with a safety-calibrated **abstain** decision;
3. urgency prioritization;
4. resource-aware routing;

with **active follow-up** as the mechanism that links uncertainty (2) to action.

## 2. Workflow & coordination (the USP, diagrammed)

```mermaid
flowchart TD
    A["Free-text message<br/>(English / Hindi / Hinglish)"] --> B["Agent 1 — LLM extraction<br/>Ollama llama3.2:3b<br/>returns a symptoms LIST"]
    B --> C["FAISS disambiguation<br/>code-mixed clause split<br/>→ dataset tokens"]
    C --> D{"age routing"}
    D -- "2–59 mo / child" --> P["Pediatric IMNCI slice<br/>danger signs → EMERGENCY"]
    D -- "≥60 mo / adult" --> E["Conformal gate (LAC, α=0.10)<br/>non-vacuous prediction set"]
    E -- "set = 1" --> F["CONFIDENT → AHP within-tier ESI"]
    E -- "set 2–3" --> G["UNCERTAIN → worst-case label"]
    E -- "set ≥4" --> H["ABSTAIN → worst-case if EMG/SEV in set"]
    G --> Q["ACTIVE FOLLOW-UP<br/>ask a clinically-relevant yes/no<br/>(coordination: uncertainty → action)"]
    H --> Q
    Q -- "reply" --> E
    P --> R["Resource-aware routing (Agent 3)<br/>nearest eligible facility + ETA"]
    F --> R
    G --> R
    H --> R
    R --> S["Doctor handoff report<br/>(dataset dx suppressed on<br/>danger-sign / abstain paths)"]
    S --> OUT["Per-role views: Caller · ASHA · Doctor · Rule-Engine/auditor"]
```

The arrows **Q → E** (follow-up re-enters classification) and the worst-case /
danger-sign escalation across every branch are the inter-module coordination the
paper is about: no single module decides safety alone.

## 3. Claims → code → evidence

| Claim | Code mechanism | Evidence / metric |
|---|---|---|
| **(i) Robust extraction under language variation** | `agent1_extraction`: `symptoms` LIST contract + `_normalize_symptom_fields`; `_SYMPTOM_CLAUSE_SPLIT` code-mixed conjunctions (aur/और/…); D-SEV null-severity fix; terse-message safety net | `tests/test_extraction_coverage.py` (36). **Quantified (§3.4):** on the labelled Hindi/Hinglish/English fixture, schema-valid **100%**, expected-token recall **100%**, failure **0%** (was 86.7 / 80.0 / 13.3 — a single schema bug accounted for every failure). **End-to-end (§3.5):** token recall 96.2% on real code-mixed lay phrasings. |
| **(ii) Safety-calibrated abstain** (CP non-vacuous) | `disease_classifier.classify_with_cp`: split conformal / LAC on the emergency-boosted NB posterior, α=0.10, q̂≈0.022 | `tests/test_cp_calibration.py` (7): q̂<1; set shrinks with evidence (6→3→2); posteriors non-degenerate; ECE≈0/Brier≈0.009 reported. |
| **Does not miss emergencies** (under-triage bounded by α) | `rules_engine`: worst-case severity on UNCERTAIN/ABSTAIN-with-EMG/SEV; `_apply_danger_sign_override` across all adult CP decisions | `tests/test_safety_guardrail.py` (43, build-failing); **OOS short-message under-triage = 0.0 (0/142)**, vs **9.2%** for an independent NB top-1 baseline and **6.3%** when worst-case labelling is ablated (`evaluate_oos.py`, `test_eval_harness.py`). See §3.1 — this is a *design guarantee*, not an empirical discovery. |
| **Asks, not guesses** (uncertainty → a relevant question) | `followup_question_selector`: `_relevant_tokens` + `_NOT_CALLER_ASKABLE`; real CP posteriors (not a uniform prior); `agent1_extraction` Flow C loop with denials recorded | **Quantified in §3.3** — absence-aware, the loop resolves **74.8%** of uncertain cases in 2.15 questions and cuts over-triage 21.6% → 5.6%, beating random (58.3%) and frequency (42.7%). Question quality: generic fallback phrasing **53.6% → 0%**, unanswerable tokens selected **→ 0** (`test_followup_question_quality.py`). |
| **(iii) Urgency prioritization** | `emergency_scorer` AHP: within-tier ESI prioritizer (not a label); weights, λmax and CR **derived from the pairwise matrix at import** (`_derive_ahp_weights`, power iteration) rather than transcribed | `tests/test_cp_ahp_adaptive.py`: weights equal the principal eigenvector; CR = **0.0198** computed (the previously quoted 0.0205 was not reproducible); CR < 0.10 enforced at import. **Now measured (§3.6):** within-tier pair discrimination **0.659** over 1025 cases, so AHP carries information the tier does not; danger sign always → ESI-1. No clinical-correctness claim. |
| **(iv) Resource-aware routing** | `routing/router.route`: eligibility filter + haversine scoring | **Now measured (§3.6):** **1.000** optimality vs exhaustive brute force over all 5 villages × 2 urgency levels; EMERGENCY eligibility and URGENT open-hours respected 1.000; mean ETA 12.0 min. Scope: 10-facility synthetic network, mock distance provider. |
| **Robust to which symptoms get mentioned** | bag-of-token scoring + worst-case set labelling | OOS **subset robustness = 86.6%** over 277 profiles (`evaluate_oos.py`): two different 2–4 symptom subsets of the same patient agree on the tier. Honest, imperfect, and therefore informative. |
| **Safe doctor handoff** | `routing/report._dataset_diagnosis_is_primary` gates the dataset dx | `tests/test_stage4_followup_ahp_report.py`: convulsing-infant report omits "Cervical spondylosis / heating pad". |

On **F1 = 1.00**: **not reported in the paper at any point.** It is produced by the
legacy in-sample scripts (`evaluate_{100_cases,balanced,1000}.py`) scoring full
symptom rows drawn from the very data the Naive Bayes was fit on, so a perfect
score is a property of the setup rather than of the system. Those scripts are
retained **only as regression tests for pipeline wiring**. Every number the paper
quotes comes from `evaluate_oos.py`.

## 3.1 The under-triage result is a guarantee, not a measurement

This distinction decides whether the central claim survives review, so it is
stated explicitly rather than left for a reader to discover.

The framework labels a multi-disease prediction set with the **worst-case
(most severe) tier in that set**. If the true disease lies inside the set, the
surfaced tier is therefore at least as severe as the truth, and under-triage is
impossible. Under-triage can only occur when the conformal set *misses* the
truth, which split conformal prediction bounds:

> **Property.** With a conformal set `S(x)` of marginal coverage ≥ 1 − α and
> worst-case-tier labelling over `S(x)`:
> `P(under-triage) ≤ P(y ∉ S(x)) ≤ α`.

So "0/142 under-triage" is **entailed by** the measured coverage of 1.0 — it is
not independent evidence. Presented correctly, this is *stronger* than an
empirical zero: it is a distribution-free safety bound that holds by
construction, tunable through α. What the harness measures empirically is the
**premise** (coverage) and the **price** (over-triage 21.6%).

### Ablations and baseline (same held-out short messages, `evaluate_oos.py`)

| system | boost | CP set | worst-case | under-triage | over-triage | confident |
|---|:--:|:--:|:--:|---|---|---|
| **full_system** | ✓ | ✓ | ✓ | **0.0%** | 21.6% | 66.1% |
| ablation_no_emergency_boost | – | ✓ | ✓ | 0.0% | 21.6% | 65.8% |
| ablation_no_worstcase | ✓ | ✓ | – | 6.3% | 6.8% | 66.1% |
| baseline_flat_nb_top1 | – | – | – | 9.2% | 6.8% | 100% |

Three things this table establishes, each of which a reviewer would otherwise
have to take on trust:

1. **Worst-case set labelling is the mechanism that carries the safety result.**
   Removing it alone moves under-triage 0.0% → 6.3%.
2. **The baseline is genuinely independent** — plain NB argmax with none of the
   framework's machinery — and under-triages at 9.2%. (An earlier version of
   this harness scored the "baseline" and the "ablation" with *identical code*
   and reported one number, 6.3%, twice; `test_comparators_are_distinct_computations`
   now makes that failure mode build-breaking.)
3. **`EMERGENCY_SAFETY_FACTOR = 3.0` is redundant.** Ablating the emergency
   class-imbalance boost changes neither under-triage nor over-triage — the
   worst-case rule already subsumes it. This is reported as a negative result
   and is grounds for removing the constant rather than defending it.

The honest cost is visible in the same table: the safety guarantee is bought
with over-triage (21.6% vs 6.8%), and the two systems that achieve 0%
under-triage are exactly the two that pay it.

## 3.2 Sensitivity of the free constants (`tests/evaluate_sensitivity.py`)

Every hand-set constant that materially affects safety is reported as a curve
rather than defended as a value. Both sweeps use the same held-out set.

**α (conformal miscoverage).** The bound `P(under-triage) ≤ α` is checked at
every level, not just production.

| α | coverage | under-triage | over-triage | confident | mean set | LAC-empty fallback | bound holds |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.984 | 0.000 | 0.173 | 0.691 | 1.86 | **0.000** | ✓ |
| 0.02 | 0.961 | 0.007 | 0.142 | 0.743 | 1.56 | 0.000 | ✓ |
| 0.05 | 0.957 | 0.028 | 0.124 | 0.813 | 1.63 | 0.135 | ✓ |
| **0.10 (prod)** | 1.000 | 0.000 | 0.216 | 0.661 | 2.13 | **0.339** | ✓ |
| 0.15 | 1.000 | 0.000 | 0.235 | 0.612 | 2.20 | 0.388 | ✓ |
| 0.20 | 1.000 | 0.000 | 0.265 | 0.582 | 2.24 | 0.418 | ✓ |
| 0.30 | 1.000 | 0.000 | 0.265 | 0.582 | 2.24 | 0.418 | ✓ |

Two disclosures this produces, both of which would otherwise be found by a
reviewer rather than by us:

- **The LAC-empty fallback rate must be reported.** When the conformal set is
  empty the system returns *every surviving candidate*. At production α=0.10
  that happens on **33.9%** of cases — exactly the ask-or-abstain rate. So at
  α=0.10 the gate is effectively binary (confident singleton, or "return
  everything"), and the coverage of 1.000 is partly an artefact of the
  fallback. Describing that behaviour as conformal filtering would overstate it.
- **The production α is dominated.** α=0.01 is at least as good on every
  reported axis and strictly better on two: same 0.000 under-triage, lower
  over-triage (0.173 vs 0.216), higher confident rate (0.691 vs 0.661), and it
  removes the degenerate fallback entirely (0.000 vs 0.339). Nominal coverage
  is slightly under-attained at the tightest α (0.984 vs 0.99 at α=0.01;
  0.961 vs 0.98 at α=0.02) because conformal coverage here is marginal and
  finite-sample with n=304 calibration points.

**EMERGENCY_SAFETY_FACTOR** (swept with the conformal gate **re-calibrated** at
each value, so calibration and inference stay exchangeable):

| factor | coverage | under-triage | over-triage | confident |
|---|---|---|---|---|
| 1.0 (off) | 1.000 | 0.000 | 0.210 | 0.661 |
| 1.5 / 2.0 | 1.000 | 0.000 | 0.210 | 0.661 |
| 3.0 (prod) | 1.000 | 0.000 | 0.216 | 0.661 |
| 5.0 / 10.0 | 1.000 | 0.000 | 0.216 | 0.655 |

**Negative result, reported as such:** the emergency class-imbalance boost does
not reduce under-triage at *any* value — worst-case set labelling already does
that job — and at higher values it slightly *increases* over-triage. The
constant is not load-bearing and is a candidate for removal.
`test_sensitivity.py::test_emergency_boost_is_not_load_bearing_for_safety`
locks the finding.

## 3.3 The adaptive follow-up loop (`tests/evaluate_followup.py`)

The loop is the paper's claimed core contribution and previously had **no**
quantitative evaluation. It is now measured out-of-sample on the same held-out
cases, with a 3-question budget.

**Answers are ground truth, not simulation.** Each held-out case is a real
patient profile with a complete recorded symptom list; the caller is shown only
a 2–4 symptom subset. When the system asks "does the patient have X?", the
answer is read off that same real profile. Nothing is generated or imagined.

**The LLM's role here.** The question is *chosen* by `followup_question_selector`
from dataset statistics (emergency-first, else information gain over the
prediction set); the LLM only renders the chosen token as a sentence. Phrasing
cannot change which symptom is asked about, so the selection policy is
evaluated deterministically.

| scoring | policy | resolved | mean Q | set before → after | resolved correctly | under | over |
|---|---|---|---|---|---|---|---|
| present-only | **selector** | 0.204 | 2.69 | 4.33 → 3.31 | 0.204 | 0.000 | 0.216 |
| present-only | frequency | 0.107 | 2.97 | 4.33 → 3.36 | 0.068 | 0.007 | 0.191 |
| present-only | random | **0.291** | 2.68 | 4.33 → 3.43 | 0.272 | 0.000 | 0.142 |
| **absence-aware (shipped)** | **selector** | **0.748** | 2.15 | 4.33 → **1.67** | **0.670** | 0.021 | **0.056** |
| absence-aware | frequency | 0.427 | 2.62 | 4.33 → 1.97 | 0.340 | 0.007 | 0.105 |
| absence-aware | random | 0.583 | 2.45 | 4.33 → 1.94 | 0.544 | 0.007 | 0.086 |

This is the paper's most interesting result, and it is a two-part finding:

1. **Under present-only scoring the clever selector loses to random** (0.204 vs
   0.291 resolved). The cause is a genuine design inconsistency: the
   information-gain objective explicitly models `P(absent)`, but the Naive
   Bayes scored *present* tokens only, so a "no" taught the classifier nothing.
   Half the information each question was selected to obtain was discarded, and
   the optimisation was wasted.
2. **Make a "no" informative and the selector dominates.** A Laplace-smoothed
   Bernoulli absence term raises resolution **0.204 → 0.748** (3.7×), correct
   resolution 0.204 → 0.670 (3.3×), cuts questions 2.69 → 2.15, and the
   selector now beats both random (0.583) and frequency (0.427).

The loop's triage effect is a favourable trade: over-triage falls **21.6% →
5.6%** because resolved cases no longer need worst-case escalation, at the cost
of 2.1% under-triage (still within the α bound), with the truth retained in the
prediction set 97.4% of the time.

**This is the shipped configuration, not a variant.** Absence-aware scoring is
in production: `ExtractedCase.absent_symptom_tokens` is populated by
`record_followup_answer` on a parsed "no" and consumed by
`disease_classifier._nb_posteriors_with_absence`.

### A pathology found while shipping it — and why the fix matters

The obvious implementation applies the absence term across all 41 diseases.
That is correct Bayes for a true generative model, but this Naive Bayes scores
presence only, so the two terms are not commensurable: with a few denials the
penalties on the genuinely plausible diseases outweigh the thin positive
signal, the posterior flattens, and diseases the denied symptoms have nothing
to do with rise simply because they were penalised least. Measured directly,
`["cough"]` with `chest_pain`/`phlegm`/`high_fever` denied produced a
**28-disease** prediction set containing Paralysis and positional vertigo,
which the worst-case rule then escalated to **EMERGENCY**. Denials made the
system *less* certain and *more* alarmed — the exact opposite of their purpose.

The fix restricts the penalty to the candidate pool that positive evidence
supports, i.e. it conditions on the differential the patient's actual symptoms
imply. A "no" can then only narrow or re-rank, never introduce an unsupported
disease and never escalate the tier — properties asserted in
`tests/test_absence_evidence.py`. It also *improved* the metrics over the
unrestricted version (resolution 0.718 → 0.748, final set 1.97 → 1.67).

Worth stating plainly in the paper: the naive version would have passed a
casual review, because its aggregate numbers looked good. Only a monotonicity
check on a single hand-built case exposed it.

### Question quality

Selection was previously driven purely by discriminative power, with 32
templates over a 131-token vocabulary. The result was that **53.6% of questions
fell back to a generic "Is the patient experiencing {token}?"** and the loop
asked a layperson things like *"Is the patient experiencing acute liver
failure?"*, *"...coma?"*, *"...receiving blood transfusion?"* — statistically
sound, clinically incoherent, and unfixable by rewording because the chosen
*token* is not a reportable symptom. Two mechanisms now prevent it:
`_NOT_CALLER_ASKABLE` excludes 12 diagnosis / history / instrumented tokens
from the candidate pool, and templates cover all 119 askable tokens. Measured
after: generic fallback **0%**, unanswerable tokens selected **0**.

## 3.5 End-to-end: free text in, tier out (`tests/evaluate_end_to_end.py`)

Every other harness scores symptom *tokens*, which excludes the framework's own
first two stages and leaves no number for the composed system. This closes that
gap. The same folds and the same labelling logic are run twice per case —
ORACLE on ground-truth tokens, END-TO-END on tokens recovered from rendered
text — so the difference is attributable to extraction alone.

| rendering | n | token recall | exact match | under-triage | agreement w/ oracle | extraction cost (over-triage) |
|---|---|---|---|---|---|---|
| canonical (ceiling) | 304 | 99.3% | 98.0% | **0.000** | 99.7% | −0.006 |
| **lay (real code-mixed)** | 83 | **96.2%** | 89.2% | **0.000** | **97.6%** | **+0.053** |

In `lay` mode every symptom is replaced by a real caregiver phrasing from the
project's pre-existing labelled fixture, rotated across English, Hindi
(Devanagari) and Hinglish so each message is genuinely code-mixed. Profiles
lacking a labelled phrasing are skipped, never back-filled with the canonical
token.

**The headline: under-triage is 0.000 end-to-end in both modes.** The α-bound
survives real extraction through code-mixed lay language; what extraction costs
is precision (+5.3 points of over-triage, 2.4% label disagreement).

> **Read the delta, not the absolute rate.** The lay subsample is selected by
> fixture coverage rather than at random, so its absolute over-triage (0.842) is
> dominated by whichever diseases have labelled phrasings — the oracle on those
> identical rows is already 0.789. The harness emits
> `sample_is_representative` and `comparison_rule` to prevent that figure being
> quoted across modes.

## 3.6 Urgency prioritization and routing (`tests/evaluate_prioritization_routing.py`)

These two pillars previously had correctness tests but no measurement, so the
paper claimed four contributions while evidencing two.

| AHP (1025 cases) | result |
|---|---|
| Within-tier pair discrimination | **0.659** |
| Distinct scores per tier | 3–5 (range ≥ 2 in every tier) |
| Monotone in the severity attribute | yes |
| Danger sign → ESI-1 / score 10, always | yes |

The discrimination figure is the one that matters: AHP's declared role is
ordering patients *within* a tier, so a scorer that tied everyone would be a
relabelling of the tier it sits on. At 0.659 it strictly orders two-thirds of
same-tier pairs. **No clinical-correctness claim is made** — this dataset has
no ground-truth dispatch order.

| Routing (10 scenarios) | result |
|---|---|
| Optimality vs exhaustive brute force | **1.000** |
| EMERGENCY eligibility respected | **1.000** |
| URGENT open-hours respected | **1.000** |
| No eligible facility found | 0.000 |
| ETA mean / max | 12.0 / 36.8 min |

Scope: a 10-facility synthetic network with a mock haversine provider. This
evaluates the routing *logic*, not real-world travel.

## 3.7 Robustness and reply handling (`tests/evaluate_robustness.py`)

| metric | result |
|---|---|
| Repeat agreement (identical message ×3) | **1.000** |
| **Paraphrase agreement (3 real lay renderings)** | **0.750** |
| Reply not understood (37 definite forms) | **0.000** |
| Reply misread as the opposite answer | **0.000** |
| Hedges coerced into an answer (11 forms) | **0.000** |

Repeat and paraphrase are reported separately on purpose: the first isolates
LLM nondeterminism, the second is the language-robustness claim. Collapsing
them would let a perfect repeat score stand in for a language result — which is
exactly what the removed token-order metric did.

Hedge handling is a safety property, not a nicety: "maybe" / "pata nahi" must
stay unparseable so the caller is re-prompted rather than have an invented
clinical fact written into the case.

*Provenance:* the reply list is authored, so this measures **parser coverage
over documented reply forms**, not observed caller behaviour. Stated in the
output itself.

## 3.4 LLM baseline and reliability (`tests/evaluate_llm.py`)

**"Why not just prompt the LLM?"** — `llama3.2:3b` is asked for the same four
tiers on the same 304 held-out cases, given the same symptoms and a prompt that
states the tier definitions *and* the safety convention ("when uncertain,
choose the more severe"). It is not a strawman.

| system | under-triage | over-triage | exact-tier accuracy |
|---|---|---|---|
| **Framework (full)** | **0.000** | **0.216** | — |
| NB top-1 baseline | 0.092 | 0.068 | — |
| **LLM-only (llama3.2:3b)** | **0.169** | **0.827** | 0.454 |

The LLM-only baseline is simultaneously **unsafe and uninformative**: it misses
16.9% of emergency/severe cases while over-calling 82.7% of mild/moderate ones,
because it collapses onto a single answer — 251 of 304 replies are "SEVERE".
Prompted safety instructions do not produce calibrated triage; the structural
guarantee does.

**Reliability (what "hardening the LLM" means here — measurement, not extra
machinery):**

| property | result |
|---|---|
| Output validity (one of 4 tiers) | 100% (0 invalid, 0 transport errors over 304 calls) |
| Determinism at temperature 0 | **100%** label stability (25 cases × 3 identical calls) |
| Agent-1 extraction, schema-valid | **100%** (was 86.7%) |
| Agent-1 extraction, failure rate | **0%** (was 13.3%) |
| Agent-1 expected-token recall | **100%** (was 80.0%) |

These close limitation **L4**: Agent-1 error was previously unmeasured and
excluded from the token-level results.

**A finding worth reporting as a finding.** The original 13.3% failure rate was
not a model-capability limit at all — it was a single schema bug.
`_sanitize_enum_fields` repaired an *unrecognised* severity string but guarded
on `sev is not None`, so an explicit `"severity": null` passed through to a
non-Optional field and the entire case was discarded. `llama3.2:3b` emits
exactly that for bare complaints ("chest pain", "headache", "stomach pain",
"constipation"), and those four inputs were 100% of the failures. Two further
prompt defects were diagnosed the same way: the model split "mild fever" into
`symptoms:["fever"] + severity:mild` (so the token became `high_fever` — the
matcher was innocent, scoring `mild fever → mild_fever` at 1.0), and dropped
bare one-word messages entirely. The lesson for the paper is that what looked
like LLM unreliability was mostly integration defects in the layer around it,
and only measurement distinguished the two.

## 4. Logicality pass — claims that would not have survived review, now resolved

| Prior risk (plan §6) | Status after Stages 0–6 |
|---|---|
| "CP gives distribution-free ≥95% coverage" but q̂=1.0 → vacuous | FIXED (Stage 3): LAC gate, q̂≈0.022, set shrinks with evidence; coverage reported honestly (α=0.10 nominal 90%, empirical ~99%). |
| "A τ/δ reject option decides when to abstain" but it is never called | FIXED (Stage 3): the dead `classify_with_abstention` τ/δ path is removed; conformal is the sole gate. |
| "Calibrated posterior probabilities" but isotonic collapses to 0.000 | FIXED (Stage 3): displayed posteriors are the real (boosted) NB probabilities; isotonic backs only the reported top-1 confidence + ECE/Brier. |
| "AHP multi-criteria scoring prioritizes urgency" but AHP doesn't change the label | REFRAMED (Stage 4): AHP is a *within-tier ESI prioritizer*, explicitly not the triage label — documented and tested. |
| "Worst-case severity protects against under-triage" but only for 2–3 sets | FIXED (Stage 2): extended to ABSTAIN-with-EMG/SEV; guardrail proves 0 under-triage. |
| "Follow-up asks the most informative question" but questions were clinically incoherent | FIXED (Stage 4): candidate tokens constrained to characteristic discriminators of the differential. |
| UI captions claim τ/δ / isotonic posteriors | FIXED (Stage 6): Rule Engine captions rewritten; held-out evaluation view added. |
| "Baseline and ablation" were the **same computation** reported twice (both 6.3%) | FIXED (Tier 1): one parameterized scorer over a mechanism grid; independent baseline is 9.2%; `test_comparators_are_distinct_computations` makes recurrence build-breaking (§3.1). |
| "0% under-triage" presented as an empirical finding | REFRAMED (Tier 1): it is *entailed* by coverage + worst-case labelling. Stated as the bound `P(under-triage) ≤ α` and verified across the whole α grid (§3.1, §3.2). |
| "Token-order consistency = 1.0" as a robustness result | REMOVED (Tier 1): tautological on a sorted-set bag-of-words model. Replaced by subset robustness = 86.6%, which is informative because it is imperfect. |
| AHP weights + "CR = 0.0205" quoted but not reproducible from the matrix | FIXED (Tier 2): weights/λmax/CR derived by power iteration at import; CR is **0.0198**; reciprocity and CR < 0.10 enforced. |
| `EMERGENCY_SAFETY_FACTOR = 3.0` defended by a comment | DISPROVED (Tier 2): swept with re-calibration; it never improves under-triage and can worsen over-triage. Reported as a negative result (§3.2). |
| "The conformal gate abstains" — but at α=0.10 it often returns *everything* | DISCLOSED (Tier 2): the LAC-empty fallback fires on 33.9% of cases at production α; rate now reported at every α (§3.2). |
| Follow-up loop claimed as the core contribution with **no** evaluation | FIXED (Tier 3): resolution/questions/set-reduction vs two baselines, both evidence models (§3.3). |
| "Why not just prompt an LLM?" unanswered | FIXED (Tier 3): LLM-only baseline under-triages 16.9% and over-triages 82.7% (§3.4). |
| L4 "Agent-1 error is unmeasured" | FIXED (Tier 3): 86.7% schema-valid, 80.0% token recall, 13.3% failure on the labelled fixture (§3.4). |

### Mechanism-honesty checklist (all ✓)
- Conformal gate is **non-vacuous** (q̂ < 1, set varies by input).
- **No dead** τ/δ reject path.
- **AHP has a defined role** (within-tier ESI prioritization, not labelling), and
  its weights are **computed, not asserted**.
- Displayed **posteriors are meaningful** (not 0.000).
- Baseline ≠ ablation, enforced by a test.
- The under-triage result is presented as a **bound**, with its premise measured.
- The **LAC-empty fallback rate is disclosed**, so fallback behaviour is never
  described as conformal filtering.
- The follow-up selector's **advantage is stated conditionally** — it only beats
  random once negative evidence is used (§3.3).
- The disease knowledge base is described as a **Kaggle-style 41-disease dataset**,
  and the pediatric path as a **small hand-coded IMNCI danger-sign slice** — NOT a
  full IMNCI implementation (see §5).

## 5. Limitations & future work

**Stated limitations** (do not overstate):

- **(L1)** The data is still the Kaggle 41-disease CSV. The OOS harness removes
  the train-on-test problem but not the fact that these profiles are synthetic
  and not representative of real rural presentations.
- **(L2)** Only 2 EMERGENCY diseases exist in the dataset, so emergency recall
  does not generalize to conditions absent from the CSV. The EMERGENCY/SEVERE
  population (n=142) is dominated by SEVERE.
- **(L3)** `disease_severity.csv` has no documented clinical provenance; the
  severity tiers are the labels the whole safety argument is measured against.
- **(L4, now CLOSED)** Agent-1 extraction error is no longer unmeasured. On the
  labelled fixture it is **0% failure / 100% token recall** (§3.4, after the
  D-SEV and prompt fixes), and §3.5 now reports a full **end-to-end** result
  through real code-mixed lay language: 96.2% token recall, under-triage
  0.000, +5.3 points of over-triage attributable to extraction.
- **(L5)** Conformal coverage is **marginal and finite-sample** (n=304
  calibration points). Nominal coverage is slightly under-attained at tight α
  (§3.2), and at production α the gate falls back to "all candidates" on 33.9%
  of cases.
- **(L6, now RESOLVED)** "No" answers are real evidence in production (§3.3),
  which removed the selector/classifier misalignment and raised resolution
  3.7×. Residual limitations: the absence term applies only within the
  positive-evidence pool (a deliberate constraint — see the pathology in §3.3),
  and once denials exist the conformal input is no longer exchangeable with the
  calibration set, so the α bound becomes **approximate rather than exact**.
  That is inherent to adaptively collected evidence and is verified empirically
  (post-follow-up under-triage 0.021 ≤ α = 0.10) rather than assumed.
- **(L8, new)** Dataset symptom–disease associations are effectively **binary**:
  across 321 (disease, token) pairs, 85.4% hold in ≥95% of that disease's rows,
  100% hold in ≥70%, and none fall below 0.70. Real clinical associations are
  probabilistic, so the information-gain selector has markedly less headroom
  here than it would on real data, and the follow-up figures should not be
  extrapolated. This also explains why the relevance-threshold sweep is flat
  (any cut-off in [0, 0.70] selects the same pool), which is what makes the
  shipped 0.30 defensible.
- **(L7)** The pediatric IMNCI path needs exam-only signs (breaths/min, chest
  indrawing, skin pinch) that cannot come from a text message and have no UI, so
  pediatric cases are effectively untested by these harnesses.

The system makes **no clinical diagnostic-accuracy claim**.

**Future work (named in the paper, not built):** Agent 2 (DBSCAN outbreak
surveillance / RAG); real SMS/IVR channel; graceful LLM degradation / backend
failover; speed optimization; clinical validation (anchor severities to ESI or a
clinician sign-off); real road routing (ORS); full IMNCI ingestion (ear /
malaria / nutrition axes, young-infant ruleset); folding Agent-1 extraction error
into the end-to-end metric.

## 6. Reproducing the results

```bash
cd triage-poc
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # Python 3.11

# every number in this document, in the order it appears
PYTHONPATH=. ./.venv/bin/python tests/evaluate_oos.py                     # §3.1 baseline + ablations
PYTHONPATH=. ./.venv/bin/python tests/evaluate_sensitivity.py             # §3.2 alpha + boost sweeps
PYTHONPATH=. ./.venv/bin/python tests/evaluate_followup.py                # §3.3 follow-up loop
PYTHONPATH=. ./.venv/bin/python tests/evaluate_llm.py                     # §3.4 needs `ollama serve`
PYTHONPATH=. ./.venv/bin/python tests/evaluate_end_to_end.py              # §3.5 needs Ollama + FAISS
PYTHONPATH=. ./.venv/bin/python tests/evaluate_prioritization_routing.py  # §3.6 deterministic
PYTHONPATH=. ./.venv/bin/python tests/evaluate_robustness.py              # §3.7 needs `ollama serve`

./.venv/bin/python -m pytest tests -q                          # full suite (CI mirrors this)
./.venv/bin/python -m streamlit run streamlit_app.py           # UI + evaluation view
```

Add `--write` to any harness to refresh its tracked baseline JSON
(`tests/eval_{oos,sensitivity,followup,llm,end_to_end,robustness,prioritization_routing}_results.json`);
without it, results go to a timestamped `tests/eval_runs/` directory and the
committed baselines are untouched. The LLM-backed harnesses also accept
`--limit N` for a capped pass.

`evaluate_oos`, `evaluate_sensitivity`, `evaluate_followup` and
`evaluate_prioritization_routing` are fully deterministic (seed 42, no Ollama,
no FAISS) and reproduce byte-for-byte. `evaluate_llm`, `evaluate_end_to_end`
and `evaluate_robustness` call a local `llama3.2:3b`, so they are reproducible
only up to the model's own determinism — which is itself measured at 1.000
label stability at temperature 0 (§3.7).

The whole evaluation set is also browsable in the running app under the
**📊 Results** tab, which reads the same tracked JSONs.

The legacy in-sample scripts `tests/evaluate_{100_cases,balanced,1000}.py` are
retained as pipeline-wiring regression checks only and are **not** a source of
paper numbers. Data provenance for every input, including the author-assigned
severity labels all safety numbers are measured against, is in
`triage-poc/data/PROVENANCE.md`.
