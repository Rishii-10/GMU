# Results and Evaluation

> **Draft of the paper's Results & Evaluation section.** Written in the paper's
> framing (a computer-science / software-engineering framework; see
> `PAPER_FRAMING.md`): we evaluate the framework's *behaviour* — how it wraps an
> unreliable AI component into a dependable, extensible system — not clinical
> diagnostic accuracy. Every figure below is the value the current system
> produces on the demonstration dataset; regenerate with
> `PYTHONPATH=. python tests/evaluate_<name>.py --write`. Numbers are reported
> as *engineering* quantities measured against a reference severity ordering,
> not as medical claims.

## 5.0 Experimental setup

**Demonstration dataset.** All experiments use a public 41-disease symptom
dataset (4,920 rows; 304 unique symptom profiles) as a controlled test-bench for
exercising the framework. Its severity tiers (EMERGENCY/SEVERE/MODERATE/MILD) are
an author-assigned *reference ordering* used to measure framework behaviour, **not
clinical ground truth**; the dataset is synthetic and not representative of real
rural presentations. We treat this as a sandbox to show the system behaves as
designed, and state clinical validation as future work.

**Protocol.** Unless noted, evaluation is **out-of-sample** (leave-profiles-out
5-fold: every test profile is scored by a model that never trained on it) and on
**short messages** (each profile truncated to 2–4 symptoms), which is the
realistic first-contact condition — a caregiver reports a few symptoms, not a
full 17-symptom record. Token-scored experiments are deterministic (fixed seed);
language and reliability experiments use a local `llama3.2:3b` and a multilingual
sentence-embedding matcher.

**Reporting conventions.** We report two behavioural safety quantities in plain
terms: **"misses-urgent"** (a case whose reference tier is EMERGENCY/SEVERE
surfaced at a lower tier) and **"over-calls"** (a MILD/MODERATE case surfaced as
EMERGENCY/SEVERE). We also report the decision mix (confident / uncertain /
abstain) and, where relevant, reproducibility and cost.

**Reproducibility.** The system ships 627 automated tests (CI-gated) and
deterministic evaluation harnesses that write tracked baseline JSON files; the
deterministic harnesses reproduce byte-for-byte. The results below are drawn from
those baselines.

---

## 5.1 A single AI model is not sufficient (motivation)

The natural challenge to any framework of this kind is that a modern LLM might do
the task directly. We test this: `llama3.2:3b` is asked to assign the same four
tiers on the same held-out cases, with the same information and a prompt that
states the tier definitions *and* the safety convention ("when uncertain, choose
the more severe tier") — so the baseline is not a straw man.

| System | misses-urgent | over-calls | exact-tier accuracy |
|---|---|---|---|
| **LLM-only (`llama3.2:3b`)** | **0.169** | **0.827** | 0.454 |

The model is simultaneously **unsafe and uninformative**: it lets 16.9 % of
serious cases through at a lower tier while over-calling 82.7 % of mild/moderate
ones, because it **collapses onto a single answer** — 251 of 304 replies are
"SEVERE". Prompted safety instructions do not produce usable triage behaviour.
This is the motivation for the framework: the contribution is the software
structure that turns this component into a dependable system.

---

## 5.2 The framework, component by component

### 5.2.1 Composed ablation

We walk a single behavioural metric across the whole stack on the same held-out
short messages, adding one mechanism at a time (`tests/evaluate_composed.py`):

| Stage | Configuration | misses-urgent | over-calls |
|---|---|---|---|
| C0 | LLM-only (prompt the model) | 0.169 | 0.827 |
| C1 | + structured classifier (argmax) | 0.092 | 0.068 |
| C2 | + abstain gate (uncertainty set) | 0.063 | 0.068 |
| C3 | + worst-case guard (**full system**) | **0.000** | 0.216 |
| C4 | + active follow-up (**shipped**) | 0.021 | **0.056** |

Two things are visible in one table. First, structure alone (C1) already removes
most of the LLM's pathology — misses-urgent falls 0.169 → 0.092 and over-calls
0.827 → 0.068. Second, the **worst-case guard (C3)** is the mechanism that drives
misses-urgent to zero, at the honest cost of over-calling (0.216); the **active
follow-up loop (C4)** then buys most of that cost back (over-calls 0.216 → 0.056)
while keeping misses-urgent near zero (0.021).

### 5.2.2 Failure modes of naive AI use, and the design response

The framework is best understood as a set of mitigations, each removing a
concrete, observed failure of the naive approach. Every row below is a defect we
measured in the system and the design decision that closes it.

| Observed failure of naive AI use | Design response | Measured effect |
|---|---|---|
| LLM mode-collapse → everything "SEVERE" | structured classifier + abstain gate | over-calls 0.827 → 0.216 |
| Fragile extraction — one schema-handling bug discarded whole cases | schema repair + input grounding | extraction failure 13.3 % → **0.0 %**; expected-token recall 80.0 % → **100 %** |
| Intensity words corrupting a symptom ("mild fever" → the wrong token) | extraction-prompt fix | corrected (regression-locked) |
| One-word messages dropped entirely | prompt + raw-text fallback | corrected |
| Hallucinated diagnosis leaking into a clinical handoff (a convulsing infant mislabelled with an unrelated chronic disease and its home-care advice) | suppress the supplementary diagnosis on danger-sign / abstaining paths | leak removed |
| Incoherent follow-up questions put to a layperson (e.g. asking about "acute liver failure") | restrict questions to answerable, characteristic symptoms | unanswerable-question rate 53.6 % → **0.0 %** |
| Negative evidence flattening the classifier → a 28-disease set wrongly escalated to EMERGENCY | restrict absence evidence to the supported differential | escalation impossible (invariant-tested) |

### 5.2.3 The safety property is a guarantee, stated honestly

The framework labels an uncertainty set by its **most severe** member. Whenever
the true tier is in the set, the surfaced tier is therefore at least as severe as
the truth, so a miss can only happen when the set omits the truth — an event the
uncertainty gate bounds. In plain terms, **the system's miss rate is bounded by a
single tunable parameter** (the gate's miscoverage budget, α): on the held-out
set, coverage is 1.0 and misses-urgent is 0.000, with the price paid as
over-calls (0.216). We present this as a *design guarantee with a measured
premise*, not as an empirical surprise.

We report two honest disclosures a reviewer would otherwise find. (i) At the
shipped setting the gate is conservative: on 33.9 % of cases its set is empty and
the system returns all surviving candidates, so part of the coverage is a
fallback rather than selective filtering. (ii) A parameter sweep shows the bound
holds across the whole range of α and that a tighter setting (α = 0.01) is at
least as safe with less over-calling and no fallback — so α is reported as a
tunable dial, and a secondary emergency-class weighting is reported as a
**negative result** (it changes neither safety quantity and is a candidate for
removal). The statistical machinery behind the gate is an implementation choice,
detailed in the appendix; the contribution is its role in the framework.

The full-system decision mix on the held-out short messages is **201 confident /
55 uncertain / 48 abstain** (of 304), and the tier is stable across two different
symptom subsets of the same patient 86.6 % of the time.

---

## 5.3 Active follow-up: converting uncertainty into action

The follow-up loop is the framework's novel coordination mechanism: when the
classifier is unsure, the system asks the caller a small number of targeted
yes/no questions and re-decides. We evaluate it out-of-sample with a 3-question
budget; **answers are read from each held-out patient's real recorded symptom
profile — a ground-truth lookup, not a simulation** (`tests/evaluate_followup.py`).

| Evidence model | Question policy | resolved | mean Q | set before → after |
|---|---|---|---|---|
| ignores "no" (early) | information-gain selector | 0.194 | — | 4.33 → … |
| ignores "no" (early) | random relevant question | **0.291** | — | — |
| counts "no" (**shipped**) | **information-gain selector** | **0.748** | **2.15** | **4.33 → 1.67** |
| counts "no" (shipped) | random relevant question | 0.583 | — | — |

This is a two-part finding and the paper's most interesting result. Scored on
positive evidence only, the "clever" selector is **worse than asking a random
relevant question** (0.194 vs 0.291): its objective models the value of a "no",
but the classifier discarded every "no", so half the information each question
was chosen to obtain was thrown away — a genuine design inconsistency we found by
measuring. Making a "no" real evidence raises resolution **0.194 → 0.748** (≈
3.9×) and the selector then dominates both baselines, resolving three-quarters of
uncertain cases in about two questions.

Shipping that fix surfaced a second lesson worth reporting. The obvious
implementation (apply the absence term across all diseases) let a few "no"s
*flatten* the classifier, producing a 28-disease set that the worst-case guard
then escalated to EMERGENCY — denials made the system **less** certain and **more**
alarmed. Restricting absence evidence to the differential the positive symptoms
already support removes this: a "no" can now only narrow or re-rank, never
introduce an unsupported disease or raise the tier (asserted by test). The loop's
net effect on the full system is favourable: over-calls fall **0.216 → 0.056**
for 0.021 misses-urgent (within the bound), with the true tier retained in the
set 97.4 % of the time.

---

## 5.4 Extensibility: a second agent on the same framework

A framework earns the name only if others can extend it. We demonstrate this by
building a **second, different agent — outbreak surveillance (Agent 2)** — on the
same structure and showing how little it touches.

**Coupling (measured, test-enforced).** Agent 2 consumes the **same case record**
the triage pipeline already writes, through a minimal `all_cases()` interface, and
imports **none** of the triage decision code — a structural test fails the build
if it ever does. The cost to add it is a new self-contained package, one UI view,
and one additive database column, with **zero edits to the triage decision path**.
The model backend is likewise a genuine swap point (three interchangeable
implementations behind one interface).

**Behaviour (demonstration stream).** Agent 2 reads the case stream, flags
unusual disease clusters per village with simple baseline-deviation statistics,
and — matching its design intent — **raises an alert only when something is out of
hand**, autonomously writing the official a natural-language situational brief and
a recommended public-health action. On a synthetic multi-village stream with one
injected outbreak it surfaces **exactly that outbreak** (a diarrhoeal-disease
cluster, 12 cases this week versus a baseline of 0, concentrated in under-5s) as a
single alert and stays silent on a matched flat stream (0 alerts). The LLM is
invoked **only when an alert fires**, so a quiet region costs no model calls. An
optional feedback path (an active outbreak raising the triage prior for that
disease in that area) demonstrates bidirectional coordination between agents; it
ships disabled by default so the triage core is unaffected.

---

## 5.5 Practical and SDG-relevant impact (with vs without)

These results translate framework behaviour into terms relevant to the use case
and to UN SDG-3 (access to health services) and SDG-10 (reduced inequalities).
They are demonstrated on the sandbox dataset; the cost figures are an explicitly
**illustrative model** with stated assumptions, not field measurements
(`tests/evaluate_impact.py`).

**A. Access and avoided journeys.** The system turns 42.1 % of messages into a
clear "care at home" decision (potential avoided trips) and 57.9 % into "seek
care". For the latter it names the nearest *eligible* facility: without
eligibility filtering, the naive "go to the nearest clinic" choice sends an
emergency to a facility that **cannot treat it in 80 % of the villages** in the
demonstration network. Under a stated model (one avoided trip ≈ 4 h and ≈ ₹300 per
caregiver), the care-at-home share corresponds to ≈ 421 avoided trips and ≈ 1,684
hours per 1,000 callers — reported as an illustration of scale, not a measurement.

**B. Language reach and inclusion.** The pipeline serves English, Hindi and
Hinglish: on real code-mixed lay phrasings it recovers 96.2 % of the intended
symptoms, gives the same decision to repeated messages (1.00) and to paraphrases
(0.75), and never mis-reads or coerces a hedged reply (0.0 each). An English-only
pipeline would simply drop the non-English share of callers.

**C. Caller effort.** Reaching a decision costs the caregiver about **two yes/no
questions** (mean 2.15), resolving 74.8 % of uncertain cases — rather than a full
structured form or an unsafe guess.

---

## 5.6 Supporting results

**End-to-end (free text in, tier out).** Running the whole pipeline — LLM
extraction and symptom matching included — on rendered messages, misses-urgent is
**0.000** both on canonical phrasings (token recall 0.993, 99.7 % agreement with
an oracle on clean tokens) and on real code-mixed lay phrasings (token recall
0.962, 97.6 % agreement); extraction costs precision (a few points of
over-calling), not safety.

**Robustness and reply handling.** Identical messages give an identical decision
(1.00); three real paraphrases of the same content agree 0.75 of the time; across
48 documented reply forms, none is misunderstood or mis-read and no hedged reply
("maybe", "pata nahi") is coerced into an answer (so the caller is re-prompted
rather than having a fact invented for them).

**Prioritisation and routing.** The within-tier urgency score orders two-thirds of
same-tier case pairs (discrimination 0.659), i.e. it carries information the tier
label does not; the router's choice matches an exhaustive optimum (1.00) and
always respects emergency-eligibility and opening hours on the demonstration
network (mean ETA ≈ 12 min). No clinical-correctness claim is attached to these.

**Systems cost.** The framework is deliberately light: exactly **one** LLM call per
case is unavoidable (extraction, ≈ 3.8 s on the test machine); every other model
call is conditional, and surveillance's brief is event-driven. The deterministic
stages are negligible by comparison (routing ≈ 0.06 ms, a full surveillance run ≈
0.23 ms, classification a few ms per case), so the framework adds essentially no
compute over the single model call it wraps.

**Reproducibility.** 627 automated tests pass (1 expected-fail, 1 expected-pass);
the deterministic harnesses reproduce byte-for-byte and write the tracked baseline
files these numbers are read from.

---

## 5.7 Threats to validity

We state the scope plainly. The data is a **synthetic demonstration dataset**: its
severity labels are an author-assigned reference ordering, only two of its
diseases are EMERGENCY, and it is not representative of real presentations, so the
absolute rates should be read as framework behaviour on a test-bench, not as
clinical performance. The access/cost figures are an **illustrative model** with
stated assumptions. There is **no user study or field deployment**, and the
realistic rural channel (SMS/IVR) is named as future work. The follow-up results
have limited headroom because the dataset's symptom–disease associations are
near-binary, and the safety bound becomes approximate once adaptively collected
"no" answers are present (verified empirically to still hold: post-follow-up
misses-urgent 0.021 ≤ the budget). These are expanded in the Limitations section;
the system makes **no clinical diagnostic-accuracy claim**.
