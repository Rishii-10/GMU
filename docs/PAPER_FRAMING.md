# Research Paper — Framing & Positioning Guide

> **Purpose.** This is the fixed *stance* of the paper. Read it before writing or
> editing the paper, and before adding any result or feature to the project, so
> the positioning stays consistent. If a change can't be described in the terms
> below, it is probably off-scope — reframe it or move it to future work.
>
> Companion docs: [`PAPER_EVIDENCE.md`](PAPER_EVIDENCE.md) (claims → code → evidence)
> and [`../CODEBASE_CONTEXT.md`](../CODEBASE_CONTEXT.md) (what the code actually does).

---

## 1. One-sentence positioning

A **computer-science / software-engineering** paper proposing a **reusable AI
framework** — a pipeline, the coordination between its parts, and the design
logic — for message-based health **triage** (urgency sorting) in low-resource
settings, aligned with **UN SDG-3 (Good Health & Well-being)**. The approach is
**ideation / design-thinking / systems engineering**, *not* medicine and *not* a
mathematical machine-learning contribution. The framework is designed so that
**another person can take on the same problem, or extend it with new
capabilities, using the same structure.**

## 2. Who is making this claim

- A **computer-science student**, working from a **software-development /
  design-thinking** perspective.
- **No clinical / medical collaborator**, and the paper does **not** speak with
  medical authority.
- **Not** a mathematical or statistical ML contribution.

Stating this plainly in the paper is a feature, not a weakness — it sets the
reader's expectations to the right discipline.

## 3. What we DO claim

1. A **framework** that coordinates several components into one reliable pipeline:
   understand free-text message → make a structured urgency decision → **ask
   follow-up questions when unsure** → route to the right resource.
2. That good **software design** can take an inherently **unreliable AI
   component** and make it **usable and dependable as a whole system**.
3. **Novelty is the coordination** — especially the **active follow-up
   mechanism** that turns "the system is unsure" into "ask one targeted
   question, then decide again."
4. **Reusability & extensibility** — the framework is a **template**: its parts
   have clear boundaries and defined seams, so another contributor can swap a
   component or add a new capability without rewriting the core (see §4).
5. A concrete, working realization of an **SDG-3** use case.

## 4. Reusability & extensibility — a core design goal

This is a first-class claim of the paper, not an afterthought. A *framework* (as
opposed to one app) earns the name only if someone else can **adopt it, swap
pieces of it, or grow it**. So the paper must show that the design is built for
that.

**Extension points the framework already exposes** (substitution without
touching the core):
- **Model backends** are pluggable (the LLM extractor can be swapped for a
  different provider, or a rules-based stand-in).
- **Translation** and **routing providers** sit behind interfaces that can be
  replaced.
- A **shared, typed data contract** (a structured "case" → "decision" record)
  is the common language every module speaks, so new modules read and write the
  same stream rather than inventing their own format.

**Extensibility, demonstrated — the surveillance module (Agent 2) as the worked
example.** An **outbreak-surveillance** capability — watching the stream of
incoming cases for unusual clusters in an area — is the canonical *new module*
that the framework is meant to accommodate. It consumes the **same case stream**
the triage pipeline already produces and plugs in at a **defined seam**, which
is exactly how we show the framework is extensible rather than just asserting it.
This module is **currently not built**; the paper treats it as:
- a **demonstration of the extension pattern** (specify the interface it plugs
  into and how it reuses the existing case contract), and/or
- a **candidate to add as a working skeleton**, so the paper can point to a
  second module living inside the same framework.

**How reusability is evidenced in the paper** (not just claimed):
- the architecture's **module boundaries and interfaces** described explicitly;
- a **worked "how to add a module"** walkthrough (the surveillance agent);
- the **small, localized change** needed to substitute or add a part (e.g. how
  few places you touch to swap a backend or plug in a new module).

## 5. What we DO NOT claim (say this explicitly in the paper)

- **Not** clinical or diagnostic accuracy — we are not asserting the medical
  correctness of any decision.
- **Not** a new ML model, and **not** a mathematical/statistical result.
- **Not** that the dataset represents real patients.

These are written as **scope and future work**, never hidden.

## 6. Dataset stance

- The Kaggle-style 41-disease dataset is a **demonstration test-bench / sandbox**
  used to show the **framework behaves as designed**.
- Its **deterministic / synthetic** nature is **fine and intentional**: it
  isolates *framework behavior* from messy real-world confounders, which is
  exactly what a systems demonstration needs.
- Always call it a **"demonstration dataset,"** never "clinical ground truth."
  Real-world clinical data is named as future work.

## 7. Language rules — avoid two drifts

The intro already states the right frame; the danger is the **body drifting** out
of it. Guard against both:

**Drift A — sounding medical.**
Words like *under-triage / over-triage / "does not miss emergencies" / safety*
read as a clinician's claim. Reframe them as **engineering properties of the
pipeline** — i.e. *how the system handles its own uncertainty* — and keep
medical-sounding phrasing minimal and qualified.

**Drift B — sounding like a math / ML paper.**
Do **not** foreground statistical machinery (conformal-prediction theory,
calibration error, eigenvector/consistency math) as the contribution. If used,
present it as a plain **implementation choice**, in everyday language, and keep
the heavy detail in an appendix.

**Preferred vocabulary (the discipline we're actually in):**
architecture · pipeline · modularity · **separation of concerns** · coordination
/ orchestration · **interfaces / extension points / pluggability** · graceful
handling of an unreliable component · design decisions and trade-offs ·
reproducibility · robustness.

## 8. How the results should be framed (four pillars, in plain terms)

1. **Motivation — a plain AI alone is not enough.** Show that using the AI model
   directly is unreliable/unsafe for this task. This justifies building a
   framework at all.
2. **The framework fixes specific failures.** One clear table: *each concrete
   failure of the naive approach → the design response in our framework → the
   measured improvement.* This table is the heart of the paper.
3. **The new idea — active follow-up.** The loop that converts uncertainty into a
   targeted yes/no question and a better decision. This is the original
   contribution; give it room.
4. **Reusability & extensibility.** Show the framework is a template others can
   build on: the extension points, and a worked example of adding a module (the
   surveillance agent) that reuses the existing case stream (see §4).

Supporting (keep brief, not headline): robustness to language variation (English
/ Hindi / Hinglish), reliability/determinism of the pipeline, reproducibility,
and correct behavior of the prioritization and routing stages.

## 9. Checklist for future changes

Before adding a result, feature, or paragraph, confirm it passes all four:

- [ ] It can be described as a **framework / coordination / design** property.
- [ ] It does **not** quietly become a **medical** claim or a **pure-ML/math**
      claim (if it only makes sense as one of those, reframe or move to future
      work).
- [ ] The dataset is treated as a **demonstration sandbox**, not clinical truth.
- [ ] It **respects the framework's seams** — a new capability plugs into a
      defined interface and reuses the shared case contract, rather than
      special-casing the core (this is what keeps the reusability claim true).

Keep this file in sync with the paper — it is the **context anchor** for the
project's positioning.
