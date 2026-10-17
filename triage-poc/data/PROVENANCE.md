# Data provenance

Written for reviewers and anyone reproducing the evaluation. Every safety
number this project reports is measured **against the labels described here**,
so their origin is part of the result and not a footnote.

## `disease_symptoms.csv`

- **Shape:** 4,920 rows; 41 diseases x 120 rows each; 131 unique symptom tokens
  after `app.disease_kb.normalize_symptom_token`; 3-17 symptoms per row.
- **Origin:** a public Kaggle-style disease/symptom tabulation. It is a
  **synthetic teaching dataset**, not a clinical register, and was not
  collected from patients.
- **Structure that matters for the results:** the 120 rows per disease collapse
  to only **304 unique (disease, symptom-set) profiles** (5-10 per disease).
  All evaluation therefore splits on *profiles*, never on rows -- a row-level
  split would place near-duplicate rows on both sides and leak.
- **Measured characteristic:** the symptom-disease relation is effectively
  **binary**. Across all 321 (disease, present-token) pairs, 85.4% of
  associations hold in >= 95% of that disease's rows, 100% hold in >= 70%, and
  none fall below 0.70. Real clinical associations are probabilistic, so this
  dataset makes the discrimination problem easier than reality.
  (Computed by `tests/evaluate_followup.py::characterize_symptom_associations`.)

## `disease_severity.csv` -- the evaluation ground truth

- **Shape:** 41 rows, `Disease,Severity` in {EMERGENCY, SEVERE, MODERATE, MILD}.
- **Distribution:** EMERGENCY 2, SEVERE 16, MODERATE 18, MILD 5.
- **Origin: assigned by the project authors**, mapping each disease name in
  `disease_symptoms.csv` onto the four triage tiers this system outputs. The
  source dataset carries no severity column.
- **NOT clinically validated.** No clinician reviewed these tiers and they are
  not derived from ESI, WHO IMNCI, or any published triage standard.

**Why this is the most important limitation in the project.** Under-triage and
over-triage are defined relative to this file. Every safety claim is therefore
of the form *"relative to these tier assignments, the framework does not
under-triage"* -- a statement about the framework's internal consistency and
its uncertainty handling, **not** evidence of clinical safety. The honest
reading is that this validates the *mechanism*, and that anchoring the tiers to
a published standard with clinician sign-off is required future work.

Two structural consequences:
- **Only 2 EMERGENCY diseases exist** (Heart attack, Paralysis/brain
  haemorrhage), so the EMERGENCY/SEVERE population used for under-triage
  (n=142) is dominated by SEVERE, and emergency recall cannot generalise to
  conditions absent from the CSV.
- Name spellings in this file are **authoritative join keys** across all three
  CSVs plus `app/emergency_scorer.py::_DISEASE_ATTRS` (e.g. the double space in
  `(vertigo) Paroymsal  Positional Vertigo`, `Peptic ulcer diseae`,
  `Osteoarthristis`, lowercase `hepatitis A`). Correcting a spelling requires
  updating every one of those together.

## `disease_precautions.csv`

- 41 rows, up to 4 precautions each (Allergy and Heart attack have 3).
- Same Kaggle-style origin; shown to callers for MILD/MODERATE outcomes only.
- Advisory text, not a clinical protocol.

## Expert-elicited values in code (not from any dataset)

These are **author judgements**, documented as such because they cannot be
derived from the data and should not be presented as if they were:

| Where | What |
|---|---|
| `emergency_scorer._AHP_PAIRWISE` | Saaty pairwise matrix over 6 acuity attributes. The weights, lambda_max and CR = 0.0198 are **computed** from it at import, so only the pairwise judgements are elicited. |
| `emergency_scorer._DISEASE_ATTRS` | Per-disease (time-to-treatment, complication probability, transmissibility), 41 x 3 values. Informed by WHO IHR / IMNCI and the cited treatment-window literature, but the individual numbers are assigned, not measured. |
| `emergency_scorer._SEVERITY_SCORE` | Tier -> 0-1 mapping (0.15 / 0.45 / 0.75 / 1.00). Monotone by construction. |
| `emergency_scorer._age_vulnerability` | Age bands, following WHO IMNCI age structure and the AHRQ ESI geriatric adjustment. |
| `routing/router.py` score weights | 24h-doctor 5.0, emergency-care 3.0, blood-bank 2.0, radius 60 km, mock speed 40 km/h. Routing is evaluated for **logic** (optimality vs brute force), with no claim about real travel time. |

AHP is reported as a **within-tier dispatch prioritizer** and makes no
clinical-correctness claim; see `tests/evaluate_prioritization_routing.py`.

## Facility network

`app/routing/demo_facilities.py` holds **10 synthetic facilities and 5
villages** in the Denkanikottai/Hosur area (TN/KA). Coordinates are
approximate and the opening hours, equipment flags and the distance provider
(`MockRoutingProvider`, haversine at 40 km/h) are all invented for the demo.
Routing results evaluate the selection logic over this fixture, nothing more.

## Reproducibility

All harnesses are seeded (42). The three deterministic ones
(`evaluate_oos`, `evaluate_sensitivity`, `evaluate_followup`) reproduce
byte-for-byte; `evaluate_llm`, `evaluate_end_to_end` and
`evaluate_robustness` call a local `llama3.2:3b` and so are reproducible only
up to the model's own determinism (measured at 1.0 label stability at
temperature 0 -- see `tests/eval_robustness_results.json`).
