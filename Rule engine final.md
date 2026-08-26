# Rule Engine — Final Design

**Document scope:** Complete design specification for the two-stage rule engine sitting inside the rural triage AI framework. This document covers the intellectual contribution, all concepts used, step-by-step working in chronological order, one full worked example, evaluation strategy, and a fully annotated reference list explaining where each citation is used and what is derived from it.

---

## 1. Position in the Framework

The rule engine is **not** the headline contribution of the paper. It is the trustworthy, auditable decision organ of a multi-agent coordination framework whose central thesis is:

> *Uncertainty — whether a symptom is not-assessed, a diagnosis is low-confidence, or no facility is reachable — is treated as a first-class signal that triggers follow-up, abstention, or escalation instead of a silent guess, propagated end-to-end via a shared validated schema.*

The rule engine's job inside that framework is to:
1. Produce a **calibrated, probabilistic diagnosis** from a structured symptom vector delivered by Agent 1.
2. Classify the diagnosed disease into an **emergency level** using a multi-criteria, guideline-grounded scoring rubric.
3. **Abstain rather than guess** when neither stage can answer confidently — handing control back to the coordination layer, which then triggers a follow-up, escalation, or routing decision.

Claiming raw accuracy as the contribution is explicitly avoided. The paper's evaluative claim is **selective/safe accuracy**: correct when it answers, abstaining when it cannot.

---

## 2. Input: What the Rule Engine Receives

Agent 1 delivers a validated `ExtractedCase` — a structured object carrying:
- A binary symptom vector (132 symptoms drawn from the disease-symptom CSV, each `True` / `False` / `None`).
- Patient metadata available at runtime: **age**, **region**, **season**.
- A `followup_trail` audit log of what was asked and answered during multi-turn extraction.

The tri-state boolean (`True` = present, `False` = assessed and absent, `None` = not yet assessed) is the single most load-bearing invariant in the pipeline. The rule engine **never** conflates `None` with `False`. A symptom that was not asked about is not the same as a symptom that was asked about and found absent.

---

## 3. Stage 1 — Probabilistic Diagnosis

### 3.1 Core Model: Bernoulli Naive Bayes

**What it is:** A generative probabilistic classifier. For each disease, it learns from the CSV how often each symptom co-occurs with that disease. At inference time it computes a posterior probability for every disease given the observed symptom vector.

**Why Naive Bayes and not a more powerful model:**
- It produces a true `P(disease | symptoms)` posterior — the exact quantity the abstention and Bayesian prior fusion layers need.
- Its per-symptom likelihoods are human-readable (symptom weights per disease), which fits the rule-engine auditability requirement of the paper.
- It unifies the whole Stage-1 story into one coherent Bayesian narrative: `NB likelihood × epidemiological prior → calibrated posterior`.
- On this dataset (41 diseases, 132 binary symptoms), all models converge to ~99% raw accuracy, so interpretability and calibration are the differentiating factors — not discriminative power.

**The math:**
```
P(disease d | symptoms x) ∝ P(d) · ∏ P(sᵢ | d)^xᵢ · (1 − P(sᵢ | d))^(1−xᵢ)
```
- `P(sᵢ | d)` = fraction of disease d's rows in the CSV that contain symptom sᵢ (learned during training).
- `P(d)` = class prior, initially uniform (1/41), replaced by the epidemiological prior in Step 3.
- `xᵢ` = 1 if symptom present, 0 if absent (`None` symptoms are masked out — not counted for or against).
- Normalize the raw scores across all 41 diseases so posteriors sum to 1.

**Baseline comparison model:** Calibrated Logistic Regression (one-vs-rest, L2). Reported alongside NB to demonstrate the architecture is model-agnostic and to compare calibration quality (Brier score, ECE). No additional models — this is a framework paper, not a model bake-off.

---

### 3.2 Isotonic Calibration

**What it is:** A post-hoc correction applied to the NB model's raw probability outputs.

**Why it is non-negotiable:** Naive Bayes's conditional-independence assumption causes it to be systematically overconfident — it assigns posteriors closer to 0 or 1 than the true probabilities warrant. Without calibration, the abstention threshold and the reliability diagram are both meaningless, because the probabilities do not represent real likelihoods.

**How it works:** After training, a held-out calibration set is used to fit a monotonic function (isotonic regression) that maps NB's raw posteriors to corrected probabilities. The corrected probabilities are validated using a **reliability diagram** (predicted probability vs. observed frequency) — a perfectly calibrated model lies on the diagonal.

**Metrics reported:** Brier score (lower = better), ECE (Expected Calibration Error, lower = better), reliability diagram.

*Citation: Niculescu-Mizil & Caruana, ICML 2005 — the foundational paper establishing that Naive Bayes is systematically overconfident and that isotonic regression is the correct post-hoc fix.*

---

### 3.3 Epidemiological Prior Fusion (Bayesian Layer)

**What it is:** The hybrid probabilistic layer that contextualizes the NB likelihood with local disease prevalence.

**Why it exists:** The symptom-only NB model is data-blind to geography and season. The same symptoms — fever, chills, vomiting — can be Malaria in a monsoon-endemic district or Typhoid in an enteric-fever-burdened urban setting. The prior breaks ambiguous ties in a medically sensible, explainable way rather than arbitrarily.

**Where the priors come from:** WHO Global Health Observatory regional disease-incidence data and national health-ministry surveillance reports — declared as configurable parameters in the system. If a region-specific prior is unavailable, the system falls back to a uniform prior (standard NB).

**The fusion formula (plain Bayes):**
```
posterior ∝ NB_likelihood(d | symptoms) × P(d | region, season)
```
Renormalize after multiplication so all posteriors still sum to 1. The NB likelihood is the discriminative evidence from symptoms; the prior is the epidemiological context. Together they are strictly more informative than either alone.

---

### 3.4 Reject Option — Abstention Layer

**What it is:** The safety mechanism that prevents the engine from producing a confident-but-wrong diagnosis.

**The core rule:** output a diagnosis only if **both** conditions hold on the calibrated, prior-fused posterior:
1. **Top posterior ≥ τ** (minimum confidence threshold).
2. **Gap between top and second posterior ≥ δ** (minimum separation, ruling out near-ties).

If either condition fails → **ABSTAIN**. The engine returns no diagnosis and hands control to the coordination layer, which routes to the follow-up gate or escalation path.

**How τ and δ are chosen — the citable method:**
The thresholds are **not hand-picked**. They are derived from the validation data using the **Youden's J index**:
```
J = max over all candidate thresholds of (Sensitivity + Specificity − 1)
```
The Youden cutoff is the standard ROC-based threshold-selection method in medical diagnostics. The chosen τ is the value of the calibrated posterior at which J is maximized on the validation ROC curve. The value is reported as a result, not declared as a design choice.

For the gap δ (multi-class abstention), the **risk–coverage trade-off** framework is used: sweep δ across candidate values, plot selective error (error among answered cases) vs. coverage (fraction of cases answered). The operating point is selected to meet a pre-declared safety target, and the coverage paid for it is reported. The paper states the method and the data-derived value — never an invented number.

**Why this protects against confident-wrong outputs:** the engine only answers when both its confidence is high and its top-two candidates are well-separated. Cases where symptoms overlap multiple diseases (the hardest cases) systematically fail the gap check and abstain, rather than coin-flipping.

*Citations: Youden (1950) for J index; BMC Medical Research Methodology 2024 for the clinical cutoff-selection review; Chow (1970) and Geifman & El-Yaniv (2017) for the reject-option / selective-classification framework.*

---

## 4. Stage 2 — Emergency Classification

The old three-attribute additive design is fully replaced. The new design uses **six grounded attributes** scored on a normalized 0–1 scale and combined via **AHP-derived weights** (Analytic Hierarchy Process, Saaty 1980). No weight is hand-picked; every attribute traces to a published clinical source.

---

### 4.1 The Six Attributes

Each attribute is scored 0–1. The six dimensions collectively represent what clinical triage instruments measure: individual patient risk, disease trajectory, and public health impact.

| # | Attribute | What it measures | Source |
|---|---|---|---|
| 1 | **Disease intrinsic severity** | How clinically dangerous is the disease in general? | disease_severity.csv mapped to clinical severity grading; WHO disease classification |
| 2 | **Patient age vulnerability** | How much does the patient's age amplify risk? | ESI v5 AHRQ Handbook; WHO IMCI age-band protocol |
| 3 | **Onset acuity** | How rapidly did symptoms appear? | ESI Decision Point B ("high-risk situation / should not wait"); acute-onset as ESI Level 2 discriminator |
| 4 | **Time-to-treatment sensitivity** | Does delayed treatment significantly worsen outcomes? Is there a narrow treatment window? | Golden Hour principle (Lerner & Moscati, 2001); treatment-delay mortality literature (Kumar et al. 2006 — sepsis) |
| 5 | **Complication / deterioration probability** | How likely is this condition to escalate to a life-threatening state if untreated? | WHO IMCI danger-signs framework; clinical deterioration prediction literature |
| 6 | **Transmissibility / public health impact** | Can this disease spread to others and pose a community-level threat? | WHO International Health Regulations (IHR 2005) — epidemic potential and international spread classification |

---

### 4.2 Scoring Each Attribute (0–1 normalized)

**Attribute 1 — Disease intrinsic severity:**
The CSV labels (MILD / MODERATE / SEVERE / EMERGENCY) map to normalized scores that correspond to clinical care-level requirements — not arbitrary integers:
- MILD → 0.15 (self-limiting, outpatient management)
- MODERATE → 0.45 (requires medical attention)
- SEVERE → 0.75 (requires hospital care)
- EMERGENCY → 1.0 (requires immediate intervention)

**Attribute 2 — Patient age vulnerability:**
Five bands drawn from WHO IMCI age-band structure and ESI v5's geriatric vital-sign adjustment:
- < 2 months → 1.0 (neonates: WHO IMCI separate high-risk protocol)
- 2–59 months → 0.8 (young children: primary IMCI target population)
- 5–14 years → 0.3 (school-age: lowest general vulnerability)
- 15–64 years → 0.2 (baseline adult)
- > 65 years → 0.85 (geriatric: masked compensatory responses, comorbidity burden — ESI v5 Handbook explicitly flags this group)

**Attribute 3 — Onset acuity:**
Captured at intake by Agent 1 ("when did symptoms start?"):
- Acute (hours) → 1.0
- Subacute (days) → 0.5
- Chronic / gradual (weeks+) → 0.2

Acute onset is a direct ESI Level 2 trigger — the ESI algorithm's "should not wait" decision point. This attribute encodes that clinical logic numerically.

**Attribute 4 — Time-to-treatment sensitivity:**
Per-disease, based on whether delayed treatment causes irreversible harm, drawn from clinical treatment-delay literature:
- Critical window (< 6 hours, irreversible harm likely) → 1.0 (e.g., meningitis, sepsis)
- Moderate window (6–48 hours) → 0.6 (e.g., Malaria — delays increase cerebral malaria risk)
- Flexible (> 48 hours, treatment effective regardless of timing) → 0.2 (e.g., fungal infection)

*Grounded in the Golden Hour principle (Lerner & Moscati, 2001) and the sepsis antibiotic-delay mortality data (Kumar et al., 2006: 7.6% mortality increase per hour of delay).*

**Attribute 5 — Complication / deterioration probability:**
Per-disease, based on likelihood of progression to life-threatening complications if untreated:
- High (> 50% risk of serious complication without treatment) → 1.0 (e.g., pneumonia → respiratory failure)
- Moderate (10–50%) → 0.6 (e.g., typhoid → bowel perforation)
- Low (< 10%) → 0.2 (e.g., acne, mild fungal infection)

*Sourced from WHO IMCI danger-sign classifications and per-disease complication-rate data in clinical literature.*

**Attribute 6 — Transmissibility / public health impact:**
Classified per WHO IHR (2005) notifiable-disease and epidemic-potential frameworks:
- Airborne / highly contagious → 1.0 (e.g., tuberculosis)
- Vector-borne / moderate epidemic potential → 0.6 (e.g., malaria, dengue — regional outbreak risk)
- Contact / limited spread → 0.3 (e.g., fungal infection)
- Non-communicable → 0.0 (e.g., hypertension, diabetes)

---

### 4.3 Weight Derivation via AHP (Analytic Hierarchy Process)

**What AHP is:** The standard multi-criteria decision-making method in healthcare prioritization, developed by Thomas L. Saaty (1980) with over 170,000 citations. AHP derives weights from pairwise comparisons — asking "how much more important is criterion A than criterion B?" — and produces a mathematically consistent priority vector. It is not opinion; it is a structured encoding of published clinical priority logic.

**Why AHP and not hand-picked weights:** Every weight is reproducible, auditable, and backed by a consistency check. A reviewer can re-derive the weights from the pairwise matrix. Hand-picked weights cannot be defended this way.

**The five steps of AHP as applied here:**

**Step 1 — Build the pairwise comparison matrix.**
Compare all six attributes against each other using Saaty's 1–9 scale:
- 1 = equally important
- 3 = moderately more important
- 5 = strongly more important
- 7 = very strongly more important
- 9 = extremely more important

The comparisons are derived from the ESI decision algorithm's own priority ordering (life-threat > high acuity > resource need) cross-referenced with WHO IMCI priority ordering. This is not expert opinion — it is a structured encoding of two published clinical frameworks.

**Step 2 — Extract the priority vector (weights).**
Compute the principal eigenvector of the 6×6 pairwise matrix. This eigenvector gives the six weights w₁ through w₆, which sum to 1.0. The full pairwise matrix is reported in the paper's appendix so the weights are fully reproducible.

**Step 3 — Consistency check.**
Compute the Consistency Ratio:
```
CR = Consistency Index / Random Index
```
Saaty's threshold: CR ≤ 0.10 means the pairwise comparisons are acceptably consistent. If CR > 0.10, revise the pairwise judgements until the condition is met. The CR value is reported in the paper.

**Illustrative weight ordering** (actual values emerge from the matrix — not pre-decided):
complication/deterioration > time-to-treatment sensitivity > disease intrinsic severity > age vulnerability > onset acuity > transmissibility.

This ordering reflects that *individual patient trajectory* (will this patient deteriorate?) dominates *population risk* (will it spread?) in a triage context — consistent with ESI's patient-first philosophy, where life-threat and high-risk acuity are the top two decision points before resource prediction.

**Step 4 — Sensitivity analysis.**
Vary each weight ±20% and report how often the final band assignment changes. A robust system is stable under this perturbation. This is standard AHP practice and is reported in the evaluation section.

---

### 4.4 The Scoring Formula

```
Acuity Score (0–1) = Σ (wᵢ × attribute_scoreᵢ)   for i = 1..6
```

Map to the 1–10 emergency scale:
```
Emergency Score = round(Acuity Score × 9) + 1
```

**Score → band mapping:**
- 1–3: **Non-urgent** → ESI 4–5
- 4–7: **Urgent** → ESI 3
- 8–10: **Emergency** → ESI 1–2

**Honest framing for the paper:** this is an **ESI-aligned mapping**, not the literal ESI algorithm. True ESI v5 also uses vital signs and predicted resource count, which the dataset does not contain. This limitation is explicitly stated in the paper's limitations section — reviewers reward that honesty. The contribution is the principled, AHP-grounded multi-attribute scoring that operationalizes ESI's intent in a data-constrained rural setting.

---

### 4.5 WHO IMCI Danger-Sign Hard Overrides

**What they are:** A set of categorical, clinically non-negotiable gates that force Emergency regardless of the AHP-weighted score.

**Why they exist alongside AHP:** AHP handles graduated urgency well but is vulnerable to weight-sensitivity at the extremes — a misconfigured weight could theoretically suppress the score for a life-threatening presentation. The danger-sign overrides are the unconditional safety net. They bypass arithmetic entirely. This dual-layer design (weighted score + categorical override) is itself an architectural contribution: the system is robust to weight perturbation in the exact cases where a wrong answer would be catastrophic.

**The danger signs (sourced directly from WHO IMCI Chart Booklet, 2014):**
- Unable to drink or breastfeed
- Persistent vomiting (cannot keep anything down)
- Convulsions (current or recent history)
- Lethargic or unconscious
- Severe respiratory distress (chest indrawing, stridor at rest)
- Severe dehydration with shock signs

Any one present → **score overridden to 10 → Emergency band → ESI 1**, regardless of the AHP total.

---

## 5. Chronological Working of the Engine

```
1. Receive ExtractedCase from Agent 1
       |
       v
2. Build binary symptom vector
   (True = present, False = absent, None = masked — never conflated)
       |
       v
3. Bernoulli Naive Bayes
   Compute raw likelihood scores for all 41 diseases
       |
       v
4. Epidemiological Prior Fusion
   Multiply likelihoods by P(disease | region, season)
   Renormalize → posterior over 41 diseases
       |
       v
5. Isotonic Calibration correction
   Map raw posteriors to corrected probabilities
   (removes NB overconfidence bias)
       |
       v
6. Reject Option Check
   top_posterior ≥ τ  AND  (top − second) ≥ δ ?
       |                              |
      YES                            NO
       |                              |
   Diagnosis + confidence       ABSTAIN →
                                return to coordination layer
                                (trigger follow-up or escalation)
       |
       v
7. Stage 2: WHO IMCI Danger-sign override check
   Any danger sign present in symptom vector?
       |                              |
      YES                            NO
       |                              |
   Force Emergency (10)        AHP-weighted acuity score
   → ESI 1                     (6 attributes × derived weights)
                                → band → ESI level
       |
       v
8. Output: diagnosis, confidence, emergency score, band, ESI level, reasoning trail
```

---

## 6. Full Worked Example

**Scenario:** A 4-year-old child in Odisha district, India, during monsoon season. Symptoms reported to Agent 1: fever, chills, vomiting, headache, muscle pain.

### Step 1 — Agent 1 hands over
```
Symptom vector: {fever: True, chills: True, vomiting: True,
                 headache: True, muscle_pain: True,
                 itching: None, skin_rash: None, ...}
Patient metadata: {age: 4, region: "Odisha", season: "monsoon"}
Onset: acute (hours)
Danger signs assessed: none present
```

### Step 2 — Bernoulli NB raw likelihoods
fever, chills, vomiting, headache, muscle_pain overlap across several febrile diseases. Raw posteriors (illustrative — actual values derived from the CSV at training time):
- Malaria ≈ 0.44
- Typhoid ≈ 0.39
- Dengue ≈ 0.10
- Others ≈ 0.07

### Step 3 — Epidemiological prior fusion
WHO incidence data for Odisha, monsoon: Malaria prevalence high, Typhoid moderate, Dengue lower.
Priors: Malaria = 0.60, Typhoid = 0.28, Dengue = 0.08, others = 0.04

```
Malaria:  0.44 × 0.60 = 0.264
Typhoid:  0.39 × 0.28 = 0.109
Dengue:   0.10 × 0.08 = 0.008
Others:   0.07 × 0.04 = 0.003
Total:    0.384

Renormalized:
  Malaria = 0.264 / 0.384 = 0.687
  Typhoid = 0.109 / 0.384 = 0.284
  Dengue  = 0.008 / 0.384 = 0.021
  Others  = 0.003 / 0.384 = 0.008
```

### Step 4 — Isotonic calibration
Corrected Malaria posterior = 0.71 (isotonic regression adjusts for NB overconfidence).

### Step 5 — Reject option check
- τ derived from Youden's J on validation ROC curve. Say τ = 0.63. Top posterior 0.71 ≥ 0.63 ✅
- δ derived from risk–coverage curve at target selective error. Say δ = 0.18. Gap = 0.71 − 0.28 = 0.43 ≥ 0.18 ✅

**→ Diagnosis: Malaria, calibrated confidence 0.71.** Engine does not abstain.

### Step 6 — Danger-sign override check
No WHO IMCI danger signs present. Proceed to AHP scoring.

### Step 7 — AHP-weighted acuity score

Six attributes scored for this case:

| Attribute | Score | Notes |
|---|---|---|
| Disease intrinsic severity | 0.75 | Malaria = SEVERE |
| Patient age vulnerability | 0.80 | Age 4 = 2–59 months band (WHO IMCI) |
| Onset acuity | 1.00 | Acute onset (hours) — ESI Level 2 trigger |
| Time-to-treatment sensitivity | 0.60 | Malaria: 6–48 hour window (cerebral malaria risk) |
| Complication / deterioration prob. | 0.60 | Malaria: moderate complication rate if treated promptly |
| Transmissibility | 0.60 | Vector-borne (Anopheles) — regional epidemic potential |

Applying AHP-derived weights (illustrative — actual weights from eigenvector of pairwise matrix):
```
Acuity Score = (w₁×0.75) + (w₂×0.80) + (w₃×1.00)
             + (w₄×0.60) + (w₅×0.60) + (w₆×0.60)
           ≈ 0.72   (computed from the full weight vector)

Emergency Score = round(0.72 × 9) + 1 = round(6.48) + 1 = 7
```

→ Band: **Urgent → ESI 3**

*(Note: if a danger sign had been present — e.g., lethargic child — the override would have forced 10 → Emergency → ESI 1 regardless of this score.)*

### Step 8 — Output
```
Diagnosis:       Malaria
Confidence:      0.71 (calibrated posterior)
Emergency score: 7 / 10
Emergency band:  Urgent
ESI level:       3
Reasoning:       SEVERE disease, child age (IMCI high-risk band), acute onset,
                 moderate treatment window, vector-borne; no danger-sign override
Action:          Route to nearest facility with antimalarial capability;
                 monitor for danger-sign onset
```

**What happens if the prior had been uniform (no regional data)?**
Without the prior, Malaria ≈ 0.44 < τ → ABSTAIN → coordination layer asks a discriminating follow-up: "Is the fever coming in cyclic bouts with sweating?" Answer yes → rerun → Malaria posterior rises above τ → answers. The prior is the fast path; the follow-up is the fallback. Both converge to the same diagnosis; the prior does it without burning an extra conversation turn.

---

## 7. Evaluation Plan

The evaluation proves the safety claim, not just accuracy.

### Primary metric — Selective accuracy + risk–coverage curve
Report accuracy *only among cases the engine answered* (selective accuracy), and the fraction of cases it answered (coverage). Plot selective error vs. coverage as δ is varied. This is the headline result — it directly demonstrates that abstaining protects against wrong outputs and is the core safety contribution.

### Calibration metrics
- Reliability diagram (predicted probability vs. observed frequency across probability bins)
- **Brier score** (proper scoring rule for probabilistic classifiers — measures both accuracy and calibration jointly)
- **ECE** (Expected Calibration Error — average deviation of predicted probability from observed frequency)

Reported for both NB (primary) and LR (baseline). A perfectly calibrated model lies on the reliability diagram diagonal.

### Per-class sensitivity and specificity
For each of the 41 diseases. Identifies which diseases are hard to separate (the ones that most often trigger abstention). These are the high-risk pairs that the epidemiological prior is most valuable for.

### Hard-case test suite — demonstrating true positives AND true negatives
Three constructed case categories:
- **Clean cases** (symptoms map unambiguously to one disease) → must answer correctly → demonstrates true positives.
- **Ambiguous cases** (overlapping symptom sets, e.g. Malaria / Typhoid without regional prior) → must abstain → demonstrates true negatives (not confidently wrong).
- **Out-of-scope cases** (symptom vectors that match no disease in the CSV) → must abstain → demonstrates true negatives (no spurious confident output). These are constructed, as the synthetic dataset contains none natively. This category makes the true-negative claim falsifiable.

### Stage 2 validation
- **Cohen's κ** of the system's band assignment vs. a documented ESI-level reference mapping or a clinician-reviewed subset.
- **AHP sensitivity analysis**: vary each weight ±20% and report the fraction of cases whose band assignment changes. A robust design is stable under perturbation.
- **Benchmark context**: real-world human ESI correct-triage rate is only ~60% (PMC7211387); inter-rater κ ≈ 0.87 (same source). These are the comparison benchmarks.

### External sanity check
Run the frozen trained model on one small real symptom→diagnosis dataset (not the training CSV). Prevents the ~99% synthetic accuracy from being the sole evidence.

---

## 8. Limitations (to state explicitly in the paper)

- Primary dataset is synthetic and near-separable; real-world symptom presentations are noisier and more overlapping.
- No vitals (temperature, pulse, SpO₂, respiratory rate) in the dataset; the ESI-aligned mapping is approximate by construction. True ESI v5 requires vital signs for all patients not immediately classified as Level 1 or 2.
- Epidemiological priors are approximate (WHO/national-ministry data, not local clinical records); declared as configurable parameters rather than fixed values.
- AHP pairwise comparisons encode the priority ordering of ESI and WHO IMCI — any deviation from these guidelines would require a revised pairwise matrix and a re-verified CR.
- Single-region pilot assumption; multi-region validation is future work.
- Prospective clinical validation against real patient cases is the required next step before deployment.

Reviewers reward explicit limitations. Do not hide them.

---

## 9. References — Annotated

Every reference below is listed with: where it is used in the rule engine, and exactly what is derived from it.

---

### [R1] Naive Bayes for Medical Symptom Classification
**Citation:** Rish, I. (2001). *An empirical study of the naive Bayes classifier.* IJCAI Workshop on Empirical Methods in AI. Also: Peng, Y. et al. (arXiv:2106.02813) — ML web-based disease prediction.
**Used in:** Section 3.1 — Core Model.
**What is derived:** Justification for using Bernoulli Naive Bayes as the primary generative probabilistic classifier for symptom→disease prediction. Establishes that NB is the standard, well-understood model for binary-feature (symptom present/absent) medical classification.

---

### [R2] Isotonic Calibration for Naive Bayes
**Citation:** Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting good probabilities with supervised learning.* ICML 2005.
**Used in:** Section 3.2 — Isotonic Calibration.
**What is derived:** (a) The finding that Naive Bayes is among the most systematically overconfident classifiers due to its conditional-independence assumption. (b) The recommendation that isotonic regression is the correct post-hoc calibration method for NB. This citation is the direct justification for why calibration is non-negotiable in this system.

---

### [R3] Youden's J Index — Optimal Threshold Selection
**Citation:** Youden, W.J. (1950). *Index for rating diagnostic tests.* Cancer, 3(1), 32–35. Also: Unal, I. (2017). *Defining an Optimal Cut-Point Value in ROC Analysis.* Computational and Mathematical Methods in Medicine.
**Used in:** Section 3.4 — Reject Option (derivation of τ).
**What is derived:** The method for selecting the confidence threshold τ. The Youden index J = max(Sensitivity + Specificity − 1) is computed across all candidate thresholds on the validation ROC curve; the threshold that maximizes J is τ. No threshold is hand-picked — it is always data-derived using this method.

---

### [R4] Cutoff Selection Methods — Clinical Diagnostic Review
**Citation:** Uçar, M. et al. (2024). *Methods of determining optimal cut-point of diagnostic biomarkers with application of clinical data in ROC analysis: an update review.* BMC Medical Research Methodology.
**Used in:** Section 3.4 — Reject Option (supporting the Youden's J choice).
**What is derived:** Confirmation that Youden's J is the most widely used and stable method for optimal cut-point selection in medical diagnostic tests. Used to frame the threshold-selection subsection in the paper.

---

### [R5] Reject Option / Selective Classification — Chow's Rule
**Citation:** Chow, C.K. (1970). *On optimum recognition error and reject tradeoff.* IEEE Transactions on Information Theory, 16(1), 41–46.
**Used in:** Section 3.4 — Reject Option (derivation of δ via risk–coverage trade-off).
**What is derived:** The foundational theoretical result that a reject (abstention) option minimizes expected misclassification cost when the abstention cost is set appropriately. Establishes the framework for the gap threshold δ: sweep δ, plot the risk–coverage curve, select the operating point that meets the declared safety target.

---

### [R6] Selective Prediction / Risk–Coverage Trade-off
**Citation:** Geifman, Y. & El-Yaniv, R. (2017). *Selective prediction in deep neural networks.* NeurIPS 2017. Also: NeurIPS 2024 — *Overcoming Common Flaws in the Evaluation of Selective Classification Systems.*
**Used in:** Section 3.4 — Reject Option (risk–coverage curve as the headline evaluation result).
**What is derived:** The modern treatment of the selective-classification framework: the risk–coverage curve, how to evaluate it, and how to select an operating point. The NeurIPS 2024 paper specifically flags common mistakes in evaluating selective classifiers (e.g., reporting coverage without risk, or vice versa) — guarding against those mistakes in the paper's evaluation section.

---

### [R7] Analytic Hierarchy Process (AHP)
**Citation:** Saaty, T.L. (1980). *The Analytic Hierarchy Process.* McGraw-Hill, New York.
**Used in:** Section 4.3 — Weight Derivation via AHP.
**What is derived:** The entire weight-derivation method for Stage 2. Pairwise comparison matrix construction, Saaty's 1–9 scale, principal eigenvector extraction to produce the weight vector, and the Consistency Ratio (CR ≤ 0.10) as the acceptability threshold for the pairwise judgements. No weight in Stage 2 is hand-picked; all weights are derived by this method.

---

### [R8] AHP in Healthcare — Systematic Review
**Citation:** Liberatore, M.J. & Nydick, R.L. (2008). *The analytic hierarchy process in medical and health care decision making: A literature review.* European Journal of Operational Research, 189(1), 194–207. Also: Blaizot, A. et al. (2015). *Applying the Analytic Hierarchy Process in healthcare research: A systematic literature review.* BMC Medical Informatics and Decision Making (PMC4690361).
**Used in:** Section 4.3 — Weight Derivation via AHP.
**What is derived:** Precedent and justification for using AHP in healthcare prioritization settings. These reviews confirm AHP is the established multi-criteria method in healthcare decision support and provide the citation a reviewer would expect when AHP is used in a medical-system paper.

---

### [R9] AHP Applied to ICU Triage — COVID-19 Case
**Citation:** Almutairi, A.F. et al. (2021). *Automated Triage System for Intensive Care Admissions during the COVID-19 Pandemic Using Hybrid XGBoost-AHP Approach.* Computational Intelligence and Neuroscience (PMC8512533).
**Used in:** Section 4.3 — Weight Derivation via AHP.
**What is derived:** A direct published example of AHP being used to derive weights for clinical triage priority scoring (ICU admissions). Demonstrates that the AHP approach has been applied to triage specifically — not just general healthcare — and that it is accepted in peer-reviewed medical informatics literature.

---

### [R10] Emergency Severity Index (ESI) — Primary Triage Citation
**Citation:** Gilboy, N. et al. (2012). *Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4. Implementation Handbook.* AHRQ Publication No. 12-0014. Agency for Healthcare Research and Quality. Also: ESI v5 Handbook (EMSC Improvement Center, 2024).
**Used in:** Sections 4.1 (Attribute 2 — age), 4.1 (Attribute 3 — onset acuity), 4.4 (band-to-ESI-level mapping), 4.5 (override logic), 6 (worked example).
**What is derived:** (a) The five-level ESI framework as the primary triage citation. (b) Age-based vital-sign adjustment flags for geriatric patients. (c) The "should not wait / high-risk situation" decision point as the clinical grounding for the onset-acuity attribute. (d) The ESI 1–5 level definitions that the band mapping is aligned to. (e) The honest framing that this system is ESI-aligned, not ESI-equivalent, because vitals and resource-count data are absent.

---

### [R11] ESI Validity and Reliability Study
**Citation:** Validity and Reliability of Emergency Severity Index. PMC7409571. Hospital Universiti Sains Malaysia.
**Used in:** Section 7 (Stage 2 evaluation benchmark).
**What is derived:** The benchmark figures used to contextualize Stage 2 evaluation: real-world human ESI correct-triage rate is only approximately 60%, and inter-rater agreement κ ≈ 0.87. These are the numbers that set the human-performance bar that the system is compared against.

---

### [R12] ESI Accuracy Study — Predicting Patient Outcome
**Citation:** Accuracy of the Emergency Department Triage System using the Emergency Severity Index for Predicting Patient Outcome. PMC7211387.
**Used in:** Section 7 (Stage 2 evaluation benchmark) and framing in paper introduction.
**What is derived:** Additional confirmation of ESI's real-world ~60% correct-triage rate. Used in the paper to motivate why a structured, auditable scoring system with explicit uncertainty handling is needed — even expert human triage using the gold-standard tool is imperfect.

---

### [R13] ESI Geriatric Modification
**Citation:** Modification of the Emergency Severity Index Improves Mortality Prediction in Older Patients. PMC6625680.
**Used in:** Section 4.2 — Attribute 2 (age vulnerability, >65 band).
**What is derived:** The finding that the ESI systematically under-triages older patients because they may not produce expected vital-sign responses due to comorbidities and medications. This justifies the elevated vulnerability score (0.85) assigned to the >65 age band, which is higher than the naive expectation.

---

### [R14] WHO IMCI Chart Booklet — Danger Signs
**Citation:** World Health Organization. (2014). *Integrated Management of Childhood Illness: Chart Booklet.* WHO Press, Geneva. Updated guidelines.
**Used in:** Section 4.1 (Attribute 5 grounding), Section 4.5 (danger-sign hard overrides), Section 5 (flow), Section 6 (worked example).
**What is derived:** (a) The age-band structure (< 2 months, 2–59 months) that defines the highest vulnerability scores in Attribute 2. (b) The complete list of general danger signs that trigger the hard Emergency override in Section 4.5. (c) The clinical grounding for the complication / deterioration probability attribute — conditions flagged as danger signs are the high-deterioration-probability cases.

---

### [R15] WHO International Health Regulations (IHR 2005) — Epidemic Potential
**Citation:** World Health Organization. (2005). *International Health Regulations, Second Edition.* WHO Press, Geneva. Also: Annex 2 evaluation — PMC3313850.
**Used in:** Section 4.1 (Attribute 6 — transmissibility / public health impact), Section 4.2 (transmissibility scoring scale).
**What is derived:** The classification framework for disease transmissibility and epidemic potential. The 0–1 transmissibility scores (non-communicable → 0.0, contact → 0.3, vector-borne → 0.6, airborne → 1.0) are grounded in the IHR's distinction between diseases that pose local-only risk vs. those with epidemic / international-spread potential.

---

### [R16] Golden Hour Principle — Time-to-Treatment
**Citation:** Lerner, E.B. & Moscati, R.M. (2001). *The Golden Hour: Scientific Fact or Medical Urban Legend?* Academic Emergency Medicine, 8(7), 758–760.
**Used in:** Section 4.1 (Attribute 4 — time-to-treatment sensitivity), Section 4.2 (time-window scoring).
**What is derived:** The concept that a narrow time window exists in emergency conditions during which treatment significantly improves outcomes. Used to ground the critical-window (< 6 hours → score 1.0) tier of the time-to-treatment attribute.

---

### [R17] Treatment-Delay Mortality — Sepsis
**Citation:** Kumar, A. et al. (2006). *Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock.* Critical Care Medicine, 34(6), 1589–1596.
**Used in:** Section 4.2 — Attribute 4 (time-to-treatment sensitivity, critical window tier).
**What is derived:** Concrete quantitative grounding for the critical-window concept: each hour of delay in antibiotic administration in septic shock is associated with a 7.6% increase in mortality. This is the sharpest published example of a dose-response relationship between treatment delay and mortality, used to justify the high score assigned to short-window diseases.

---

### [R18] WHO Global Health Observatory — Epidemiological Priors
**Citation:** World Health Organization. *Global Health Observatory Data Repository.* https://www.who.int/data/gho
**Used in:** Section 3.3 — Epidemiological Prior Fusion.
**What is derived:** The source of the regional disease-incidence priors `P(disease | region, season)` fed into the Bayesian prior fusion layer. These are declared as configurable parameters — the WHO GHO and national health-ministry surveillance reports are the authoritative sources for deriving them per deployment region.

---

## 10. One-Line Summary for the Paper

> A calibrated Bernoulli Naive Bayes diagnostic layer, fused with an epidemiological Bayesian prior and a Youden-J-derived reject option, feeds a six-attribute AHP-weighted emergency scorer with WHO-IMCI danger-sign overrides and ESI-level mapping — evaluated on selective accuracy, calibration, and AHP weight sensitivity — functioning as the trustworthy, auditable decision organ of the rural triage coordination framework.
