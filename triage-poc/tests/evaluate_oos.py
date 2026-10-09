"""Stage 5 evaluation harness -- the numbers the paper rests on.

The legacy evals (evaluate_1000/balanced/100_cases) score FULL in-sample symptom
rows and report F1 = 1.00. That measures internal consistency, not real caller
performance -- it is kept only as a "pipeline wiring sanity check". This harness
instead measures the regime that matters for a triage-safety claim:

  * OUT-OF-SAMPLE: leave-profiles-out k-fold, so every test profile is scored by
    a model that never trained on it.
  * SHORT MESSAGES: each test profile truncated to 2-4 tokens (the real caller
    condition -- few symptoms, not a full 17-symptom row).
  * SUBSET ROBUSTNESS: two DIFFERENT short subsets drawn from the same patient
    profile must yield the same triage label (does the tier depend on which
    symptoms the caller happens to mention first?).

Metrics (guiding principle: for triage, selective risk and under-triage are the
headline, not accuracy):
  - under-triage rate   : true EMERGENCY/SEVERE surfaced at a lower tier
  - over-triage rate    : true MILD/MODERATE surfaced as EMERGENCY/SEVERE  (bounded, reported as the trade-off)
  - risk-coverage        : selective error at each coverage (CONFIDENT fraction)
  - silent-error vs abstain: wrong-and-confident rate vs uncertain-and-asked rate
  - subset robustness    : label agreement across two different symptom subsets

HONESTY NOTE (read before quoting under-triage). The framework labels a
multi-disease prediction set with the WORST-CASE tier in that set. So whenever
the true disease is inside the set, the surfaced tier is by construction at
least as severe as the truth, and under-triage cannot occur. Under-triage is
therefore bounded by the conformal miscoverage rate, i.e. P(under-triage) <=
P(true not in set) <= alpha -- it is a PROPERTY OF THE DESIGN, not an
independent empirical discovery. Report it as a guarantee whose premise
(coverage) is what this harness measures empirically, and report over-triage as
the price paid for it.

A token-order "consistency" metric was REMOVED: _FoldModel.posteriors() reduces
tokens to a sorted set, so a bag-of-words Naive Bayes is order-invariant by
construction and that metric could only ever return 1.0. Genuine paraphrase
robustness requires running the Agent-1 / FAISS text path and is out of scope
for this deterministic harness (stated limitation L4).

Comparators (scored on the SAME out-of-sample short messages, see _VARIANTS):
  - baseline_flat_nb_top1       : plain NB argmax -- none of the safety machinery
  - ablation_no_worstcase       : boost + CP set, but surface the top candidate
  - ablation_no_emergency_boost : CP set + worst-case, but no class-imbalance boost
Each is a DIFFERENT mechanism combination, so no two can collapse into the same
computation (an earlier version scored the baseline and the ablation with
identical code and reported one number twice).

Deterministic: scores tokens directly through app.rules_engine.classify (adult
route), no Ollama / FAISS. Agent-1 extraction error is a separate, stated
limitation (L4) and is not measured here.

Run:  PYTHONPATH=. python tests/evaluate_oos.py            # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_oos.py --write    # -> tests/eval_oos_results.json (tracked baseline)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import (
    DEFAULT_CP_ALPHA,
    EMERGENCY_SAFETY_FACTOR,
    DiseaseClassifier,
    _fit_naive_bayes,
    _clean_candidate_scores,
    _boost_emergency_scores,
    _nb_posteriors,
    get_default_classifier,
    severity_for_disease,
)
from app.disease_kb import DiseaseKB
from app.schemas import AgeGroup, ClassificationLabel, ExtractedCase, SEVERITY_RANK

_SEED = 42
_N_FOLDS = 5
_SHORT_MIN, _SHORT_MAX = 2, 4
_ADULT_AGE_MONTHS = 600

# The systems compared on the same out-of-sample short messages. Each row is a
# distinct combination of the three mechanisms the framework claims credit for:
#   boost      -- EMERGENCY class-imbalance boost (EMERGENCY_SAFETY_FACTOR)
#   use_cp     -- conformal (LAC) prediction set rather than a bare top-1
#   worst_case -- label a multi-disease set with its most severe tier
# Because every comparator differs from every other in at least one flag, the
# baseline and the ablations are structurally incapable of producing the same
# number by accident (tests/test_eval_harness.py enforces this).
_FULL_SYSTEM = "full_system"
_VARIANTS: dict[str, dict[str, bool]] = {
    _FULL_SYSTEM:                  {"boost": True,  "use_cp": True,  "worst_case": True},
    "ablation_no_worstcase":       {"boost": True,  "use_cp": True,  "worst_case": False},
    "ablation_no_emergency_boost": {"boost": False, "use_cp": True,  "worst_case": True},
    "baseline_flat_nb_top1":       {"boost": False, "use_cp": False, "worst_case": False},
}


def _unique_profiles(rows_by_disease):
    out = []
    for d, rows in rows_by_disease.items():
        seen = set()
        for row in rows:
            key = tuple(sorted(row))
            if key not in seen:
                seen.add(key)
                out.append((d, key))
    return out


def _assign_folds(profiles, rng):
    by_d = {}
    for d, p in profiles:
        by_d.setdefault(d, []).append((d, p))
    fold_of = {}
    for d, lst in by_d.items():
        shuffled = list(lst)
        rng.shuffle(shuffled)
        for i, kp in enumerate(shuffled):
            fold_of[kp] = i % _N_FOLDS
    return fold_of


def _truncate(profile, rng, k_min=_SHORT_MIN, k_max=_SHORT_MAX):
    toks = list(profile)
    rng.shuffle(toks)
    k = min(len(toks), rng.randint(k_min, k_max))
    return toks[:k]


def run_evaluation(alpha: float = DEFAULT_CP_ALPHA, classifier=None) -> dict:
    """Score the OOS short-message set.

    `alpha` and `classifier` exist for the sensitivity analysis
    (tests/evaluate_sensitivity.py). Defaults reproduce production exactly.
    Passing a classifier built with a different `emergency_factor` sweeps the
    boost WITH its conformal calibration rebuilt, preserving exchangeability.
    """
    kb = DiseaseKB.load()
    full = classifier or get_default_classifier()  # production model (for severity lookups)
    rows_by_disease = full._rows_by_disease
    vocab = full.vocabulary
    vocab_set = set(vocab)
    emergency_names = full._emergency_disease_names

    profiles = _unique_profiles(rows_by_disease)
    rng = random.Random(_SEED)
    fold_of = _assign_folds(profiles, rng)

    # One record stream per comparator, all scored on the SAME short messages.
    by_variant: dict[str, list[dict]] = {name: [] for name in _VARIANTS}
    # Subset robustness: does the tier change when the caller happens to mention
    # a different subset of the same true symptom profile?
    robust_hits = robust_total = 0

    trunc_rng = random.Random(_SEED + 1)
    # Separate, explicitly seeded stream for the second subset. (The previous
    # version seeded this from hash(tuple(tokens)); str hashing is salted per
    # process unless PYTHONHASHSEED is set, so that draw was not reproducible
    # across runs.)
    alt_rng = random.Random(_SEED + 2)

    for fold_idx in range(_N_FOLDS):
        held = [k for k, f in fold_of.items() if f == fold_idx]
        held_by_d = {}
        for d, p in held:
            held_by_d.setdefault(d, set()).add(p)
        train = {}
        for d, rows in rows_by_disease.items():
            ex = held_by_d.get(d, set())
            kept = [r for r in rows if tuple(sorted(r)) not in ex]
            if kept:
                train[d] = kept
        lp, ll = _fit_naive_bayes(train, vocab)
        # A fold-local classifier that reuses the production calibration/q̂ but a
        # fold-trained NB -- so the test profile is genuinely out-of-sample for
        # the NB model (the conformal calibration stays fixed, as in deployment).
        fold_clf = _FoldModel(full, lp, ll, emergency_names, alpha=alpha)

        for true_d, profile in held:
            true_rank = SEVERITY_RANK[severity_for_disease(true_d, kb)]
            short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
            if not short:
                continue

            # Every comparator sees exactly the same input tokens.
            for name, flags in _VARIANTS.items():
                res = _score_variant(fold_clf, short, kb, **flags)
                by_variant[name].append({
                    "true": true_d, "true_rank": true_rank,
                    "label": res["label"],
                    "label_rank": SEVERITY_RANK.get(ClassificationLabel(res["label"]), 99),
                    "decision": res["decision"], "set_size": res["set_size"],
                    "correct_in_set": true_d in res["prediction_set"],
                    "lac_empty_fallback": res["lac_empty_fallback"],
                })

            # Subset robustness: a SECOND, independently drawn short subset of
            # the same true profile. Only pairs that actually differ are
            # counted -- comparing a subset with itself would be vacuous.
            alt = [t for t in _truncate(profile, alt_rng) if t in vocab_set]
            if alt and set(alt) != set(short):
                a = _score_variant(fold_clf, short, kb, **_VARIANTS[_FULL_SYSTEM])
                b = _score_variant(fold_clf, alt, kb, **_VARIANTS[_FULL_SYSTEM])
                robust_total += 1
                robust_hits += 1 if a["label"] == b["label"] else 0

    return _summarize(by_variant, robust_hits, robust_total, alpha,
                      getattr(full, "_emergency_factor", EMERGENCY_SAFETY_FACTOR))


class _FoldModel:
    """A DiseaseClassifier-like shim: fold-trained NB posteriors + the
    production conformal calibration (q̂) and emergency set."""
    def __init__(self, full, lp, ll, emergency_names, alpha: float = DEFAULT_CP_ALPHA):
        self._lp, self._ll = lp, ll
        self._diseases = list(lp.keys())
        self._cp = sorted(full._cp_cal_nonconformity_scores)
        # Defaults to the production alpha rather than restating it, so the
        # harness can never silently evaluate a different gate than the one
        # shipped; the sensitivity sweep overrides it explicitly.
        self._alpha = alpha
        self._emergency_names = emergency_names
        # Follow the classifier's own boost so a re-calibrated sweep model is
        # scored with the factor it was calibrated under.
        self._emergency_factor = getattr(full, "_emergency_factor", EMERGENCY_SAFETY_FACTOR)
        self._vocab = full.vocabulary

    def posteriors(self, tokens):
        rec = sorted({t for t in tokens if t in set(self._vocab)})
        if not rec:
            return {}
        return _nb_posteriors(rec, self._diseases, self._lp, self._ll)

    def q_hat(self):
        import math
        n = len(self._cp)
        qi = min(int(math.ceil((n + 1) * (1.0 - self._alpha))), n) - 1
        return self._cp[qi]


def _score_variant(fold_clf, tokens, kb, *, boost: bool, use_cp: bool, worst_case: bool,
                   probs: dict[str, float] | None = None) -> dict:
    """Score one short message under a chosen combination of the framework's
    three mechanisms. With all three enabled this reproduces the production
    path (classify_with_cp + classify_via_dataset labelling) on a fold-trained
    NB; disabling flags yields the baseline and the ablations.

    Parameterising a single implementation (rather than hand-writing one
    scorer per comparator) is deliberate: it is the reason the baseline and
    the ablations cannot silently become the same computation again.

    `probs` lets a caller inject posteriors computed some other way (the
    follow-up harness injects absence-aware posteriors) while reusing the
    identical set-construction and labelling logic."""
    if probs is None:
        probs = fold_clf.posteriors(tokens)
    if not probs:
        return {"label": ClassificationLabel.INCOMPLETE_ASSESSMENT.value, "decision": "ABSTAIN",
                "set_size": 0, "prediction_set": [], "top": None}

    scores = _clean_candidate_scores(probs)
    if boost:
        scores = _boost_emergency_scores(
            scores, fold_clf._emergency_names, fold_clf._emergency_factor
        )
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[0][0]

    fallback = False
    if use_cp:
        # Conformal (LAC) set; an empty set means nothing cleared the bar, so
        # fall back to the honest wide differential (production behaviour).
        q = fold_clf.q_hat()
        lac = [d for d, s in scores.items() if (1.0 - s) <= q]
        if lac:
            in_set = lac
        else:
            # DIAGNOSTIC: when this fires the output is NOT a conformal set --
            # it is every surviving candidate. Tracked because a high rate
            # means the "conformal gate" is mostly inactive at this alpha.
            in_set = [d for d, _ in ranked]
            fallback = True
    else:
        in_set = [top]  # no uncertainty quantification at all: bare argmax

    set_size = len(in_set)
    decision = "CONFIDENT" if set_size == 1 else ("UNCERTAIN" if set_size <= 3 else "ABSTAIN")

    if worst_case and set_size > 1:
        # rules_engine F3 + Stage-2 escalation: surface the most severe tier
        # present in the set.
        worst_rank = min(SEVERITY_RANK[severity_for_disease(d, kb)] for d in in_set)
        label_enum = next(l for l, r in SEVERITY_RANK.items() if r == worst_rank)
        label = label_enum.value if hasattr(label_enum, "value") else label_enum
    else:
        label = severity_for_disease(top, kb).value

    return {"label": label, "decision": decision, "set_size": set_size,
            "prediction_set": in_set, "top": top, "lac_empty_fallback": fallback}


def _rate(num, den):
    return round(num / den, 4) if den else 0.0


def _variant_metrics(records: list[dict]) -> dict:
    """Triage-safety metrics for one comparator. Identical metric code is
    applied to every comparator; only the record stream differs."""
    n = len(records)
    emg_sev = [r for r in records if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]]
    mild_mod = [r for r in records if r["true_rank"] >= SEVERITY_RANK[ClassificationLabel.MODERATE]]
    under = sum(1 for r in emg_sev if r["label_rank"] > r["true_rank"])
    over = sum(1 for r in mild_mod if r["label_rank"] < r["true_rank"])

    confident = [r for r in records if r["decision"] == "CONFIDENT"]
    conf_wrong = sum(1 for r in confident if r["label_rank"] != r["true_rank"])
    abstain_or_ask = [r for r in records if r["decision"] != "CONFIDENT"]

    return {
        "n": n,
        "under_triage_rate_emergency_severe": _rate(under, len(emg_sev)),
        "under_triage_count": under,
        "emergency_severe_n": len(emg_sev),
        "over_triage_rate_mild_moderate": _rate(over, len(mild_mod)),
        "over_triage_count": over,
        "mild_moderate_n": len(mild_mod),
        "cp_coverage_true_in_set": _rate(sum(1 for r in records if r["correct_in_set"]), n),
        "decision_counts": dict(Counter(r["decision"] for r in records)),
        "coverage_confident_fraction": _rate(len(confident), n),
        "selective_risk_on_confident": _rate(conf_wrong, len(confident)),
        "silent_error_rate": _rate(conf_wrong, n),            # wrong AND confident
        "abstain_or_ask_rate": _rate(len(abstain_or_ask), n),
        "mean_set_size": round(sum(r["set_size"] for r in records) / n, 4) if n else 0.0,
        # How often the LAC set was empty and the system returned every
        # candidate instead. A high rate means the conformal gate is largely
        # inactive and the behaviour is driven by the fallback + worst-case
        # rule, which must be disclosed rather than described as "conformal".
        "lac_empty_fallback_rate": _rate(sum(1 for r in records if r.get("lac_empty_fallback")), n),
        "label_distribution": dict(Counter(r["label"] for r in records)),
    }


def _summarize(by_variant: dict[str, list[dict]], r_hits: int, r_total: int,
               alpha: float, emergency_factor: float) -> dict:
    per_variant = {name: _variant_metrics(recs) for name, recs in by_variant.items()}
    full = per_variant[_FULL_SYSTEM]

    out = dict(full)  # full-system metrics stay at the top level
    out["regime"] = "out-of-sample (leave-profiles-out 5-fold) + short message (2-4 tokens)"
    out["alpha"] = alpha
    out["emergency_safety_factor"] = emergency_factor
    out["subset_robustness_label_agreement"] = _rate(r_hits, r_total)
    out["subset_robustness_pairs_compared"] = r_total
    # Under-triage is bounded by conformal miscoverage BY DESIGN (worst-case
    # labelling of a set that contains the truth cannot land below the truth).
    # Surfacing the bound next to the measurement keeps the two from being
    # confused in the write-up -- see this module's HONESTY NOTE.
    out["under_triage_design_bound_alpha"] = alpha
    out["under_triage_is_bounded_by_miscoverage"] = True
    out["comparisons"] = {
        name: m for name, m in per_variant.items() if name != _FULL_SYSTEM
    }
    out["variant_definitions"] = _VARIANTS
    return out


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 72)
    print("OUT-OF-SAMPLE + SHORT-MESSAGE EVALUATION (Stage 5)")
    print("=" * 72)
    for k in ("n", "regime", "alpha", "under_triage_rate_emergency_severe", "under_triage_count",
              "emergency_severe_n", "over_triage_rate_mild_moderate", "over_triage_count",
              "cp_coverage_true_in_set", "coverage_confident_fraction",
              "selective_risk_on_confident", "silent_error_rate", "abstain_or_ask_rate",
              "subset_robustness_label_agreement", "subset_robustness_pairs_compared",
              "decision_counts"):
        print(f"  {k}: {ev[k]}")
    print("\n  NOTE: worst-case labelling of a set that contains the truth cannot")
    print("  under-triage, so under-triage <= conformal miscoverage <= alpha BY")
    print("  DESIGN. Report it as a guarantee, not an empirical discovery.")

    print("\n  " + "-" * 68)
    print(f"  {'comparator':<30}{'under-tri':>11}{'over-tri':>11}{'confident':>11}")
    print("  " + "-" * 68)
    rows = [(_FULL_SYSTEM, ev)] + sorted(ev["comparisons"].items())
    for name, m in rows:
        print(f"  {name:<30}"
              f"{m['under_triage_rate_emergency_severe']:>11.4f}"
              f"{m['over_triage_rate_mild_moderate']:>11.4f}"
              f"{m['coverage_confident_fraction']:>11.4f}")
    print("  " + "-" * 68)
    print("  mechanisms per comparator (boost / conformal set / worst-case):")
    for name, flags in ev["variant_definitions"].items():
        on = ", ".join(k for k, v in flags.items() if v) or "none"
        print(f"    {name:<30} {on}")
    out = out_dir / "eval_oos_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("out-of-sample + short-message evaluation (Stage 5)")
    print("Running out-of-sample short-message evaluation...")
    ev = run_evaluation()
    print_report(ev, out_dir)
