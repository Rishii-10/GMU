"""Tier 3 evaluation of the ADAPTIVE FOLLOW-UP LOOP -- the mechanism the paper
claims as its core contribution, and the one that previously had no numbers at
all.

WHAT IS MEASURED
    The loop's job is to convert uncertainty into a resolved triage decision by
    asking the fewest, most discriminating yes/no questions. So:
      * resolution rate     -- fraction of uncertain cases driven to a single
                               disease within the turn budget
      * questions to resolve-- how many were needed
      * set-size reduction  -- uncertainty actually removed
      * triage effect       -- under/over-triage before vs after the loop

WHERE THE ANSWERS COME FROM (no simulation, nothing invented)
    Each held-out case IS a real patient profile from data/disease_symptoms.csv
    with a known, complete symptom list. The caller is shown only a short 2-4
    token subset of it (the realistic first-message condition). When the system
    asks "does the patient have X?", the answer is READ OFF that same real
    profile: X is either in the patient's recorded symptom list or it is not.
    Nothing is generated, sampled or imagined -- it is a lookup of ground truth
    the dataset already contains, which is the standard way a sequential
    questioning policy is evaluated offline.

WHAT THE LLM DOES AND DOES NOT DO HERE
    In production the question is chosen by followup_question_selector from
    dataset statistics (emergency-first, else information gain over the
    prediction set); the LLM only renders the chosen token as a natural-language
    sentence. Phrasing cannot change WHICH symptom is asked about, so the
    selection policy -- the part that determines resolution -- is evaluated
    deterministically here, with no Ollama dependency. LLM phrasing quality is
    measured separately in tests/evaluate_llm.py.

COMPARATORS (the selector has to beat something to be worth its complexity)
    selector  -- production policy (emergency-first / information gain)
    frequency -- ask about the most common symptom in the differential
    random    -- ask about any clinically relevant symptom, seeded
    All three draw from the SAME clinically-relevant token pool, so the
    comparison isolates the selection rule rather than the relevance filter.

NEGATIVE EVIDENCE (Tier 4)
    Production Naive Bayes scores PRESENT tokens only, so answering "no"
    currently teaches the classifier nothing -- while the information-gain
    selector explicitly models P(absent). That is an internal inconsistency.
    `use_negative_evidence=True` adds a Bernoulli absence term, so a "no" is
    real evidence, and the two are reported side by side.

Out-of-sample throughout: the same leave-profiles-out folds, fold-trained NB
and fold-local selector statistics as tests/evaluate_oos.py. Deterministic.

Run:  PYTHONPATH=. python tests/evaluate_followup.py           # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_followup.py --write   # -> tests/eval_followup_results.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import (
    DEFAULT_CP_ALPHA,
    _fit_naive_bayes,
    _nb_posteriors_with_absence,
    get_default_classifier,
    severity_for_disease,
)
from app.disease_kb import DiseaseKB
from app.followup_question_selector import (
    _MIN_RELEVANCE_FREQ,
    _relevant_tokens,
    _symptom_freq,
    select_followup_question,
)
from app.schemas import SEVERITY_RANK, ClassificationLabel
from tests.evaluate_oos import (
    _FULL_SYSTEM,
    _VARIANTS,
    _FoldModel,
    _assign_folds,
    _rate,
    _score_variant,
    _truncate,
    _unique_profiles,
)

_SEED = 42
_N_FOLDS = 5
MAX_TURNS = 3  # production cap in agent1_extraction.next_followup_question
STRATEGIES = ("selector", "frequency", "random")


def _posteriors(fold_clf, present, absent, train_rows):
    """NB posterior over the fold-trained model.

    With `absent` empty this is the production present-only posterior. With
    `absent` non-empty it delegates to the PRODUCTION absence-aware scorer
    `app.disease_classifier._nb_posteriors_with_absence` -- the same function
    `DiseaseClassifier._raw_posteriors` calls at inference.

    This harness previously carried its OWN copy of the absence math, which
    applied the Bernoulli penalty across all 41 diseases. That version let
    denials flatten the posterior and admit diseases the positive evidence
    never supported (see the production docstring for the measured
    ["cough"] -> 28-disease case). Delegating guarantees the number this
    harness reports is the number the shipped system produces, and that the
    two can never diverge again.
    """
    vocab = set(fold_clf._vocab)
    pres = sorted({t for t in present if t in vocab})
    absent_only = sorted({t for t in absent if t in vocab} - set(pres))
    if not absent_only:
        return fold_clf.posteriors(pres)
    if not pres:
        return {}
    return _nb_posteriors_with_absence(
        pres,
        absent_only,
        fold_clf._diseases,
        fold_clf._lp,
        fold_clf._ll,
        train_rows,
    )


def _pick_token(strategy, prediction_set, severity_map, known, train_rows, vocab, rng,
                min_relevance=_MIN_RELEVANCE_FREQ):
    """Choose the next symptom to ask about. All strategies draw from the same
    clinically-relevant pool so only the ranking rule differs."""
    if strategy == "selector":
        q = select_followup_question(
            prediction_set=list(prediction_set),
            severity_map=severity_map,
            known_tokens=list(known),
            rows_by_disease=train_rows,
            vocabulary=vocab,
            min_relevance=min_relevance,
        )
        return q.symptom_token if q else None

    pool = _relevant_tokens(list(prediction_set), set(known), train_rows, vocab, min_relevance)
    if not pool:
        return None
    if strategy == "random":
        return rng.choice(sorted(pool))
    if strategy == "frequency":
        return max(
            sorted(pool),
            key=lambda t: sum(_symptom_freq(d, t, train_rows) for d in prediction_set),
        )
    raise ValueError(f"unknown strategy {strategy}")


def _run_one(strategy, use_negative_evidence, kb, full, rng_seed=_SEED,
             min_relevance=_MIN_RELEVANCE_FREQ):
    """One full pass over the held-out set under a given questioning policy."""
    rows_by_disease = full._rows_by_disease
    vocab = full.vocabulary
    vocab_set = set(vocab)
    emergency_names = full._emergency_disease_names
    severity_map_all = {d: severity_for_disease(d, kb).value for d in kb.diseases}

    profiles = _unique_profiles(rows_by_disease)
    fold_of = _assign_folds(profiles, random.Random(_SEED))
    trunc_rng = random.Random(_SEED + 1)
    pick_rng = random.Random(rng_seed + 7)
    flags = _VARIANTS[_FULL_SYSTEM]

    records = []
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
        fold_clf = _FoldModel(full, lp, ll, emergency_names, alpha=DEFAULT_CP_ALPHA)

        for true_d, profile in held:
            short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
            if not short:
                continue
            true_rank = SEVERITY_RANK[severity_for_disease(true_d, kb)]
            profile_set = set(profile)  # the patient's REAL symptom list

            present, absent, asked = list(short), [], []
            first = _score_variant(fold_clf, present, kb, **flags,
                                   probs=_posteriors(fold_clf, present, absent, train))
            initial = first

            current = first
            for _ in range(MAX_TURNS):
                if current["set_size"] <= 1:
                    break
                token = _pick_token(
                    strategy, current["prediction_set"],
                    {d: severity_map_all[d] for d in current["prediction_set"]},
                    present + absent, train, vocab, pick_rng, min_relevance,
                )
                if token is None:
                    break
                asked.append(token)
                # GROUND TRUTH ANSWER: is this symptom in the real profile?
                if token in profile_set:
                    present.append(token)
                elif use_negative_evidence:
                    absent.append(token)
                else:
                    # Production behaviour: a "no" is only remembered so the
                    # same question is not re-asked; it does not inform NB.
                    absent.append(token)
                probs = _posteriors(
                    fold_clf, present, absent if use_negative_evidence else [], train
                )
                current = _score_variant(fold_clf, present, kb, **flags, probs=probs)

            records.append({
                "true": true_d,
                "true_rank": true_rank,
                "questions_asked": len(asked),
                "initial_set_size": initial["set_size"],
                "final_set_size": current["set_size"],
                "initial_label_rank": SEVERITY_RANK.get(ClassificationLabel(initial["label"]), 99),
                "final_label_rank": SEVERITY_RANK.get(ClassificationLabel(current["label"]), 99),
                "resolved": current["set_size"] == 1,
                "started_uncertain": initial["set_size"] > 1,
                "final_correct": current["set_size"] == 1 and current["top"] == true_d,
                "truth_kept_in_set": true_d in current["prediction_set"],
            })
    return records


def _summarize(records) -> dict:
    n = len(records)
    uncertain = [r for r in records if r["started_uncertain"]]
    nu = len(uncertain)

    def triage(rs, key):
        emg = [r for r in rs if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]]
        mild = [r for r in rs if r["true_rank"] >= SEVERITY_RANK[ClassificationLabel.MODERATE]]
        return (
            _rate(sum(1 for r in emg if r[key] > r["true_rank"]), len(emg)),
            _rate(sum(1 for r in mild if r[key] < r["true_rank"]), len(mild)),
        )

    u_before, o_before = triage(records, "initial_label_rank")
    u_after, o_after = triage(records, "final_label_rank")

    return {
        "n_cases": n,
        "n_started_uncertain": nu,
        "resolution_rate_of_uncertain": _rate(sum(1 for r in uncertain if r["resolved"]), nu),
        "mean_questions_on_uncertain": round(
            sum(r["questions_asked"] for r in uncertain) / nu, 4) if nu else 0.0,
        "mean_set_size_before": round(
            sum(r["initial_set_size"] for r in uncertain) / nu, 4) if nu else 0.0,
        "mean_set_size_after": round(
            sum(r["final_set_size"] for r in uncertain) / nu, 4) if nu else 0.0,
        "set_size_reduction": round(
            (sum(r["initial_set_size"] for r in uncertain)
             - sum(r["final_set_size"] for r in uncertain)) / nu, 4) if nu else 0.0,
        "resolved_correctly_rate": _rate(sum(1 for r in uncertain if r["final_correct"]), nu),
        "truth_retained_in_set_rate": _rate(sum(1 for r in records if r["truth_kept_in_set"]), n),
        "under_triage_before": u_before,
        "under_triage_after": u_after,
        "over_triage_before": o_before,
        "over_triage_after": o_after,
    }


def run_followup_evaluation() -> dict:
    kb = DiseaseKB.load()
    full = get_default_classifier()
    out = {
        "regime": "out-of-sample (leave-profiles-out 5-fold) + short message (2-4 tokens); "
                  "answers read from the held-out patient's real symptom profile",
        "max_turns": MAX_TURNS,
        "answer_source": "ground-truth lookup in data/disease_symptoms.csv (no simulation)",
        "strategies": {},
        "negative_evidence": {},
    }
    # Full grid: every questioning policy under both evidence models. Running
    # the cross-product is what answers the question that matters -- whether
    # the information-gain selector earns its complexity. Scored present-only
    # it cannot, because half the information it optimises for ("no") is
    # discarded by the classifier.
    for s in STRATEGIES:
        for use_neg in (False, True):
            key = "with_negative_evidence" if use_neg else "present_only_production"
            out["strategies"].setdefault(s, {})[key] = _summarize(
                _run_one(s, use_neg, kb, full)
            )
    out["negative_evidence"] = {
        k: out["strategies"]["selector"][k]
        for k in ("present_only_production", "with_negative_evidence")
    }
    out["relevance_threshold_sweep"] = sweep_relevance_threshold(kb, full)
    out["symptom_association_characterization"] = characterize_symptom_associations(full)
    return out


# Grid for _MIN_RELEVANCE_FREQ: a token is an eligible question only if it
# appears in at least this fraction of some prediction-set disease's rows.
RELEVANCE_GRID = [0.10, 0.20, 0.30, 0.50, 0.70]


def sweep_relevance_threshold(kb, full) -> list[dict]:
    """Sensitivity of the loop to the clinical-relevance cut-off.

    0.30 was chosen when the constraint was introduced to stop the selector
    asking questions that split the dataset but made no sense for the
    differential. It was never swept, so it was an unexamined constant sitting
    in front of the paper's core mechanism. Reported as a curve: if resolution
    is flat across the grid the exact value does not matter and the choice is
    defensible; if it is peaked, the value needs justifying.
    """
    rows = []
    for thr in RELEVANCE_GRID:
        m = _summarize(_run_one("selector", True, kb, full, min_relevance=thr))
        rows.append({
            "min_relevance": thr,
            "resolution_rate_of_uncertain": m["resolution_rate_of_uncertain"],
            "mean_questions_on_uncertain": m["mean_questions_on_uncertain"],
            "mean_set_size_after": m["mean_set_size_after"],
            "resolved_correctly_rate": m["resolved_correctly_rate"],
            "under_triage_after": m["under_triage_after"],
            "over_triage_after": m["over_triage_after"],
        })
    return rows


def characterize_symptom_associations(full) -> dict:
    """Why the relevance sweep above is flat -- and it is not a wiring bug.

    Measured over every (disease, present-token) pair in the dataset: 85.4% of
    associations hold in >= 95% of that disease's rows, 100% hold in >= 70%,
    and NONE fall below 0.70. The symptom-disease relation in this Kaggle-style
    data is therefore effectively BINARY: a symptom is in essentially all of a
    disease's rows or none of them.

    Two consequences worth stating rather than discovering later:
      * any relevance cut-off in [0, 0.70] selects the identical token pool, so
        the shipped 0.30 is insensitive -- defensible precisely because it does
        not matter, which a sweep can show and an assertion cannot.
      * real clinical symptom-disease relations are probabilistic, not binary,
        so the information-gain selector has far less headroom here than it
        would on real data. This limits how much the follow-up numbers can be
        extrapolated, and belongs in the limitations rather than the results.
    """
    buckets = {">=0.95": 0, "0.70-0.95": 0, "0.30-0.70": 0, "<0.30": 0}
    total = 0
    for _disease, rows in full._rows_by_disease.items():
        n = len(rows)
        if not n:
            continue
        for token in {t for r in rows for t in r}:
            freq = sum(1 for r in rows if token in r) / n
            total += 1
            if freq >= 0.95:
                buckets[">=0.95"] += 1
            elif freq >= 0.70:
                buckets["0.70-0.95"] += 1
            elif freq >= 0.30:
                buckets["0.30-0.70"] += 1
            else:
                buckets["<0.30"] += 1
    return {
        "n_disease_token_pairs": total,
        "frequency_distribution": buckets,
        "relation_is_effectively_binary": buckets["0.30-0.70"] == 0 and buckets["<0.30"] == 0,
        "implication": "any relevance cut-off in [0, 0.70] yields the same token "
                       "pool, so _MIN_RELEVANCE_FREQ=0.30 is insensitive; and the "
                       "near-binary associations give the information-gain "
                       "selector less headroom than real probabilistic data would",
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 86)
    print("ADAPTIVE FOLLOW-UP LOOP EVALUATION (Tier 3)")
    print("=" * 86)
    print(f"  answers: {ev['answer_source']}")
    print(f"  turn budget: {ev['max_turns']}")

    for mode, title in (
        ("present_only_production", "PRESENT-ONLY scoring (production: a 'no' teaches the model nothing)"),
        ("with_negative_evidence", "ABSENCE-AWARE scoring (a 'no' is real evidence)"),
    ):
        print(f"\n  {title}")
        print(f"  {'policy':<12}{'resolved':>10}{'mean Q':>9}{'set before':>12}"
              f"{'set after':>11}{'correct':>9}{'under':>8}{'over':>8}")
        print("  " + "-" * 79)
        for name in STRATEGIES:
            m = ev["strategies"][name][mode]
            print(f"  {name:<12}{m['resolution_rate_of_uncertain']:>10.4f}"
                  f"{m['mean_questions_on_uncertain']:>9.2f}"
                  f"{m['mean_set_size_before']:>12.3f}{m['mean_set_size_after']:>11.3f}"
                  f"{m['resolved_correctly_rate']:>9.4f}"
                  f"{m['under_triage_after']:>8.4f}{m['over_triage_after']:>8.4f}")

    sweep = ev.get("relevance_threshold_sweep") or []
    if sweep:
        print("\n  clinical-relevance threshold sensitivity (selector, absence-aware)")
        print(f"  {'min_rel':<10}{'resolved':>10}{'mean Q':>9}{'set after':>11}"
              f"{'correct':>9}{'under':>8}{'over':>8}")
        print("  " + "-" * 65)
        for r in sweep:
            print(f"  {r['min_relevance']:<10.2f}{r['resolution_rate_of_uncertain']:>10.4f}"
                  f"{r['mean_questions_on_uncertain']:>9.2f}{r['mean_set_size_after']:>11.3f}"
                  f"{r['resolved_correctly_rate']:>9.4f}"
                  f"{r['under_triage_after']:>8.4f}{r['over_triage_after']:>8.4f}")

    ch = ev.get("symptom_association_characterization")
    if ch:
        print(f"\n  why that sweep is flat: over {ch['n_disease_token_pairs']} "
              f"(disease, token) pairs the association frequency is "
              f"{ch['frequency_distribution']}")
        print(f"  effectively binary: {ch['relation_is_effectively_binary']} -- "
              f"any cut-off in [0, 0.70] gives the same pool")

    base = ev["strategies"]["selector"]["with_negative_evidence"]
    print(f"\n  triage effect of the loop (selector, absence-aware): under-triage "
          f"{base['under_triage_before']:.4f} -> {base['under_triage_after']:.4f}, "
          f"over-triage {base['over_triage_before']:.4f} -> {base['over_triage_after']:.4f}")
    print(f"  truth retained in the set after questioning: {base['truth_retained_in_set_rate']:.4f}")

    out = out_dir / "eval_followup_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("adaptive follow-up loop evaluation (Tier 3)")
    print("Running adaptive follow-up evaluation...")
    print_report(run_followup_evaluation(), out_dir)
