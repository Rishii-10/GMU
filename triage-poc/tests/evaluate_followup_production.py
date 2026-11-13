"""Active follow-up measured through the SHIPPED code path, not a harness copy.

tests/evaluate_followup.py (the source of the paper's 74.8% / 2.15 / 0.056)
re-implements the loop: its own stop rule, a uniform prior for question
selection (production passes the real posteriors), the decision logic of
`_score_variant` instead of rules_engine.classify, and the conformal threshold
calibrated on the FULL dataset. This harness removes every one of those
differences:

  * per fold, a real `DiseaseClassifier` is built from that fold's training rows
    only (its own NB fit AND its own conformal calibration -- nothing from the
    held-out profiles), and installed as the process-wide default classifier;
  * each held-out short message (same folds, same 2-4 token draw as
    evaluate_oos / evaluate_followup) becomes an adult ExtractedCase and goes
    through rules_engine.classify -> next_followup_question ->
    apply_followup_reply -> classify ... -> finalize_after_followup, exactly
    the sequence the Streamlit chat runs (_start_adaptive/_answer_adaptive);
  * the simulated caller answers "yes" iff the asked symptom is in the
    patient's REAL dataset profile, else "no" (a ground-truth oracle, as in
    evaluate_followup -- the paper must say the answers are simulated this way).

Labels: INCOMPLETE_ASSESSMENT on a truly EMERGENCY/SEVERE case is counted as an
under-call (it was not surfaced at its level), the conservative reading.

Run (from triage-poc/):  PYTHONPATH=. python tests/evaluate_followup_production.py [--write]
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.disease_classifier as dc
from app.agent1_extraction import (
    RegexBackend,
    apply_followup_reply,
    finalize_after_followup,
    next_followup_question,
)
from app.disease_kb import DEFAULT_DATA_DIR, DiseaseKB
from app.rules_engine import classify
from app.schemas import SEVERITY_RANK, ClassificationLabel, ExtractedCase
from tests.evaluate_oos import _N_FOLDS, _SEED, _assign_folds, _rate, _truncate, _unique_profiles

MAX_TURNS = 3


def _rank(label: ClassificationLabel) -> int:
    return SEVERITY_RANK.get(label, 99)


def _write_fold_data(dst: Path, train_rows: dict[str, list[list[str]]]) -> None:
    for name in ("disease_precautions.csv", "disease_severity.csv"):
        shutil.copy(Path(DEFAULT_DATA_DIR) / name, dst / name)
    with open(Path(DEFAULT_DATA_DIR) / "disease_symptoms.csv", newline="") as f:
        header = next(csv.reader(f))
    width = len(header) - 1
    with open(dst / "disease_symptoms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for d, rows in train_rows.items():
            for r in rows:
                w.writerow([d] + list(r) + [""] * (width - len(r)))


def _run_case(true_d, short, profile, kb, case: ExtractedCase | None = None):
    """Run one case through the shipped loop. `case` lets the end-to-end harness
    pass the case Agent 1 actually extracted (tokens, denials and all);
    otherwise an adult case is built from the ground-truth short tokens."""
    if case is None:
        case = ExtractedCase(raw_symptom_text=" ".join(short), symptom=", ".join(short),
                             age_group="adult", symptom_tokens=list(short))
    first = classify(case)
    result, trail, backend = first, [], RegexBackend()
    while True:
        nxt = next_followup_question(case, result, trail, backend, max_turns=MAX_TURNS)
        if nxt is None:
            break
        token, question = nxt
        answer = "yes" if token in profile else "no"
        case, _ = apply_followup_reply(case, trail, token, question, answer)
        result = classify(case)
    result = finalize_after_followup(case, result, trail)
    true_rank = _rank(dc.severity_for_disease(true_d, kb))
    # rules_engine leaves prediction_set EMPTY on a CONFIDENT decision and puts
    # the one condition in probable_disease; normalise both to "the set".
    def _set(r):
        if r.prediction_set:
            return list(r.prediction_set)
        return [r.probable_disease] if r.probable_disease else []
    first_set, final_set = _set(first), _set(result)
    return {
        "true_rank": true_rank,
        "initial_rank": _rank(first.label),
        "final_rank": _rank(result.label),
        "started_uncertain": len(first_set) > 1,
        "resolved": len(final_set) == 1,
        "final_correct": final_set == [true_d],
        "truth_kept": true_d in final_set,
        "questions": len(trail),
        "initial_set": len(first_set),
        "final_set": len(final_set),
        "deferred": result.deferred,
    }


def fold_classifier(kb, train_rows, tmp: Path):
    """A production DiseaseClassifier fitted AND conformally calibrated on this
    fold's training rows only."""
    _write_fold_data(tmp, train_rows)
    return dc.DiseaseClassifier(kb, data_dir=tmp)


def _triage(records, key):
    emg = [r for r in records if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]]
    mild = [r for r in records if r["true_rank"] >= SEVERITY_RANK[ClassificationLabel.MODERATE]]
    return (_rate(sum(r[key] > r["true_rank"] for r in emg), len(emg)),
            _rate(sum(r[key] < r["true_rank"] for r in mild), len(mild)))


def run_production_followup_evaluation() -> dict:
    kb = DiseaseKB.load()
    full = dc.get_default_classifier()
    rows_by_disease = full._rows_by_disease
    vocab_set = set(full.vocabulary)
    profiles = _unique_profiles(rows_by_disease)
    fold_of = _assign_folds(profiles, random.Random(_SEED))
    trunc_rng = random.Random(_SEED + 1)  # identical draw order to evaluate_oos

    records = []
    saved = dc._default_classifier
    try:
        for fold_idx in range(_N_FOLDS):
            held = [k for k, f in fold_of.items() if f == fold_idx]
            held_by_d = {}
            for d, p in held:
                held_by_d.setdefault(d, set()).add(p)
            train = {d: [r for r in rows if tuple(sorted(r)) not in held_by_d.get(d, set())]
                     for d, rows in rows_by_disease.items()}
            train = {d: rows for d, rows in train.items() if rows}
            with tempfile.TemporaryDirectory() as tmp:
                dc._default_classifier = fold_classifier(kb, train, Path(tmp))
                for true_d, profile in held:
                    short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
                    if short:
                        records.append(_run_case(true_d, short, set(profile), kb))
    finally:
        dc._default_classifier = saved

    unc = [r for r in records if r["started_uncertain"]]
    nu = len(unc)
    u0, o0 = _triage(records, "initial_rank")
    u1, o1 = _triage(records, "final_rank")
    harness = json.loads((Path(__file__).resolve().parent / "eval_followup_results.json").read_text())
    h = harness["strategies"]["selector"]["with_negative_evidence"]
    return {
        "regime": "out-of-sample per-fold production DiseaseClassifier (own NB + own conformal "
                  "calibration); same held-out short messages as evaluate_oos; shipped step functions",
        "answer_source": "simulated caller: 'yes' iff the asked symptom is in the patient's real "
                         "dataset profile, else 'no' (ground-truth oracle, no hedges)",
        "max_turns": MAX_TURNS,
        "n_cases": len(records),
        "n_started_uncertain": nu,
        "resolution_rate_of_uncertain": _rate(sum(r["resolved"] for r in unc), nu),
        "resolved_correctly_rate": _rate(sum(r["final_correct"] for r in unc), nu),
        "truth_retained_in_set_rate": _rate(sum(r["truth_kept"] for r in records), len(records)),
        "mean_questions_on_uncertain": round(sum(r["questions"] for r in unc) / nu, 4) if nu else 0.0,
        "mean_set_size_before": round(sum(r["initial_set"] for r in unc) / nu, 4) if nu else 0.0,
        "mean_set_size_after": round(sum(r["final_set"] for r in unc) / nu, 4) if nu else 0.0,
        "deferred_rate_of_uncertain": _rate(sum(r["deferred"] for r in unc), nu),
        "under_triage_before": u0,
        "under_triage_after": u1,
        "over_triage_before": o0,
        "over_triage_after": o1,
        "harness_copy_for_comparison": {
            "source": "tests/eval_followup_results.json strategies.selector.with_negative_evidence",
            "n_started_uncertain": h["n_started_uncertain"],
            "resolution_rate_of_uncertain": h["resolution_rate_of_uncertain"],
            "mean_questions_on_uncertain": h["mean_questions_on_uncertain"],
            "under_triage_after": h["under_triage_after"],
            "over_triage_after": h["over_triage_after"],
        },
    }


def print_report(ev: dict, out_dir: Path) -> None:
    h = ev["harness_copy_for_comparison"]
    print("=" * 74)
    print("ACTIVE FOLLOW-UP THROUGH THE SHIPPED CODE PATH")
    print("=" * 74)
    rows = [
        ("started uncertain", ev["n_started_uncertain"], h["n_started_uncertain"]),
        ("resolved (of uncertain)", ev["resolution_rate_of_uncertain"], h["resolution_rate_of_uncertain"]),
        ("mean questions", ev["mean_questions_on_uncertain"], h["mean_questions_on_uncertain"]),
        ("under-call after", ev["under_triage_after"], h["under_triage_after"]),
        ("over-call after", ev["over_triage_after"], h["over_triage_after"]),
        ("deferred (of uncertain)", ev["deferred_rate_of_uncertain"], "-"),
    ]
    print(f"  {'metric':<26}{'shipped path':>14}{'harness copy':>14}")
    for name, a, b in rows:
        print(f"  {name:<26}{a!s:>14}{b!s:>14}")
    out = out_dir / "eval_followup_production_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    out_dir = resolve_out_dir("active follow-up through the shipped step functions")
    print_report(run_production_followup_evaluation(), out_dir)
