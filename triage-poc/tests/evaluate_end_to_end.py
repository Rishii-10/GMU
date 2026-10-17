"""Phase 3 END-TO-END evaluation: free text -> triage tier, through the real
extraction and matching path.

WHY THIS EXISTS
    Every other harness scores SYMPTOM TOKENS directly. That isolates the
    classifier cleanly, but it means the framework's own first two stages --
    Agent 1 LLM extraction (E1) and FAISS symptom matching (E2) -- are excluded
    from the headline safety numbers, and no single number describes the
    composed system. For a framework paper whose claim is coordination between
    stages, that was the central evidential gap. This harness closes it.

WHAT IT MEASURES
    The SAME held-out profiles, the SAME leave-profiles-out folds, and the SAME
    labelling logic (`_score_variant` from evaluate_oos) are used twice:

      ORACLE      : scored on the ground-truth short token list
      END-TO-END  : scored on tokens actually recovered from rendered text

    Holding the classifier regime fixed means the DIFFERENCE between the two is
    attributable to extraction + matching, and nothing else. That delta is the
    number the paper should quote as the real-world cost of the NLP front end.

HOW THE TEXT IS PRODUCED (no simulation, nothing invented) -- TWO MODES
    CANONICAL (n = all held-out profiles). The profile's tokens are
    de-underscored and joined by a fixed template. This is close to feeding the
    vocabulary back verbatim, so it measures a CEILING -- "can the pipeline
    carry symptoms it is handed in clean prose" -- and is reported as a wiring
    check, not as a language result. Token recall here is ~1.0 by construction.

    LAY (n = profiles whose every token has a real lay phrasing on file). Each
    token is replaced by an actual caregiver-style phrase drawn from the repo's
    existing labelled fixture, tests/fixtures/symptom_calibration.py -- e.g.
    high_fever -> "tez bukhar" / "बुखार" / "I have a really high temperature",
    chest_pain -> "chhati mein dard" / "छाती में दर्द" / "my chest hurts".
    Phrasings are rotated across English / Hindi / Hinglish so a single message
    is genuinely code-mixed. These phrases were labelled in-repo before this
    harness existed and are NOT authored here, so this is a real measurement of
    the language robustness the paper claims, not a favourable restatement of
    the vocabulary.

    LAY is the mode to quote. Its sample is smaller (the fixture covers 34 of
    131 tokens, so ~50 of 304 profiles are fully coverable) and that limit is
    reported alongside the number rather than hidden.

Needs a running Ollama (llama3.2:3b) and FAISS. Skipped automatically when
Ollama is unreachable.

Run:  PYTHONPATH=. python tests/evaluate_end_to_end.py              # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_end_to_end.py --write      # -> tests/eval_end_to_end_results.json
      PYTHONPATH=. python tests/evaluate_end_to_end.py --limit 40   # quick pass
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
    _fit_naive_bayes,
    get_default_classifier,
    severity_for_disease,
)
from app.disease_kb import DiseaseKB
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
_ADULT_AGE_YEARS = 35  # keeps every case on the adult/dataset pathway
MODES = ("canonical", "lay")


def _lay_phrasings() -> dict[str, list[str]]:
    """token -> real caregiver phrasings, from the repo's existing labelled
    fixture. Sorted for determinism. Nothing is authored here."""
    from tests.fixtures.symptom_calibration import CALIBRATION_SAMPLE

    by_token: dict[str, list[str]] = {}
    for text, token in CALIBRATION_SAMPLE:
        if token:
            by_token.setdefault(token, []).append(text)
    return {t: sorted(set(v)) for t, v in by_token.items()}


def _render_message(tokens: list[str], mode: str, lay: dict[str, list[str]],
                    turn: int = 0) -> "str | None":
    """Render a token list as a complaint. Returns None when `mode` is "lay"
    and some token has no real phrasing on file (that profile is skipped
    rather than silently falling back to the canonical token, which would
    quietly inflate the lay result)."""
    if mode == "canonical":
        parts = [t.replace("_", " ") for t in tokens]
    else:
        if any(t not in lay for t in tokens):
            return None
        # Rotate through the available phrasings so successive tokens pull
        # from different languages -> genuinely code-mixed messages.
        parts = [lay[t][(turn + i) % len(lay[t])] for i, t in enumerate(tokens)]

    body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"I am {_ADULT_AGE_YEARS} years old. I have {body}."


def run_end_to_end(limit: int | None = None) -> dict:
    """Run both rendering modes over the same held-out folds."""
    out = {
        "alpha": DEFAULT_CP_ALPHA,
        "regime": "END-TO-END: rendered free text -> Agent 1 (Ollama) -> FAISS -> "
                  "conformal gate -> tier; leave-profiles-out folds, same labelling "
                  "logic as evaluate_oos.py",
        "modes": {},
    }
    for mode in MODES:
        out["modes"][mode] = _run_mode(mode, limit)
    return out


def _run_mode(mode: str, limit: int | None = None) -> dict:
    from app.agent1_extraction import (
        ExtractionValidationError,
        OllamaBackend,
        _apply_disambiguation,
        extract_case,
    )
    from app.disambiguation import FAISSDisambiguator

    backend = OllamaBackend()
    disambiguator = FAISSDisambiguator()
    lay = _lay_phrasings()

    kb = DiseaseKB.load()
    full = get_default_classifier()
    rows_by_disease = full._rows_by_disease
    vocab = full.vocabulary
    vocab_set = set(vocab)
    emergency_names = full._emergency_disease_names
    flags = _VARIANTS[_FULL_SYSTEM]

    profiles = _unique_profiles(rows_by_disease)
    fold_of = _assign_folds(profiles, random.Random(_SEED))
    trunc_rng = random.Random(_SEED + 1)

    records: list[dict] = []
    extraction_failures = 0
    attempted = 0
    skipped_no_phrasing = 0
    turn = 0

    for fold_idx in range(_N_FOLDS):
        held = [k for k, f in fold_of.items() if f == fold_idx]
        held_by_d: dict[str, set] = {}
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
            if limit is not None and attempted >= limit:
                break
            short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
            if not short:
                continue
            message = _render_message(short, mode, lay, turn)
            if message is None:
                skipped_no_phrasing += 1
                continue
            turn += 1
            attempted += 1

            try:
                case = _apply_disambiguation(extract_case(message, backend), disambiguator)
                extracted = [t for t in (case.symptom_tokens or []) if t in vocab_set]
            except (ExtractionValidationError, ValueError, KeyError):
                extraction_failures += 1
                extracted = []

            true_rank = SEVERITY_RANK[severity_for_disease(true_d, kb)]
            oracle = _score_variant(fold_clf, short, kb, **flags)
            e2e = (
                _score_variant(fold_clf, extracted, kb, **flags)
                if extracted
                else {"label": ClassificationLabel.INCOMPLETE_ASSESSMENT.value,
                      "decision": "ABSTAIN", "set_size": 0,
                      "prediction_set": [], "top": None, "lac_empty_fallback": False}
            )

            want, got = set(short), set(extracted)
            records.append({
                "true_rank": true_rank,
                "oracle_label_rank": SEVERITY_RANK.get(ClassificationLabel(oracle["label"]), 99),
                "e2e_label_rank": SEVERITY_RANK.get(ClassificationLabel(e2e["label"]), 99),
                "oracle_label": oracle["label"],
                "e2e_label": e2e["label"],
                "n_want": len(want),
                "n_recovered": len(want & got),
                "n_spurious": len(got - want),
                "n_extracted": len(got),
                "exact_token_match": want == got,
                "e2e_decision": e2e["decision"],
            })
        if limit is not None and attempted >= limit:
            break

    return _summarize(records, extraction_failures, attempted, mode, skipped_no_phrasing)


def _summarize(records: list[dict], extraction_failures: int, attempted: int,
               mode: str, skipped_no_phrasing: int) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "mode": mode, "error": "no cases scored"}

    emg = [r for r in records if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]]
    mild = [r for r in records if r["true_rank"] >= SEVERITY_RANK[ClassificationLabel.MODERATE]]

    def under(rs, key):
        return sum(1 for r in rs if r[key] > r["true_rank"])

    def over(rs, key):
        return sum(1 for r in rs if r[key] < r["true_rank"])

    want_total = sum(r["n_want"] for r in records)
    recovered = sum(r["n_recovered"] for r in records)
    extracted_total = sum(r["n_extracted"] for r in records)
    spurious = sum(r["n_spurious"] for r in records)

    e2e_under = _rate(under(emg, "e2e_label_rank"), len(emg))
    oracle_under = _rate(under(emg, "oracle_label_rank"), len(emg))
    e2e_over = _rate(over(mild, "e2e_label_rank"), len(mild))
    oracle_over = _rate(over(mild, "oracle_label_rank"), len(mild))

    return {
        "n": n,
        "mode": mode,
        "text_source": (
            "canonical: de-underscored dataset tokens in a fixed template -- a "
            "CEILING / wiring check, recall ~1.0 by construction"
            if mode == "canonical" else
            "lay: every symptom replaced by a real caregiver phrasing from the "
            "pre-existing labelled fixture tests/fixtures/symptom_calibration.py "
            "(English / Hindi / Hinglish, rotated so messages are code-mixed); "
            "profiles lacking a phrasing for any token are SKIPPED, not "
            "back-filled with the canonical token"
        ),
        "profiles_skipped_no_lay_phrasing": skipped_no_phrasing,
        # The lay subsample is selected by FIXTURE COVERAGE, not at random, so
        # its absolute rates are not comparable to the full 304-profile set --
        # whichever diseases happen to have labelled lay phrasings dominate it.
        # Only the ORACLE-vs-END-TO-END delta on the same rows is meaningful,
        # which is why both are always reported side by side. (Concretely: lay
        # over-triage reads 0.84 while the oracle on those identical rows reads
        # 0.79 -- a 0.05 extraction cost, not an 84% failure.)
        "sample_is_representative": mode == "canonical",
        "comparison_rule": "compare end_to_end against oracle WITHIN a mode; do "
                           "not compare absolute rates ACROSS modes",

        # --- extraction + matching quality (E1 + E2) ---
        "extraction_failure_rate": _rate(extraction_failures, attempted),
        "token_recall": _rate(recovered, want_total),
        "token_precision": _rate(recovered, extracted_total),
        "spurious_tokens_per_case": round(spurious / n, 4),
        "exact_token_match_rate": _rate(sum(1 for r in records if r["exact_token_match"]), n),

        # --- the composed system ---
        "end_to_end_under_triage": e2e_under,
        "end_to_end_over_triage": e2e_over,
        "end_to_end_decision_counts": dict(Counter(r["e2e_decision"] for r in records)),

        # --- the same cases scored on ground-truth tokens ---
        "oracle_under_triage": oracle_under,
        "oracle_over_triage": oracle_over,

        # --- the delta: what the NLP front end costs ---
        "under_triage_cost_of_extraction": round(e2e_under - oracle_under, 4),
        "over_triage_cost_of_extraction": round(e2e_over - oracle_over, 4),
        "label_agreement_with_oracle": _rate(
            sum(1 for r in records if r["e2e_label"] == r["oracle_label"]), n
        ),
        "emergency_severe_n": len(emg),
        "mild_moderate_n": len(mild),
    }


def print_report(ev: dict, out_dir: Path) -> None:
    print("=" * 78)
    print("END-TO-END EVALUATION (Phase 3) -- text in, tier out")
    print("=" * 78)
    print(f"  alpha = {ev['alpha']}")

    for mode in MODES:
        m = ev["modes"].get(mode, {})
        if not m or m.get("error"):
            print(f"\n  [{mode}] no cases scored")
            continue
        label = ("CEILING / wiring check" if mode == "canonical"
                 else "REAL LAY LANGUAGE -- quote this one")
        print(f"\n  --- {mode.upper()}  (n = {m['n']})  {label} ---")
        if mode == "lay":
            print(f"      skipped for lack of a labelled phrasing: "
                  f"{m['profiles_skipped_no_lay_phrasing']}")
            print("      NOTE: this subsample is selected by fixture coverage, not at")
            print("      random -- read the oracle/end-to-end DELTA, not absolute rates.")
        print(f"      {'extraction failure':<26}{m['extraction_failure_rate']:>9}")
        print(f"      {'token recall':<26}{m['token_recall']:>9}")
        print(f"      {'token precision':<26}{m['token_precision']:>9}")
        print(f"      {'exact token match':<26}{m['exact_token_match_rate']:>9}")
        print(f"      {'spurious tokens/case':<26}{m['spurious_tokens_per_case']:>9}")
        print(f"      {'':<26}{'end-to-end':>12}{'oracle':>10}{'cost':>9}")
        print("      " + "-" * 57)
        print(f"      {'under-triage':<26}{m['end_to_end_under_triage']:>12.4f}"
              f"{m['oracle_under_triage']:>10.4f}"
              f"{m['under_triage_cost_of_extraction']:>9.4f}")
        print(f"      {'over-triage':<26}{m['end_to_end_over_triage']:>12.4f}"
              f"{m['oracle_over_triage']:>10.4f}"
              f"{m['over_triage_cost_of_extraction']:>9.4f}")
        print(f"      label agreement with oracle: {m['label_agreement_with_oracle']:.4f}"
              f"   decisions: {m['end_to_end_decision_counts']}")

    out = out_dir / "eval_end_to_end_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.evaluate_llm import ollama_available
    from tests.eval_output import resolve_out_dir_and_limit

    out_dir, limit = resolve_out_dir_and_limit("end-to-end evaluation (Phase 3)")

    if not ollama_available():
        print("Ollama not reachable (needs `ollama serve` + llama3.2:3b).")
        raise SystemExit(1)

    print("Running end-to-end evaluation (one LLM extraction per case)...")
    print_report(run_end_to_end(limit), out_dir)
