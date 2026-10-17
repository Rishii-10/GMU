"""Phase 3 robustness evaluation -- the two planned metrics that were never
built (EVALUATION_METRICS.md rows 4c and 6).

PART 1 -- DECISION CONSISTENCY (planned metric 6)
    "% of cases where the same message (repeated or paraphrased) gets the same
    label, running the full pipeline (Ollama + FAISS + classifier)."

    Two distinct questions, reported separately because they are not the same
    claim:

      REPEAT     the identical message submitted N times. Any disagreement is
                 pure LLM nondeterminism. At temperature 0 this should be
                 near-perfect, and saying so is only meaningful because it is
                 measured end-to-end rather than asserted.

      PARAPHRASE the same clinical content expressed differently -- a different
                 real lay phrasing per symptom, drawn from the repo's existing
                 labelled fixture (tests/fixtures/symptom_calibration.py), so
                 successive variants shift between English, Hindi (Devanagari)
                 and Hinglish. THIS is the number that supports a language
                 robustness claim; REPEAT does not.

    An earlier "consistency = 1.0" figure came from shuffling token ORDER on a
    bag-of-words model, which is order-invariant by construction and could not
    return anything else. That metric was removed. This one can fail, and does.

PART 2 -- REPLY-NOT-UNDERSTOOD RATE (planned metric 4c)
    "% of caller replies the system cannot parse as yes/no."

    PROVENANCE, stated plainly: the reply list in `REPLY_VARIANTS` is authored
    here, not collected from callers. It is therefore a measurement of the
    PARSER'S COVERAGE against documented reply forms -- affirmations,
    negations, hedges and code-mixed variants -- and must not be presented as
    an observed rate of real caller behaviour. Its value is diagnostic: it
    names exactly which reply forms `parse_yes_no` mishandles, including
    hedges ("I think so", "maybe") which SHOULD be unparseable so the caller
    is re-prompted rather than guessed at.

Deterministic apart from the LLM itself. Part 2 needs nothing; Part 1 needs a
running Ollama + FAISS.

Run:  PYTHONPATH=. python tests/evaluate_robustness.py             # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_robustness.py --write     # -> tests/eval_robustness_results.json
      PYTHONPATH=. python tests/evaluate_robustness.py --limit 12  # fewer cases in Part 1
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import parse_yes_no
from app.disease_classifier import get_default_classifier
from tests.evaluate_oos import _rate, _truncate, _unique_profiles

_SEED = 42
REPEATS = 3
PARAPHRASE_VARIANTS = 3
DEFAULT_CASES = 20


# ---------------------------------------------------------------------------
# Part 2 -- yes/no reply coverage
# ---------------------------------------------------------------------------
# (reply, expected) where expected is True / False / None.
# None means "must NOT parse": a hedge has to be re-prompted, never coerced
# into a clinical answer. Those rows are assertions about safety, not misses.
REPLY_VARIANTS: list[tuple[str, "bool | None"]] = [
    # --- affirmations: English ---
    ("yes", True), ("Yes", True), ("YES", True), ("y", True), ("yeah", True),
    ("yep", True), ("yes doctor", True), ("yes, he does", True), ("true", True),
    ("1", True),
    # --- affirmations: Hindi / Hinglish ---
    ("haan", True), ("haan ji", True), ("han", True), ("haa", True), ("ha", True),
    ("ji", True), ("ji haan", True), ("हाँ", True), ("हां", True), ("हा", True),
    # --- negations: English ---
    ("no", False), ("No", False), ("NO", False), ("n", False), ("nope", False),
    ("no doctor", False), ("no, not really", False), ("false", False), ("0", False),
    # --- negations: Hindi / Hinglish ---
    ("nahi", False), ("nahin", False), ("nai", False), ("na", False),
    ("nahi ji", False), ("नहीं", False), ("नही", False), ("ना", False),
    # --- hedges / non-answers: MUST stay unparseable (None) ---
    ("maybe", None), ("I think so", None), ("not sure", None), ("don't know", None),
    ("pata nahi", None), ("sometimes", None), ("a little", None), ("thoda", None),
    ("", None), ("?", None), ("what do you mean", None),
]


def run_reply_coverage() -> dict:
    rows = []
    for reply, expected in REPLY_VARIANTS:
        got = parse_yes_no(reply)
        rows.append({"reply": reply, "expected": expected, "got": got,
                     "correct": got == expected})

    definite = [r for r in rows if r["expected"] is not None]
    hedges = [r for r in rows if r["expected"] is None]
    unparsed_definite = [r for r in definite if r["got"] is None]
    misread = [r for r in definite if r["got"] is not None and not r["correct"]]
    coerced_hedges = [r for r in hedges if r["got"] is not None]

    return {
        "n_variants": len(rows),
        "n_definite": len(definite),
        "n_hedges": len(hedges),
        # The planned metric 4c, over the documented reply forms.
        "reply_not_understood_rate": _rate(len(unparsed_definite), len(definite)),
        "unparsed_definite_replies": [r["reply"] for r in unparsed_definite],
        # A definite reply read as the OPPOSITE answer -- the dangerous failure.
        "misread_rate": _rate(len(misread), len(definite)),
        "misread_replies": [r["reply"] for r in misread],
        # A hedge silently treated as yes/no instead of re-prompting.
        "hedge_coercion_rate": _rate(len(coerced_hedges), len(hedges)),
        "coerced_hedges": [r["reply"] for r in coerced_hedges],
        "provenance": "authored reply list -- measures PARSER COVERAGE over "
                      "documented reply forms, NOT an observed caller rate",
    }


# ---------------------------------------------------------------------------
# Part 1 -- decision consistency (full pipeline)
# ---------------------------------------------------------------------------
def _lay_phrasings() -> dict[str, list[str]]:
    from tests.fixtures.symptom_calibration import CALIBRATION_SAMPLE

    by_token: dict[str, list[str]] = {}
    for text, token in CALIBRATION_SAMPLE:
        if token:
            by_token.setdefault(token, []).append(text)
    return {t: sorted(set(v)) for t, v in by_token.items()}


def _variants(tokens: list[str], lay: dict[str, list[str]], k: int) -> list[str]:
    """k different renderings of the same clinical content, rotating each
    symptom through its real labelled phrasings so variants differ in wording
    AND language. Returns [] when any token has no phrasing on file."""
    if any(t not in lay for t in tokens):
        return []
    out = []
    for v in range(k):
        parts = [lay[t][(v + i) % len(lay[t])] for i, t in enumerate(tokens)]
        body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        out.append(f"I am 35 years old. I have {body}.")
    return out if len({*out}) > 1 else []


def run_decision_consistency(limit: int = DEFAULT_CASES) -> dict:
    from app.agent1_extraction import OllamaBackend, _apply_disambiguation, extract_case
    from app.disambiguation import FAISSDisambiguator
    from app.rules_engine import classify

    backend = OllamaBackend()
    disambiguator = FAISSDisambiguator()
    lay = _lay_phrasings()
    full = get_default_classifier()
    vocab_set = set(full.vocabulary)

    profiles = _unique_profiles(full._rows_by_disease)
    rng = random.Random(_SEED)
    rng.shuffle(profiles)
    trunc = random.Random(_SEED + 1)

    def label_for(message: str) -> "str | None":
        try:
            case = _apply_disambiguation(extract_case(message, backend), disambiguator)
            return classify(case).label.value
        except Exception:
            return None

    repeat_hits = repeat_total = 0
    para_hits = para_total = 0
    para_labels: list[int] = []

    for _, profile in profiles:
        if repeat_total >= limit and para_total >= limit:
            break
        short = [t for t in _truncate(profile, trunc) if t in vocab_set]
        if not short:
            continue

        variants = _variants(short, lay, PARAPHRASE_VARIANTS)
        if not variants:
            continue

        # REPEAT: the first variant, submitted REPEATS times.
        if repeat_total < limit:
            labels = [label_for(variants[0]) for _ in range(REPEATS)]
            if all(l is not None for l in labels):
                repeat_total += 1
                repeat_hits += 1 if len(set(labels)) == 1 else 0

        # PARAPHRASE: one submission of each distinct rendering.
        if para_total < limit:
            labels = [label_for(v) for v in variants]
            if all(l is not None for l in labels):
                para_total += 1
                para_hits += 1 if len(set(labels)) == 1 else 0
                para_labels.append(len(set(labels)))

    return {
        "repeat_cases": repeat_total,
        "repeats_per_case": REPEATS,
        "repeat_label_agreement": _rate(repeat_hits, repeat_total),
        "paraphrase_cases": para_total,
        "variants_per_case": PARAPHRASE_VARIANTS,
        "paraphrase_label_agreement": _rate(para_hits, para_total),
        "distinct_labels_per_paraphrase_set": dict(Counter(para_labels)),
        "note": "REPEAT isolates LLM nondeterminism; PARAPHRASE is the "
                "language-robustness number (real labelled English / Hindi / "
                "Hinglish phrasings of the same content)",
    }


def run_robustness(limit: int | None = None) -> dict:
    return {
        "reply_coverage": run_reply_coverage(),
        "decision_consistency": run_decision_consistency(limit or DEFAULT_CASES),
    }


def print_report(ev: dict, out_dir: Path) -> None:
    r = ev["reply_coverage"]
    print("=" * 78)
    print("ROBUSTNESS EVALUATION (Phase 3)")
    print("=" * 78)
    print("\n  PART 2: yes/no reply coverage (planned metric 4c)")
    print(f"    variants: {r['n_variants']} ({r['n_definite']} definite, {r['n_hedges']} hedges)")
    print(f"    reply-not-understood rate : {r['reply_not_understood_rate']}")
    print(f"    misread (answered wrong)  : {r['misread_rate']}  {r['misread_replies']}")
    print(f"    hedges wrongly coerced    : {r['hedge_coercion_rate']}  {r['coerced_hedges']}")
    if r["unparsed_definite_replies"]:
        print(f"    NOT UNDERSTOOD            : {r['unparsed_definite_replies']}")
    print(f"    provenance: {r['provenance']}")

    c = ev["decision_consistency"]
    print("\n  PART 1: decision consistency, full pipeline (planned metric 6)")
    print(f"    repeat     : {c['repeat_label_agreement']} over {c['repeat_cases']} cases "
          f"x {c['repeats_per_case']} identical submissions")
    print(f"    paraphrase : {c['paraphrase_label_agreement']} over {c['paraphrase_cases']} cases "
          f"x {c['variants_per_case']} real lay renderings")
    print(f"    distinct labels per paraphrase set: {c['distinct_labels_per_paraphrase_set']}")

    out = out_dir / "eval_robustness_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.evaluate_llm import ollama_available
    from tests.eval_output import resolve_out_dir_and_limit

    out_dir, limit = resolve_out_dir_and_limit("robustness evaluation (Phase 3)")

    if not ollama_available():
        print("Ollama not reachable (needs `ollama serve` + llama3.2:3b).")
        raise SystemExit(1)

    print("Running robustness evaluation...")
    print_report(run_robustness(limit), out_dir)
