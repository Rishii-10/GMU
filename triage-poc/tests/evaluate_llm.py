"""Tier 3 evaluation of the LLM component itself -- the baseline a reviewer
will demand, plus the reliability numbers that "hardening an LLM" actually
means for a paper.

PART 1 -- LLM-ONLY TRIAGE BASELINE ("why not just prompt the model?")
    The obvious challenge to this framework is that a modern LLM could do the
    triage directly. So llama3.2:3b is asked to assign the same four tiers on
    exactly the same held-out cases, and scored with the same metrics.

    FAIRNESS: the LLM is given the SAME information the framework's harness
    gets -- the patient's 2-4 observed symptoms -- rendered as plain text, and
    a system prompt that states the tier definitions and the clinical
    convention that missing an emergency is worse than over-calling one. It is
    not a strawman: nothing is withheld, and the safety instruction the
    framework encodes structurally is given to the LLM in words.

PART 2 -- LLM RELIABILITY (hardening = measuring, not adding machinery)
    * determinism     -- the same prompt at temperature 0, repeated: does the
                         label change between identical calls?
    * output validity -- how often the model returns something outside the
                         four permitted labels (a failure the framework must
                         handle)
    * extraction      -- app.agent1_extraction.extract_case() over the repo's
                         existing labelled fixture (tests/fixtures/
                         symptom_calibration.py, 80 real Hindi/Hinglish/English
                         phrases): does it return schema-valid output, and is
                         it stable across identical calls?

Needs a running Ollama (`ollama serve`, llama3.2:3b). Skipped automatically
when unreachable.

Run:  PYTHONPATH=. python tests/evaluate_llm.py                 # -> tests/eval_runs/<ts>/
      PYTHONPATH=. python tests/evaluate_llm.py --write         # -> tests/eval_llm_results.json
      PYTHONPATH=. python tests/evaluate_llm.py --limit 40      # quick pass
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app.disease_classifier import get_default_classifier, severity_for_disease
from app.disease_kb import DiseaseKB
from app.schemas import SEVERITY_RANK, ClassificationLabel
from tests.evaluate_oos import _assign_folds, _rate, _truncate, _unique_profiles

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"
_SEED = 42
_N_FOLDS = 5

_LABELS = ("EMERGENCY", "SEVERE", "MODERATE", "MILD")

# Tier definitions mirror data/disease_severity.csv's semantics, and the
# safety convention the framework enforces structurally is stated explicitly
# so the baseline is not handicapped relative to the framework.
TRIAGE_SYSTEM_PROMPT = (
    "You are a clinical triage classifier for a rural health service.\n"
    "Given a patient's reported symptoms, assign exactly ONE urgency tier:\n"
    "  EMERGENCY - immediately life-threatening; needs emergency care now "
    "(e.g. heart attack, stroke/brain haemorrhage).\n"
    "  SEVERE    - serious; needs a doctor the same day "
    "(e.g. pneumonia, tuberculosis, dengue, typhoid, hepatitis).\n"
    "  MODERATE  - needs medical attention soon, not same-day "
    "(e.g. malaria, migraine, arthritis, hypertension).\n"
    "  MILD      - can be managed at home (e.g. common cold, acne, allergy).\n"
    "Missing an emergency is far worse than over-calling one: when uncertain "
    "between two tiers, choose the more severe.\n"
    "Reply with ONLY the single tier word. No explanation, no punctuation."
)


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200 and MODEL.split(":")[0] in r.text
    except requests.RequestException:
        return False


def _ask(prompt: str, system: str, temperature: float = 0.0) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "system": system,
              "stream": False, "options": {"temperature": temperature}},
        timeout=60,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def _parse_label(text: str) -> str | None:
    """First permitted tier word appearing in the reply, else None (counted as
    an invalid response rather than silently coerced)."""
    up = text.upper()
    hits = [(up.find(l), l) for l in _LABELS if up.find(l) >= 0]
    return min(hits)[1] if hits else None


def _render(tokens: list[str]) -> str:
    return ", ".join(t.replace("_", " ") for t in tokens)


def oos_cases(limit: int | None = None) -> list[tuple[str, list[str]]]:
    """The SAME held-out short messages evaluate_oos.py scores (same folds,
    same seeds, same truncations), so the comparison is like-for-like."""
    full = get_default_classifier()
    rows_by_disease = full._rows_by_disease
    vocab_set = set(full.vocabulary)
    profiles = _unique_profiles(rows_by_disease)
    fold_of = _assign_folds(profiles, random.Random(_SEED))
    trunc_rng = random.Random(_SEED + 1)

    cases = []
    for fold_idx in range(_N_FOLDS):
        for true_d, profile in [k for k, f in fold_of.items() if f == fold_idx]:
            short = [t for t in _truncate(profile, trunc_rng) if t in vocab_set]
            if short:
                cases.append((true_d, short))
    return cases[:limit] if limit else cases


def run_llm_triage_baseline(limit: int | None = None) -> dict:
    kb = DiseaseKB.load()
    cases = oos_cases(limit)
    records, invalid, errors = [], 0, 0

    for true_d, tokens in cases:
        try:
            reply = _ask(f"Patient's symptoms: {_render(tokens)}", TRIAGE_SYSTEM_PROMPT)
        except requests.RequestException:
            errors += 1
            continue
        label = _parse_label(reply)
        if label is None:
            invalid += 1
            continue
        records.append({
            "true_rank": SEVERITY_RANK[severity_for_disease(true_d, kb)],
            "label_rank": SEVERITY_RANK[ClassificationLabel(label)],
            "label": label,
        })

    n = len(records)
    emg = [r for r in records if r["true_rank"] <= SEVERITY_RANK[ClassificationLabel.SEVERE]]
    mild = [r for r in records if r["true_rank"] >= SEVERITY_RANK[ClassificationLabel.MODERATE]]
    return {
        "n_attempted": len(cases),
        "n_scored": n,
        "invalid_response_rate": _rate(invalid, len(cases)),
        "transport_error_rate": _rate(errors, len(cases)),
        "under_triage_rate_emergency_severe": _rate(
            sum(1 for r in emg if r["label_rank"] > r["true_rank"]), len(emg)),
        "under_triage_count": sum(1 for r in emg if r["label_rank"] > r["true_rank"]),
        "emergency_severe_n": len(emg),
        "over_triage_rate_mild_moderate": _rate(
            sum(1 for r in mild if r["label_rank"] < r["true_rank"]), len(mild)),
        "exact_tier_accuracy": _rate(
            sum(1 for r in records if r["label_rank"] == r["true_rank"]), n),
        "label_distribution": dict(Counter(r["label"] for r in records)),
    }


def run_llm_determinism(n_cases: int = 25, repeats: int = 3) -> dict:
    """Same prompt, temperature 0, repeated. Any disagreement is
    non-determinism the surrounding system has to tolerate."""
    cases = oos_cases(n_cases)
    stable = comparable = 0
    for _, tokens in cases:
        replies = []
        for _ in range(repeats):
            try:
                replies.append(_parse_label(
                    _ask(f"Patient's symptoms: {_render(tokens)}", TRIAGE_SYSTEM_PROMPT)))
            except requests.RequestException:
                pass
        if len(replies) == repeats:
            comparable += 1
            stable += 1 if len(set(replies)) == 1 else 0
    return {
        "cases": comparable,
        "repeats_per_case": repeats,
        "label_stability_rate": _rate(stable, comparable),
    }


def run_extraction_reliability(limit: int = 30) -> dict:
    """Schema validity + token accuracy of Agent 1 over the repo's existing
    labelled fixture (real English/Hindi/Hinglish phrases, not invented here).
    This is the end-to-end E1+E2 quality that the token-level OOS harness
    deliberately does not measure (limitation L4)."""
    from app.agent1_extraction import (
        ExtractionValidationError,
        OllamaBackend,
        _apply_disambiguation,
        extract_case,
    )
    from app.disambiguation import FAISSDisambiguator
    from tests.fixtures.symptom_calibration import CALIBRATION_SAMPLE

    backend = OllamaBackend()
    disamb = FAISSDisambiguator()
    sample = [(t, e) for t, e in CALIBRATION_SAMPLE if e is not None][:limit]

    valid = matched = failures = 0
    for text, expected in sample:
        try:
            case = _apply_disambiguation(extract_case(text, backend), disamb)
            valid += 1
            if expected in (case.symptom_tokens or []):
                matched += 1
        except (ExtractionValidationError, requests.RequestException, ValueError):
            failures += 1
    n = len(sample)
    return {
        "n": n,
        "schema_valid_rate": _rate(valid, n),
        "extraction_failure_rate": _rate(failures, n),
        "expected_token_recall": _rate(matched, n),
        "note": "end-to-end E1+E2 on labelled real phrases; the OOS harness "
                "scores tokens directly and excludes this error source (L4)",
    }


def run_llm_evaluation(limit: int | None = None) -> dict:
    return {
        "model": MODEL,
        "triage_baseline": run_llm_triage_baseline(limit),
        "determinism": run_llm_determinism(),
        "extraction_reliability": run_extraction_reliability(),
    }


def print_report(ev: dict, out_dir: Path) -> None:
    b = ev["triage_baseline"]
    print("=" * 78)
    print(f"LLM EVALUATION (Tier 3) -- model {ev['model']}")
    print("=" * 78)
    print("\n  PART 1: LLM-only triage baseline (same held-out cases as the framework)")
    for k in ("n_attempted", "n_scored", "invalid_response_rate", "transport_error_rate",
              "under_triage_rate_emergency_severe", "under_triage_count",
              "emergency_severe_n", "over_triage_rate_mild_moderate",
              "exact_tier_accuracy", "label_distribution"):
        print(f"    {k}: {b[k]}")

    d = ev["determinism"]
    print("\n  PART 2a: determinism at temperature 0")
    print(f"    {d['cases']} cases x {d['repeats_per_case']} identical calls -> "
          f"label stability {d['label_stability_rate']:.4f}")

    e = ev["extraction_reliability"]
    print("\n  PART 2b: Agent-1 extraction on the labelled fixture")
    for k in ("n", "schema_valid_rate", "extraction_failure_rate", "expected_token_recall"):
        print(f"    {k}: {e[k]}")

    out = out_dir / "eval_llm_results.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    from tests.eval_output import resolve_out_dir

    if not ollama_available():
        print(f"Ollama not reachable at {OLLAMA_URL} with {MODEL}. Start `ollama serve`.")
        raise SystemExit(1)

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    out_dir = resolve_out_dir("LLM baseline + reliability (Tier 3)")
    print(f"Running LLM evaluation ({MODEL})... this calls the model many times.")
    print_report(run_llm_evaluation(limit), out_dir)
