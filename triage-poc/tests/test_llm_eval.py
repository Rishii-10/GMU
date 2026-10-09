"""Tier 3: the LLM evaluation harness is wired correctly.

Gated on a reachable Ollama, like the other live-model tests, so CI (and any
checkout without the model pulled) skips rather than fails. The expensive full
run is NOT executed here -- tests/evaluate_llm.py --write produces the tracked
baseline. These tests cover the parts that can silently rot: prompt contract,
response parsing, and like-for-like case selection against the OOS harness.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.evaluate_llm import (
    _LABELS,
    TRIAGE_SYSTEM_PROMPT,
    _parse_label,
    _render,
    oos_cases,
    ollama_available,
    run_llm_triage_baseline,
)

requires_ollama = pytest.mark.skipif(
    not ollama_available(), reason="Ollama/llama3.2 not reachable"
)


def test_parse_label_accepts_every_permitted_tier():
    for label in _LABELS:
        assert _parse_label(label) == label
        assert _parse_label(f"  {label.lower()}  ") == label


def test_parse_label_rejects_rather_than_coerces_unknown_output():
    # An unparseable reply must be counted as invalid, never silently mapped
    # onto a tier -- that would hide a real LLM failure mode.
    assert _parse_label("I cannot determine this") is None
    assert _parse_label("") is None


def test_parse_label_takes_the_first_mentioned_tier():
    assert _parse_label("MILD, but could be SEVERE") == "MILD"


def test_prompt_states_every_tier_and_the_safety_convention():
    for label in _LABELS:
        assert label in TRIAGE_SYSTEM_PROMPT
    # The baseline must be given the same safety instruction the framework
    # encodes structurally, or the comparison is a strawman.
    assert "more severe" in TRIAGE_SYSTEM_PROMPT


def test_render_is_plain_readable_text():
    assert _render(["chest_pain", "high_fever"]) == "chest pain, high fever"


def test_baseline_scores_the_same_cases_as_the_oos_harness():
    from tests.evaluate_oos import run_evaluation

    assert len(oos_cases()) == run_evaluation()["n"]


@requires_ollama
def test_llm_baseline_runs_and_reports_required_metrics():
    ev = run_llm_triage_baseline(limit=8)
    for key in ("n_attempted", "n_scored", "invalid_response_rate",
                "under_triage_rate_emergency_severe", "over_triage_rate_mild_moderate",
                "exact_tier_accuracy"):
        assert key in ev
    assert ev["n_attempted"] == 8
