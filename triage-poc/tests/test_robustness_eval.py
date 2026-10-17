"""Phase 3: the yes/no reply parser, and the robustness harness wiring.

Part 2 of the harness (reply coverage) needs no model, so it is asserted
directly and runs in CI. Part 1 (full-pipeline decision consistency) needs
Ollama + FAISS and is gated.

The safety-relevant assertion here is `hedge_coercion_rate == 0`: a reply like
"maybe" or "pata nahi" must stay UNPARSEABLE so the caller is re-prompted.
Silently reading a hedge as a yes or a no would write an invented clinical fact
into the case -- exactly the failure the Optional[bool] schema invariant exists
to prevent.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import parse_yes_no
from tests.evaluate_robustness import REPLY_VARIANTS, run_reply_coverage

COVERAGE = run_reply_coverage()


def test_every_documented_reply_form_is_classified_as_intended():
    wrong = [(r, e, parse_yes_no(r)) for r, e in REPLY_VARIANTS if parse_yes_no(r) != e]
    assert wrong == [], f"reply forms handled incorrectly: {wrong}"


def test_no_definite_reply_is_left_unparsed():
    # Planned metric 4c over the documented forms.
    assert COVERAGE["reply_not_understood_rate"] == 0.0, (
        f"not understood: {COVERAGE['unparsed_definite_replies']}"
    )


def test_no_definite_reply_is_read_as_the_opposite_answer():
    assert COVERAGE["misread_rate"] == 0.0, f"misread: {COVERAGE['misread_replies']}"


def test_hedges_are_never_coerced_into_a_clinical_answer():
    assert COVERAGE["hedge_coercion_rate"] == 0.0, (
        f"hedges silently answered instead of re-prompted: {COVERAGE['coerced_hedges']}"
    )


@pytest.mark.parametrize("hedge", ["maybe", "I think so", "not sure", "pata nahi",
                                   "sometimes", "a little", "thoda", "", "?"])
def test_hedge_returns_none_so_the_ui_re_prompts(hedge):
    assert parse_yes_no(hedge) is None


def test_coverage_sample_is_balanced_enough_to_be_meaningful():
    assert COVERAGE["n_definite"] >= 30
    assert COVERAGE["n_hedges"] >= 8


def test_provenance_of_the_reply_list_is_declared():
    # The list is authored, not collected; the harness must say so rather than
    # let the number be read as observed caller behaviour.
    assert "NOT an observed caller rate" in COVERAGE["provenance"]


# --- full-pipeline consistency (gated) ------------------------------------
def _ollama_up() -> bool:
    from tests.evaluate_llm import ollama_available

    return ollama_available()


@pytest.mark.skipif(not _ollama_up(), reason="Ollama/llama3.2 not reachable")
def test_decision_consistency_reports_repeat_and_paraphrase_separately():
    """They are different claims: REPEAT measures LLM nondeterminism,
    PARAPHRASE measures language robustness. Collapsing them would let a
    near-perfect repeat score stand in for a language result."""
    from tests.evaluate_robustness import run_decision_consistency

    ev = run_decision_consistency(limit=4)
    for key in ("repeat_label_agreement", "paraphrase_label_agreement",
                "repeat_cases", "paraphrase_cases"):
        assert key in ev
    assert ev["repeat_cases"] > 0
    assert 0.0 <= ev["paraphrase_label_agreement"] <= 1.0
