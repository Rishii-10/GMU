"""DEC-2 production tests: an explicitly DENIED symptom is real evidence.

Before this, a caller answering "no" changed nothing -- the token was recorded
in the trail purely so it would not be asked twice. That left the
information-gain question selector optimising for evidence the classifier then
discarded, and it showed: the selector resolved no more uncertain cases than
asking a random clinically-relevant question (0.204 vs 0.243). With denials
scored, resolution rises to 0.767 and the selector dominates both baselines
(tests/evaluate_followup.py).

The tests below lock the two properties that make this safe to ship:

  1. NO REGRESSION -- with no denials, every number is bit-identical to the
     present-only model. The first classification of any case is unaffected.
  2. DENIALS ONLY NARROW -- a "no" can shrink or re-rank the differential but
     can never introduce a disease the positive evidence did not support, and
     can never escalate the triage tier.

Property 2 is not theoretical. Applying the Bernoulli absence term across all
41 diseases (the obvious implementation) let denials flatten the posterior:
["cough"] with chest_pain/phlegm/high_fever denied produced a 28-disease set
containing Paralysis and positional vertigo, which the worst-case rule then
escalated to EMERGENCY. Denials made the system less certain and more alarmed.
_nb_posteriors_with_absence therefore restricts the penalty to the
positive-evidence candidate pool; these tests fail if that restriction is lost.

Classifier-only (no Ollama, no FAISS) -- runs in CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import record_followup_answer
from app.disease_classifier import get_default_classifier
from app.rules_engine import classify
from app.schemas import SEVERITY_RANK, ClassificationLabel, ExtractedCase

clf = get_default_classifier()


def _case(tokens, absent=None, age_months=420):
    return ExtractedCase(
        raw_symptom_text="test",
        age_months=age_months,
        symptom_tokens=list(tokens),
        absent_symptom_tokens=list(absent or []),
    )


# --- Property 1: no denials => no behaviour change ------------------------


def test_no_denials_is_identical_to_present_only():
    for tokens in (["cough"], ["cough", "high_fever"], ["chest_pain"],
                   ["chest_pain", "sweating", "breathlessness", "vomiting"]):
        a = clf.classify_with_cp(tokens)
        b = clf.classify_with_cp(tokens, absent_tokens=[])
        c = clf.classify_with_cp(tokens, absent_tokens=None)
        assert a.prediction_set == b.prediction_set == c.prediction_set
        assert a.decision == b.decision == c.decision
        assert a.q_hat == b.q_hat == c.q_hat


def test_empty_case_absent_list_is_the_default():
    assert _case(["cough"]).absent_symptom_tokens == []


def test_a_denied_token_already_present_is_ignored():
    # An explicit mention outranks a denial; scoring must not change.
    plain = clf.classify_with_cp(["cough", "high_fever"])
    conflicted = clf.classify_with_cp(["cough", "high_fever"], absent_tokens=["cough"])
    assert plain.prediction_set == conflicted.prediction_set


# --- Property 2: denials only narrow, never widen or escalate -------------


def test_denials_never_widen_the_prediction_set():
    base = clf.classify_with_cp(["cough"])
    progressive = [
        ["phlegm"],
        ["phlegm", "high_fever"],
        ["chest_pain", "phlegm", "high_fever"],
        ["chest_pain", "phlegm", "high_fever", "breathlessness", "fatigue"],
    ]
    previous = base.set_size
    for denied in progressive:
        got = clf.classify_with_cp(["cough"], absent_tokens=denied)
        assert got.set_size <= base.set_size, (
            f"denying {denied} widened the set {base.set_size} -> {got.set_size}"
        )
        assert got.set_size <= previous  # monotone as evidence accumulates
        previous = got.set_size


def test_denials_never_introduce_unsupported_diseases():
    """The regression that motivated the pool restriction: Paralysis and
    vertigo must not appear in a cough differential because other symptoms
    were denied."""
    base = set(clf.classify_with_cp(["cough"]).prediction_set)
    got = clf.classify_with_cp(
        ["cough"], absent_tokens=["chest_pain", "phlegm", "high_fever"]
    )
    assert set(got.prediction_set) <= base, (
        f"denials admitted diseases positive evidence never supported: "
        f"{set(got.prediction_set) - base}"
    )


def test_denials_never_escalate_the_triage_tier():
    baseline = classify(_case(["cough"]))
    for denied in (["phlegm"], ["chest_pain", "phlegm", "high_fever"],
                   ["chest_pain", "phlegm", "high_fever", "breathlessness"]):
        got = classify(_case(["cough"], absent=denied))
        assert SEVERITY_RANK[got.label] >= SEVERITY_RANK[baseline.label], (
            f"denying {denied} escalated {baseline.label.value} -> {got.label.value}"
        )


def test_ruling_out_severe_diseases_de_escalates_the_label():
    """The point of the loop: as severe candidates are excluded, the
    worst-case label should relax."""
    before = classify(_case(["cough"]))
    after = classify(
        _case(["cough"], absent=["phlegm", "high_fever", "chest_pain",
                                 "breathlessness", "fatigue"])
    )
    assert before.label in (ClassificationLabel.SEVERE, ClassificationLabel.EMERGENCY)
    assert SEVERITY_RANK[after.label] > SEVERITY_RANK[before.label]


# --- The capture path: record_followup_answer ------------------------------


def test_no_answer_is_recorded_as_a_denial():
    case, trail = _case(["cough"]), []
    out = record_followup_answer(case, trail, "phlegm", "Any phlegm?", "no")
    assert out.absent_symptom_tokens == ["phlegm"]
    assert out.symptom_tokens == ["cough"]       # untouched
    assert trail[-1]["answer"] == "no"


def test_yes_answer_still_only_touches_present_tokens():
    out = record_followup_answer(_case(["cough"]), [], "phlegm", "Any phlegm?", "yes")
    assert out.symptom_tokens == ["cough", "phlegm"]
    assert out.absent_symptom_tokens == []


def test_unparseable_answer_updates_neither_list():
    out = record_followup_answer(_case(["cough"]), [], "phlegm", "Any phlegm?", "maybe")
    assert out.symptom_tokens == ["cough"]
    assert out.absent_symptom_tokens == []


def test_hindi_no_is_recorded_as_a_denial():
    for reply in ("nahi", "nahin", "ना", "नहीं"):
        out = record_followup_answer(_case(["cough"]), [], "phlegm", "Q?", reply)
        assert out.absent_symptom_tokens == ["phlegm"], f"{reply!r} not parsed as no"


def test_a_token_is_never_denied_twice():
    case, trail = _case(["cough"]), []
    case = record_followup_answer(case, trail, "phlegm", "Q?", "no")
    case = record_followup_answer(case, trail, "phlegm", "Q?", "no")
    assert case.absent_symptom_tokens == ["phlegm"]


def test_present_and_absent_lists_stay_disjoint_across_a_conversation():
    case, trail = _case(["cough"]), []
    for token, answer in (("phlegm", "no"), ("high_fever", "yes"),
                          ("chest_pain", "no"), ("fatigue", "yes")):
        case = record_followup_answer(case, trail, token, "Q?", answer)
    assert set(case.symptom_tokens).isdisjoint(case.absent_symptom_tokens)
    assert case.symptom_tokens == ["cough", "high_fever", "fatigue"]
    assert case.absent_symptom_tokens == ["phlegm", "chest_pain"]
