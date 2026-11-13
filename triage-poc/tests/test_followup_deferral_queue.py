"""Paper §4.2-§5 behaviours that previously had no code path:

A2  Paediatric danger-sign questions run INSIDE the follow-up loop (no
    "add it to your message and press Assess again"), and the §5 worked example
    -- "child, fever and loose motions for two days" -> "able to drink?" -> "no"
    -> EMERGENCY -> routed -- runs end to end.
A4  A case the exchange cannot finish is DEFERRED to the health worker: a flag
    with a reason, never a lowered label, visible in the case log and queue.
A5  An unclear reply gets the same question again; a second unclear reply
    leaves it unanswered (no evidence) and the exchange moves on.
B4  The health-worker queue orders by urgency tier, then AHP score within the
    tier, and never lets the score cross a tier.
C1  The case log row is updated in place with the FINAL decision.
All deterministic (RegexBackend: template question text, no LLM).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import (
    DANGER_SIGN_FOLLOWUP_QUESTIONS,
    DANGER_SIGN_TOKEN_PREFIX,
    DANGER_SIGN_YES_MEANS_PRESENT,
    MAX_UNCLEAR_REPLIES,
    RegexBackend,
    apply_followup_reply,
    classify_with_adaptive_followup,
    finalize_after_followup,
    next_followup_question,
    record_followup_answer,
)
from app.case_store import CaseStore
from app.prioritization import QUEUE_TIERS, priority_for, queue_key
from app.routing.demo_facilities import build_demo_facility_db
from app.routing.router import route
from app.rules_engine import classify
from app.schemas import ClassificationLabel, DangerSigns, DiarrheaAssessment, ExtractedCase


def _child(**kw):
    base = dict(
        raw_symptom_text="child, fever and loose motions for two days",
        symptom="fever, loose motions",
        symptoms=["fever", "loose motions"],
        duration="2 days",
        age_months=24,
        diarrhea=DiarrheaAssessment(present=True, duration_days=2),
        symptom_tokens=["high_fever", "diarrhoea"],
    )
    base.update(kw)
    return ExtractedCase(**base)


def _adult(tokens, **kw):
    return ExtractedCase(raw_symptom_text="x", symptom="x", age_group="adult", symptom_tokens=tokens, **kw)


def _scripted(answers):
    it = iter(answers)
    asked = []

    def provider(q):
        asked.append(q)
        return next(it)

    return provider, asked


# --- A2: paediatric danger-sign loop + the §5 worked example --------------------
def test_child_case_first_question_is_ability_to_drink():
    case = _child()
    result = classify(case)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    token, question = next_followup_question(case, result, [], RegexBackend())
    assert token == DANGER_SIGN_TOKEN_PREFIX + "not_able_to_drink_or_breastfeed"
    assert question == DANGER_SIGN_FOLLOWUP_QUESTIONS["not_able_to_drink_or_breastfeed"]


def test_section5_worked_example_end_to_end():
    """'no' to 'able to drink?' -> danger-sign override -> EMERGENCY -> routed."""
    provider, asked = _scripted(["no"])
    result, trail = classify_with_adaptive_followup(_child(), RegexBackend(), answer_provider=provider)
    assert asked == [DANGER_SIGN_FOLLOWUP_QUESTIONS["not_able_to_drink_or_breastfeed"]]
    assert result.label == ClassificationLabel.EMERGENCY
    assert result.condition == "GENERAL_DANGER_SIGN"
    assert not result.deferred
    dispatch = route(result.label, "Anchetty", build_demo_facility_db(), case=_child(
        danger_signs=DangerSigns(not_able_to_drink_or_breastfeed=True)))
    assert not dispatch.no_facility_found


@pytest.mark.parametrize("field, yes_present", sorted(DANGER_SIGN_YES_MEANS_PRESENT.items()))
def test_danger_sign_answer_polarity(field, yes_present):
    tok = DANGER_SIGN_TOKEN_PREFIX + field
    yes = record_followup_answer(_child(), [], tok, "q", "yes")
    no = record_followup_answer(_child(), [], tok, "q", "nahi")
    assert getattr(yes.danger_signs, field) is yes_present
    assert getattr(no.danger_signs, field) is (not yes_present)


def test_all_four_denied_lets_imnci_classify_and_symptom_budget_is_untouched():
    provider, asked = _scripted(["yes", "no", "no", "no"])
    result, trail = classify_with_adaptive_followup(_child(), RegexBackend(), answer_provider=provider)
    assert len(asked) == 4
    assert result.label != ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert all(t["token"].startswith(DANGER_SIGN_TOKEN_PREFIX) for t in trail)


def test_danger_sign_never_set_false_by_an_unclear_reply():
    case, status = apply_followup_reply(
        _child(), [], DANGER_SIGN_TOKEN_PREFIX + "convulsions", "q", "not sure", unclear_so_far=0
    )
    assert status == "reask" and case.danger_signs.convulsions is None


# --- A5: unclear replies --------------------------------------------------------
def test_unclear_reply_reasks_then_moves_on_without_evidence():
    tok = DANGER_SIGN_TOKEN_PREFIX + "convulsions"
    trail: list = []
    case, s1 = apply_followup_reply(_child(), trail, tok, "q", "maybe", 0)
    assert s1 == "reask" and trail == []
    case, s2 = apply_followup_reply(case, trail, tok, "q", "not sure", MAX_UNCLEAR_REPLIES - 1)
    assert s2 == "unanswered" and trail[-1]["unanswered"] == "true"
    assert case.danger_signs.convulsions is None


def test_blocking_loop_repeats_the_same_question_on_a_hedge():
    provider, asked = _scripted(["yes", "no", "hmm", "no", "no"])
    classify_with_adaptive_followup(_child(), RegexBackend(), answer_provider=provider)
    assert asked[2] == asked[3] == DANGER_SIGN_FOLLOWUP_QUESTIONS["convulsions"]


def test_unanswered_symptom_question_adds_no_evidence():
    case = _adult(["cough"])
    trail: list = []
    out, status = apply_followup_reply(case, trail, "phlegm", "q", "?", MAX_UNCLEAR_REPLIES - 1)
    assert status == "unanswered"
    assert out.symptom_tokens == ["cough"] and out.absent_symptom_tokens == []


# --- A4: deferral ----------------------------------------------------------------
def test_unanswered_danger_sign_defers_to_health_worker():
    provider, _ = _scripted(["yes", "no", "maybe", "not sure", "no"])
    result, _ = classify_with_adaptive_followup(_child(), RegexBackend(), answer_provider=provider)
    assert result.label == ClassificationLabel.INCOMPLETE_ASSESSMENT
    assert result.deferred and result.deferral_reason == "DANGER_SIGNS_UNANSWERED"
    assert any("DEFERRED" in r for r in result.reasoning)


def test_young_infant_is_deferred():
    infant = _child(age_months=1)
    result = finalize_after_followup(infant, classify(infant), [])
    assert result.deferred and result.deferral_reason == "YOUNG_INFANT_NO_VALIDATED_RULESET"


def test_unresolved_differential_defers_but_keeps_the_worst_case_label():
    case = _adult(["cough"])
    before = classify(case)
    assert len(before.prediction_set) > 1
    after = finalize_after_followup(case, before, [])
    assert after.deferred and after.deferral_reason == "DIFFERENTIAL_UNRESOLVED"
    assert after.label == before.label  # deferring never lowers the label


def test_resolved_case_is_not_deferred_and_finalize_is_idempotent():
    case = _adult(["altered_sensorium"])
    r = classify(case)
    assert len(r.prediction_set) <= 1
    assert not finalize_after_followup(case, r, []).deferred
    deferred = finalize_after_followup(_child(age_months=1), classify(_child(age_months=1)), [])
    again = finalize_after_followup(_child(age_months=1), deferred, [])
    assert again.reasoning == deferred.reasoning


# --- C1 + B4: case log update and the health-worker queue -----------------------------
def test_case_log_row_is_updated_with_the_final_decision():
    store = CaseStore(":memory:")
    case = _child(location="Anchetty")
    first = classify(case)
    cid = store.record(case, first)
    final_case = case.model_copy(update={"danger_signs": DangerSigns(not_able_to_drink_or_breastfeed=True)})
    trail = [{"question": "q", "answer": "no", "token": DANGER_SIGN_TOKEN_PREFIX + "x"}]
    store.record(final_case, classify(final_case), trail=trail, case_id=cid, final=True)
    rows = store.all_cases()
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "EMERGENCY" and row["initial_label"] == "INCOMPLETE_ASSESSMENT"
    assert row["is_final"] == 1
    assert row["followup_trail"] == [{"question": "q", "token": DANGER_SIGN_TOKEN_PREFIX + "x", "verdict": "no"}]
    assert "answer" not in row["followup_trail"][0]  # raw reply text never stored


def test_every_case_gets_a_priority_score_even_when_uncertain():
    case = _adult(["cough"])
    result = classify(case)
    assert result.emergency_result is None
    assert 1 <= priority_for(case, result).score <= 10


def test_queue_orders_by_tier_then_ahp_and_score_never_crosses_tiers():
    store = CaseStore(":memory:")
    severe_low = _adult(["cough"], location="A")
    mild = _adult(["continuous_sneezing", "chills", "watering_from_eyes"], location="A")
    emergency = _adult(["altered_sensorium", "weakness_of_one_body_side"], location="A")
    infant = _child(age_months=1, location="A")
    for c in (mild, severe_low, emergency):
        store.record(c, classify(c))
    store.record(infant, finalize_after_followup(infant, classify(infant), []), final=True)
    q = store.health_worker_queue()
    tiers = [r["queue_tier"] for r in q]
    assert tiers == sorted(tiers, key=QUEUE_TIERS.index)
    assert tiers[0] == "EMERGENCY"
    # the unassessed infant is above MODERATE/MILD, never ranked as routine
    for routine in ("MODERATE", "MILD"):
        if routine in tiers:
            assert tiers.index("NEEDS_REVIEW") < tiers.index(routine)
    assert next(r for r in q if r["queue_tier"] == "NEEDS_REVIEW")["deferred"] is True


def test_queue_key_score_only_orders_within_a_tier():
    hi_mild = queue_key("MILD", False, 10, "2026-01-01T00:00:00")
    lo_severe = queue_key("SEVERE", False, 1, "2026-01-02T00:00:00")
    assert lo_severe < hi_mild
    assert queue_key("SEVERE", False, 9, "2026-01-02T00:00:00") < lo_severe
    assert queue_key("INCOMPLETE_ASSESSMENT", True, None, "2026-01-01T00:00:00") < queue_key("MODERATE", False, 10, "2026-01-01T00:00:00")


def test_mark_reviewed_removes_case_from_open_queue():
    store = CaseStore(":memory:")
    c = _adult(["cough"], location="A")
    cid = store.record(c, classify(c))
    assert store.mark_reviewed(cid, "called back")
    assert store.health_worker_queue() == []
    assert store.health_worker_queue(include_reviewed=True)[0]["review_note"] == "called back"


def test_legacy_rows_without_case_id_are_migrated_and_kept_out_of_the_queue(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE cases (case_id TEXT, area TEXT NOT NULL, recorded_at TEXT NOT NULL, "
        "symptom_tokens TEXT NOT NULL, label TEXT NOT NULL, probable_disease TEXT, language TEXT, "
        "raw_text_hash TEXT NOT NULL);"
        "INSERT INTO cases VALUES (NULL, 'A', '2026-10-17T00:00:00+00:00', 'cough', 'SEVERE', NULL, 'en', 'h');"
    )
    con.commit()
    con.close()
    store = CaseStore(db)
    rows = store.all_cases()
    assert rows[0]["case_id"].startswith("legacy-")
    assert store.health_worker_queue() == []
