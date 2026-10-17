"""Phase 4: the follow-up loop must ask questions a real caller can answer.

Measured problem this locks down. With only 32 hand-written templates over a
131-token vocabulary, 53.6% of selected questions fell through to the generic
"Is the patient experiencing {token}?" phrasing, and because selection was
driven purely by discriminative power the loop asked things like:

    "Is the patient experiencing acute liver failure?"   (a diagnosis)
    "Is the patient experiencing coma?"                  (not self-reportable)
    "Is the patient experiencing receiving blood transfusion?"  (history)

Statistically excellent, clinically incoherent -- and no amount of LLM
rewording fixes it, because the chosen TOKEN is not a reportable symptom.

Two mechanisms now prevent it, and these tests keep both honest:
  * `_NOT_CALLER_ASKABLE` removes diagnosis / history / instrumented tokens
    from the candidate pool entirely.
  * `_QUESTION_TEMPLATES` covers every remaining askable token, so the generic
    fallback is unreachable in practice.

Deterministic, classifier-only (no Ollama, no FAISS) -- runs in CI.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.disease_classifier import get_default_classifier, severity_for_disease
from app.disease_kb import DiseaseKB
from app.followup_question_selector import (
    _NOT_CALLER_ASKABLE,
    _QUESTION_TEMPLATES,
    _relevant_tokens,
    select_followup_question,
    template_for,
)

kb = DiseaseKB.load()
clf = get_default_classifier()
VOCAB = set(kb.vocabulary)
ASKABLE = VOCAB - _NOT_CALLER_ASKABLE


# --- the two token lists must stay consistent with the dataset ------------


def test_no_template_is_unreachable():
    """Every template key must be a real vocabulary token, or it can never
    fire. `radiating_pain` and `seizures` were dead entries for exactly this
    reason."""
    dead = sorted(t for t in _QUESTION_TEMPLATES if t not in VOCAB)
    assert dead == [], f"templates for tokens that do not exist: {dead}"


def test_every_non_askable_token_exists():
    """A stale exclusion silently stops excluding anything."""
    missing = sorted(t for t in _NOT_CALLER_ASKABLE if t not in VOCAB)
    assert missing == [], f"_NOT_CALLER_ASKABLE lists unknown tokens: {missing}"


def test_every_askable_token_has_a_human_phrasing():
    missing = sorted(t for t in ASKABLE if t not in _QUESTION_TEMPLATES)
    assert missing == [], f"askable tokens with no template: {missing}"


def test_templates_are_phrased_as_questions_in_plain_language():
    for token, text in _QUESTION_TEMPLATES.items():
        assert text.endswith("?"), f"{token} template is not a question: {text!r}"
        # The raw snake_case token leaking into the text is the signature of a
        # machine-generated phrasing rather than a human one.
        assert "_" not in text, f"{token} template leaks a raw token: {text!r}"


# --- the selector must never reach a non-askable token --------------------


def test_relevant_token_pool_excludes_non_askable_tokens():
    for disease in ("Heart attack", "Tuberculosis", "Hepatitis B", "Malaria"):
        pool = _relevant_tokens([disease], set(), clf._rows_by_disease, list(kb.vocabulary))
        leaked = sorted(set(pool) & _NOT_CALLER_ASKABLE)
        assert leaked == [], f"{disease} pool offered non-askable tokens: {leaked}"


def test_selector_never_asks_an_unanswerable_question_across_many_sets():
    """Sweep real prediction sets and assert every selected token is both
    askable and has a human phrasing."""
    rng = random.Random(42)
    diseases = list(kb.diseases)
    checked = 0
    for _ in range(120):
        size = rng.randint(2, 4)
        pset = rng.sample(diseases, size)
        severity_map = {d: severity_for_disease(d, kb).value for d in pset}
        fq = select_followup_question(
            prediction_set=pset,
            severity_map=severity_map,
            known_tokens=[],
            rows_by_disease=clf._rows_by_disease,
            vocabulary=list(kb.vocabulary),
        )
        if fq is None:
            continue
        checked += 1
        assert fq.symptom_token not in _NOT_CALLER_ASKABLE, (
            f"selected unanswerable token {fq.symptom_token!r} for {pset}"
        )
        assert fq.symptom_token in _QUESTION_TEMPLATES, (
            f"selected {fq.symptom_token!r} with no human phrasing, for {pset}"
        )
        assert fq.template_question.endswith("?")
    assert checked > 50, "too few prediction sets produced a question to be meaningful"


def test_template_for_still_degrades_gracefully_for_an_unknown_token():
    # The generic fallback must remain correct for anything unforeseen, even
    # though the coverage tests above mean it should not fire in practice.
    assert template_for("some_new_token") == "Is the patient experiencing some new token?"


def test_known_tokens_are_never_re_asked():
    pset = ["Tuberculosis", "Pneumonia"]
    severity_map = {d: severity_for_disease(d, kb).value for d in pset}
    first = select_followup_question(
        prediction_set=pset, severity_map=severity_map, known_tokens=[],
        rows_by_disease=clf._rows_by_disease, vocabulary=list(kb.vocabulary),
    )
    assert first is not None
    second = select_followup_question(
        prediction_set=pset, severity_map=severity_map,
        known_tokens=[first.symptom_token],
        rows_by_disease=clf._rows_by_disease, vocabulary=list(kb.vocabulary),
    )
    assert second is None or second.symptom_token != first.symptom_token
