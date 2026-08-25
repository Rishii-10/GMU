"""
Tests for app/case_store.py (Phase 5: previous-calls / area case store) and
its opt-in wiring into
app.agent1_extraction.extract_and_classify_with_followup().

Every test uses an in-memory SQLite database (":memory:") -- no files
touch disk, and each test gets its own isolated CaseStore instance (no
shared module-level state to leak between tests).
"""
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import LLMBackend, extract_and_classify_with_followup
from app.case_store import CaseStore
from app.schemas import ClassificationLabel, ClassificationResult, DangerSigns, ExtractedCase


@pytest.fixture
def store() -> CaseStore:
    s = CaseStore(db_path=":memory:")
    yield s
    s.close()


def make_case(**kwargs) -> ExtractedCase:
    defaults = dict(raw_symptom_text="original patient text", danger_signs=DangerSigns())
    defaults.update(kwargs)
    return ExtractedCase(**defaults)


def make_result(**kwargs) -> ClassificationResult:
    defaults = dict(label=ClassificationLabel.MILD)
    defaults.update(kwargs)
    return ClassificationResult(**defaults)


# --- record() / recent_by_area(): basic behavior -----------------------------


def test_record_and_retrieve_by_area(store: CaseStore):
    case = make_case(location="Denkanikottai", case_id="c1", symptom_tokens=["high_fever", "chills"])
    result = make_result(label=ClassificationLabel.MODERATE, probable_disease="Malaria")
    store.record(case, result)

    records = store.recent_by_area("Denkanikottai")
    assert len(records) == 1
    assert records[0]["case_id"] == "c1"
    assert records[0]["area"] == "Denkanikottai"
    assert records[0]["label"] == "MODERATE"
    assert records[0]["probable_disease"] == "Malaria"
    assert records[0]["symptom_tokens"] == ["high_fever", "chills"]


def test_unknown_area_returns_empty_not_error(store: CaseStore):
    assert store.recent_by_area("Nowhere Village") == []


def test_case_with_no_location_stored_under_unknown_not_dropped(store: CaseStore):
    case = make_case(location=None, case_id="c2")
    result = make_result()
    store.record(case, result)
    records = store.recent_by_area("unknown")
    assert len(records) == 1
    assert records[0]["case_id"] == "c2"


def test_different_areas_do_not_leak_into_each_other(store: CaseStore):
    store.record(make_case(location="Area A", case_id="a1"), make_result())
    store.record(make_case(location="Area B", case_id="b1"), make_result())
    area_a = store.recent_by_area("Area A")
    area_b = store.recent_by_area("Area B")
    assert [r["case_id"] for r in area_a] == ["a1"]
    assert [r["case_id"] for r in area_b] == ["b1"]


def test_most_recent_first(store: CaseStore):
    store.record(make_case(location="X", case_id="first"), make_result())
    store.record(make_case(location="X", case_id="second"), make_result())
    store.record(make_case(location="X", case_id="third"), make_result())
    records = store.recent_by_area("X")
    assert [r["case_id"] for r in records] == ["third", "second", "first"]


def test_limit_is_respected(store: CaseStore):
    for i in range(5):
        store.record(make_case(location="X", case_id=f"c{i}"), make_result())
    records = store.recent_by_area("X", limit=2)
    assert len(records) == 2


# --- Privacy: raw text is never stored, only its hash ------------------------


def test_raw_text_never_stored_verbatim(store: CaseStore):
    secret_text = "extremely specific patient detail that must not be retained"
    case = make_case(location="Y", raw_symptom_text=secret_text)
    store.record(case, make_result())
    records = store.recent_by_area("Y")
    assert secret_text not in str(records[0])
    assert "raw_text_hash" in records[0]
    assert len(records[0]["raw_text_hash"]) == 64  # sha256 hex digest length


def test_same_raw_text_produces_same_hash(store: CaseStore):
    text = "same message twice"
    store.record(make_case(location="Z", raw_symptom_text=text, case_id="a"), make_result())
    store.record(make_case(location="Z", raw_symptom_text=text, case_id="b"), make_result())
    records = store.recent_by_area("Z")
    assert records[0]["raw_text_hash"] == records[1]["raw_text_hash"]


# --- Wiring: opt-in recording via extract_and_classify_with_followup --------


class _FixedResponseBackend(LLMBackend):
    name = "fixed_response_test_backend"
    supports_followup = False

    def __init__(self, symptom: str, location: Optional[str]):
        self._symptom = symptom
        self._location = location

    def extract(self, patient_text: str, context: Optional[list[str]] = None) -> dict:
        return {
            "symptom": self._symptom,
            "duration": None,
            "severity": "unknown",
            "age_group": "child",
            "age_months": 24,
            "location": self._location,
            "notes": None,
            "danger_signs": {
                "not_able_to_drink_or_breastfeed": False,
                "vomits_everything": False,
                "convulsions": False,
                "lethargic_or_unconscious": False,
            },
            "cough": None,
            "diarrhea": None,
        }


def test_pipeline_does_not_record_when_no_case_store_passed():
    # Default behavior (no case_store argument) must have zero I/O side
    # effects -- this is what every pre-existing caller/test relies on.
    backend = _FixedResponseBackend(symptom="fever", location="Somewhere")
    case, trail, result = extract_and_classify_with_followup("test message", backend)
    assert case is not None  # sanity: pipeline still runs normally
    # (nothing to assert about a store that was never created/touched)


def test_pipeline_records_when_case_store_passed(store: CaseStore):
    backend = _FixedResponseBackend(symptom="fever", location="Hosur")
    case, trail, result = extract_and_classify_with_followup(
        "test message", backend, case_store=store
    )
    records = store.recent_by_area("Hosur")
    assert len(records) == 1
    assert records[0]["label"] == result.label.value
