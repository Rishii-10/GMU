"""Turn logged cases into SurveillanceCase records the agent reasons over.

Two sources, one shape out:
  * from_case_store() -- the live region-wide stream (app.case_store.CaseStore),
    the real extensibility point: Agent 2 reads exactly what the triage pipeline
    already writes, nothing bespoke.
  * from_records()    -- any list of plain dicts (the synthetic demo stream, or
    a test fixture), so the agent can be exercised with no database at all.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from app.schemas import ClassificationLabel
from app.surveillance.schemas import SurveillanceCase


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _to_label(value) -> ClassificationLabel:
    if isinstance(value, ClassificationLabel):
        return value
    try:
        return ClassificationLabel(str(value))
    except ValueError:
        return ClassificationLabel.INCOMPLETE_ASSESSMENT


def record_to_case(rec: dict) -> SurveillanceCase:
    """Project one logged-case dict (CaseStore row or demo record) onto a
    SurveillanceCase. Unknown/absent fields degrade to None rather than raising
    -- a surveillance stream should tolerate imperfect history."""
    return SurveillanceCase(
        area=rec.get("area") or "unknown",
        disease=rec.get("probable_disease") or rec.get("disease"),
        symptom_tokens=list(rec.get("symptom_tokens") or []),
        severity=_to_label(rec.get("label") or rec.get("severity")),
        age_months=rec.get("age_months"),
        recorded_at=_parse_dt(rec.get("recorded_at")),
        language=rec.get("language"),
    )


def from_records(records: Iterable[dict]) -> list[SurveillanceCase]:
    return [record_to_case(r) for r in records]


def from_case_store(store, since: Optional[str] = None) -> list[SurveillanceCase]:
    """Read the whole case stream (optionally since an ISO timestamp)."""
    return from_records(store.all_cases(since=since))
