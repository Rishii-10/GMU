"""
Previous-calls / area case store (plan diagram 3, "Extends"): a persistent
log of past cases keyed by area/location, so a new call from the SAME AREA
can be mapped to existing cases. Also the data source a future Agent 2
(outbreak surveillance -- out of scope for this build, per user decision)
would consume, per the diagram's own "this info also used in agent 2" note.

Deterministic, no LLM. SQLite via the stdlib `sqlite3` module -- no new
dependency, consistent with this project's "add a dependency only when a
milestone actually needs it" requirements.txt policy.

PRIVACY -- decision, stated explicitly
------------------------------------------------
Only `area` (a caregiver-reported location string, e.g. a village name --
already collected as ExtractedCase.location) plus a SHA-256 hash of the raw
message text are stored, never the raw patient-reported text itself. The
hash lets an operator confirm "was this exact message logged before"
without this store becoming a second place raw patient text is retained
(agent1_extraction.py's ExtractedCase already carries raw_symptom_text
in-memory for the duration of a request; this store is not meant to
duplicate that as a second, permanent copy).
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas import ClassificationResult, ExtractedCase

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "case_store.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT,
    area TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    symptom_tokens TEXT NOT NULL,
    label TEXT NOT NULL,
    probable_disease TEXT,
    language TEXT,
    raw_text_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_area ON cases(area);
"""


class CaseRecord(dict):
    """A logged case as returned by recent_by_area() -- a plain dict
    subclass (not a pydantic model) since this is read-only query output,
    not something the rest of the pipeline validates or mutates."""


class CaseStore:
    """Interface: record() appends one case; recent_by_area() looks up
    prior cases from the same area, most recent first. Backed by SQLite;
    every write is committed immediately (no batching) -- case volume in
    this POC's scope does not warrant the complexity of deferred writes,
    and losing a case record silently on a crash would defeat the store's
    own purpose.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, case: ExtractedCase, result: ClassificationResult) -> None:
        """Logs one case. No-op-safe on a case with no reported area
        (case.location is None) -- an unlocated case cannot usefully be
        matched to "same area" later, but it still isn't dropped: it is
        stored under the literal area value "unknown" rather than silently
        discarded, so nothing about this call is lossy."""
        area = case.location or "unknown"
        raw_hash = hashlib.sha256(case.raw_symptom_text.encode("utf-8")).hexdigest()
        self._conn.execute(
            "INSERT INTO cases (case_id, area, recorded_at, symptom_tokens, label, "
            "probable_disease, language, raw_text_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                case.case_id,
                area,
                datetime.now(timezone.utc).isoformat(),
                ",".join(case.symptom_tokens),
                result.label.value,
                result.probable_disease,
                case.language,
                raw_hash,
            ),
        )
        self._conn.commit()

    def recent_by_area(self, area: str, limit: int = 20) -> list[CaseRecord]:
        """Prior cases from the same area, most recent first. Empty list
        (not an error) when the area has no recorded history -- an honest
        "nothing found," matching this project's "no forced guess" pattern
        elsewhere (app.disambiguation, app.disease_classifier)."""
        cursor = self._conn.execute(
            "SELECT case_id, area, recorded_at, symptom_tokens, label, probable_disease, "
            "language, raw_text_hash FROM cases WHERE area = ? ORDER BY recorded_at DESC LIMIT ?",
            (area, limit),
        )
        columns = [d[0] for d in cursor.description]
        records = []
        for row in cursor.fetchall():
            rec = CaseRecord(zip(columns, row))
            rec["symptom_tokens"] = rec["symptom_tokens"].split(",") if rec["symptom_tokens"] else []
            records.append(rec)
        return records

    def close(self) -> None:
        self._conn.close()
