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
    raw_text_hash TEXT NOT NULL,
    age_months INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cases_area ON cases(area);
"""

# Columns added after the original schema shipped. Each is applied with a
# guarded ALTER TABLE on open so a database created by an earlier version keeps
# working (old rows read the new column as NULL). Additive and back-compatible
# by design -- never drop or rename, only add nullable columns here.
_ADDED_COLUMNS = {
    # age_months: enables Agent-2 (surveillance) age-band analytics on live
    # cases. Nullable: cases logged before this column, or with no reported
    # age, read as NULL (an honest "age not recorded"), never 0.
    "age_months": "INTEGER",
}


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
        self._migrate_columns()
        self._conn.commit()

    def _migrate_columns(self) -> None:
        """Add any post-ship columns missing from an existing database. Guarded
        so a fresh DB (which already has them from _SCHEMA) is a no-op and an
        older DB is upgraded in place without data loss."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(cases)")}
        for name, coltype in _ADDED_COLUMNS.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE cases ADD COLUMN {name} {coltype}")

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
            "probable_disease, language, raw_text_hash, age_months) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                case.case_id,
                area,
                datetime.now(timezone.utc).isoformat(),
                ",".join(case.symptom_tokens),
                result.label.value,
                result.probable_disease,
                case.language,
                raw_hash,
                case.age_months,
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
            "language, raw_text_hash, age_months FROM cases WHERE area = ? ORDER BY recorded_at DESC, rowid DESC LIMIT ?",
            (area, limit),
        )
        columns = [d[0] for d in cursor.description]
        records = []
        for row in cursor.fetchall():
            rec = CaseRecord(zip(columns, row))
            rec["symptom_tokens"] = rec["symptom_tokens"].split(",") if rec["symptom_tokens"] else []
            records.append(rec)
        return records

    def all_cases(self, since: Optional[str] = None, limit: Optional[int] = None) -> list[CaseRecord]:
        """Every logged case, newest first -- the region-wide stream Agent 2
        (surveillance) consumes. `since` is an ISO timestamp lower bound
        (inclusive); `limit` caps the count. Empty list when nothing matches,
        same honest-absence convention as recent_by_area()."""
        sql = (
            "SELECT case_id, area, recorded_at, symptom_tokens, label, probable_disease, "
            "language, raw_text_hash, age_months FROM cases"
        )
        params: list = []
        if since is not None:
            sql += " WHERE recorded_at >= ?"
            params.append(since)
        sql += " ORDER BY recorded_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = self._conn.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        records = []
        for row in cursor.fetchall():
            rec = CaseRecord(zip(columns, row))
            rec["symptom_tokens"] = rec["symptom_tokens"].split(",") if rec["symptom_tokens"] else []
            records.append(rec)
        return records

    def close(self) -> None:
        self._conn.close()
