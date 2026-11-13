"""
Previous-calls / area case store (plan diagram 3, "Extends"): a persistent
log of past cases keyed by area/location, so a new call from the SAME AREA
can be mapped to existing cases. Also the data source a future Agent 2
(app.surveillance) consumes, and the health worker's prioritised queue of
open and deferred cases (health_worker_queue()).

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
import json
import sqlite3
import uuid
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
    # Final-state + health-worker-queue fields (2026-11-13). A case is written
    # once on arrival and UPDATED in place when the follow-up exchange ends, so
    # the stored label is the final one (surveillance and the queue read it),
    # while `initial_label` keeps the first decision for audit.
    "initial_label": "TEXT",
    "updated_at": "TEXT",
    "condition": "TEXT",
    "prediction_set": "TEXT",
    "absent_symptom_tokens": "TEXT",
    "deferred": "INTEGER",
    "deferral_reason": "TEXT",
    "emergency_score": "INTEGER",
    "esi_level": "INTEGER",
    "followup_trail": "TEXT",
    "is_final": "INTEGER",
    "reviewed_at": "TEXT",
    "review_note": "TEXT",
}

_READ_COLUMNS = (
    "case_id, area, recorded_at, symptom_tokens, label, probable_disease, language, "
    "raw_text_hash, age_months, initial_label, updated_at, condition, prediction_set, "
    "absent_symptom_tokens, deferred, deferral_reason, emergency_score, esi_level, "
    "followup_trail, is_final, reviewed_at, review_note"
)


def _trail_for_storage(trail: Optional[list[dict]]) -> str:
    """Store each question with its PARSED verdict only (yes/no/unanswered),
    never the caller's raw reply text -- same privacy stance as raw_text_hash."""
    from app.agent1_extraction import parse_yes_no

    out = []
    for t in trail or []:
        verdict = None if t.get("unanswered") else parse_yes_no(t.get("answer", ""))
        out.append({
            "question": t.get("question"),
            "token": t.get("token"),
            "verdict": "unanswered" if verdict is None else ("yes" if verdict else "no"),
        })
    return json.dumps(out)


class CaseRecord(dict):
    """A logged case as returned by recent_by_area() -- a plain dict
    subclass (not a pydantic model) since this is read-only query output,
    not something the rest of the pipeline validates or mutates."""


class CaseStore:
    """Interface: record() logs a case (or updates it in place by case_id);
    recent_by_area() looks up prior cases from the same area, most recent
    first; health_worker_queue() lists open cases in priority order. Backed by SQLite;
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
        # Rows logged before case_id was always set have NULL ids; give each a
        # unique one so it can be addressed (nothing else about the row changes).
        self._conn.execute(
            "UPDATE cases SET case_id = 'legacy-' || lower(hex(randomblob(8))) WHERE case_id IS NULL"
        )

    def record(
        self,
        case: ExtractedCase,
        result: ClassificationResult,
        trail: Optional[list[dict]] = None,
        case_id: Optional[str] = None,
        final: bool = False,
    ) -> str:
        """Log a case, or UPDATE it in place when `case_id` (or case.case_id)
        is already stored. Returns the case_id (generated when none is given).

        Call once on arrival and again with the same id when follow-up ends
        (`final=True`): the stored label is then the final decision, and
        `initial_label` keeps the first one. A case with no reported area is
        stored under "unknown" rather than dropped."""
        from app.prioritization import priority_for

        case_id = case_id or case.case_id or uuid.uuid4().hex
        area = case.location or "unknown"
        raw_hash = hashlib.sha256(case.raw_symptom_text.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        prio = priority_for(case, result)
        fields = {
            "area": area,
            "symptom_tokens": ",".join(case.symptom_tokens),
            "label": result.label.value,
            "probable_disease": result.probable_disease,
            "language": case.language,
            "raw_text_hash": raw_hash,
            "age_months": case.age_months,
            "updated_at": now,
            "condition": result.condition,
            "prediction_set": json.dumps(list(result.prediction_set)),
            "absent_symptom_tokens": ",".join(case.absent_symptom_tokens),
            "deferred": int(result.deferred),
            "deferral_reason": result.deferral_reason,
            "emergency_score": prio.score,
            "esi_level": prio.esi_level,
            "followup_trail": _trail_for_storage(trail),
            "is_final": int(final),
        }
        exists = self._conn.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if exists:
            sets = ", ".join(f"{k} = ?" for k in fields)
            self._conn.execute(f"UPDATE cases SET {sets} WHERE case_id = ?", (*fields.values(), case_id))
        else:
            fields.update(case_id=case_id, recorded_at=now, initial_label=result.label.value)
            cols = ", ".join(fields)
            marks = ", ".join("?" for _ in fields)
            self._conn.execute(f"INSERT INTO cases ({cols}) VALUES ({marks})", tuple(fields.values()))
        self._conn.commit()
        return case_id

    def _rows(self, sql: str, params) -> list[CaseRecord]:
        cursor = self._conn.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        records = []
        for row in cursor.fetchall():
            rec = CaseRecord(zip(columns, row))
            rec["symptom_tokens"] = rec["symptom_tokens"].split(",") if rec.get("symptom_tokens") else []
            if "absent_symptom_tokens" in rec:
                rec["absent_symptom_tokens"] = (
                    rec["absent_symptom_tokens"].split(",") if rec["absent_symptom_tokens"] else []
                )
            for key in ("prediction_set", "followup_trail"):
                if key in rec:
                    rec[key] = json.loads(rec[key]) if rec[key] else []
            if "deferred" in rec:
                rec["deferred"] = bool(rec["deferred"])
            records.append(rec)
        return records

    def get(self, case_id: str) -> Optional[CaseRecord]:
        rows = self._rows(f"SELECT {_READ_COLUMNS} FROM cases WHERE case_id = ?", (case_id,))
        return rows[0] if rows else None

    def health_worker_queue(
        self, area: Optional[str] = None, include_reviewed: bool = False, limit: int = 50
    ) -> list[CaseRecord]:
        """Open cases for the health worker, ordered by app.prioritization.queue_key:
        urgency tier first (EMERGENCY, SEVERE, NEEDS_REVIEW, MODERATE, MILD -- an
        unassessed case is never ranked as routine), then the AHP score within the
        tier, then oldest first. The AHP score never moves a case across tiers.
        Rows logged before the queue existed are excluded. Each record carries
        `queue_tier`."""
        from app.prioritization import queue_key, queue_tier

        sql = f"SELECT {_READ_COLUMNS} FROM cases"
        where, params = [], []
        if area is not None:
            where.append("area = ?")
            params.append(area)
        if not include_reviewed:
            where.append("reviewed_at IS NULL")
        # rows logged before the queue existed (no updated_at) are history, not open work
        where.append("updated_at IS NOT NULL")
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._rows(sql, params)
        for r in rows:
            r["queue_tier"] = queue_tier(r["label"], bool(r.get("deferred")))
        rows.sort(key=lambda r: queue_key(r["label"], bool(r.get("deferred")), r.get("emergency_score"), r["recorded_at"]))
        return rows[:limit]

    def mark_reviewed(self, case_id: str, note: Optional[str] = None) -> bool:
        """The health worker has taken the case; it leaves the open queue."""
        cur = self._conn.execute(
            "UPDATE cases SET reviewed_at = ?, review_note = ? WHERE case_id = ?",
            (datetime.now(timezone.utc).isoformat(), note, case_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def recent_by_area(self, area: str, limit: int = 20) -> list[CaseRecord]:
        """Prior cases from the same area, most recent first. Empty list
        (not an error) when the area has no recorded history -- an honest
        "nothing found," matching this project's "no forced guess" pattern
        elsewhere (app.disambiguation, app.disease_classifier)."""
        return self._rows(
            f"SELECT {_READ_COLUMNS} FROM cases WHERE area = ? "
            "ORDER BY recorded_at DESC, rowid DESC LIMIT ?",
            (area, limit),
        )

    def all_cases(self, since: Optional[str] = None, limit: Optional[int] = None) -> list[CaseRecord]:
        """Every logged case, newest first -- the region-wide stream Agent 2
        (surveillance) consumes. `since` is an ISO timestamp lower bound
        (inclusive); `limit` caps the count. Empty list when nothing matches,
        same honest-absence convention as recent_by_area()."""
        sql = f"SELECT {_READ_COLUMNS} FROM cases"
        params: list = []
        if since is not None:
            sql += " WHERE recorded_at >= ?"
            params.append(since)
        sql += " ORDER BY recorded_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._rows(sql, params)

    def close(self) -> None:
        self._conn.close()
