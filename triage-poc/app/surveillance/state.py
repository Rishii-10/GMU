"""Persistent memory for Agent 2: watchlist, alert history, escalation log.

Before this module the watchlist lived in a Python set on the agent object, so
it vanished on every restart and the same outbreak re-alerted (and would have
re-notified) on every run. SQLite via stdlib, same as app.case_store.

Re-notification rule (`should_notify`): an alert for an (area, disease) pair is
acted on with notifications only when it is NEW for this window, its level has
RISEN, or its severity-weighted score has grown by at least
RENOTIFY_SCORE_GROWTH since the last notified alert. Otherwise it is still
recorded (the official sees it) but no one is messaged again.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.surveillance.schemas import AlertLevel

DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "surveillance_state.sqlite"
RENOTIFY_SCORE_GROWTH = 0.25

_LEVEL_RANK = {AlertLevel.INFO.value: 0, AlertLevel.WATCH.value: 1, AlertLevel.ALERT.value: 2}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    area TEXT PRIMARY KEY,
    since TEXT NOT NULL,
    last_level TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    area TEXT NOT NULL,
    disease TEXT NOT NULL,
    level TEXT NOT NULL,
    weighted_score REAL NOT NULL,
    window_label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notified INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_pair ON alerts(area, disease);
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    area TEXT NOT NULL,
    disease TEXT NOT NULL,
    level TEXT NOT NULL,
    created_at TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class SurveillanceState:
    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- watchlist -------------------------------------------------------------
    def watch(self, area: str, level: AlertLevel, at: datetime) -> None:
        now = at.isoformat()
        self._conn.execute(
            "INSERT INTO watchlist (area, since, last_level, last_seen) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(area) DO UPDATE SET last_level = excluded.last_level, last_seen = excluded.last_seen",
            (area, now, level.value, now),
        )
        self._conn.commit()

    def watchlist(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT area FROM watchlist ORDER BY area")]

    # --- alert history ------------------------------------------------------------
    def last_notified(self, area: str, disease: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT level, weighted_score, window_label, created_at FROM alerts "
            "WHERE area = ? AND disease = ? AND notified = 1 ORDER BY alert_id DESC LIMIT 1",
            (area, disease),
        ).fetchone()
        if row is None:
            return None
        return {"level": row[0], "weighted_score": row[1], "window_label": row[2], "created_at": row[3]}

    def should_notify(self, area: str, disease: str, level: AlertLevel, weighted_score: float, window_label: str) -> tuple[bool, str]:
        """(notify?, reason). See the module docstring for the rule."""
        prev = self.last_notified(area, disease)
        if prev is None:
            return True, "first alert for this area and disease"
        if _LEVEL_RANK[level.value] > _LEVEL_RANK[prev["level"]]:
            return True, f"level rose {prev['level']} -> {level.value}"
        if prev["weighted_score"] > 0 and weighted_score >= prev["weighted_score"] * (1 + RENOTIFY_SCORE_GROWTH):
            return True, f"worsened: score {prev['weighted_score']} -> {weighted_score}"
        if window_label != prev["window_label"] and _LEVEL_RANK[level.value] >= _LEVEL_RANK[prev["level"]]:
            return True, f"still active in a new window ({window_label})"
        return False, (
            f"already notified at {prev['level']} (score {prev['weighted_score']}) in "
            f"{prev['window_label']}; not worse -- recorded, no repeat message"
        )

    def record_alert(self, area: str, disease: str, level: AlertLevel, weighted_score: float,
                     window_label: str, at: datetime, notified: bool) -> int:
        cur = self._conn.execute(
            "INSERT INTO alerts (area, disease, level, weighted_score, window_label, created_at, notified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (area, disease, level.value, weighted_score, window_label, at.isoformat(), int(notified)),
        )
        self._conn.commit()
        return cur.lastrowid

    def alert_history(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT alert_id, area, disease, level, weighted_score, window_label, created_at, notified "
            "FROM alerts ORDER BY alert_id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # --- escalation log -------------------------------------------------------------
    def log_escalation(self, area: str, disease: str, level: AlertLevel, detail: dict, at: Optional[datetime] = None) -> int:
        at = at or datetime.now(timezone.utc)
        cur = self._conn.execute(
            "INSERT INTO escalations (area, disease, level, created_at, detail) VALUES (?, ?, ?, ?, ?)",
            (area, disease, level.value, at.isoformat(), json.dumps(detail, default=str)),
        )
        self._conn.commit()
        return cur.lastrowid

    def escalations(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT escalation_id, area, disease, level, created_at, detail FROM escalations "
            "ORDER BY escalation_id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        out = []
        for r in cur.fetchall():
            rec = dict(zip(cols, r))
            rec["detail"] = json.loads(rec["detail"])
            out.append(rec)
        return out

    def close(self) -> None:
        self._conn.close()
