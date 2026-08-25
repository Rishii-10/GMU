"""
Facility master data + village-to-coordinates lookup (plan diagram 2, step
2: "Facility Database (SQLite)"). SQLite via the stdlib sqlite3 module, no
new dependency -- same choice as app/case_store.py.

Deterministic, no LLM, no network. Distance/radius filtering happens in
Python after fetching rows (haversine great-circle distance) rather than
in SQL -- stock SQLite has no geo functions, and at this dataset's scale
(tens to low hundreds of facilities for a POC) an application-layer filter
costs nothing worth optimizing away.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Optional, Union

from app.routing.schemas import Facility, FacilityType

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "facility_db.sqlite"
EARTH_RADIUS_KM = 6371.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facilities (
    facility_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    facility_type TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    has_emergency_care INTEGER NOT NULL DEFAULT 0,
    has_doctor_24hr INTEGER NOT NULL DEFAULT 0,
    has_blood_bank INTEGER NOT NULL DEFAULT 0,
    is_24hr INTEGER NOT NULL DEFAULT 0,
    open_time TEXT,
    close_time TEXT,
    open_days TEXT,
    phone TEXT
);
CREATE TABLE IF NOT EXISTS villages (
    name TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL
);
"""

Location = Union[str, tuple]  # village name, or an explicit (lat, lon) pair


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Exact for a spherical-earth
    approximation -- adequate for radius filtering / ranking-proxy
    purposes; NOT a substitute for the real road distance a
    RoutingProvider computes for the final chosen facility (see
    app.routing.router)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _facility_to_row(f: Facility) -> tuple:
    open_days = ",".join(str(d) for d in f.open_days) if f.open_days is not None else None
    return (
        f.name,
        f.facility_type.value,
        f.lat,
        f.lon,
        int(f.has_emergency_care),
        int(f.has_doctor_24hr),
        int(f.has_blood_bank),
        int(f.is_24hr),
        f.open_time,
        f.close_time,
        open_days,
        f.phone,
    )


def _row_to_facility(row: tuple) -> Facility:
    (
        facility_id, name, facility_type, lat, lon,
        has_emergency_care, has_doctor_24hr, has_blood_bank, is_24hr,
        open_time, close_time, open_days, phone,
    ) = row
    return Facility(
        facility_id=facility_id,
        name=name,
        facility_type=FacilityType(facility_type),
        lat=lat,
        lon=lon,
        has_emergency_care=bool(has_emergency_care),
        has_doctor_24hr=bool(has_doctor_24hr),
        has_blood_bank=bool(has_blood_bank),
        is_24hr=bool(is_24hr),
        open_time=open_time,
        close_time=close_time,
        open_days=[int(d) for d in open_days.split(",")] if open_days else None,
        phone=phone,
    )


class FacilityDB:
    """SQLite-backed facility master data + village->coordinates table.

    Construction is empty by default -- call add_facility()/add_village()
    to seed data (tests build small in-memory datasets this way), or point
    db_path at a pre-populated file for production/fixture use (Phase 8
    supplies real fixture data, seeded the same way via these methods).
    """

    def __init__(self, db_path: Union[Path, str] = DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_facility(self, facility: Facility) -> int:
        cursor = self._conn.execute(
            "INSERT INTO facilities (name, facility_type, lat, lon, has_emergency_care, "
            "has_doctor_24hr, has_blood_bank, is_24hr, open_time, close_time, open_days, phone) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _facility_to_row(facility),
        )
        self._conn.commit()
        return cursor.lastrowid

    def add_village(self, name: str, lat: float, lon: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO villages (name, lat, lon) VALUES (?, ?, ?)", (name, lat, lon)
        )
        self._conn.commit()

    def resolve_location(self, location: Location) -> Optional[tuple[float, float]]:
        """Village name -> coords via the villages table (plan diagram 2,
        step 1: "Village name is converted to coordinates using
        village-to-coordinates table"), or pass an explicit (lat, lon)
        tuple straight through for GPS input. Returns None -- not a guess
        -- when a village name isn't in the table."""
        if isinstance(location, tuple) and len(location) == 2:
            return location
        row = self._conn.execute(
            "SELECT lat, lon FROM villages WHERE name = ?", (location,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def all_facilities(self) -> list[Facility]:
        rows = self._conn.execute(
            "SELECT facility_id, name, facility_type, lat, lon, has_emergency_care, "
            "has_doctor_24hr, has_blood_bank, is_24hr, open_time, close_time, open_days, phone "
            "FROM facilities"
        ).fetchall()
        return [_row_to_facility(r) for r in rows]

    def facilities_within_radius(self, coords: tuple[float, float], radius_km: float) -> list[Facility]:
        """Plan diagram 2, step 2 output: facilities within a configurable
        radius (default 60km, see app.routing.router.DEFAULT_RADIUS_KM) of
        the given coordinates."""
        return [
            f for f in self.all_facilities()
            if haversine_km(coords[0], coords[1], f.lat, f.lon) <= radius_km
        ]

    def close(self) -> None:
        self._conn.close()
