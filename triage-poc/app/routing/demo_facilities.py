"""Synthetic demo facility network for the frontend (streamlit_app.py).

Real facility master data (name/type/coordinates/capabilities/hours) for
rural PHCs is not available to this project yet -- app/routing/facility_db.py
is built to be seeded from a real dataset once one exists (see its own
docstring: "Construction is empty by default"). Until then, this module
gives the demo UI something concrete to route against: the same
Denkanikottai/Hosur taluk sample used by tests/fixtures/facilities.py,
duplicated here (not imported from the tests package) so the app layer
doesn't depend on test-only code.
"""
from __future__ import annotations

from app.routing.facility_db import FacilityDB
from app.routing.schemas import Facility, FacilityType

VILLAGES: dict[str, tuple[float, float]] = {
    "Denkanikottai": (12.5333, 77.7667),
    "Anchetty": (12.4667, 77.6333),
    "Thally": (12.6167, 77.8333),
    "Kelamangalam": (12.4833, 77.7333),
    "Bannerghatta": (12.8000, 77.5667),
}

FACILITIES: list[Facility] = [
    Facility(
        name="Denkanikottai PHC", facility_type=FacilityType.PHC, lat=12.5666, lon=77.7999,
        has_emergency_care=False, has_doctor_24hr=False, has_blood_bank=False,
        open_time="09:00", close_time="17:00", open_days=[0, 1, 2, 3, 4, 5],
        phone="04347-220001",
    ),
    Facility(
        name="Thally CHC", facility_type=FacilityType.CHC, lat=12.5883, lon=77.8217,
        has_emergency_care=True, has_doctor_24hr=True, has_blood_bank=False, is_24hr=True,
        phone="04347-220002",
    ),
    Facility(
        name="Anchetty PHC", facility_type=FacilityType.PHC, lat=12.4720, lon=77.6410,
        has_emergency_care=False, has_doctor_24hr=False, has_blood_bank=False,
        open_time="08:00", close_time="14:00", open_days=[0, 1, 2, 3, 4, 5],
        phone="04347-220003",
    ),
    Facility(
        name="Kelamangalam PHC", facility_type=FacilityType.PHC, lat=12.4900, lon=77.7400,
        has_emergency_care=False, has_doctor_24hr=False, has_blood_bank=False,
        open_time="09:00", close_time="18:00", open_days=[0, 1, 2, 3, 4, 5, 6],
        phone="04347-220004",
    ),
    Facility(
        name="Hosur District Hospital", facility_type=FacilityType.DH, lat=12.7350, lon=77.8258,
        has_emergency_care=True, has_doctor_24hr=True, has_blood_bank=True, is_24hr=True,
        phone="04344-220005",
    ),
    Facility(
        name="Krishnagiri SDH", facility_type=FacilityType.SDH, lat=12.5186, lon=78.2137,
        has_emergency_care=True, has_doctor_24hr=False, has_blood_bank=True,
        open_time="00:00", close_time="23:59", open_days=[0, 1, 2, 3, 4, 5, 6],
        phone="04347-220006",
    ),
    Facility(
        name="Bannerghatta CHC", facility_type=FacilityType.CHC, lat=12.7980, lon=77.5700,
        has_emergency_care=True, has_doctor_24hr=False, has_blood_bank=False,
        open_time="08:00", close_time="20:00", open_days=[0, 1, 2, 3, 4, 5, 6],
        phone="080-220007",
    ),
    Facility(
        name="St. John's Medical College", facility_type=FacilityType.MEDICAL_COLLEGE,
        lat=12.9165, lon=77.6101,
        has_emergency_care=True, has_doctor_24hr=True, has_blood_bank=True, is_24hr=True,
        phone="080-220008",
    ),
    Facility(
        name="Unregistered Rural Clinic", facility_type=FacilityType.PHC, lat=12.55, lon=77.75,
        has_emergency_care=False, has_doctor_24hr=False, has_blood_bank=False,
        phone=None,
    ),
    Facility(
        name="Bangalore City Medical College", facility_type=FacilityType.MEDICAL_COLLEGE,
        lat=12.9716, lon=77.5946,
        has_emergency_care=True, has_doctor_24hr=True, has_blood_bank=True, is_24hr=True,
        phone="080-220009",
    ),
]


def build_demo_facility_db() -> FacilityDB:
    """Fresh in-memory FacilityDB seeded with the demo dataset above."""
    db = FacilityDB(db_path=":memory:")
    for name, (lat, lon) in VILLAGES.items():
        db.add_village(name, lat, lon)
    for facility in FACILITIES:
        db.add_facility(facility)
    return db
