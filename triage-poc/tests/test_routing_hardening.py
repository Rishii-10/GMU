"""Routing hardening (paper §4.4: "filters candidate facilities by what the case
needs and then selects by travel distance").

B1  the facility is chosen by TRAVEL TIME from the routing provider, not by
    straight-line distance (a provider where the nearer facility is slower by
    road must change the choice);
B2  the case's needs (paediatrics / neonatal / cardiac / stroke) filter the
    candidates, the radius widens for a missing service, and a shortfall is
    recorded rather than leaving an emergency unrouted;
B3  opening hours are checked in the facility's LOCAL time zone (D-TZ).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.routing.facility_db import FacilityDB
from app.routing.router import (
    MAX_ROUTED_CANDIDATES,
    RoutingProvider,
    facility_services,
    is_facility_open,
    required_services,
    route,
)
from app.routing.schemas import Facility, FacilityType, RouteInfo
from app.schemas import ClassificationLabel, ClassificationResult, DangerSigns, ExtractedCase

HOME = (12.5333, 77.7667)
DAY = datetime(2026, 8, 24, 6, 0, tzinfo=timezone.utc)  # Monday 11:30 IST


def _fac(name, km_north, **kw):
    kw.setdefault("facility_type", FacilityType.CHC)
    kw.setdefault("has_emergency_care", True)
    kw.setdefault("has_doctor_24hr", True)
    kw.setdefault("is_24hr", True)
    return Facility(name=name, lat=HOME[0] + km_north / 111.0, lon=HOME[1], **kw)


def _db(*facilities):
    db = FacilityDB(":memory:")
    db.add_village("Home", *HOME)
    for f in facilities:
        db.add_facility(f)
    return db


class _RoadProvider(RoutingProvider):
    """ETA from a per-facility road table (name looked up by coordinates)."""

    def __init__(self, facilities, eta_by_name):
        self.by_coord = {(round(f.lat, 6), round(f.lon, 6)): f.name for f in facilities}
        self.eta = eta_by_name
        self.calls = 0

    def get_route(self, origin, destination):
        self.calls += 1
        name = self.by_coord[(round(destination[0], 6), round(destination[1], 6))]
        return RouteInfo(road_distance_km=self.eta[name] * 0.6, eta_minutes=self.eta[name], provider="fake_road")


def _child(age=24, **kw):
    return ExtractedCase(raw_symptom_text="x", age_months=age, **kw)


# --- B1: travel time decides ------------------------------------------------------
def test_slower_by_road_loses_even_when_nearer_in_straight_line():
    near, far = _fac("Near but across the river", 5), _fac("Far but on the highway", 15)
    provider = _RoadProvider([near, far], {"Near but across the river": 70.0, "Far but on the highway": 25.0})
    d = route(ClassificationLabel.EMERGENCY, "Home", _db(near, far), at=DAY, routing_provider=provider)
    assert d.facility.name == "Far but on the highway"
    assert d.route.eta_minutes == 25.0
    assert [a["name"] for a in d.alternatives] == ["Far but on the highway", "Near but across the river"]


def test_routing_calls_are_capped_to_the_nearest_candidates():
    facs = [_fac(f"F{i}", 2 + i) for i in range(MAX_ROUTED_CANDIDATES + 3)]
    provider = _RoadProvider(facs, {f.name: 10.0 + i for i, f in enumerate(facs)})
    route(ClassificationLabel.EMERGENCY, "Home", _db(*facs), at=DAY, routing_provider=provider)
    assert provider.calls == MAX_ROUTED_CANDIDATES


def test_offline_provider_ranks_by_distance_consistently():
    a, b = _fac("A", 3), _fac("B", 9)
    d = route(ClassificationLabel.EMERGENCY, "Home", _db(a, b), at=DAY)
    assert d.facility.name == "A" and d.route.provider == "mock_haversine"


# --- B2: what the case needs --------------------------------------------------------
def test_required_services_derived_from_case_and_label():
    assert required_services(_child(), None, ClassificationLabel.SEVERE) == ["paediatrics"]
    assert required_services(_child(age=1), None, ClassificationLabel.EMERGENCY) == ["emergency", "neonatal"]
    adult = ExtractedCase(raw_symptom_text="x", age_group="adult")
    mi = ClassificationResult(label=ClassificationLabel.EMERGENCY, probable_disease="Heart attack")
    assert required_services(adult, mi, ClassificationLabel.EMERGENCY) == ["emergency", "cardiac"]
    stroke = ClassificationResult(label=ClassificationLabel.EMERGENCY, prediction_set=["Paralysis (brain hemorrhage)", "Migraine"])
    assert "stroke" in required_services(adult, stroke, ClassificationLabel.EMERGENCY)


def test_child_is_not_sent_to_a_facility_without_paediatrics():
    adult_only = _fac("Adult-only ER", 3, services=["general", "emergency"])
    paeds = _fac("Paediatric CHC", 8)
    d = route(ClassificationLabel.EMERGENCY, "Home", _db(adult_only, paeds), at=DAY,
              case=_child(danger_signs=DangerSigns(not_able_to_drink_or_breastfeed=True)))
    assert d.facility.name == "Paediatric CHC"
    assert d.required_services == ["emergency", "paediatrics"] and d.unmet_services == []


def test_radius_widens_for_a_missing_service():
    chc = _fac("CHC nearby", 5)  # derived services: no neonatal
    dh = _fac("District hospital", 90, facility_type=FacilityType.DH)  # neonatal, 90 km away
    d = route(ClassificationLabel.EMERGENCY, "Home", _db(chc, dh), at=DAY, case=_child(age=1))
    assert d.facility.name == "District hospital"
    assert d.radius_km_used == 120.0 and d.unmet_services == []
    assert "neonatal" in facility_services(d.facility)


def test_shortfall_is_recorded_and_the_case_is_still_routed():
    only = _fac("CHC nearby", 5)
    adult = ExtractedCase(raw_symptom_text="x", age_group="adult")
    mi = ClassificationResult(label=ClassificationLabel.EMERGENCY, probable_disease="Heart attack")
    d = route(ClassificationLabel.EMERGENCY, "Home", _db(only), at=DAY, case=adult, result=mi)
    assert d.facility.name == "CHC nearby"
    assert d.unmet_services == ["cardiac"]
    assert any("recording the shortfall" in r for r in d.reasoning)


def test_no_eligible_facility_at_requested_radius_is_still_honest():
    far = _fac("Far ER", 100)
    d = route(ClassificationLabel.EMERGENCY, "Home", _db(far), at=DAY, radius_km=60)
    assert d.no_facility_found


def test_derived_services_by_facility_type():
    phc = Facility(name="p", facility_type=FacilityType.PHC, lat=0, lon=0)
    dh = Facility(name="d", facility_type=FacilityType.DH, lat=0, lon=0, has_emergency_care=True)
    assert facility_services(phc) == {"general", "paediatrics"}
    assert {"neonatal", "cardiac", "stroke", "emergency"} <= facility_services(dh)
    explicit = Facility(name="e", facility_type=FacilityType.DH, lat=0, lon=0, services=["general"])
    assert facility_services(explicit) == {"general"}


def test_services_and_timezone_round_trip_through_the_db():
    f = _fac("X", 1, services=["general", "neonatal"], timezone="Asia/Kathmandu")
    got = _db(f).all_facilities()[0]
    assert got.services == ["general", "neonatal"] and got.timezone == "Asia/Kathmandu"


# --- B3: local opening hours ----------------------------------------------------------
_PHC = Facility(name="PHC", facility_type=FacilityType.PHC, lat=0, lon=0,
                open_time="09:00", close_time="17:00", open_days=[0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "utc_hour, utc_minute, is_open",
    [
        (4, 0, True),    # 09:30 IST -- open; the old UTC compare said closed (04:00)
        (12, 0, False),  # 17:30 IST -- closed; the old UTC compare said open (12:00)
        (11, 25, True),  # 16:55 IST
        (3, 0, False),   # 08:30 IST
    ],
)
def test_hours_are_checked_in_facility_local_time(utc_hour, utc_minute, is_open):
    at = datetime(2026, 8, 24, utc_hour, utc_minute, tzinfo=timezone.utc)
    assert is_facility_open(_PHC, at) is is_open


def test_local_weekday_is_used_across_midnight():
    # Saturday 20:00 UTC = Sunday 01:30 IST; Sunday is not an open day, and an
    # overnight clinic open 00:00-06:00 every day except Sunday must be closed.
    night = Facility(name="N", facility_type=FacilityType.PHC, lat=0, lon=0,
                     open_time="00:00", close_time="06:00", open_days=[0, 1, 2, 3, 4, 5])
    assert is_facility_open(night, datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc)) is False
    assert is_facility_open(night, datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)) is True


def test_naive_datetime_is_treated_as_utc():
    assert is_facility_open(_PHC, datetime(2026, 8, 24, 4, 0)) is True


def test_urgent_routing_respects_local_hours():
    phc = Facility(name="Near PHC", facility_type=FacilityType.PHC, lat=HOME[0] + 0.02, lon=HOME[1],
                   open_time="09:00", close_time="17:00")
    chc = _fac("CHC", 10)
    db = _db(phc, chc)
    at_1730_ist = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
    at_1000_ist = datetime(2026, 8, 24, 4, 30, tzinfo=timezone.utc)
    assert route(ClassificationLabel.SEVERE, "Home", db, at=at_1730_ist).facility.name == "CHC"
    assert route(ClassificationLabel.SEVERE, "Home", db, at=at_1000_ist).facility.name == "Near PHC"
