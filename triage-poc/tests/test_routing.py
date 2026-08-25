"""
Tests for app/routing/ (Phase 6: Routing Agent / Agent 3) -- the
deterministic facility_db.py/router.py core and the report.py LLM layer's
fallback behavior. No network, no LLM calls in the deterministic-core
tests (in-memory SQLite via FacilityDB(":memory:"), MockRoutingProvider).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import LLMBackend, RegexBackend
from app.routing.facility_db import FacilityDB, haversine_km
from app.routing.report import generate_doctor_report
from app.routing.router import (
    MockRoutingProvider,
    filter_eligible_facilities,
    is_facility_open,
    route,
    score_facility,
    urgency_for_label,
)
from app.routing.schemas import DispatchResult, Facility, FacilityType, RouteInfo
from app.schemas import ClassificationLabel, ClassificationResult, DiseaseCandidate, ExtractedCase


# --- Fixtures: the diagram's own worked example ------------------------------


@pytest.fixture
def db() -> FacilityDB:
    d = FacilityDB(db_path=":memory:")
    d.add_village("Denkanikottai", 12.5333, 77.7667)
    d.add_facility(
        Facility(
            name="PHC A (Nearest)", facility_type=FacilityType.PHC, lat=12.5666, lon=77.7999,
            has_doctor_24hr=False, has_emergency_care=False, has_blood_bank=False,
            open_time="09:00", close_time="17:00", phone="04343-000001",
        )
    )
    d.add_facility(
        Facility(
            name="CHC B", facility_type=FacilityType.CHC, lat=12.5883, lon=77.8217,
            has_doctor_24hr=True, has_emergency_care=True, has_blood_bank=False,
            is_24hr=True, phone="04343-123456",
        )
    )
    d.add_facility(
        Facility(
            name="District Hospital C", facility_type=FacilityType.DH, lat=12.65, lon=77.90,
            has_doctor_24hr=True, has_emergency_care=True, has_blood_bank=True,
            is_24hr=True, phone="04343-999999",
        )
    )
    d.add_facility(
        Facility(
            name="PHC D (Far)", facility_type=FacilityType.PHC, lat=13.5, lon=78.5,
            has_doctor_24hr=False, has_emergency_care=False, open_time="09:00", close_time="17:00",
        )
    )
    yield d
    d.close()


NIGHT = datetime(2026, 8, 24, 22, 0, tzinfo=timezone.utc)  # 10pm -- PHC A/D (9-5) closed
DAY = datetime(2026, 8, 24, 11, 0, tzinfo=timezone.utc)  # 11am -- everything open


# --- haversine_km --------------------------------------------------------------


def test_haversine_zero_distance_for_same_point():
    assert haversine_km(12.5, 77.5, 12.5, 77.5) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_roughly_correct():
    # Denkanikottai to CHC B in the fixture above -- roughly 8-9km.
    d = haversine_km(12.5333, 77.7667, 12.5883, 77.8217)
    assert 8 < d < 9


# --- urgency_for_label ---------------------------------------------------------


def test_urgency_mapping():
    assert urgency_for_label(ClassificationLabel.EMERGENCY) == "EMERGENCY"
    assert urgency_for_label(ClassificationLabel.SEVERE) == "URGENT"
    assert urgency_for_label(ClassificationLabel.MODERATE) is None
    assert urgency_for_label(ClassificationLabel.MILD) is None
    assert urgency_for_label(ClassificationLabel.INCOMPLETE_ASSESSMENT) is None


# --- is_facility_open -----------------------------------------------------------


def test_is_24hr_always_open():
    f = Facility(name="X", facility_type=FacilityType.CHC, lat=0, lon=0, is_24hr=True)
    assert is_facility_open(f, NIGHT) is True
    assert is_facility_open(f, DAY) is True


def test_hours_based_facility_closed_at_night():
    f = Facility(name="X", facility_type=FacilityType.PHC, lat=0, lon=0, open_time="09:00", close_time="17:00")
    assert is_facility_open(f, NIGHT) is False
    assert is_facility_open(f, DAY) is True


def test_overnight_hours_span():
    f = Facility(name="X", facility_type=FacilityType.PHC, lat=0, lon=0, open_time="20:00", close_time="06:00")
    assert is_facility_open(f, NIGHT) is True  # 22:00 is within 20:00-06:00
    assert is_facility_open(f, DAY) is False  # 11:00 is not


def test_unknown_hours_treated_as_not_verifiably_open():
    f = Facility(name="X", facility_type=FacilityType.PHC, lat=0, lon=0)
    assert is_facility_open(f, DAY) is False


def test_open_days_restricts_to_specific_weekdays():
    # NIGHT/DAY are both a Monday (2026-08-24) -- weekday() == 0.
    f = Facility(
        name="X", facility_type=FacilityType.PHC, lat=0, lon=0,
        open_time="09:00", close_time="17:00", open_days=[1, 2, 3],  # Tue/Wed/Thu only
    )
    assert is_facility_open(f, DAY) is False


# --- filter_eligible_facilities --------------------------------------------------


def test_emergency_filter_requires_both_capabilities(db: FacilityDB):
    all_facilities = db.all_facilities()
    eligible = filter_eligible_facilities(all_facilities, "EMERGENCY", NIGHT)
    names = {f.name for f in eligible}
    assert names == {"CHC B", "District Hospital C"}  # PHC A/D lack emergency+24hr doctor


def test_urgent_filter_requires_only_open(db: FacilityDB):
    eligible_night = filter_eligible_facilities(db.all_facilities(), "URGENT", NIGHT)
    eligible_day = filter_eligible_facilities(db.all_facilities(), "URGENT", DAY)
    assert {f.name for f in eligible_night} == {"CHC B", "District Hospital C"}  # is_24hr ones
    assert {f.name for f in eligible_day} == {"PHC A (Nearest)", "CHC B", "District Hospital C", "PHC D (Far)"}


# --- score_facility --------------------------------------------------------------


def test_scoring_prefers_equipped_facility_over_nearer_unequipped(db: FacilityDB):
    coords = (12.5333, 77.7667)
    phc_a = next(f for f in db.all_facilities() if f.name == "PHC A (Nearest)")
    chc_b = next(f for f in db.all_facilities() if f.name == "CHC B")
    scored_phc = score_facility(phc_a, coords)
    scored_chc = score_facility(chc_b, coords)
    # PHC A is geographically nearer but has none of the capability bonuses;
    # CHC B's -10 total bonus should still win despite being farther.
    assert scored_phc.straight_line_distance_km < scored_chc.straight_line_distance_km
    assert scored_chc.score < scored_phc.score


def test_score_reasoning_lists_every_bonus_applied():
    f = Facility(
        name="X", facility_type=FacilityType.DH, lat=12.6, lon=77.85,
        has_doctor_24hr=True, has_emergency_care=True, has_blood_bank=True,
    )
    scored = score_facility(f, (12.5333, 77.7667))
    assert any("has_doctor_24hr" in r for r in scored.reasoning)
    assert any("has_emergency_care" in r for r in scored.reasoning)
    assert any("has_blood_bank" in r for r in scored.reasoning)


# --- route(): full orchestration, mirroring the diagram's worked example -------


def test_full_route_matches_diagram_worked_example(db: FacilityDB):
    result = route(ClassificationLabel.EMERGENCY, "Denkanikottai", db, at=NIGHT)
    assert result.facility.name == "CHC B"  # beats the nearer PHC and the farther DH
    assert result.urgency == "EMERGENCY"
    assert result.no_facility_found is False
    assert result.route is not None
    assert result.route.road_distance_km > 0
    assert result.route.eta_minutes > 0
    assert result.route.maps_link.startswith("https://www.google.com/maps")


def test_route_non_urgent_label_does_not_route(db: FacilityDB):
    for label in (ClassificationLabel.MODERATE, ClassificationLabel.MILD, ClassificationLabel.INCOMPLETE_ASSESSMENT):
        result = route(label, "Denkanikottai", db, at=DAY)
        assert result.no_facility_found is True
        assert result.facility is None
        assert result.urgency == "NON_URGENT"


def test_route_unresolvable_location(db: FacilityDB):
    result = route(ClassificationLabel.EMERGENCY, "Nonexistent Village", db, at=DAY)
    assert result.no_facility_found is True
    assert result.facility is None
    assert any("could not resolve" in r for r in result.reasoning)


def test_route_gps_tuple_location_works_without_village_lookup(db: FacilityDB):
    result = route(ClassificationLabel.EMERGENCY, (12.5333, 77.7667), db, at=DAY)
    assert result.facility is not None


def test_route_no_eligible_facility_in_radius():
    db = FacilityDB(db_path=":memory:")
    db.add_village("Remote", 0.0, 0.0)
    db.add_facility(
        Facility(name="Far Away Hospital", facility_type=FacilityType.DH, lat=50.0, lon=50.0,
                 has_doctor_24hr=True, has_emergency_care=True)
    )
    result = route(ClassificationLabel.EMERGENCY, "Remote", db, at=DAY, radius_km=60)
    assert result.no_facility_found is True
    assert "no eligible facility" in result.reasoning[-1]
    db.close()


def test_route_urgent_uses_hours_based_filter(db: FacilityDB):
    result_day = route(ClassificationLabel.SEVERE, "Denkanikottai", db, at=DAY)
    result_night = route(ClassificationLabel.SEVERE, "Denkanikottai", db, at=NIGHT)
    assert result_day.facility is not None
    assert result_night.facility is not None
    # PHC A is nearest and open during the day -- but CHC B's capability
    # bonuses should still make it win the composite score.
    assert result_day.facility.name == "CHC B"
    assert result_night.facility.name == "CHC B"


def test_route_reasoning_trail_is_populated(db: FacilityDB):
    result = route(ClassificationLabel.EMERGENCY, "Denkanikottai", db, at=NIGHT)
    assert len(result.reasoning) >= 5  # one per step roughly
    assert any("urgency" in r for r in result.reasoning)
    assert any("resolved location" in r for r in result.reasoning)
    assert any("eligible" in r for r in result.reasoning)
    assert any("top-scored" in r for r in result.reasoning)


def test_route_deterministic(db: FacilityDB):
    a = route(ClassificationLabel.EMERGENCY, "Denkanikottai", db, at=NIGHT)
    b = route(ClassificationLabel.EMERGENCY, "Denkanikottai", db, at=NIGHT)
    assert a.facility.name == b.facility.name
    assert a.route.road_distance_km == b.route.road_distance_km


def test_mock_routing_provider_directly():
    provider = MockRoutingProvider()
    info = provider.get_route((12.5333, 77.7667), (12.60, 77.85))
    assert info.provider == "mock_haversine"
    assert info.road_distance_km > 0
    assert info.eta_minutes > 0
    assert "google.com/maps" in info.maps_link


# --- report.py: fallback path (no live LLM required) ---------------------------


def _make_case_and_result():
    case = ExtractedCase(raw_symptom_text="chest pain and sweating", age_group="adult")
    result = ClassificationResult(
        label=ClassificationLabel.EMERGENCY,
        condition="Heart attack",
        probable_disease="Heart attack",
        candidates=[
            DiseaseCandidate(name="Heart attack", score=0.7, matched_symptoms=["chest_pain"], precautions=["call ambulance"])
        ],
    )
    dispatch = DispatchResult(
        facility=Facility(name="CHC B", facility_type=FacilityType.CHC, lat=12.6, lon=77.85,
                           has_doctor_24hr=True, has_emergency_care=True),
        route=RouteInfo(road_distance_km=11.69, eta_minutes=17.5, maps_link="https://maps.example", provider="mock_haversine"),
        urgency="EMERGENCY",
    )
    return case, result, dispatch


def test_report_falls_back_for_regex_backend():
    case, result, dispatch = _make_case_and_result()
    report = generate_doctor_report(case, result, dispatch, RegexBackend())
    assert "EMERGENCY" in report
    assert "Heart attack" in report
    assert "CHC B" in report


def test_report_fallback_never_invents_a_different_label():
    case, result, dispatch = _make_case_and_result()
    report = generate_doctor_report(case, result, dispatch, RegexBackend())
    assert "MILD" not in report
    assert "MODERATE" not in report


def test_report_generation_is_polymorphic_no_isinstance_coupling():
    # generate_doctor_report() must work with ANY LLMBackend subclass that
    # overrides generate_text() -- not just OllamaBackend/GroqBackend by
    # name. Proves report.py has zero isinstance-based dispatch on
    # concrete backend types.
    class _CustomBackend(LLMBackend):
        name = "custom_test_backend"

        def extract(self, patient_text, context=None):
            return {}

        def generate_text(self, system_prompt: str, user_prompt: str) -> str:
            return "custom backend report text"

    case, result, dispatch = _make_case_and_result()
    report = generate_doctor_report(case, result, dispatch, _CustomBackend())
    assert report == "custom backend report text"


def test_report_backend_generate_text_failure_falls_back():
    class _FailingBackend(LLMBackend):
        name = "failing_test_backend"

        def extract(self, patient_text, context=None):
            return {}

        def generate_text(self, system_prompt: str, user_prompt: str) -> str:
            from app.agent1_extraction import BackendUnavailable

            raise BackendUnavailable("simulated outage")

    case, result, dispatch = _make_case_and_result()
    report = generate_doctor_report(case, result, dispatch, _FailingBackend())
    assert "EMERGENCY" in report  # fell back to the deterministic summary
