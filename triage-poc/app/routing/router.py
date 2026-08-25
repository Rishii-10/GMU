"""
Routing Agent (Agent 3) -- deterministic core. Implements the 6 steps from
plan diagram 2 ("Routing Agent -- Data Flow Overview"):

  1. Receive input: urgency (EMERGENCY/URGENT) + patient location (village
     name or GPS) + current date/time.
  2. Query facility DB: facilities within a configurable radius (default
     60km) of the resolved location.
  3. Filter by availability & capability: URGENT -> currently open;
     EMERGENCY -> must have emergency care AND a 24-hr doctor.
  4. Score facilities: composite function, lower is better --
     score = straight_line_distance_km - 5*has_doctor_24hr
                                        - 3*has_emergency_care
                                        - 2*has_blood_bank
  5. Route & ETA: real road distance/driving time for the ONE top-scored
     facility only, via a RoutingProvider (Google Maps link included).
  6. Prepare result: facility + route info, packaged for the dispatch
     pipeline, with a full step-by-step reasoning trail for audit.

No LLM anywhere in this file (plan diagram 2's own "Key characteristics":
deterministic, transparent, auditable, No LLM, reproducible, privacy-safe).
The LLM doctor-report layer the user asked to add ON TOP of this core lives
in app/routing/report.py instead, deliberately kept out of this module.

STEP 4 vs STEP 5 DISTANCE -- decision, stated explicitly
------------------------------------------------------------
Step 4 scores every ELIGIBLE facility using straight-line (haversine)
distance as an efficient proxy for ranking -- calling a real routing API
once per candidate before knowing which one wins would be wasteful network
use for no benefit. Step 5 then calls a RoutingProvider exactly once, for
the single top-scored facility, to get the real road distance/ETA/maps
link the dispatch output actually needs. This is a stated simplification,
not silently glossed over: two facilities that are close by straight-line
distance but far by actual road distance could in principle be
misordered by step 4's scoring. Acceptable for a POC; a future iteration
could re-rank the top-K by real road distance if this proves to matter.

ROUTING PROVIDER -- decision, stated explicitly
------------------------------------------------------------
RoutingProvider (ABC) + MockRoutingProvider (offline default: haversine
distance + a fixed average-speed ETA assumption + a Google-Maps-link built
from raw coordinates, which works without any API key) are defined HERE,
not deferred to Phase 7. Phase 7 (app/integrations/geo.py) adds the real
OpenRouteService-backed provider and re-exports this same interface --
kept in one place so Phase 7 is purely additive, no rework of this module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time as dt_time, timezone
from typing import Optional

from app.routing.facility_db import FacilityDB, Location, haversine_km
from app.routing.schemas import DispatchResult, Facility, RouteInfo, ScoredFacility
from app.schemas import ClassificationLabel

DEFAULT_RADIUS_KM = 60.0

_SCORE_WEIGHT_HAS_DOCTOR_24HR = 5.0
_SCORE_WEIGHT_HAS_EMERGENCY_CARE = 3.0
_SCORE_WEIGHT_HAS_BLOOD_BANK = 2.0


def urgency_for_label(label: ClassificationLabel) -> Optional[str]:
    """Maps a Rules-Engine/dataset-classifier severity label onto the
    routing agent's own two-tier urgency notion (plan diagram 1: only
    Emergency and Urgent branches reach the routing agent at all --
    Non-urgent gets a Home care SMS instead, upstream of this module).

    EMERGENCY -> "EMERGENCY". SEVERE -> "URGENT" (serious enough to need
    prompt facility care, per data/disease_severity.csv's own SEVERE-tier
    rationale -- see app/disease_kb.py's module docstring -- but not
    necessarily an ambulance-grade emergency).
    MODERATE/MILD/INCOMPLETE_ASSESSMENT -> None: these do not
    warrant routing at all -- INCOMPLETE_ASSESSMENT specifically means
    "not enough info yet," which should trigger more follow-up questions
    (app.followup_policy), not a facility dispatch.
    """
    if label == ClassificationLabel.EMERGENCY:
        return "EMERGENCY"
    if label == ClassificationLabel.SEVERE:
        return "URGENT"
    return None


class RoutingProvider(ABC):
    """Single-method interface so a real routing API can be swapped in
    without touching router.route() -- same pattern as
    app.disambiguation.Disambiguator."""

    @abstractmethod
    def get_route(self, origin: tuple[float, float], destination: tuple[float, float]) -> RouteInfo:
        """origin/destination are (lat, lon) pairs. Must never raise on
        ordinary coordinate input -- ABCs elsewhere in this codebase treat
        an honest degraded result as preferable to a crash; this is no
        different, and callers should be able to always get a road_distance
        estimate."""


class MockRoutingProvider(RoutingProvider):
    """Offline default (no network, no API key): straight-line (haversine)
    distance, ETA from a fixed average-speed assumption, and a Google Maps
    directions link built from raw coordinates (this link format works
    without any API key -- it's a plain user-facing maps URL, not an API
    call). Real OpenRouteService-backed provider: Phase 7's
    app.integrations.geo.ORSProvider.
    """

    # Rough rural-road average speed assumption for the ETA estimate --
    # uncalibrated, same "flagged, not validated" status as other
    # placeholder constants in this codebase (e.g.
    # app.disambiguation.DEFAULT_CONFIDENCE_THRESHOLD).
    AVERAGE_SPEED_KMH = 40.0

    def get_route(self, origin: tuple[float, float], destination: tuple[float, float]) -> RouteInfo:
        distance_km = haversine_km(origin[0], origin[1], destination[0], destination[1])
        eta_minutes = (distance_km / self.AVERAGE_SPEED_KMH) * 60.0
        maps_link = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}"
        )
        return RouteInfo(
            road_distance_km=round(distance_km, 2),
            eta_minutes=round(eta_minutes, 1),
            maps_link=maps_link,
            provider="mock_haversine",
        )


def _parse_hhmm(value: str) -> dt_time:
    hour, minute = value.split(":")
    return dt_time(int(hour), int(minute))


def is_facility_open(facility: Facility, at: datetime) -> bool:
    """Step 3's "is facility currently open?" check. is_24hr short-
    circuits to always-open. A facility with open_days/open_time/
    close_time all None (hours genuinely unknown) is treated as NOT
    verifiably open -- same "don't guess, report honestly" stance as the
    rest of this codebase, since falsely routing a caregiver to a closed
    facility is worse than filtering it out."""
    if facility.is_24hr:
        return True
    if facility.open_days is not None and at.weekday() not in facility.open_days:
        return False
    if facility.open_time is None or facility.close_time is None:
        return False
    current = at.time()
    open_t = _parse_hhmm(facility.open_time)
    close_t = _parse_hhmm(facility.close_time)
    if open_t <= close_t:
        return open_t <= current <= close_t
    # Overnight span, e.g. "20:00"-"06:00".
    return current >= open_t or current <= close_t


def filter_eligible_facilities(facilities: list[Facility], urgency: str, at: datetime) -> list[Facility]:
    """Step 3: EMERGENCY must have emergency care AND a 24-hr doctor
    (regardless of open_time/close_time -- a facility with a 24-hr doctor
    is, by definition, available for an emergency at any hour). URGENT
    just needs to be currently open."""
    if urgency == "EMERGENCY":
        return [f for f in facilities if f.has_emergency_care and f.has_doctor_24hr]
    return [f for f in facilities if is_facility_open(f, at)]


def score_facility(facility: Facility, coords: tuple[float, float]) -> ScoredFacility:
    """Step 4's composite scoring function (lower = better)."""
    distance_km = haversine_km(coords[0], coords[1], facility.lat, facility.lon)
    score = distance_km
    reasoning = [f"straight-line distance: {distance_km:.2f}km"]
    if facility.has_doctor_24hr:
        score -= _SCORE_WEIGHT_HAS_DOCTOR_24HR
        reasoning.append(f"has_doctor_24hr: -{_SCORE_WEIGHT_HAS_DOCTOR_24HR}")
    if facility.has_emergency_care:
        score -= _SCORE_WEIGHT_HAS_EMERGENCY_CARE
        reasoning.append(f"has_emergency_care: -{_SCORE_WEIGHT_HAS_EMERGENCY_CARE}")
    if facility.has_blood_bank:
        score -= _SCORE_WEIGHT_HAS_BLOOD_BANK
        reasoning.append(f"has_blood_bank: -{_SCORE_WEIGHT_HAS_BLOOD_BANK}")
    return ScoredFacility(
        facility=facility,
        straight_line_distance_km=round(distance_km, 2),
        score=round(score, 2),
        reasoning=reasoning,
    )


def route(
    label: ClassificationLabel,
    location: Location,
    facility_db: FacilityDB,
    at: Optional[datetime] = None,
    radius_km: float = DEFAULT_RADIUS_KM,
    routing_provider: Optional[RoutingProvider] = None,
    case_id: Optional[str] = None,
) -> DispatchResult:
    """Top-level orchestrator for all 6 steps. This is the only function
    the dispatch pipeline (or app.routing.report's LLM layer) should call.

    `location`: village name (looked up via facility_db) or an explicit
    (lat, lon) tuple. `at`: defaults to current UTC time -- pass an
    explicit datetime for testable/reproducible "is it open" checks.
    """
    at = at or datetime.now(timezone.utc)
    routing_provider = routing_provider or MockRoutingProvider()
    reasoning: list[str] = []

    # Step 1 (partial): urgency from the classification label.
    urgency = urgency_for_label(label)
    if urgency is None:
        return DispatchResult(
            urgency="NON_URGENT",
            no_facility_found=True,
            reasoning=[
                f"classification label {label.value} does not warrant routing -- "
                "only EMERGENCY/SEVERE (mapped to EMERGENCY/URGENT) reach the "
                "routing agent; MODERATE/MILD go to a Home care SMS instead, and "
                "INCOMPLETE_ASSESSMENT needs more follow-up first, not dispatch."
            ],
            case_id=case_id,
        )
    reasoning.append(f"urgency: {label.value} -> {urgency}")

    # Step 1 (rest): resolve location to coordinates.
    coords = facility_db.resolve_location(location)
    if coords is None:
        return DispatchResult(
            urgency=urgency,
            no_facility_found=True,
            reasoning=reasoning + [f"could not resolve location {location!r} to coordinates"],
            case_id=case_id,
        )
    reasoning.append(f"resolved location {location!r} -> coords {coords}")

    # Step 2: facilities within radius.
    nearby = facility_db.facilities_within_radius(coords, radius_km)
    reasoning.append(f"{len(nearby)} facility(ies) within {radius_km}km")

    # Step 3: availability & capability filter.
    eligible = filter_eligible_facilities(nearby, urgency, at)
    reasoning.append(f"{len(eligible)} facility(ies) eligible for {urgency} (availability+capability filter)")
    if not eligible:
        return DispatchResult(
            urgency=urgency,
            no_facility_found=True,
            reasoning=reasoning + ["no eligible facility found within radius"],
            case_id=case_id,
        )

    # Step 4: score and rank.
    scored = sorted((score_facility(f, coords) for f in eligible), key=lambda s: s.score)
    top = scored[0]
    reasoning.append(f"top-scored facility: {top.facility.name} (score={top.score})")
    for other in scored[1:]:
        reasoning.append(f"also considered: {other.facility.name} (score={other.score})")

    # Step 5: real route & ETA for the chosen facility only.
    route_info = routing_provider.get_route(coords, (top.facility.lat, top.facility.lon))
    reasoning.append(
        f"route via {route_info.provider}: {route_info.road_distance_km}km, "
        f"ETA {route_info.eta_minutes}min"
    )

    # Step 6: package for the dispatch pipeline.
    return DispatchResult(
        facility=top.facility,
        route=route_info,
        urgency=urgency,
        reasoning=reasoning,
        case_id=case_id,
    )
