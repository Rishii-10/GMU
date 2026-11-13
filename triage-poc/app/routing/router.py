"""
Routing Agent (Agent 3) -- deterministic core. Implements the 6 steps from
plan diagram 2 ("Routing Agent -- Data Flow Overview"):

  1. Receive input: urgency (EMERGENCY/URGENT) + patient location (village
     name or GPS) + current date/time.
  2. Query facility DB: facilities within a configurable radius (default
     60km) of the resolved location.
  3. Filter by availability & capability: URGENT -> currently open (in the
     FACILITY'S local time zone); EMERGENCY -> must have emergency care AND a
     24-hr doctor; then by the SERVICES the case needs (required_services:
     paediatrics for a child, neonatal care for a young infant, cardiac/stroke
     care for those emergencies). If no facility within the radius offers
     them, the radius widens (WIDEN_RADII_KM); if still none, the case is
     routed on the urgency filter alone and the shortfall is recorded in
     `unmet_services` -- an emergency is never left unrouted for want of a
     specialty.
  4. Rank by TRAVEL TIME: the RoutingProvider is called for the nearest
     MAX_ROUTED_CANDIDATES eligible facilities (by straight line, to bound
     calls to a real routing API) and the one with the shortest ETA wins
     (ties: 24-hr doctor, then road distance).
  5. Route & ETA come from that same call -- the winner's road route is the
     one already computed, and every routed candidate is kept in
     `alternatives` for audit.
  6. Prepare result: facility + route info, packaged for the dispatch
     pipeline, with a full step-by-step reasoning trail for audit.

No LLM anywhere in this file (plan diagram 2's own "Key characteristics":
deterministic, transparent, auditable, No LLM, reproducible, privacy-safe).
The LLM doctor-report layer the user asked to add ON TOP of this core lives
in app/routing/report.py instead, deliberately kept out of this module.

SELECTION BY TRAVEL TIME -- decision, stated explicitly (2026-11-13)
------------------------------------------------------------
The facility is chosen by travel time from the RoutingProvider, not by
straight-line distance. Earlier versions ranked by straight-line distance
minus fixed capability bonuses and fetched a road route only for the winner,
so a facility near by straight line but far by road could be chosen. Capability
is now a FILTER (what the case needs), not a bonus, and travel time decides
among the facilities that meet it. Straight-line distance is used only to cap
how many candidates are sent to the routing provider (MAX_ROUTED_CANDIDATES).
With the offline MockRoutingProvider, travel time is proportional to
straight-line distance; a real provider (app.integrations.geo.ORSProvider)
uses the road network. `score_facility` (the old composite) is kept for
callers that still report it, but no longer selects the facility.

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
from zoneinfo import ZoneInfo

from app.routing.facility_db import FacilityDB, Location, haversine_km
from app.routing.schemas import DispatchResult, Facility, RouteInfo, ScoredFacility
from app.schemas import ClassificationLabel, ClassificationResult, ExtractedCase

DEFAULT_RADIUS_KM = 60.0
WIDEN_RADII_KM: tuple[float, ...] = (120.0, 200.0)
MAX_ROUTED_CANDIDATES = 5

# Controlled service vocabulary for Facility.services.
SERVICES: tuple[str, ...] = (
    "general", "paediatrics", "neonatal", "emergency", "inpatient", "blood_bank", "cardiac", "stroke",
)

# Emergencies that need a specific service beyond general emergency care.
_DISEASE_SERVICE: dict[str, str] = {
    "Heart attack": "cardiac",
    "Paralysis (brain hemorrhage)": "stroke",
}

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


def _local_time(facility: Facility, at: datetime) -> datetime:
    """`at` expressed in the facility's own time zone. A naive `at` is taken
    as UTC. Fixes D-TZ: opening hours are local ("09:00" in Hosur is IST), but
    the check used to compare them against the UTC clock -- 5h30m off."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(ZoneInfo(facility.timezone))


def is_facility_open(facility: Facility, at: datetime) -> bool:
    """Step 3's "is facility currently open?" check, in the facility's LOCAL
    time. is_24hr short-circuits to always-open. A facility with open_days/
    open_time/close_time all None (hours genuinely unknown) is treated as NOT
    verifiably open -- same "don't guess, report honestly" stance as the
    rest of this codebase, since falsely routing a caregiver to a closed
    facility is worse than filtering it out."""
    if facility.is_24hr:
        return True
    local = _local_time(facility, at)
    if facility.open_days is not None and local.weekday() not in facility.open_days:
        return False
    if facility.open_time is None or facility.close_time is None:
        return False
    current = local.time()
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


def facility_services(facility: Facility) -> set[str]:
    """Services a facility offers: its explicit `services`, else derived from
    the capability flags and facility type (PHCs and up run IMNCI child care;
    district hospitals and medical colleges have newborn units)."""
    if facility.services:
        return set(facility.services)
    out = {"general", "paediatrics"}
    if facility.has_emergency_care:
        out.add("emergency")
    if facility.has_blood_bank:
        out.add("blood_bank")
    if facility.facility_type.value in ("CHC", "SDH", "DH", "Medical College"):
        out.add("inpatient")
    if facility.facility_type.value in ("DH", "Medical College"):
        out.update({"neonatal", "cardiac", "stroke"})
    return out


def required_services(
    case: Optional[ExtractedCase], result: Optional[ClassificationResult], label: ClassificationLabel
) -> list[str]:
    """Services this case needs, derived only from what the decision module
    already established -- age band, the urgency label, a confirmed danger
    sign, and the condition that set an EMERGENCY label. No new clinical
    judgement is made here."""
    needs: list[str] = []
    if label == ClassificationLabel.EMERGENCY:
        needs.append("emergency")
    if case is not None:
        age = case.age_months
        group = case.age_group.value if case.age_group is not None else None
        if age is not None and age < 2:
            needs.append("neonatal")
        elif (age is not None and age < 60) or (age is None and group in ("infant", "child")):
            needs.append("paediatrics")
    if result is not None and label == ClassificationLabel.EMERGENCY:
        for disease in [result.probable_disease, *result.prediction_set]:
            svc = _DISEASE_SERVICE.get(disease or "")
            if svc and svc not in needs:
                needs.append(svc)
    return needs


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
    case: Optional[ExtractedCase] = None,
    result: Optional[ClassificationResult] = None,
) -> DispatchResult:
    """Top-level orchestrator for all 6 steps. This is the only function
    the dispatch pipeline (or app.routing.report's LLM layer) should call.

    `location`: village name (looked up via facility_db) or an explicit
    (lat, lon) tuple. `at`: defaults to current UTC time -- pass an
    explicit datetime for testable/reproducible "is it open" checks.
    `case`/`result` (optional) let step 3 match the services the case needs.
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

    needs = required_services(case, result, label)
    if needs:
        reasoning.append(f"services required by the case: {needs}")

    # Step 2-3: radius query + availability/capability filter (urgency).
    nearby = facility_db.facilities_within_radius(coords, radius_km)
    base_eligible = filter_eligible_facilities(nearby, urgency, at)
    reasoning.append(f"{len(nearby)} facility(ies) within {radius_km}km")
    reasoning.append(
        f"{len(base_eligible)} facility(ies) eligible for {urgency} (availability+capability filter)"
    )
    if not base_eligible:
        return DispatchResult(
            urgency=urgency,
            no_facility_found=True,
            reasoning=reasoning + ["no eligible facility found within radius"],
            case_id=case_id,
            required_services=needs,
            unmet_services=list(needs),
            radius_km_used=radius_km,
        )

    # Step 3b: match the services the case needs. The radius widens only for a
    # missing SERVICE (a reachable facility exists, but none offers it); if none
    # within the widest radius does, route on the urgency filter alone and
    # record the shortfall -- an emergency is never left unrouted for a specialty.
    eligible, radius_used, unmet = [], radius_km, []
    for r in (radius_km, *[w for w in WIDEN_RADII_KM if w > radius_km]):
        pool = base_eligible if r == radius_km else filter_eligible_facilities(
            facility_db.facilities_within_radius(coords, r), urgency, at
        )
        matched = [f for f in pool if set(needs) <= facility_services(f)]
        if matched:
            eligible, radius_used = matched, r
            if needs:
                reasoning.append(f"{len(matched)} eligible facility(ies) within {r}km offer {needs}")
            break
    if not eligible:
        eligible = base_eligible
        unmet = [n for n in needs if not any(n in facility_services(f) for f in base_eligible)]
        reasoning.append(
            f"no eligible facility within {max((radius_km, *WIDEN_RADII_KM))}km offers {unmet}; "
            f"routing to the fastest facility that meets the {urgency} filter and recording the shortfall"
        )

    # Step 4: rank by travel time over the nearest MAX_ROUTED_CANDIDATES.
    by_line = sorted(eligible, key=lambda f: haversine_km(coords[0], coords[1], f.lat, f.lon))
    routed = []
    for f in by_line[:MAX_ROUTED_CANDIDATES]:
        info = routing_provider.get_route(coords, (f.lat, f.lon))
        eta = info.eta_minutes if info.eta_minutes is not None else float("inf")
        dist = info.road_distance_km if info.road_distance_km is not None else float("inf")
        routed.append((eta, not f.has_doctor_24hr, dist, f.name, f, info))
    routed.sort(key=lambda t: t[:4])
    eta, _, _, _, top, route_info = routed[0]
    reasoning.append(
        f"fastest by travel time ({route_info.provider}): {top.name} -- "
        f"{route_info.road_distance_km}km, ETA {route_info.eta_minutes}min"
    )
    for e, _, d, name, _, _ in routed[1:]:
        reasoning.append(f"also considered: {name} (ETA {e}min, {d}km)")

    # Steps 5-6: the winner's route is the one already computed; package.
    return DispatchResult(
        facility=top,
        route=route_info,
        urgency=urgency,
        reasoning=reasoning,
        case_id=case_id,
        required_services=needs,
        unmet_services=unmet,
        radius_km_used=radius_used,
        alternatives=[
            {"name": name, "eta_minutes": info.eta_minutes, "road_distance_km": info.road_distance_km}
            for _, _, _, name, _, info in routed
        ],
    )
