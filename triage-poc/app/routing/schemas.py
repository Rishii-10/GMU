"""
Pydantic models for the Routing Agent (Agent 3). Kept in their own module
rather than added to the top-level app/schemas.py -- that file is
specifically the shared Agent-1/Rules-Engine contract; routing is a
separate agent with its own concerns (facility master data, route/dispatch
results) that Agent 1 and the Rules Engine never need to see.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FacilityType(str, Enum):
    PHC = "PHC"
    CHC = "CHC"
    DH = "DH"
    SDH = "SDH"
    MEDICAL_COLLEGE = "Medical College"


class Facility(BaseModel):
    """Facility master data (plan diagram 2, step 2: "Name, Type, Coordinates,
    Capabilities, Doctor availability schedule, Contact details")."""

    facility_id: Optional[int] = None
    name: str
    facility_type: FacilityType
    lat: float
    lon: float
    has_emergency_care: bool = False
    has_doctor_24hr: bool = False
    has_blood_bank: bool = False
    is_24hr: bool = Field(
        default=False,
        description="Facility itself (not just the doctor) is open around the clock -- overrides open_time/close_time/open_days entirely.",
    )
    open_time: Optional[str] = Field(default=None, description='"HH:MM", 24hr clock, local time. None if not is_24hr and hours unknown.')
    close_time: Optional[str] = Field(default=None, description='"HH:MM", 24hr clock. May be earlier than open_time to express an overnight span (e.g. 20:00-06:00).')
    open_days: Optional[list[int]] = Field(
        default=None,
        description="0=Monday..6=Sunday. None means open every day (subject to open_time/close_time).",
    )
    phone: Optional[str] = None


class ScoredFacility(BaseModel):
    """One facility's routing score (plan diagram 2, step 4). `score` uses
    STRAIGHT-LINE (haversine) distance as an efficient proxy for ranking
    all eligible facilities -- calling a real routing API for every
    candidate before knowing which one wins would be wasteful. The real
    road distance/ETA is fetched only for the single top-scored facility
    (step 5, see app.routing.router.route())."""

    facility: Facility
    straight_line_distance_km: float
    score: float
    reasoning: list[str] = Field(default_factory=list)


class RouteInfo(BaseModel):
    """Real road route for the chosen facility (plan diagram 2, step 5).
    `provider` names which RoutingProvider produced this -- "mock_haversine"
    (Phase 6 default, offline) or "openrouteservice" (Phase 7, real API) --
    so a caller can tell an estimate from a live routed result."""

    road_distance_km: Optional[float] = None
    eta_minutes: Optional[float] = None
    maps_link: Optional[str] = None
    provider: str


class DispatchResult(BaseModel):
    """Final routing-agent output (plan diagram 2, step 6: "facility
    name/type/phone, road distance, ETA, Google Maps link"). `reasoning` is
    the full audit trail across all 6 steps, same "no black-box scoring"
    convention as app.schemas.ClassificationResult.reasoning.

    `no_facility_found=True` covers every honest "couldn't route" case
    (label doesn't warrant routing, location unresolvable, no eligible
    facility within radius) -- `facility`/`route` stay None rather than a
    forced guess; `reasoning` always explains which case it was.
    """

    facility: Optional[Facility] = None
    route: Optional[RouteInfo] = None
    urgency: str  # "EMERGENCY" | "URGENT" | "NON_URGENT"
    reasoning: list[str] = Field(default_factory=list)
    case_id: Optional[str] = None
    no_facility_found: bool = False
