"""The tools Agent 2 calls when it acts on an alert.

The agent's decide step (alert_policy) chooses WHICH tools to run for a signal
(`plan_for`); the tools then gather facts and take actions in the world, and
every call is returned as a ToolCall audit record. Tool selection is a
deterministic, inspectable policy -- consistent with the paper's definition of
"agentic" (a sense-decide-act loop whose decisions are auditable), not a
language model choosing tools. The LLM is still used only to word the brief.

Tools:
  query_cases            line list for the (area, disease) cluster: count, age
                         bands, severity mix, first/last seen, recent weekly trend
  check_facility_capacity nearest facilities to the area that offer the services
                         this kind of outbreak needs (inpatient, paediatrics,
                         blood bank, ...), with straight-line distance
  notify_health_workers  message the area's ASHAs/ANM (ALERT also copies the
                         district officer) through a MessagingProvider
                         (MockProvider by default -- nothing leaves the machine)
  log_escalation         persist the escalation (signal + tool results) in
                         SurveillanceState for the official's record
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Callable, Optional

from app.integrations.messaging import MessagingProvider, MockProvider
from app.routing.facility_db import FacilityDB, haversine_km
from app.routing.router import facility_services
from app.surveillance import recommend, roster
from app.surveillance.analytics import window_index
from app.surveillance.schemas import AlertLevel, OutbreakSignal, SurveillanceCase, ToolCall
from app.surveillance.state import SurveillanceState

# Services an outbreak of each response category puts demand on.
_CATEGORY_SERVICES: dict[str, tuple[str, ...]] = {
    "waterborne": ("inpatient",),
    "vector": ("inpatient", "blood_bank"),
    "respiratory": ("inpatient",),
    "contact": ("general",),
    "bloodborne": ("general",),
    "adverse_event": ("general",),
    "ncd": ("general",),
    "acute_cardiac": ("emergency", "cardiac"),
    "acute_neuro": ("emergency", "stroke"),
    "unknown": ("general",),
}
CAPACITY_RADIUS_KM = 60.0


def services_for_outbreak(disease: str, dominant_age_band: Optional[str]) -> list[str]:
    needs = list(_CATEGORY_SERVICES[recommend.category_for(disease)])
    if dominant_age_band == "<5y" and "paediatrics" not in needs:
        needs.append("paediatrics")
    return needs


class SurveillanceTools:
    """Bound tool set. Dependencies are injected so tests and deployments can
    swap them: the case stream, the facility network, the messaging provider,
    the persistent state, and a village -> coordinates lookup."""

    def __init__(
        self,
        state: SurveillanceState,
        facility_db: Optional[FacilityDB] = None,
        messenger: Optional[MessagingProvider] = None,
        coords_for: Optional[Callable[[str], Optional[tuple[float, float]]]] = None,
    ):
        from app.routing.demo_facilities import VILLAGES, build_demo_facility_db

        self.state = state
        self.facility_db = facility_db or build_demo_facility_db()
        self.messenger = messenger or MockProvider()
        self.coords_for = coords_for or (lambda area: VILLAGES.get(area))

    # -- tools ---------------------------------------------------------------------------
    def query_cases(self, cases: list[SurveillanceCase], area: str, disease: str, now: datetime) -> ToolCall:
        group = [c for c in cases if c.area == area and c.disease == disease]
        current = [c for c in group if window_index(c, now) == 0]
        weekly = Counter(window_index(c, now) for c in group)
        trend = [weekly.get(i, 0) for i in range(5, -1, -1)]  # oldest -> current
        out = {
            "total": len(group),
            "this_week": len(current),
            "weekly_trend_oldest_to_current": trend,
            "age_bands": dict(Counter(c.age_band for c in current)),
            "severity_mix": dict(Counter(c.severity.value for c in current)),
            "first_seen": min((c.recorded_at for c in current), default=None),
            "last_seen": max((c.recorded_at for c in current), default=None),
        }
        return ToolCall(
            tool="query_cases", arguments={"area": area, "disease": disease}, ok=True, at=now,
            summary=f"{out['this_week']} case(s) this week; weekly trend {trend}", output=out,
        )

    def check_facility_capacity(self, area: str, disease: str, dominant_age_band: Optional[str], now: datetime) -> ToolCall:
        needs = services_for_outbreak(disease, dominant_age_band)
        coords = self.coords_for(area)
        args = {"area": area, "disease": disease, "services": needs}
        if coords is None:
            return ToolCall(tool="check_facility_capacity", arguments=args, ok=False, at=now,
                            summary=f"no coordinates for {area}")
        nearby = self.facility_db.facilities_within_radius(coords, CAPACITY_RADIUS_KM)
        rows = []
        for f in nearby:
            svc = facility_services(f)
            rows.append({
                "name": f.name, "type": f.facility_type.value,
                "km": round(haversine_km(coords[0], coords[1], f.lat, f.lon), 1),
                "offers_all_needed": set(needs) <= svc,
                "missing": sorted(set(needs) - svc),
                "phone": f.phone,
            })
        rows.sort(key=lambda r: (not r["offers_all_needed"], r["km"]))
        capable = [r for r in rows if r["offers_all_needed"]]
        summary = (
            f"{len(capable)} facility(ies) within {CAPACITY_RADIUS_KM:.0f}km offer {needs}"
            + (f"; nearest: {capable[0]['name']} ({capable[0]['km']}km)" if capable else " -- CAPACITY GAP")
        )
        return ToolCall(tool="check_facility_capacity", arguments=args, ok=True, at=now, summary=summary,
                        output={"needed_services": needs, "facilities": rows, "capacity_gap": not capable})

    def notify_health_workers(self, area: str, disease: str, level: AlertLevel, message: str, now: datetime) -> ToolCall:
        roles = (roster.ROLE_ASHA, roster.ROLE_ANM) + (
            (roster.ROLE_DISTRICT_OFFICER,) if level == AlertLevel.ALERT else ()
        )
        recipients = roster.contacts_for(area, roles)
        sent = []
        for c in recipients:
            ok = bool(self.messenger.send_message(c["phone"], message))
            sent.append({"name": c["name"], "role": c["role"], "delivered": ok})
        delivered = sum(1 for s in sent if s["delivered"])
        return ToolCall(
            tool="notify_health_workers",
            arguments={"area": area, "disease": disease, "level": level.value, "roles": list(roles)},
            ok=bool(recipients) and delivered == len(recipients), at=now,
            summary=f"messaged {delivered}/{len(recipients)} contact(s) via {type(self.messenger).__name__}",
            output={"recipients": sent, "message": message},
        )

    def log_escalation(self, signal: OutbreakSignal, level: AlertLevel, detail: dict, now: datetime) -> ToolCall:
        eid = self.state.log_escalation(signal.area, signal.disease, level, detail, at=now)
        return ToolCall(
            tool="log_escalation", arguments={"area": signal.area, "disease": signal.disease, "level": level.value},
            ok=True, at=now, summary=f"escalation #{eid} recorded", output={"escalation_id": eid},
        )


def plan_for(level: AlertLevel, notify: bool) -> list[str]:
    """Which tools the agent runs for a surfaced signal, in order.

    WATCH: gather facts and record -- a human looks, nobody is messaged.
    ALERT: gather facts, check whether local facilities can absorb the cases,
           message the area's health workers (+ district officer) unless the
           same alert was already sent and has not worsened, and record.
    """
    if level == AlertLevel.WATCH:
        return ["query_cases", "check_facility_capacity", "log_escalation"]
    if level == AlertLevel.ALERT:
        steps = ["query_cases", "check_facility_capacity"]
        if notify:
            steps.append("notify_health_workers")
        return steps + ["log_escalation"]
    return []


def outbound_message(signal: OutbreakSignal, level: AlertLevel, action: str, capacity: Optional[ToolCall]) -> str:
    """Short SMS-length text for health workers; facts only."""
    cap = ""
    if capacity is not None and capacity.ok:
        fac = next((f for f in capacity.output.get("facilities", []) if f["offers_all_needed"]), None)
        cap = f" Refer to {fac['name']}." if fac else " No nearby facility has capacity -- escalate."
    return (
        f"[{level.value}] {signal.observed} {signal.disease} cases in {signal.area} this week "
        f"(usual ~{signal.baseline_mean:.0f}). {action}{cap}"
    )[:480]

