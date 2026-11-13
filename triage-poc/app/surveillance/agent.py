"""SurveillanceAgent -- the orchestrator that ties the layers into an agent.

One run() is a full agent cycle:
  perceive  -> ingest the case stream
  reason    -> village risk table + baseline-deviation detection
  decide    -> grade each signal; keep only those worth a human's attention, and
               choose the tools to run for it (tools.plan_for), including whether
               a notification is warranted (state.should_notify: new, worse, or
               still active in a new week -- never a repeat of the same alert)
  act       -> run the tools: pull the line list, check facility capacity,
               message the area's health workers (ALERT), record the escalation;
               then write the official a situational brief (LLM, fired only here)
               grounded in those tool results
  remember  -> persistent watchlist + alert history + escalation log
               (SurveillanceState, SQLite) that survive restarts
  coordinate-> optionally expose outbreak priors back to triage (feedback.py,
               off by default)

Every tool call is returned in Alert.actions and SurveillanceReport.tool_log.
The LLM is used only when an alert is actually raised (brief.py), so a quiet
region makes no model calls and messages no one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.agent1_extraction import LLMBackend
from app.surveillance import alert_policy, analytics, detect, recommend
from app.surveillance.brief import generate_brief
from app.surveillance.ingest import from_case_store
from app.surveillance.schemas import Alert, SurveillanceCase, SurveillanceReport, ToolCall
from app.surveillance.state import SurveillanceState
from app.surveillance.tools import SurveillanceTools, outbound_message, plan_for


class SurveillanceAgent:
    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        store=None,
        state: Optional[SurveillanceState] = None,
        tools: Optional[SurveillanceTools] = None,
    ):
        """`backend` enables the LLM situational brief (None -> deterministic
        fallback brief). `store` is a CaseStore to read the live stream from
        when run() is not given explicit cases. `state` is the persistent memory
        (in-memory SQLite by default; pass a file-backed one to survive
        restarts). `tools` is the bound tool set (default: demo facility
        network + offline MockProvider messaging)."""
        self.backend = backend
        self.store = store
        self.state = state or SurveillanceState(":memory:")
        self.tools = tools or SurveillanceTools(self.state)

    def _resolve_cases(self, cases) -> list[SurveillanceCase]:
        if cases is not None:
            return list(cases)
        if self.store is not None:
            return from_case_store(self.store)
        return []

    def _act(self, sig, level, action, cases, now) -> tuple[list[ToolCall], bool, str]:
        notify, decision = (False, "WATCH: no message, recorded for the official")
        if level.value == "ALERT":
            notify, decision = self.state.should_notify(
                sig.area, sig.disease, level, sig.weighted_score, sig.window_label
            )
        calls: list[ToolCall] = []
        capacity = None
        for tool in plan_for(level, notify):
            if tool == "query_cases":
                calls.append(self.tools.query_cases(cases, sig.area, sig.disease, now))
            elif tool == "check_facility_capacity":
                capacity = self.tools.check_facility_capacity(sig.area, sig.disease, sig.dominant_age_band, now)
                calls.append(capacity)
            elif tool == "notify_health_workers":
                msg = outbound_message(sig, level, action, capacity)
                calls.append(self.tools.notify_health_workers(sig.area, sig.disease, level, msg, now))
            elif tool == "log_escalation":
                detail = {
                    "signal": sig.model_dump(mode="json"),
                    "recommended_action": action,
                    "notify_decision": decision,
                    "tools": [{"tool": c.tool, "ok": c.ok, "summary": c.summary} for c in calls],
                }
                calls.append(self.tools.log_escalation(sig, level, detail, now))
        notified = any(c.tool == "notify_health_workers" and c.ok for c in calls)
        self.state.record_alert(sig.area, sig.disease, level, sig.weighted_score,
                                sig.window_label, now, notified)
        self.state.watch(sig.area, level, now)
        return calls, notified, decision

    def run(
        self,
        cases: Optional[list[SurveillanceCase]] = None,
        now: Optional[datetime] = None,
        weeks: int = 8,
        is_demonstration: bool = False,
    ) -> SurveillanceReport:
        cases = self._resolve_cases(cases)
        if now is None:
            now = max((c.recorded_at for c in cases), default=datetime.now(timezone.utc))

        village = analytics.village_table(cases, now)
        signals = detect.detect_outbreaks(cases, now=now)

        alerts: list[Alert] = []
        tool_log: list[ToolCall] = []
        for sig in signals:
            level = alert_policy.level_for(sig)
            if not alert_policy.is_surfaced(level):
                continue  # INFO: logged in `signals`, but the agent stays silent
            action = recommend.recommend_action(sig.disease)
            calls, notified, decision = self._act(sig, level, action, cases, now)
            tool_log.extend(calls)
            facts = {c.tool: c.summary for c in calls}
            brief = generate_brief(sig, level, action, self.backend, facts=facts)  # LLM fires only here
            alerts.append(Alert(
                level=level, area=sig.area, disease=sig.disease, signal=sig,
                recommended_action=action, brief=brief, created_at=now,
                actions=calls, notified=notified, notify_decision=decision,
            ))

        notes = [
            f"{len(cases)} cases analysed across {len({c.area for c in cases})} areas",
            f"{len(signals)} raw signal(s); {len(alerts)} surfaced to the official",
            f"{len(tool_log)} tool call(s); {sum(a.notified for a in alerts)} alert(s) sent to health workers",
        ]
        if not alerts:
            notes.append("No area is out of hand this period -- nothing surfaced.")

        return SurveillanceReport(
            generated_at=now, window="week", n_cases=len(cases),
            village_table=village, signals=signals, alerts=alerts,
            watchlist=self.state.watchlist(), notes=notes,
            is_demonstration=is_demonstration, tool_log=tool_log,
        )
