"""SurveillanceAgent -- the orchestrator that ties the layers into an agent.

One run() is a full agent cycle:
  perceive  -> ingest the case stream
  reason    -> village risk table + baseline-deviation detection
  decide    -> grade each signal; keep only those worth a human's attention
  act       -> for each surfaced signal, recommend an action and (autonomously,
               via the LLM) write the official a situational brief
  remember  -> update a persistent watchlist across runs (standing goal state)
  coordinate-> optionally expose outbreak priors back to triage (feedback.py,
               off by default)

The LLM is used only when an alert is actually raised (brief.py), so a quiet
region makes no model calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.agent1_extraction import LLMBackend
from app.surveillance import alert_policy, analytics, detect, recommend
from app.surveillance.brief import generate_brief
from app.surveillance.ingest import from_case_store
from app.surveillance.schemas import Alert, SurveillanceCase, SurveillanceReport


class SurveillanceAgent:
    def __init__(self, backend: Optional[LLMBackend] = None, store=None):
        """`backend` enables the LLM situational brief (None -> deterministic
        fallback brief). `store` is a CaseStore to read the live stream from
        when run() is not given explicit cases."""
        self.backend = backend
        self.store = store
        self._watchlist: set[str] = set()  # persistent standing-goal state across runs

    def _resolve_cases(self, cases) -> list[SurveillanceCase]:
        if cases is not None:
            return list(cases)
        if self.store is not None:
            return from_case_store(self.store)
        return []

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
        for sig in signals:
            level = alert_policy.level_for(sig)
            if not alert_policy.is_surfaced(level):
                continue  # INFO: logged in `signals`, but the agent stays silent
            action = recommend.recommend_action(sig.disease)
            brief = generate_brief(sig, level, action, self.backend)  # LLM fires only here
            alerts.append(Alert(
                level=level, area=sig.area, disease=sig.disease, signal=sig,
                recommended_action=action, brief=brief, created_at=now,
            ))

        self._watchlist.update(a.area for a in alerts)

        notes = [
            f"{len(cases)} cases analysed across {len({c.area for c in cases})} areas",
            f"{len(signals)} raw signal(s); {len(alerts)} surfaced to the official",
        ]
        if not alerts:
            notes.append("No area is out of hand this period -- nothing surfaced.")

        return SurveillanceReport(
            generated_at=now, window="week", n_cases=len(cases),
            village_table=village, signals=signals, alerts=alerts,
            watchlist=sorted(self._watchlist), notes=notes,
            is_demonstration=is_demonstration,
        )
