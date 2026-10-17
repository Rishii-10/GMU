"""Pydantic models for Agent 2 (surveillance). Kept in their own module, like
app/routing/schemas.py: surveillance is a separate agent with its own concerns
(area/time aggregates, outbreak signals, official-facing alerts) that Agent 1
and the Rules Engine never need to see.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas import ClassificationLabel


class AlertLevel(str, Enum):
    """How loud a signal is. INFO is logged but NOT surfaced to the official --
    the agent stays silent on routine fluctuation; WATCH and ALERT are the
    "something is out of hand" levels that reach a human."""

    INFO = "INFO"
    WATCH = "WATCH"
    ALERT = "ALERT"


def age_band(age_months: Optional[int]) -> str:
    """Coarse band for surveillance. None -> 'unknown' (honest absence, never
    bucketed as 0)."""
    if age_months is None:
        return "unknown"
    if age_months < 60:
        return "<5y"
    if age_months < 180:
        return "5-14y"
    if age_months < 780:
        return "15-64y"
    return "65+"


class SurveillanceCase(BaseModel):
    """One case as the agent reasons over it -- a projection of a logged case
    (app.case_store) onto the fields surveillance needs. Deliberately NOT the
    raw text (privacy: the case store never holds it either)."""

    area: str
    disease: Optional[str] = None  # probable_disease; None when the triage engine abstained
    symptom_tokens: list[str] = Field(default_factory=list)
    severity: ClassificationLabel
    age_months: Optional[int] = None
    recorded_at: datetime
    language: Optional[str] = None

    @property
    def age_band(self) -> str:
        return age_band(self.age_months)


class VillageStat(BaseModel):
    """Per-area summary for the official's risk table."""

    area: str
    total_cases: int
    severe_or_emergency: int
    severe_share: float
    top_disease: Optional[str] = None
    cases_current_window: int = 0
    cases_prev_window: int = 0
    growth: float = 0.0  # current/prev ratio - 1 (0.0 when no prior cases)
    risk_score: float = 0.0


class OutbreakSignal(BaseModel):
    """A detected area+disease deviation from baseline. `score` is a
    standardized deviation (z-like); `weighted_score` multiplies it by a
    severity weight so an emergency/severe disease trips the policy sooner."""

    area: str
    disease: str
    window_label: str
    observed: int
    baseline_mean: float
    baseline_std: float
    score: float
    severity: ClassificationLabel
    severity_weight: float
    weighted_score: float
    dominant_age_band: Optional[str] = None
    detector: str = "baseline_deviation"


class Alert(BaseModel):
    """What actually reaches the official: a signal the policy judged worth
    surfacing, with an autonomously generated brief + recommended action."""

    level: AlertLevel
    area: str
    disease: str
    signal: OutbreakSignal
    recommended_action: str
    brief: str
    created_at: datetime


class SurveillanceReport(BaseModel):
    """Full output of one agent run (app.surveillance.agent.SurveillanceAgent.run)."""

    generated_at: datetime
    window: str
    n_cases: int
    village_table: list[VillageStat] = Field(default_factory=list)
    signals: list[OutbreakSignal] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    is_demonstration: bool = False
