"""The 'alert only when things get out of hand' policy -- the judgement that
makes Agent 2 an agent rather than a dashboard.

Every detected signal is graded; only WATCH and ALERT reach the official. INFO
signals are returned for the audit log but the agent stays silent on them, so a
human's attention is spent only on genuine deviations.
"""
from __future__ import annotations

from app.surveillance.schemas import AlertLevel, OutbreakSignal

# Thresholds on the severity-weighted deviation score. Named constants so they
# can be tuned/swept; defaults chosen so a clear severe-disease cluster (weighted
# score well into double digits) is an ALERT, a milder or smaller deviation is a
# WATCH, and routine noise stays INFO (silent).
ALERT_SCORE = 6.0
WATCH_SCORE = 3.0


def level_for(signal: OutbreakSignal) -> AlertLevel:
    if signal.weighted_score >= ALERT_SCORE:
        return AlertLevel.ALERT
    if signal.weighted_score >= WATCH_SCORE:
        return AlertLevel.WATCH
    return AlertLevel.INFO


def is_surfaced(level: AlertLevel) -> bool:
    """WATCH and ALERT reach the official; INFO is logged silently."""
    return level in (AlertLevel.WATCH, AlertLevel.ALERT)
