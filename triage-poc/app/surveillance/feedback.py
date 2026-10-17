"""OPTIONAL closed-loop coordination: Agent 2 -> Agent 1.

When an outbreak is active in an area, new callers from that area are more
likely to have that disease, so the triage classifier could raise its prior for
it. That is the bidirectional-coordination demonstration the paper's USP is
about -- the surveillance agent does not just observe, it can improve triage.

SHIPPED OFF BY DEFAULT. `SURVEILLANCE_FEEDBACK_ENABLED` gates whether the triage
path consults these priors, so with the flag off the triage core and its tests
are completely untouched. Turning it on is a single guarded multiply at one call
site in the classifier; this module only computes the priors, it does not reach
into triage itself.
"""
from __future__ import annotations

from app.surveillance.schemas import AlertLevel, SurveillanceReport

# Master switch. Off by default: triage behaves exactly as before unless a
# deployment explicitly opts in.
SURVEILLANCE_FEEDBACK_ENABLED = False

# How much an active ALERT / WATCH in an area multiplies the prior for its
# disease when scoring a new case from that area.
_BOOST = {AlertLevel.ALERT: 1.5, AlertLevel.WATCH: 1.2}


def priors_for_area(report: SurveillanceReport, area: str) -> dict[str, float]:
    """{disease: boost_factor} for active alerts in `area`. Empty dict when the
    flag is off or the area has no active alert -- so a caller can multiply
    unconditionally and get a no-op when disabled."""
    if not SURVEILLANCE_FEEDBACK_ENABLED:
        return {}
    out: dict[str, float] = {}
    for a in report.alerts:
        if a.area == area:
            out[a.disease] = max(out.get(a.disease, 1.0), _BOOST.get(a.level, 1.0))
    return out
