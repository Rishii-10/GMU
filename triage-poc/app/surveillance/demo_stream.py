"""Synthetic demonstration case stream for Agent 2.

A live POC has no case volume, so this generates a plausible multi-village,
multi-week stream to drive the surveillance view -- an endemic baseline of mixed
mild/moderate complaints plus ONE injected outbreak (a diarrhoeal spike in one
village among under-5s in the most recent weeks) so the detector visibly fires.

This is DEMONSTRATION DATA, labelled as such everywhere it surfaces (the UI and
SurveillanceReport.is_demonstration). It is a sandbox to show the agent behaves
as designed, not real surveillance -- consistent with docs/PAPER_FRAMING.md.
Deterministic (seeded).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.disease_classifier import severity_for_disease
from app.routing.demo_facilities import VILLAGES
from app.surveillance.schemas import SurveillanceCase

# Endemic background: routine complaints that should NOT trigger an alert.
_ENDEMIC = [
    "Common Cold", "Fungal infection", "Allergy", "Migraine",
    "GERD", "Bronchial Asthma", "Urinary tract infection",
]

_OUTBREAK_AREA = "Anchetty"
_OUTBREAK_DISEASE = "Gastroenteritis"  # SEVERE in data/disease_severity.csv
_OUTBREAK_WEEKS = 1                     # spike concentrated in the current week
_OUTBREAK_CASES_PER_WEEK = (9, 13)     # vs an endemic ~0-1 of this disease/village/week


def _case(area: str, disease: str, when: datetime, age_months: int) -> SurveillanceCase:
    return SurveillanceCase(
        area=area,
        disease=disease,
        symptom_tokens=[],
        severity=severity_for_disease(disease),
        age_months=age_months,
        recorded_at=when,
        language="en",
    )


def generate_demo_stream(
    weeks: int = 8,
    seed: int = 42,
    now: Optional[datetime] = None,
    outbreak: bool = True,
) -> list[SurveillanceCase]:
    """Build the demo stream ending at `now` (UTC default). `outbreak=False`
    yields a flat endemic stream (used to prove the detector stays silent)."""
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    cases: list[SurveillanceCase] = []

    def _when_in_week(w: int) -> datetime:
        # Anchor each week at `now - 7*(weeks-1-w)` days, then shift BACKWARD by
        # 0-6 days so every case lands strictly in that week's bucket (never
        # past `now`, never bleeding into a newer window).
        anchor = now - timedelta(days=7 * (weeks - 1 - w))
        return anchor - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 23))

    for w in range(weeks):
        # week 0 = oldest, week (weeks-1) = current
        for area in VILLAGES:
            # endemic baseline: a few routine cases per village per week
            for _ in range(rng.randint(2, 4)):
                disease = rng.choice(_ENDEMIC)
                age = rng.randint(1, 900)
                cases.append(_case(area, disease, _when_in_week(w), age))

        # injected outbreak in the most recent week(s), one village, under-5s
        if outbreak and w >= weeks - _OUTBREAK_WEEKS:
            n = rng.randint(*_OUTBREAK_CASES_PER_WEEK)
            for _ in range(n):
                age = rng.randint(1, 59)  # under 5 years
                cases.append(_case(_OUTBREAK_AREA, _OUTBREAK_DISEASE, _when_in_week(w), age))

    cases.sort(key=lambda c: c.recorded_at)
    return cases
