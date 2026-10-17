"""Descriptive analytics over the case stream -- the 'data analysis' layer.

Pure stdlib aggregation (no pandas): counts by area/disease/week/age-band, a
per-village risk table, and weekly trend series for charting. `window_index`
buckets a case into whole weeks back from `now` (0 = current week), which is the
unit both the risk table and the detector (detect.py) reason in.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Iterable, Optional

from app.schemas import SEVERITY_RANK, ClassificationLabel
from app.surveillance.schemas import SurveillanceCase, VillageStat

_SEVERE_RANK = SEVERITY_RANK[ClassificationLabel.SEVERE]  # EMERGENCY/SEVERE are <= this


def window_index(case: SurveillanceCase, now: datetime) -> int:
    """Whole weeks between the case and `now` (0 = current week, 1 = last week).
    Future-dated cases clamp to 0."""
    days = (now - case.recorded_at).days
    return max(0, days // 7)


def is_severe(case: SurveillanceCase) -> bool:
    return SEVERITY_RANK.get(case.severity, 99) <= _SEVERE_RANK


def counts_by_week(cases: Iterable[SurveillanceCase], now: datetime) -> dict:
    """{(area, disease): {week_index: count}} -- the raw grid the detector reads."""
    grid: dict = defaultdict(lambda: defaultdict(int))
    for c in cases:
        if not c.disease:
            continue
        grid[(c.area, c.disease)][window_index(c, now)] += 1
    return grid


def weekly_trend(cases: Iterable[SurveillanceCase], now: datetime, weeks: int = 8) -> dict:
    """{area: [count_week_oldest ... count_week_current]} for charting."""
    per_area: dict = defaultdict(lambda: [0] * weeks)
    for c in cases:
        wi = window_index(c, now)
        if wi < weeks:
            per_area[c.area][weeks - 1 - wi] += 1
    return dict(per_area)


def _dominant_age_band(cases: list[SurveillanceCase]) -> Optional[str]:
    bands = [c.age_band for c in cases if c.age_band != "unknown"]
    return Counter(bands).most_common(1)[0][0] if bands else None


def village_table(cases: list[SurveillanceCase], now: datetime) -> list[VillageStat]:
    """One summary row per area, sorted by risk_score (highest first)."""
    by_area: dict = defaultdict(list)
    for c in cases:
        by_area[c.area].append(c)

    stats: list[VillageStat] = []
    for area, group in by_area.items():
        total = len(group)
        severe = sum(1 for c in group if is_severe(c))
        cur = sum(1 for c in group if window_index(c, now) == 0)
        prev = sum(1 for c in group if window_index(c, now) == 1)
        growth = (cur / prev - 1.0) if prev else (float(cur) if cur else 0.0)
        top = Counter(c.disease for c in group if c.disease).most_common(1)
        severe_share = round(severe / total, 4) if total else 0.0
        # Risk = weighted toward serious cases and recent growth, so a village
        # with few but severe (or sharply rising) cases ranks above a quiet one.
        risk = round(severe * 2.0 + cur * 1.0 + max(growth, 0.0) * 1.5, 3)
        stats.append(VillageStat(
            area=area, total_cases=total, severe_or_emergency=severe,
            severe_share=severe_share, top_disease=top[0][0] if top else None,
            cases_current_window=cur, cases_prev_window=prev,
            growth=round(growth, 3), risk_score=risk,
        ))
    stats.sort(key=lambda s: s.risk_score, reverse=True)
    return stats
