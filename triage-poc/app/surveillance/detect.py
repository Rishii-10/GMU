"""Outbreak detection -- deliberately light statistics, not a heavy model.

Primary detector: baseline deviation. For each (area, disease), compare the
current week's case count to the mean+spread of the prior few weeks for that
same area+disease. A standardized score (z-like) over a minimum absolute count
is the signal; a severity weight scales it so an EMERGENCY/SEVERE disease trips
sooner than a mild one. No LLM, fully deterministic.

An optional spatial-temporal DBSCAN view (cluster_spatial) is provided for
completeness; it is secondary and guarded so a missing scikit-learn or too few
points degrades to an empty list rather than an error.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import pstdev
from typing import Optional

from app.schemas import ClassificationLabel
from app.surveillance.analytics import counts_by_week, window_index
from app.surveillance.schemas import OutbreakSignal, SurveillanceCase

# How much a disease's intrinsic severity amplifies its deviation score, so the
# alert policy reacts to a severe-disease cluster at a lower raw count.
_SEVERITY_WEIGHT: dict[ClassificationLabel, float] = {
    ClassificationLabel.EMERGENCY: 2.0,
    ClassificationLabel.SEVERE: 1.5,
    ClassificationLabel.MODERATE: 1.0,
    ClassificationLabel.MILD: 0.6,
    ClassificationLabel.INCOMPLETE_ASSESSMENT: 0.8,
}

DEFAULT_BASELINE_WINDOWS = 4
DEFAULT_MIN_COUNT = 5        # never raise a signal on a handful of cases
DEFAULT_MIN_SCORE = 2.0      # standardized deviations before a signal exists


def _severity_of(cases: list[SurveillanceCase], disease: str) -> ClassificationLabel:
    for c in cases:
        if c.disease == disease:
            return c.severity
    return ClassificationLabel.MODERATE


def detect_outbreaks(
    cases: list[SurveillanceCase],
    now: Optional[datetime] = None,
    baseline_windows: int = DEFAULT_BASELINE_WINDOWS,
    min_count: int = DEFAULT_MIN_COUNT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[OutbreakSignal]:
    """Return baseline-deviation signals, strongest first. A signal requires
    BOTH a current-week count >= min_count AND a standardized deviation
    >= min_score, so routine fluctuation and tiny numbers are ignored."""
    now = now or datetime.now(tz=cases[0].recorded_at.tzinfo) if cases else datetime.now()
    grid = counts_by_week(cases, now)
    label = f"week_of_{now.date().isoformat()}"
    signals: list[OutbreakSignal] = []

    for (area, disease), by_week in grid.items():
        observed = by_week.get(0, 0)
        if observed < min_count:
            continue
        baseline = [by_week.get(i, 0) for i in range(1, baseline_windows + 1)]
        mean = sum(baseline) / len(baseline) if baseline else 0.0
        std = pstdev(baseline) if len(baseline) > 1 else 0.0
        # Guard a zero/near-zero baseline spread: a jump from ~0 to many is a
        # real signal, so floor the denominator at 1 rather than dividing by 0.
        score = (observed - mean) / max(std, 1.0)
        if score < min_score:
            continue
        severity = _severity_of(cases, disease)
        weight = _SEVERITY_WEIGHT.get(severity, 1.0)
        group = [c for c in cases if c.area == area and c.disease == disease
                 and window_index(c, now) == 0]
        bands = [c.age_band for c in group if c.age_band != "unknown"]
        dominant = Counter(bands).most_common(1)[0][0] if bands else None
        signals.append(OutbreakSignal(
            area=area, disease=disease, window_label=label,
            observed=observed, baseline_mean=round(mean, 3), baseline_std=round(std, 3),
            score=round(score, 3), severity=severity, severity_weight=weight,
            weighted_score=round(score * weight, 3), dominant_age_band=dominant,
        ))

    signals.sort(key=lambda s: s.weighted_score, reverse=True)
    return signals


def cluster_spatial(cases, now=None, eps=0.15, min_samples=5) -> list[dict]:
    """OPTIONAL secondary view: DBSCAN over (lat, lon, week) of same-disease
    cases. Guarded: returns [] if scikit-learn is unavailable, there are too few
    points, or no village coordinates are supplied. Not used by the alert policy
    -- it is an illustrative spatial lens only."""
    try:
        from sklearn.cluster import DBSCAN  # noqa: WPS433 (optional dependency)
    except ImportError:
        return []
    try:
        from app.routing.demo_facilities import VILLAGES
    except Exception:
        return []

    pts, meta = [], []
    for c in cases:
        coord = VILLAGES.get(c.area)
        if not coord or not c.disease:
            continue
        wi = window_index(c, now or datetime.now(tz=c.recorded_at.tzinfo))
        pts.append([coord[0], coord[1], wi * eps])
        meta.append(c)
    if len(pts) < min_samples:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    clusters: dict = {}
    for lbl, c in zip(labels, meta):
        if lbl == -1:
            continue
        clusters.setdefault(int(lbl), []).append(c)
    return [
        {"cluster": k, "n": len(v), "areas": sorted({c.area for c in v}),
         "diseases": sorted({c.disease for c in v})}
        for k, v in clusters.items()
    ]
