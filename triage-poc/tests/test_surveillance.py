"""Tests for Agent 2 (surveillance). Deterministic except one Ollama-gated live
brief test. Run: PYTHONPATH=. python -m pytest tests/test_surveillance.py -q
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent1_extraction import RegexBackend
from app.case_store import CaseStore
from app.disease_kb import DiseaseKB
from app.schemas import ClassificationLabel, ClassificationResult, ExtractedCase
from app.surveillance import alert_policy, recommend
from app.surveillance.agent import SurveillanceAgent
from app.surveillance.brief import generate_brief
from app.surveillance.demo_stream import generate_demo_stream
from app.surveillance.detect import detect_outbreaks
from app.surveillance.ingest import from_case_store
from app.surveillance.schemas import AlertLevel, OutbreakSignal

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


def _ollama_up() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200 and "llama3.2" in r.text
    except requests.RequestException:
        return False


def _signal(weighted: float) -> OutbreakSignal:
    return OutbreakSignal(
        area="X", disease="Gastroenteritis", window_label="w", observed=10,
        baseline_mean=1.0, baseline_std=1.0, score=weighted, severity=ClassificationLabel.SEVERE,
        severity_weight=1.0, weighted_score=weighted,
    )


# ---- demo stream + detection -------------------------------------------------

def test_demo_stream_is_deterministic():
    a = generate_demo_stream(weeks=8, now=NOW)
    b = generate_demo_stream(weeks=8, now=NOW)
    assert len(a) == len(b) and len(a) > 0


def test_detector_fires_on_injected_outbreak():
    sigs = detect_outbreaks(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    assert len(sigs) >= 1
    top = sigs[0]
    assert top.area == "Anchetty" and top.disease == "Gastroenteritis"
    assert top.observed >= 5 and top.dominant_age_band == "<5y"


def test_detector_silent_on_flat_stream():
    flat = generate_demo_stream(weeks=8, now=NOW, outbreak=False)
    assert detect_outbreaks(flat, now=NOW) == []


# ---- alert policy ------------------------------------------------------------

def test_alert_policy_bands():
    assert alert_policy.level_for(_signal(10.0)) == AlertLevel.ALERT
    assert alert_policy.level_for(_signal(4.0)) == AlertLevel.WATCH
    assert alert_policy.level_for(_signal(1.0)) == AlertLevel.INFO
    assert alert_policy.is_surfaced(AlertLevel.ALERT)
    assert alert_policy.is_surfaced(AlertLevel.WATCH)
    assert not alert_policy.is_surfaced(AlertLevel.INFO)  # INFO stays silent


# ---- recommendations ---------------------------------------------------------

def test_recommend_covers_all_41_diseases():
    for disease in DiseaseKB.load().diseases:
        action = recommend.recommend_action(disease)
        assert isinstance(action, str) and action.strip()


def test_recommend_waterborne_for_gastroenteritis():
    assert "water" in recommend.recommend_action("Gastroenteritis").lower()


# ---- brief (LLM tool, with fallback) -----------------------------------------

def test_brief_fallback_without_backend():
    sig = _signal(10.0)
    text = generate_brief(sig, AlertLevel.ALERT, recommend.recommend_action(sig.disease), None)
    assert "Anchetty" not in text  # signal area is X here
    assert "Gastroenteritis" in text and "ALERT" in text


def test_brief_fallback_on_unavailable_backend():
    # RegexBackend.generate_text raises BackendUnavailable -> deterministic fallback
    sig = _signal(10.0)
    text = generate_brief(sig, AlertLevel.ALERT, "do X", RegexBackend())
    assert "do X" in text


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")
def test_brief_live_llm():
    from app.agent1_extraction import OllamaBackend
    sig = _signal(10.0)
    text = generate_brief(sig, AlertLevel.ALERT, recommend.recommend_action(sig.disease), OllamaBackend())
    assert isinstance(text, str) and len(text.strip()) > 0


# ---- agent end-to-end --------------------------------------------------------

def test_agent_surfaces_the_outbreak():
    rep = SurveillanceAgent().run(generate_demo_stream(weeks=8, now=NOW), now=NOW, is_demonstration=True)
    assert len(rep.alerts) == 1
    assert rep.alerts[0].area == "Anchetty"
    assert rep.alerts[0].level == AlertLevel.ALERT
    assert "Anchetty" in rep.watchlist
    assert rep.is_demonstration


def test_agent_silent_on_flat_stream():
    rep = SurveillanceAgent().run(generate_demo_stream(weeks=8, now=NOW, outbreak=False), now=NOW)
    assert rep.alerts == [] and rep.watchlist == []


def test_agent_watchlist_persists_across_runs():
    agent = SurveillanceAgent()
    agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    rep2 = agent.run(generate_demo_stream(weeks=8, now=NOW, outbreak=False), now=NOW)
    assert "Anchetty" in rep2.watchlist  # remembered even though run 2 was flat


# ---- ingest from the live case store ----------------------------------------

def test_ingest_from_case_store():
    store = CaseStore(":memory:")
    case = ExtractedCase(raw_symptom_text="x", location="Thally", age_months=40,
                         symptom_tokens=["high_fever"])
    store.record(case, ClassificationResult(label=ClassificationLabel.SEVERE,
                                            probable_disease="Typhoid"))
    cases = from_case_store(store)
    assert len(cases) == 1
    assert cases[0].area == "Thally" and cases[0].disease == "Typhoid"
    assert cases[0].severity == ClassificationLabel.SEVERE and cases[0].age_months == 40


# ---- feedback hook (off by default) -----------------------------------------

def test_feedback_off_by_default():
    from app.surveillance import feedback
    rep = SurveillanceAgent().run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    assert feedback.SURVEILLANCE_FEEDBACK_ENABLED is False
    assert feedback.priors_for_area(rep, "Anchetty") == {}


def test_feedback_priors_when_enabled(monkeypatch):
    from app.surveillance import feedback
    monkeypatch.setattr(feedback, "SURVEILLANCE_FEEDBACK_ENABLED", True)
    rep = SurveillanceAgent().run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    priors = feedback.priors_for_area(rep, "Anchetty")
    assert priors.get("Gastroenteritis", 0) > 1.0  # active alert raises the prior
