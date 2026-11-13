"""Agent 2 hardening: it ACTS through tools, remembers across restarts, and
does not repeat itself.

C2  On an ALERT the agent runs query_cases -> check_facility_capacity ->
    notify_health_workers -> log_escalation, each returned as an audited
    ToolCall; on a quiet stream it calls no tool and messages no one.
C3  Watchlist, alert history and escalations persist in SQLite across agent
    instances; the same alert is not re-sent unless it is new, higher, worse,
    or still active in a new week.
"""
import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.integrations.messaging import MockProvider
from app.surveillance.agent import SurveillanceAgent
from app.surveillance.demo_stream import generate_demo_stream
from app.surveillance.schemas import AlertLevel
from app.surveillance.state import SurveillanceState
from app.surveillance.tools import SurveillanceTools, plan_for, services_for_outbreak

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


def _agent(state=None, messenger=None):
    state = state or SurveillanceState(":memory:")
    messenger = messenger or MockProvider()
    return SurveillanceAgent(state=state, tools=SurveillanceTools(state, messenger=messenger)), messenger


def test_alert_runs_the_full_tool_plan_in_order():
    agent, messenger = _agent()
    rep = agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    alert = rep.alerts[0]
    assert [c.tool for c in alert.actions] == [
        "query_cases", "check_facility_capacity", "notify_health_workers", "log_escalation"
    ]
    assert all(c.ok for c in alert.actions)
    assert alert.notified
    assert rep.tool_log == alert.actions


def test_query_cases_tool_returns_the_line_list():
    agent, _ = _agent()
    q = agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW).alerts[0].actions[0]
    assert q.output["this_week"] == q.output["weekly_trend_oldest_to_current"][-1] >= 5
    assert set(q.output["age_bands"]) == {"<5y"}  # the injected outbreak is under-5s


def test_capacity_tool_checks_services_the_outbreak_needs():
    assert services_for_outbreak("Gastroenteritis", "<5y") == ["inpatient", "paediatrics"]
    assert services_for_outbreak("Dengue", None) == ["inpatient", "blood_bank"]
    agent, _ = _agent()
    cap = agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW).alerts[0].actions[1]
    assert cap.output["needed_services"] == ["inpatient", "paediatrics"]
    assert cap.output["facilities"] and cap.output["capacity_gap"] is False
    assert cap.output["facilities"][0]["offers_all_needed"]


def test_notification_goes_to_area_health_workers_and_district_officer():
    agent, messenger = _agent()
    notify = agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW).alerts[0].actions[2]
    roles = {r["role"] for r in notify.output["recipients"]}
    assert roles == {"asha", "anm", "district_officer"}
    assert len(messenger.sent) == len(notify.output["recipients"]) == 4  # 2 ASHA + ANM + officer (Anchetty)
    text = messenger.sent[0][1]
    assert "Gastroenteritis" in text and "Anchetty" in text and len(text) <= 480


def test_quiet_stream_calls_no_tool_and_messages_no_one():
    agent, messenger = _agent()
    rep = agent.run(generate_demo_stream(weeks=8, now=NOW, outbreak=False), now=NOW)
    assert rep.alerts == [] and rep.tool_log == [] and messenger.sent == []


def test_watch_level_gathers_facts_but_sends_nothing():
    assert plan_for(AlertLevel.WATCH, notify=True) == ["query_cases", "check_facility_capacity", "log_escalation"]
    assert plan_for(AlertLevel.INFO, notify=True) == []
    assert "notify_health_workers" not in plan_for(AlertLevel.ALERT, notify=False)


def test_same_alert_is_not_resent_on_a_second_run():
    agent, messenger = _agent()
    stream = generate_demo_stream(weeks=8, now=NOW)
    agent.run(stream, now=NOW)
    first = len(messenger.sent)
    rep2 = agent.run(stream, now=NOW)
    assert len(messenger.sent) == first  # no repeat message
    assert rep2.alerts and not rep2.alerts[0].notified
    assert "no repeat message" in rep2.alerts[0].notify_decision
    assert [c.tool for c in rep2.alerts[0].actions] == ["query_cases", "check_facility_capacity", "log_escalation"]


def test_renotify_rules():
    st = SurveillanceState(":memory:")
    assert st.should_notify("A", "D", AlertLevel.WATCH, 4.0, "w1")[0]
    st.record_alert("A", "D", AlertLevel.WATCH, 4.0, "w1", NOW, notified=True)
    assert not st.should_notify("A", "D", AlertLevel.WATCH, 4.2, "w1")[0]
    assert st.should_notify("A", "D", AlertLevel.ALERT, 6.5, "w1")[0]   # level rose
    assert st.should_notify("A", "D", AlertLevel.WATCH, 5.1, "w1")[0]   # >=25% worse
    assert st.should_notify("A", "D", AlertLevel.WATCH, 4.0, "w2")[0]   # still active next week


def test_state_persists_across_agent_instances(tmp_path):
    db = tmp_path / "surv.sqlite"
    agent1, m1 = _agent(SurveillanceState(db))
    agent1.run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    assert m1.sent
    # "restart": a new agent on the same file remembers the watchlist and the
    # alert it already sent, so the flat stream keeps Anchetty on watch and the
    # repeated outbreak is not re-sent
    agent2, m2 = _agent(SurveillanceState(db))
    flat = agent2.run(generate_demo_stream(weeks=8, now=NOW, outbreak=False), now=NOW)
    assert "Anchetty" in flat.watchlist
    agent2.run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    assert m2.sent == []
    st = SurveillanceState(db)
    assert len(st.alert_history()) == 2
    assert len(st.escalations()) == 2
    assert st.escalations()[0]["detail"]["tools"]


def test_new_week_still_active_is_resent():
    agent, messenger = _agent()
    agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW)
    first = len(messenger.sent)
    later = NOW + timedelta(days=7)
    agent.run(generate_demo_stream(weeks=8, now=later), now=later)
    assert len(messenger.sent) == 2 * first


def test_brief_is_grounded_in_tool_results():
    agent, _ = _agent()
    brief = agent.run(generate_demo_stream(weeks=8, now=NOW), now=NOW).alerts[0].brief
    assert "Actions taken:" in brief and "messaged" in brief


def test_tool_seams_do_not_reach_the_triage_decision_core():
    app = Path(__file__).resolve().parent.parent / "app"
    for mod in ("routing/router.py", "routing/facility_db.py", "integrations/messaging.py"):
        tree = ast.parse((app / mod).read_text())
        names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "app.rules_engine" not in names, mod
