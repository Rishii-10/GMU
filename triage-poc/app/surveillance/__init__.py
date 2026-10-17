"""Agent 2 -- syndromic surveillance.

A monitoring agent for a public-health / government official. It consumes the
SAME case stream the triage pipeline already records (app.case_store.CaseStore),
so it adds a capability to the framework WITHOUT the triage core having to know
it exists -- the worked example of the framework's extensibility (see
docs/PAPER_FRAMING.md section 4).

What makes it an *agent*, not a dashboard:
  * it PERCEIVES the case stream (ingest.py),
  * it REASONS over it with light statistics (analytics.py, detect.py),
  * it DECIDES when a signal is worth a human's attention (alert_policy.py) --
    staying silent otherwise,
  * it ACTS autonomously by briefing the official in natural language and
    recommending a public-health action, using the LLM only when an alert
    actually fires (brief.py, recommend.py),
  * it holds STATE / a standing goal across runs (a watchlist, agent.py),
  * and it can COORDINATE back with the triage classifier via an optional,
    off-by-default prior-boost hook (feedback.py).

Detection is deliberately simple statistics, not a heavy model: consistent with
the project stance that the models are standard and the *framework/coordination*
is the contribution.
"""
