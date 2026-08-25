"""
External integration adapters -- pluggable interfaces with an offline
mock/identity default for every one, so the full test suite runs with no
network access and no API keys. Real providers drop in via environment
variables with no change to core pipeline logic.

  - messaging.py -- MessagingProvider (Twilio for Agent 1's SMS transport)
  - translation.py -- Translator (Google Translate for Agent 1's i18n)
  - geo.py -- RoutingProvider (OpenRouteService + Google Maps links for
    Agent 3's routing; re-exports app.routing.router's interface and adds
    the real ORS-backed implementation)

Every real (non-mock) provider imports its HTTP dependency (`requests`,
already a hard dependency of this project) lazily and only fails at
construction time if required credentials/env vars are missing -- never at
module import time. This mirrors app.disambiguation.FAISSDisambiguator's
lazy-import pattern, just for credentials instead of a heavy ML library.
"""
