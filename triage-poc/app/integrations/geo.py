"""
RoutingProvider: real road distance/ETA/route geometry for Agent 3 (plan
diagram 2, step 5 -- OpenRouteService free API), plus Google Maps
navigation links in the dispatch output.

RoutingProvider (ABC) and MockRoutingProvider are defined in
app.routing.router, not duplicated here -- app.routing.router.route()
needs that interface directly and Phase 6 predates this integrations
package. Re-exported below so `app.integrations.geo` is the single place
this plan's "Agent 3 -- Google Maps + OpenRouteService" integration point
is documented and imported from, per the plan's own module layout.

ORSProvider adds the real implementation: OpenRouteService's free
Directions API (https://api.openrouteservice.org/v2/directions/{profile}),
plain HTTP via `requests` with an API key header -- no SDK, consistent
with requirements.txt's existing note that ORS "uses requests ... no
extra SDK" and this package's messaging.py/translation.py adapters.
"""
from __future__ import annotations

import os
from typing import Optional

from app.routing.router import MockRoutingProvider, RoutingProvider  # re-exported
from app.routing.schemas import RouteInfo

__all__ = ["RoutingProvider", "MockRoutingProvider", "ORSProvider"]


class ORSProvider(RoutingProvider):
    """Real implementation: calls OpenRouteService's driving-car directions
    endpoint for the real road distance and duration between two points,
    and builds the same Google-Maps-link format
    MockRoutingProvider uses (a plain user-facing maps URL, no API key
    needed for that part).

    Raises ValueError immediately at construction if ORS_API_KEY is
    missing -- fail fast, not silently at first call. On any request
    failure at call time, falls back to MockRoutingProvider's haversine
    estimate rather than raising -- a routing agent must not fail to
    produce ANY distance/ETA just because the live API is unreachable;
    the fallback's `provider` field is left as "mock_haversine" so the
    degrade is visible to a caller/audit trail, not silently disguised as
    a real ORS result.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.environ.get("ORS_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("ORSProvider requires ORS_API_KEY")
        self._fallback = MockRoutingProvider()

    def get_route(self, origin: tuple[float, float], destination: tuple[float, float]) -> RouteInfo:
        import requests

        try:
            resp = requests.post(
                "https://api.openrouteservice.org/v2/directions/driving-car",
                headers={"Authorization": self.api_key, "Content-Type": "application/json"},
                json={
                    # ORS expects [lon, lat] coordinate order, opposite of
                    # this codebase's (lat, lon) convention elsewhere.
                    "coordinates": [
                        [origin[1], origin[0]],
                        [destination[1], destination[0]],
                    ]
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            return self._fallback.get_route(origin, destination)

        body = resp.json()
        try:
            summary = body["routes"][0]["summary"]
            distance_km = summary["distance"] / 1000.0
            eta_minutes = summary["duration"] / 60.0
        except (KeyError, IndexError):
            return self._fallback.get_route(origin, destination)

        maps_link = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}"
        )
        return RouteInfo(
            road_distance_km=round(distance_km, 2),
            eta_minutes=round(eta_minutes, 1),
            maps_link=maps_link,
            provider="openrouteservice",
        )
