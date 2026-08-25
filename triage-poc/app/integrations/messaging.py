"""
MessagingProvider: outbound SMS transport for Agent 1 (plan diagram 1,
"Patient input layer" -- Twilio for messaging).

SCOPE -- decision, stated explicitly
------------------------------------------------
This module provides a real, usable send_message() capability (TwilioProvider,
via Twilio's plain REST API over `requests` with HTTP Basic Auth -- no
`twilio` SDK dependency, consistent with this project's existing bias
against adding an SDK where a direct HTTP call suffices; see requirements.txt's
note on the Groq backend and OpenRouteService).

It does NOT invent a working inbound-reply/session-store layer.
app.agent1_extraction.extract_case_with_followup()'s `answer_provider`
callback already models "ask a question, synchronously get an answer" --
a real SMS-backed answer_provider would compose send_message() here with a
wait-for-the-next-inbound-message step, but that session-store/webhook
layer is explicitly flagged elsewhere in this repo (AGENT1_README.md,
"Milestone 1/6 infrastructure, not yet built") as not existing yet.
Pretending it exists by faking a synchronous wait would be dishonest about
what's actually implemented, so this module stops at the real, working
half: sending a message out. See the module-level example at the bottom
for exactly how a caller wires this into extract_case_with_followup() once
that session-store layer exists.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class MessagingProvider(ABC):
    """Single-method interface so a real transport can be swapped in
    without touching callers -- same pattern as app.disambiguation.
    Disambiguator and app.routing.router.RoutingProvider."""

    @abstractmethod
    def send_message(self, to: str, text: str) -> bool:
        """Sends `text` to `to` (a phone number in E.164 format, e.g.
        "+919876543210"). Returns True on confirmed send, False on a
        provider-reported failure -- never raises for an ordinary
        delivery failure (a triage system must not crash because one SMS
        didn't send); only construction-time misconfiguration (missing
        credentials) should raise."""


class MockProvider(MessagingProvider):
    """Offline default: records every send in `self.sent` instead of
    making a network call. Always returns True (send "succeeds")."""

    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def send_message(self, to: str, text: str) -> bool:
        self.sent.append((to, text))
        return True


class TwilioProvider(MessagingProvider):
    """Real implementation: Twilio's REST API
    (POST /2010-04-01/Accounts/{AccountSid}/Messages.json), plain HTTP
    Basic Auth, via `requests` -- no `twilio` SDK dependency.

    Reads credentials from environment variables at construction time
    (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER) unless
    passed explicitly. Raises ValueError immediately if any are missing --
    fail fast at construction, not silently at first send.
    """

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        timeout: int = 10,
    ):
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.environ.get("TWILIO_FROM_NUMBER")
        self.timeout = timeout
        missing = [
            name
            for name, v in (
                ("TWILIO_ACCOUNT_SID", self.account_sid),
                ("TWILIO_AUTH_TOKEN", self.auth_token),
                ("TWILIO_FROM_NUMBER", self.from_number),
            )
            if not v
        ]
        if missing:
            raise ValueError(f"TwilioProvider missing required credentials: {missing}")

    def send_message(self, to: str, text: str) -> bool:
        import requests

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        try:
            resp = requests.post(
                url,
                data={"From": self.from_number, "To": to, "Body": text},
                auth=(self.account_sid, self.auth_token),
                timeout=self.timeout,
            )
        except requests.RequestException:
            return False
        return resp.status_code in (200, 201)


# --- Intended integration shape (documentation, not executed here) ----------
#
# Once a real session-store/webhook layer exists (Milestone 1/6), a caller
# would wire Twilio into the follow-up loop like this:
#
#     provider = TwilioProvider()
#     def answer_provider(question: str) -> str:
#         provider.send_message(patient_phone, question)
#         return session_store.wait_for_next_inbound(patient_phone)  # NOT BUILT
#     case, trail = extract_case_with_followup(raw_text, backend, answer_provider=answer_provider)
#
# `session_store.wait_for_next_inbound` is exactly the piece this repo does
# not have yet -- see AGENT1_README.md's open-work list.
