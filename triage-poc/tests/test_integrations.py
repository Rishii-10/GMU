"""
Tests for app/integrations/ (Phase 7: pluggable external adapters).

No real network calls anywhere: mock providers are tested directly, and
real providers (TwilioProvider, GoogleTranslateProvider, ORSProvider) are
tested for construction-time credential validation plus their HTTP call
shape via unittest.mock.patch on `requests` -- never an actual request.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.integrations.geo import ORSProvider, MockRoutingProvider as GeoMockRoutingProvider
from app.integrations.messaging import MessagingProvider, MockProvider, TwilioProvider
from app.integrations.translation import GoogleTranslateProvider, IdentityTranslator, Translator


# --- messaging.py --------------------------------------------------------------


def test_mock_provider_records_sent_messages():
    provider = MockProvider()
    ok = provider.send_message("+919876543210", "hello")
    assert ok is True
    assert provider.sent == [("+919876543210", "hello")]


def test_mock_provider_records_multiple_sends_in_order():
    provider = MockProvider()
    provider.send_message("+1", "a")
    provider.send_message("+2", "b")
    assert provider.sent == [("+1", "a"), ("+2", "b")]


def test_twilio_provider_raises_without_credentials(monkeypatch):
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        TwilioProvider()


def test_twilio_provider_constructs_with_explicit_credentials():
    provider = TwilioProvider(account_sid="AC123", auth_token="token", from_number="+15551234567")
    assert provider.account_sid == "AC123"


@patch("requests.post")
def test_twilio_provider_send_success(mock_post):
    mock_post.return_value = MagicMock(status_code=201)
    provider = TwilioProvider(account_sid="AC123", auth_token="token", from_number="+15551234567")
    ok = provider.send_message("+919876543210", "test question")
    assert ok is True
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["data"]["To"] == "+919876543210"
    assert call_kwargs["data"]["Body"] == "test question"
    assert call_kwargs["auth"] == ("AC123", "token")


@patch("requests.post")
def test_twilio_provider_send_failure_returns_false_not_raise(mock_post):
    mock_post.return_value = MagicMock(status_code=400)
    provider = TwilioProvider(account_sid="AC123", auth_token="token", from_number="+15551234567")
    assert provider.send_message("+1", "x") is False


def test_twilio_provider_network_error_returns_false_not_raise():
    import requests

    provider = TwilioProvider(account_sid="AC123", auth_token="token", from_number="+15551234567")
    with patch("requests.post", side_effect=requests.RequestException("boom")):
        assert provider.send_message("+1", "x") is False


def test_messaging_provider_is_abstract():
    with pytest.raises(TypeError):
        MessagingProvider()


# --- translation.py --------------------------------------------------------------


def test_identity_translator_passthrough():
    t = IdentityTranslator()
    result = t.translate("पेट दर्द", target_language="en")
    assert result.translated_text == "पेट दर्द"
    assert result.original_text == "पेट दर्द"
    assert result.source_language is None


def test_identity_translator_detect_language_always_none():
    t = IdentityTranslator()
    assert t.detect_language("any text") is None


def test_google_translate_provider_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_TRANSLATE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GoogleTranslateProvider()


def test_google_translate_provider_constructs_with_explicit_key():
    provider = GoogleTranslateProvider(api_key="fake-key")
    assert provider.api_key == "fake-key"


@patch("requests.post")
def test_google_translate_provider_translate_success(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "data": {
                "translations": [{"translatedText": "stomach pain", "detectedSourceLanguage": "hi"}]
            }
        },
    )
    provider = GoogleTranslateProvider(api_key="fake-key")
    result = provider.translate("पेट दर्द", target_language="en")
    assert result.translated_text == "stomach pain"
    assert result.source_language == "hi"
    assert result.original_text == "पेट दर्द"


def test_google_translate_provider_falls_back_to_original_on_error():
    import requests

    provider = GoogleTranslateProvider(api_key="fake-key")
    with patch("requests.post", side_effect=requests.RequestException("boom")):
        result = provider.translate("original text", target_language="en")
    assert result.translated_text == "original text"
    assert result.source_language is None


@patch("requests.post")
def test_google_translate_provider_detect_language(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"data": {"detections": [[{"language": "hi", "confidence": 0.9}]]}},
    )
    provider = GoogleTranslateProvider(api_key="fake-key")
    lang = provider.detect_language("पेट दर्द")
    assert lang == "hi"


def test_translator_is_abstract():
    with pytest.raises(TypeError):
        Translator()


# --- geo.py -----------------------------------------------------------------------


def test_geo_reexports_mock_routing_provider():
    provider = GeoMockRoutingProvider()
    info = provider.get_route((12.5, 77.5), (12.6, 77.6))
    assert info.provider == "mock_haversine"


def test_ors_provider_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ORS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ORSProvider()


def test_ors_provider_constructs_with_explicit_key():
    provider = ORSProvider(api_key="fake-ors-key")
    assert provider.api_key == "fake-ors-key"


@patch("requests.post")
def test_ors_provider_get_route_success(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"routes": [{"summary": {"distance": 11690.0, "duration": 1050.0}}]},
    )
    provider = ORSProvider(api_key="fake-ors-key")
    route_info = provider.get_route((12.5333, 77.7667), (12.60, 77.85))
    assert route_info.provider == "openrouteservice"
    assert route_info.road_distance_km == pytest.approx(11.69, abs=0.01)
    assert route_info.eta_minutes == pytest.approx(17.5, abs=0.1)
    assert "google.com/maps" in route_info.maps_link


def test_ors_provider_falls_back_to_haversine_on_error():
    import requests

    provider = ORSProvider(api_key="fake-ors-key")
    with patch("requests.post", side_effect=requests.RequestException("boom")):
        route_info = provider.get_route((12.5333, 77.7667), (12.60, 77.85))
    assert route_info.provider == "mock_haversine"
    assert route_info.road_distance_km > 0


@patch("requests.post")
def test_ors_provider_falls_back_on_malformed_response(mock_post):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"unexpected": "shape"})
    provider = ORSProvider(api_key="fake-ors-key")
    route_info = provider.get_route((12.5333, 77.7667), (12.60, 77.85))
    assert route_info.provider == "mock_haversine"


def test_ors_uses_coordinate_order_lon_lat(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["coordinates"] = json["coordinates"]
        return MagicMock(status_code=200, json=lambda: {"routes": [{"summary": {"distance": 1000.0, "duration": 60.0}}]})

    with patch("requests.post", side_effect=fake_post):
        provider = ORSProvider(api_key="fake-ors-key")
        provider.get_route((12.5333, 77.7667), (12.60, 77.85))

    # ORS expects [lon, lat] -- opposite of this codebase's (lat, lon) convention.
    assert captured["coordinates"][0] == [77.7667, 12.5333]
    assert captured["coordinates"][1] == [77.85, 12.60]
