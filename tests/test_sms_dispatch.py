"""Emergency SMS dispatch (TEST MODE) — app/notifications/.

Pure-function/no-DB tests only (this suite runs without Postgres — see
test_gateway.py's module docstring for why): recipient selection,
HttpSMS client error handling, and SMS body composition. The DB-backed
parts (dispatch_emergency_sms's Alert read/write, the /sms-dispatch
route) are exercised live, the same way the rest of this codebase's
PostGIS-dependent logic is verified.
"""
import pytest

from app.config import Settings
from app.notifications.sms_client import send_sms
from app.notifications.sms_dispatch import GOOGLE_MAPS_NAV_URL, build_sms_body
from app.notifications.sms_recipient import SmsModeNotImplemented, get_sms_recipient


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


# --- recipient selection: TEST MODE ONLY ------------------------------

def test_get_sms_recipient_returns_the_configured_test_number():
    settings = _settings(sms_mode="test", test_sms_recipient="+919773207681")
    assert get_sms_recipient(cluster_id=123, settings=settings) == "+919773207681"


def test_get_sms_recipient_returns_none_when_unconfigured():
    settings = _settings(sms_mode="test", test_sms_recipient="")
    assert get_sms_recipient(cluster_id=123, settings=settings) is None


def test_get_sms_recipient_takes_cluster_id_even_though_test_mode_ignores_it():
    """Signature already matches what nearest-fire-station routing will
    need later (the whole point of routing selection behind this one
    function) — same recipient regardless of which cluster asks, in
    test mode, but the parameter is there and accepted."""
    settings = _settings(sms_mode="test", test_sms_recipient="+919773207681")
    assert get_sms_recipient(1, settings) == get_sms_recipient(999999, settings)


def test_get_sms_recipient_refuses_an_unimplemented_mode():
    """Do NOT implement nearest-fire-station selection yet — any mode
    other than 'test' must fail loudly, not silently fall back to
    something that looks like real routing."""
    settings = _settings(sms_mode="station", test_sms_recipient="+919773207681")
    with pytest.raises(SmsModeNotImplemented):
        get_sms_recipient(1, settings)


# --- HttpSMS client: never claims success it doesn't have -------------

def test_send_sms_fails_honestly_with_no_credentials():
    settings = _settings(httpsms_api_key="", httpsms_from_number="")
    result = send_sms("+919773207681", "test", settings=settings)
    assert result.ok is False
    assert result.provider_message_id is None
    assert "HTTPSMS_API_KEY" in result.error


def test_send_sms_never_reaches_the_network_with_no_credentials(monkeypatch):
    """Regression guard: must fail fast on missing config, not attempt
    the HTTP call and fail some other way."""
    import requests

    def _fail_if_called(*a, **k):
        raise AssertionError("requests.post must not be called with no API key configured")

    monkeypatch.setattr(requests, "post", _fail_if_called)
    settings = _settings(httpsms_api_key="", httpsms_from_number="")
    result = send_sms("+919773207681", "test", settings=settings)
    assert result.ok is False


# --- SMS body composition: real data in, honest gaps out --------------

def _incident(**overrides):
    base = {
        "cluster_id": 42,
        "lat": 21.8212,
        "lon": 83.9968,
        "location_name": "SPS Steel and Power Limited",
        "predicted_class": "industrial_fire",
        "suppression": {
            "resource_kind": "foam",
            "resource_label": "Foam (AFFF, Class B)",
            "primary_volume_l": 45000.0,
            "tanker_trips": 9,
            "air_drop_trips": 2,
            "air_support_recommended": True,
        },
    }
    base.update(overrides)
    return base


def test_sms_body_includes_all_required_fields_with_real_values():
    body = build_sms_body(_incident())
    assert "FIREOPS EMERGENCY ALERT" in body
    assert "Location: SPS Steel and Power Limited" in body
    assert "Coordinates: 21.82120, 83.99680" in body
    assert "Water Required: 45,000 L" in body
    assert "Fire Truck Trips: 9" in body
    assert "Ground Support: Required" in body
    assert "Air Support: Required" in body
    assert "Aerial Tanks: 2" in body
    assert GOOGLE_MAPS_NAV_URL.format(lat=21.8212, lon=83.9968) in body
    assert "Respond immediately." in body


def test_sms_body_no_longer_carries_the_dashboard_link():
    """By request: the "Fire Location:" section (a link back to our own
    dashboard) is removed entirely -- NAVIGATE (Google Maps) is the only
    link in the message now."""
    body = build_sms_body(_incident())
    assert "Fire Location:" not in body
    assert "?cluster=" not in body


def test_sms_body_never_fabricates_a_missing_location_or_water_value():
    """A cluster with no facility/district attribution and no
    suppression estimate must read as an honest placeholder, never a
    guessed name or a 0 that looks like a real measurement."""
    incident = _incident(location_name=None, suppression={
        "resource_kind": None, "primary_volume_l": None, "primary_volume_m3": None,
        "tanker_trips": None, "air_drop_trips": None, "air_support_recommended": False,
    })
    body = build_sms_body(incident)
    assert "Location: Location pending confirmation" in body
    assert "Water Required: Not available" in body
    assert "Fire Truck Trips: Not available" in body
    assert "Ground Support: Not Required" in body
    assert "Air Support: Not Required" in body
    assert "Aerial Tanks" not in body  # only shown when air support applies


def test_sms_body_reports_mining_fill_volume_not_a_fake_litre_count():
    """Mining's suppression estimate is m3 of fill material, not litres
    of water — the SMS must say so honestly rather than printing a
    made-up "X L" for a class that doesn't use water."""
    incident = _incident(predicted_class="mining", suppression={
        "resource_kind": "excavation_and_smothering",
        "resource_label": "Inert fill + smothering",
        "primary_volume_l": None,
        "primary_volume_m3": 320.0,
        "truck_loads": 32,
        "tanker_trips": None,
        "air_drop_trips": None,
        "air_support_recommended": False,
    })
    body = build_sms_body(incident)
    assert "Water Required: Not applicable" in body
    assert "320" in body
    assert "Fire Truck Trips: 32" in body
    assert "Ground Support: Required" in body


def test_google_maps_nav_url_uses_the_real_incident_coordinates():
    url = GOOGLE_MAPS_NAV_URL.format(lat=21.8212, lon=83.9968)
    assert url == "https://www.google.com/maps/dir/?api=1&destination=21.8212,83.9968"
