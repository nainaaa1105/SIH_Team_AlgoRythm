"""Cloud-fraction source selection: CDSE (real, verified-working
credentials) must be tried before GEE (unconfigured in this project).
Network calls are mocked — live-fetch correctness was checked manually
against the real CDSE API (see cdse_client.py docstring).
"""
from datetime import datetime, timezone
from unittest.mock import patch

from app.config import Settings
from app.ingestion.cloud_gate import get_cloud_fraction


def test_uses_cdse_result_when_available():
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret")
    with patch(
        "app.ingestion.cdse_client.search_least_cloudy_scene",
        return_value={"id": "x", "cloud_cover": 45.0, "datetime": "2026-09-13T00:00:00Z"},
    ) as search, patch("app.ingestion.cdse_client.is_configured", return_value=True):
        result = get_cloud_fraction(77.2, 28.6, datetime.now(timezone.utc), settings)

    assert result == 0.45
    search.assert_called_once()


def test_falls_back_to_gee_when_cdse_has_no_scene():
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret",
                         gee_service_account="acct")
    with patch("app.ingestion.cdse_client.is_configured", return_value=True), \
         patch("app.ingestion.cdse_client.search_least_cloudy_scene", return_value=None), \
         patch("app.ingestion.cloud_gate._cloud_fraction_via_gee", return_value=0.2) as gee:
        result = get_cloud_fraction(77.2, 28.6, datetime.now(timezone.utc), settings)

    assert result == 0.2
    gee.assert_called_once()


def test_returns_none_when_neither_source_is_available():
    settings = Settings(cdse_client_id="", cdse_client_secret="", gee_service_account="")
    result = get_cloud_fraction(77.2, 28.6, datetime.now(timezone.utc), settings)
    assert result is None


def test_cdse_not_queried_when_unconfigured():
    """No credentials -> no network call at all, not even an attempt."""
    settings = Settings(cdse_client_id="", cdse_client_secret="")
    with patch("app.ingestion.cdse_client.search_least_cloudy_scene") as search:
        get_cloud_fraction(77.2, 28.6, datetime.now(timezone.utc), settings)
    search.assert_not_called()
