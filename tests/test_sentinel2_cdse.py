"""Sentinel-2 patch fetch must try CDSE (real, verified-working
credentials) before Earth Engine (unconfigured in this project).

Live-fetch correctness of the underlying CDSE calls was checked manually
against real credentials: fetch_patch(lon=86.43, lat=23.8) returned a
real 2026-09-08 Sentinel-2 scene over Dhanbad with real reflectance
values (B4 mean ~0.359) via app.ingestion.cdse_client. These tests mock
that client so the suite has no network dependency.
"""
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from imagery.optical.sentinel2 import bbox_around, fetch_patch


def _fake_stack(n_bands=6, size=8):
    return np.random.rand(n_bands, size, size).astype("float32")


def test_uses_cdse_when_a_clear_enough_scene_is_found():
    box = bbox_around(86.43, 23.8, 2000.0)
    with patch("app.ingestion.cdse_client.is_configured", return_value=True), \
         patch("app.ingestion.cdse_client.search_least_cloudy_scene",
               return_value={"id": "x", "cloud_cover": 12.0, "datetime": "2026-09-08T05:02:00Z"}), \
         patch("app.ingestion.cdse_client.fetch_bands", return_value=_fake_stack()):
        result = fetch_patch(lon=86.43, lat=23.8, spatial_extent_km=3.0)

    assert result.available is True
    assert result.source == "CDSE_sentinel-2-l2a"
    assert result.cloud_percentage == 12.0
    assert set(result.bands) == {"B2", "B3", "B4", "B8", "B11", "B12"}
    assert result.acquired_at == datetime(2026, 9, 8, 5, 2, tzinfo=timezone.utc)


def test_rejects_a_scene_over_the_cloud_threshold_and_does_not_fetch_bands():
    with patch("app.ingestion.cdse_client.is_configured", return_value=True), \
         patch("app.ingestion.cdse_client.search_least_cloudy_scene",
               return_value={"id": "x", "cloud_cover": 90.0, "datetime": "2026-09-08T05:02:00Z"}), \
         patch("app.ingestion.cdse_client.fetch_bands") as fetch_bands:
        result = fetch_patch(lon=86.43, lat=23.8, spatial_extent_km=3.0)

    assert result.available is False
    assert "90" in result.reason
    fetch_bands.assert_not_called()  # cloud check must short-circuit before the heavier call


def test_falls_back_to_gee_when_cdse_is_unconfigured():
    with patch("app.ingestion.cdse_client.is_configured", return_value=False), \
         patch("imagery.optical.sentinel2._initialise_earth_engine", return_value=False):
        result = fetch_patch(lon=86.43, lat=23.8)

    assert result.available is False
    assert "CDSE not configured" in result.reason
    assert "Earth Engine unavailable" in result.reason


def test_no_scene_found_reports_a_clear_reason():
    with patch("app.ingestion.cdse_client.is_configured", return_value=True), \
         patch("app.ingestion.cdse_client.search_least_cloudy_scene", return_value=None), \
         patch("imagery.optical.sentinel2._initialise_earth_engine", return_value=False):
        result = fetch_patch(lon=86.43, lat=23.8)

    assert result.available is False
    assert "No Sentinel-2 scene found" in result.reason


def test_missing_band_data_degrades_cleanly():
    with patch("app.ingestion.cdse_client.is_configured", return_value=True), \
         patch("app.ingestion.cdse_client.search_least_cloudy_scene",
               return_value={"id": "x", "cloud_cover": 5.0, "datetime": "2026-09-08T05:02:00Z"}), \
         patch("app.ingestion.cdse_client.fetch_bands", return_value=None), \
         patch("imagery.optical.sentinel2._initialise_earth_engine", return_value=False):
        result = fetch_patch(lon=86.43, lat=23.8)

    assert result.available is False
    assert "CDSE Process API fetch failed" in result.reason
