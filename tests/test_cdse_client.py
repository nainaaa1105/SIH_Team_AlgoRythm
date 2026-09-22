"""Unit tests for the CDSE/Sentinel Hub client — the real GEE
replacement. Network is mocked here; the actual endpoints, auth flow and
response shapes were verified live against real credentials (2026-09-13):
token exchange succeeded, a real same-day Sentinel-2 scene was found over
Dhanbad with real cloud_cover, and the Process API returned a decodable
GeoTIFF with real (non-zero, physically plausible) reflectance values.
"""
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from app.config import Settings
from app.ingestion import cdse_client


def test_is_configured_requires_both_id_and_secret():
    assert cdse_client.is_configured(Settings(cdse_client_id="", cdse_client_secret="")) is False
    assert cdse_client.is_configured(Settings(cdse_client_id="x", cdse_client_secret="")) is False
    assert cdse_client.is_configured(Settings(cdse_client_id="x", cdse_client_secret="y")) is True


def test_get_access_token_returns_none_when_unconfigured():
    cdse_client._token_cache.update({"token": None, "expires_at": 0.0})
    settings = Settings(cdse_client_id="", cdse_client_secret="")
    assert cdse_client.get_access_token(settings) is None


def test_get_access_token_caches_across_calls():
    cdse_client._token_cache.update({"token": None, "expires_at": 0.0})
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret")
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {"access_token": "tok123", "expires_in": 600}
    fake_response.raise_for_status = Mock()

    with patch("requests.post", return_value=fake_response) as post:
        first = cdse_client.get_access_token(settings)
        second = cdse_client.get_access_token(settings)

    assert first == "tok123"
    assert second == "tok123"
    post.assert_called_once()  # second call served from cache, no second HTTP request


def test_search_picks_the_least_cloudy_of_several_candidates():
    cdse_client._token_cache.update({"token": "tok", "expires_at": 1e18})
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret")
    fake = Mock(status_code=200)
    fake.raise_for_status = Mock()
    fake.json.return_value = {
        "features": [
            {"id": "cloudy", "properties": {"eo:cloud_cover": 90.0, "datetime": "2026-09-01T00:00:00Z"}},
            {"id": "clear", "properties": {"eo:cloud_cover": 5.0, "datetime": "2026-09-10T00:00:00Z"}},
        ]
    }

    with patch("requests.post", return_value=fake):
        result = cdse_client.search_least_cloudy_scene(
            [86.0, 23.0, 87.0, 24.0],
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 13, tzinfo=timezone.utc),
            settings=settings,
        )

    assert result["id"] == "clear"
    assert result["cloud_cover"] == 5.0


def test_search_returns_none_when_no_features():
    cdse_client._token_cache.update({"token": "tok", "expires_at": 1e18})
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret")
    fake = Mock(status_code=200)
    fake.raise_for_status = Mock()
    fake.json.return_value = {"features": []}

    with patch("requests.post", return_value=fake):
        result = cdse_client.search_least_cloudy_scene(
            [0, 0, 1, 1], datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc), settings=settings,
        )
    assert result is None


def test_fetch_bands_translates_gee_band_names_to_sentinel_hub_names():
    """This project's indices code uses GEE-style keys ('B4', 'B8'); the
    evalscript sent to Sentinel Hub must use the zero-padded spelling
    ('B04', 'B08') or the request is rejected server-side."""
    cdse_client._token_cache.update({"token": "tok", "expires_at": 1e18})
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret")

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["evalscript"] = json["evalscript"]
        resp = Mock(status_code=200, content=b"not-a-real-tiff")
        resp.raise_for_status = Mock()
        return resp

    with patch("requests.post", side_effect=fake_post), \
         patch("rasterio.io.MemoryFile") as memfile_cls:
        memfile_cls.side_effect = Exception("not a real tiff, decode fails as expected")
        result = cdse_client.fetch_bands(
            bbox=[86.0, 23.0, 87.0, 24.0], bands=["B4", "B8"],
            start=datetime(2026, 9, 1, tzinfo=timezone.utc),
            end=datetime(2026, 9, 13, tzinfo=timezone.utc),
            width=64, height=64, settings=settings,
        )

    assert result is None  # decode failed, degrades to None rather than raising
    assert '"B04"' in captured["evalscript"]
    assert '"B08"' in captured["evalscript"]
    assert captured["url"] == cdse_client.PROCESS_URL


def test_fetch_bands_returns_none_without_rasterio():
    cdse_client._token_cache.update({"token": "tok", "expires_at": 1e18})
    settings = Settings(cdse_client_id="id", cdse_client_secret="secret")
    with patch.dict("sys.modules", {"rasterio": None, "rasterio.io": None}):
        result = cdse_client.fetch_bands(
            bbox=[0, 0, 1, 1], bands=["B4"],
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            width=8, height=8, settings=settings,
        )
    assert result is None
