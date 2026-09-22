"""Copernicus Data Space Ecosystem (Sentinel Hub) client.

This is the real, working replacement for Google Earth Engine, which
this project has no GCP service account for. `CDSE_CLIENT_ID` /
`CDSE_CLIENT_SECRET` in .env ARE configured with real credentials —
verified live on 2026-09-13 (OAuth token exchange succeeds; a real
Sentinel-2 scene search over Dhanbad returned a same-day acquisition;
the Process API returned real decoded reflectance values, not zeros or
an error page). GEE-dependent code should try this first and treat GEE
as the fallback tier, not the other way around.

Two Sentinel Hub APIs, both public/documented (not a guessed vendor
format the way MOSDAC's was):
  - Catalog API (STAC-compliant): scene search by bbox/date, returns
    `eo:cloud_cover` and `datetime` per scene.
  - Process API: actual band pixel data for a bbox + time range,
    returned as a GeoTIFF, decoded here with rasterio.
Docs: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/

CDSE rolled out a shorter path structure in March 2026 (`/catalog/v1`,
`/process/v1`) alongside the legacy `/api/v1/...` paths; both work, this
module uses the new ones. If CDSE later deprecates the old paths this
does not need to change.

Note on Catalog `sortby`: the API's documented sortby parameter returned
a 400 in testing against this project's real credentials (exact
accepted syntax unconfirmed), so `search_least_cloudy_scene` fetches a
batch of candidates unsorted and picks the least-cloudy one client-side
— slightly more bandwidth, zero schema risk.
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import requests

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_SEARCH_URL = f"{SH_BASE_URL}/catalog/v1/search"
PROCESS_URL = f"{SH_BASE_URL}/process/v1"

# GEE-style band keys (used throughout this project's indices/model code,
# e.g. imagery.optical.indices.SENTINEL2_BANDS) -> Sentinel Hub evalscript
# band identifiers. Only B2/B3/B4/B8 differ (need zero-padding); B11/B12/
# B8A are already the same spelling in both systems.
_GEE_TO_SH_BAND = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09",
    "B10": "B10", "B11": "B11", "B12": "B12",
}

_token_cache: Dict[str, object] = {"token": None, "expires_at": 0.0}


def is_configured(settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.cdse_client_id and settings.cdse_client_secret)


def get_access_token(settings: Optional[Settings] = None) -> Optional[str]:
    """OAuth2 client-credentials exchange. Cached in-process until ~60s
    before expiry, same pattern used for the EUMETSAT backup source."""
    settings = settings or get_settings()
    if not is_configured(settings):
        return None

    now = time.monotonic()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]  # type: ignore[return-value]

    try:
        resp = requests.post(
            settings.cdse_token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": settings.cdse_client_id,
                "client_secret": settings.cdse_client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException:
        logger.exception("CDSE token request failed — check CDSE_CLIENT_ID/SECRET")
        return None

    token = payload.get("access_token")
    expires_in = float(payload.get("expires_in", 600))
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + max(expires_in - 60.0, 30.0)
    return token


def search_least_cloudy_scene(
    bbox: Sequence[float],
    start: datetime,
    end: datetime,
    collection: str = "sentinel-2-l2a",
    settings: Optional[Settings] = None,
) -> Optional[Dict]:
    """Least-cloudy scene covering bbox within [start, end].

    Returns {"id", "cloud_cover", "datetime"}, or None if unconfigured,
    unreachable, or nothing is published for that window.
    """
    settings = settings or get_settings()
    token = get_access_token(settings)
    if token is None:
        return None

    try:
        resp = requests.post(
            CATALOG_SEARCH_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "collections": [collection],
                "datetime": f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                "bbox": list(bbox),
                "limit": 20,
            },
            timeout=30,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except requests.RequestException:
        logger.exception("CDSE catalog search failed for bbox=%s", bbox)
        return None

    if not features:
        return None

    def _cloud_cover(feature: Dict) -> float:
        value = feature.get("properties", {}).get("eo:cloud_cover")
        return float(value) if value is not None else 100.0

    best = min(features, key=_cloud_cover)
    props = best.get("properties", {})
    return {
        "id": best.get("id"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "datetime": props.get("datetime"),
    }


def fetch_bands(
    bbox: Sequence[float],
    bands: Sequence[str],
    start: datetime,
    end: datetime,
    width: int,
    height: int,
    collection: str = "sentinel-2-l2a",
    max_cloud_coverage: float = 100.0,
    settings: Optional[Settings] = None,
):
    """Least-cloudy-mosaic band stack for bbox as a real (n_bands, H, W)
    float32 reflectance array, keyed in the caller's band order — or
    None if unconfigured, unreachable, or rasterio is missing.

    Band names are accepted in this project's GEE-style spelling
    ("B2", "B4", ...) and translated to Sentinel Hub's evalscript names
    internally, so callers written against the old GEE path don't need
    to change their band lists.
    """
    settings = settings or get_settings()
    token = get_access_token(settings)
    if token is None:
        return None

    try:
        import numpy as np  # noqa: F401 - imported to fail fast if missing
        from rasterio.io import MemoryFile
    except ImportError:
        logger.info("rasterio not installed — CDSE band fetch unavailable")
        return None

    sh_bands = [_GEE_TO_SH_BAND.get(b, b) for b in bands]
    band_list = ", ".join(f'"{b}"' for b in sh_bands)
    return_list = ", ".join(f"sample.{b}" for b in sh_bands)
    evalscript = (
        "//VERSION=3\n"
        "function setup() {\n"
        f"  return {{ input: [{band_list}], output: {{ bands: {len(sh_bands)}, sampleType: \"FLOAT32\" }} }};\n"
        "}\n"
        "function evaluatePixel(sample) {\n"
        f"  return [{return_list}];\n"
        "}\n"
    )

    body = {
        "input": {
            "bounds": {"bbox": list(bbox)},
            "data": [{
                "type": collection,
                "dataFilter": {
                    "timeRange": {
                        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    "maxCloudCoverage": max_cloud_coverage,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": width, "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    try:
        resp = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {token}"}, json=body, timeout=60)
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("CDSE Process API request failed for bbox=%s", bbox)
        return None

    try:
        with MemoryFile(resp.content) as memfile, memfile.open() as src:
            return src.read()  # (n_bands, H, W), in the order `bands` was given
    except Exception:
        logger.exception("Failed to decode CDSE Process API response for bbox=%s", bbox)
        return None
