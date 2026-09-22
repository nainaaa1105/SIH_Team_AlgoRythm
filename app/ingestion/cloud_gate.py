"""Cloud-fraction gate: decides whether a cluster's location/time is
optically clear enough to trust a Sentinel-2 fetch, and applies a
confidence-decay formula so cloudy clusters don't silently vanish —
they get down-weighted instead (feeding M2's evidence-weighting engine).

Primary path is the CDSE (Sentinel Hub) catalog search in
`app/ingestion/cdse_client.py` — real, verified-working credentials,
no GCP account needed. Earth Engine is kept as a second-tier fallback
(untouched below) purely in case a GEE service account is ever
configured later; it is not required and not currently used.
"""
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_ee_initialized = False
_ee_lock = threading.Lock()


class CloudGateError(RuntimeError):
    pass


def _ensure_ee_initialized(settings: Settings) -> None:
    global _ee_initialized
    if _ee_initialized:
        return
    with _ee_lock:
        if _ee_initialized:  # double-check after acquiring lock
            return
        try:
            import ee
        except ImportError as exc:
            raise CloudGateError("earthengine-api is required (pip install earthengine-api)") from exc

        if not settings.gee_service_account:
            raise CloudGateError("GEE_SERVICE_ACCOUNT not configured")

        credentials = ee.ServiceAccountCredentials(settings.gee_service_account, settings.gee_private_key_file)
        ee.Initialize(credentials)
        _ee_initialized = True


def _cloud_fraction_via_cdse(lon: float, lat: float, date: datetime, settings: Settings) -> Optional[float]:
    from app.ingestion import cdse_client

    if not cdse_client.is_configured(settings):
        return None

    buffer_deg = 0.02  # ~2 km box around the point — enough to land in one scene's footprint
    bbox = (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)
    scene = cdse_client.search_least_cloudy_scene(
        bbox, date - timedelta(days=3), date + timedelta(days=3), settings=settings,
    )
    if scene is None or scene.get("cloud_cover") is None:
        return None
    return max(0.0, min(1.0, float(scene["cloud_cover"]) / 100.0))


def _cloud_fraction_via_gee(lon: float, lat: float, date: datetime, settings: Settings) -> Optional[float]:
    try:
        _ensure_ee_initialized(settings)
        import ee
    except CloudGateError:
        return None

    point = ee.Geometry.Point([lon, lat])
    start = (date - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=3)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    try:
        image = collection.first()
        cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    except Exception:
        logger.exception("GEE query failed for (%s, %s) around %s", lon, lat, date)
        return None

    if cloud_pct is None:
        return None
    return max(0.0, min(1.0, float(cloud_pct) / 100.0))


def get_cloud_fraction(lon: float, lat: float, date: datetime, settings: Optional[Settings] = None) -> Optional[float]:
    """Cloud fraction (0-1) of the least-cloudy Sentinel-2 scene covering
    (lon, lat) within a +/-3 day window of `date`.

    Tries CDSE first (real, verified-working credentials), then GEE only
    if a service account is ever configured. Returns None if neither is
    available — callers must treat None as "unknown", not "clear".
    """
    settings = settings or get_settings()

    # Cached on disk: this is a CDSE catalogue round-trip per cluster,
    # and a live cycle touches thousands of clusters. The key includes
    # the date because the answer is a property of a moment, not just a
    # place. A cached None ("no scene covers this point in the window")
    # is a real answer and is replayed as such — see geo_cache.MISS.
    from app import geo_cache

    key = geo_cache.make_key("cloud_fraction", lon, lat, date.strftime("%Y-%m-%d"))
    cached = geo_cache.get(key)
    if cached is not geo_cache.MISS:
        return cached

    def _lookup():
        cdse_result = _cloud_fraction_via_cdse(lon, lat, date, settings)
        if cdse_result is not None:
            return cdse_result

        gee_result = _cloud_fraction_via_gee(lon, lat, date, settings)
        if gee_result is not None:
            return gee_result

        logger.warning("No cloud-cover source available — unknown for (%s, %s)", lon, lat)
        return None

    value = _lookup()
    geo_cache.put(key, value)
    return value


def is_optically_available(cloud_fraction: Optional[float], settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    if cloud_fraction is None:
        return False
    return cloud_fraction <= settings.cloud_fraction_threshold


def confidence_decay(base_confidence: float, cloud_fraction: Optional[float]) -> float:
    """Down-weight a detection's effective confidence as cloud cover
    rises, rather than dropping it. Linear decay: at 0% cloud, full
    confidence is retained; at 100% cloud, confidence is halved (thermal
    signal from VIIRS/MODIS/INSAT-3DS still penetrates cloud reasonably
    well at night/via IR, so it is not zeroed out — only the *optical*
    corroboration is unavailable).
    """
    if cloud_fraction is None:
        return base_confidence * 0.85  # unknown cloud state: mild, not severe, penalty
    decay_factor = 1.0 - 0.5 * cloud_fraction
    return max(0.0, base_confidence * decay_factor)
