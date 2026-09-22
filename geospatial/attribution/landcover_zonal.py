"""Land-cover context via zonal statistics.

Upgrades M1's `app/enrichment/landcover.py`, which samples a fixed 5x5
window of MODIS MCD12Q1 (500 m pixels, so ~2.5 km regardless of the
event). Two problems with that for India:

  * 500 m is coarse for a fragmented landscape where a village, its
    fields and a factory can all sit inside one pixel;
  * a fixed window ignores how big the event actually is. A 30 km
    wildfire front and a single flare stack get the same context.

So this uses ESA WorldCover (10 m) and sizes the sampling buffer from
the cluster's real footprint — M2 already computes `spatial_extent_km`
on `cluster_features`, so we read it rather than guessing.

The ESA WorldCover class codes are a different scheme from MODIS IGBP;
both mappings are kept here so either raster can be used, and the source
is recorded alongside the result.
"""
import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ESA WorldCover v100/v200 class codes.
WORLDCOVER_CLASSES = {
    10: "forest",       # Tree cover
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "urban",        # Built-up
    60: "barren",       # Bare / sparse vegetation
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss_lichen",
}

# MODIS MCD12Q1 IGBP codes, matching M1's existing mapping so both
# implementations agree on what counts as forest/cropland/urban.
_MODIS_FOREST = {1, 2, 3, 4, 5, 8, 9}
_MODIS_CROPLAND = {12, 14}
_MODIS_URBAN = {13}

MIN_BUFFER_M = 500.0
MAX_BUFFER_M = 20000.0


def buffer_radius_m(spatial_extent_km: Optional[float], default_m: float = 1000.0) -> float:
    """Sampling radius sized to the event.

    Uses the footprint radius plus a margin, so context is sampled around
    the fire rather than only within it. Clamped so a degenerate extent
    can't produce a sub-pixel or continent-sized buffer.
    """
    if spatial_extent_km is None or spatial_extent_km <= 0:
        return default_m
    radius_m = (spatial_extent_km * 1000.0) / 2.0
    return max(MIN_BUFFER_M, min(radius_m * 1.5, MAX_BUFFER_M))


def summarise_worldcover(counts: Dict[int, int]) -> Dict[str, float]:
    """Class-code histogram -> fractional cover by category."""
    total = sum(counts.values())
    if total <= 0:
        return {"pct_cropland": 0.0, "pct_forest": 0.0, "pct_urban": 0.0,
                "pct_water": 0.0, "pct_barren": 0.0}

    def fraction(*names: str) -> float:
        wanted = set(names)
        matched = sum(n for code, n in counts.items() if WORLDCOVER_CLASSES.get(code) in wanted)
        return matched / total

    return {
        "pct_cropland": fraction("cropland"),
        "pct_forest": fraction("forest", "mangrove"),
        "pct_urban": fraction("urban"),
        "pct_water": fraction("water", "wetland"),
        "pct_barren": fraction("barren", "shrubland", "grassland"),
    }


def summarise_modis(counts: Dict[int, int]) -> Dict[str, float]:
    """MCD12Q1 IGBP histogram -> the same fractional categories."""
    total = sum(counts.values())
    if total <= 0:
        return {"pct_cropland": 0.0, "pct_forest": 0.0, "pct_urban": 0.0,
                "pct_water": 0.0, "pct_barren": 0.0}

    def fraction(codes: set) -> float:
        return sum(n for code, n in counts.items() if code in codes) / total

    return {
        "pct_cropland": fraction(_MODIS_CROPLAND),
        "pct_forest": fraction(_MODIS_FOREST),
        "pct_urban": fraction(_MODIS_URBAN),
        "pct_water": fraction({17}),
        "pct_barren": fraction({16, 7, 10}),
    }


def zonal_landcover_gee(
    lon: float, lat: float, radius_m: float, collection: str = "ESA/WorldCover/v200"
) -> Optional[Dict[str, float]]:
    """Zonal land-cover fractions from Earth Engine.

    Returns None (never raises) when GEE is unavailable or
    unconfigured, so the caller can fall back to M1's local-raster
    sampling rather than failing the whole enrichment.
    """
    try:
        import ee
    except ImportError:
        logger.info("earthengine-api not installed — skipping WorldCover zonal stats")
        return None

    try:
        from app.config import get_settings

        settings = get_settings()
        if settings.gee_service_account:
            credentials = ee.ServiceAccountCredentials(
                settings.gee_service_account, settings.gee_private_key_file
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize()

        region = ee.Geometry.Point([lon, lat]).buffer(radius_m)
        image = ee.ImageCollection(collection).first().select("Map")

        histogram = image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=10,
            maxPixels=1e9,
        ).getInfo()

        raw = (histogram or {}).get("Map") or {}
        counts = {int(code): int(float(count)) for code, count in raw.items()}
        if not counts:
            return None
        return summarise_worldcover(counts)
    except Exception:
        logger.warning("WorldCover zonal stats failed for (%s, %s)", lon, lat, exc_info=True)
        return None


def landcover_with_fallback(
    lon: float, lat: float, spatial_extent_km: Optional[float] = None
) -> Dict[str, object]:
    """Best available land-cover context, with the source recorded.

    Tries 10 m WorldCover zonal stats first, falls back to M1's local
    MODIS raster sampler. The `source` key matters: a downstream
    consumer should know whether it is looking at 10 m zonal statistics
    or a coarse 500 m window.
    """
    radius_m = buffer_radius_m(spatial_extent_km)

    zonal = zonal_landcover_gee(lon, lat, radius_m)
    if zonal is not None:
        return {**zonal, "source": "ESA_WorldCover_10m", "radius_m": radius_m}

    try:
        from app.enrichment.landcover import sample_landcover_at_point

        sampled = sample_landcover_at_point("data/landcover/mcd12q1_india.tif", lon, lat)
        if sampled:
            return {**sampled, "source": "MODIS_MCD12Q1_500m", "radius_m": radius_m}
    except Exception:
        logger.warning("MODIS fallback sampling failed for (%s, %s)", lon, lat, exc_info=True)

    return {"source": "unavailable", "radius_m": radius_m}
