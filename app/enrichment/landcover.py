"""Land-cover context enrichment: samples % cropland/forest/urban around a
cluster centroid from a MODIS MCD12Q1 (IGBP classification) raster.

MCD12Q1 is a yearly global 500m land-cover product. Download once (see
scripts/bulk_load_landcover.py) via Google Earth Engine or NASA Earthdata
and store locally/in object storage — this is a one-time/annual-refresh
base layer, not a live feed, per the architecture ("Layer 6: land-cover
context" is static reference data, unlike the real-time hotspot feeds).

IGBP class codes relevant here (MCD12Q1 LC_Type1):
    1-5, 8-9   = various forest/savanna/woody types -> "forest"
    10         = grassland
    12, 14     = cropland / cropland-natural mosaic -> "cropland"
    13         = urban and built-up -> "urban"
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_FOREST_CODES = {1, 2, 3, 4, 5, 8, 9}
_CROPLAND_CODES = {12, 14}
_URBAN_CODES = {13}


def classify_pixel(igbp_code: int) -> str:
    if igbp_code in _FOREST_CODES:
        return "forest"
    if igbp_code in _CROPLAND_CODES:
        return "cropland"
    if igbp_code in _URBAN_CODES:
        return "urban"
    return "other"


def summarize_landcover_window(igbp_codes: np.ndarray) -> Dict[str, float]:
    """Given a 2D (or flattened) array of IGBP codes sampled from a window
    around a point (e.g. a 3x3 or 5x5 pixel kernel at 500m resolution ~=
    1.5-2.5km around the centroid), return pct_forest/pct_cropland/pct_urban.
    """
    flat = igbp_codes.flatten()
    total = flat.size
    if total == 0:
        return {"pct_forest": 0.0, "pct_cropland": 0.0, "pct_urban": 0.0}

    classes = np.vectorize(classify_pixel)(flat)
    return {
        "pct_forest": float(np.sum(classes == "forest")) / total,
        "pct_cropland": float(np.sum(classes == "cropland")) / total,
        "pct_urban": float(np.sum(classes == "urban")) / total,
    }


def sample_landcover_at_point(
    raster_path: str, lon: float, lat: float, window_pixels: int = 5
) -> Optional[Dict[str, float]]:
    """Sample a window_pixels x window_pixels kernel centred on (lon, lat)
    from a local MCD12Q1 GeoTIFF and return land-cover percentages.
    Returns None if rasterio isn't installed or the point is outside the
    raster's extent (e.g. raster only covers a subset of India).
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        logger.warning("rasterio not installed — skipping land-cover sampling")
        return None

    try:
        with rasterio.open(raster_path) as src:
            row, col = src.index(lon, lat)
            half = window_pixels // 2
            window = Window(col - half, row - half, window_pixels, window_pixels)
            data = src.read(1, window=window, boundless=True, fill_value=0)
    except Exception:
        logger.exception("Land-cover sampling failed for (%s, %s)", lon, lat)
        return None

    return summarize_landcover_window(data)
