"""Population exposure inside a hazard polygon.

M1 samples WorldPop at a single point (`app/enrichment/population.py`).
For a plume you need the *sum* over an area, which is a different
operation — a zonal sum, masked by the polygon.

WorldPop's standard India product is population count per pixel, so the
sum over a masked region is a headcount directly. That matters: if the
raster were population *density* the pixel values would need multiplying
by pixel area first, and silently mixing the two is an easy way to be
wrong by orders of magnitude. `raster_is_density` makes the assumption
explicit rather than implicit.
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def polygon_area_km2(ring: Sequence[Tuple[float, float]]) -> float:
    """Approximate area of a small (lon, lat) ring, in km^2.

    Uses the planar shoelace formula on degrees scaled to metres at the
    ring's mean latitude. Fine for plume-sized polygons; not for
    anything continental.
    """
    import math

    if len(ring) < 3:
        return 0.0

    mean_lat = sum(lat for _, lat in ring) / len(ring)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))

    points = list(ring)
    if points[0] != points[-1]:
        points.append(points[0])

    area2 = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(points, points[1:]):
        x1, y1 = lon1 * m_per_deg_lon, lat1 * m_per_deg_lat
        x2, y2 = lon2 * m_per_deg_lon, lat2 * m_per_deg_lat
        area2 += x1 * y2 - x2 * y1

    return abs(area2) / 2.0 / 1_000_000.0


def population_in_polygon(
    raster_path: str,
    ring: Sequence[Tuple[float, float]],
    raster_is_density: bool = False,
) -> Optional[int]:
    """Total population inside a (lon, lat) ring, or None if unavailable.

    Returns None rather than 0 when the raster or rasterio is missing —
    "we don't know" and "nobody lives there" must not look the same to a
    responder.
    """
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
    except ImportError:
        logger.info("rasterio not installed — population exposure unavailable")
        return None

    geometry = {"type": "Polygon", "coordinates": [[[lon, lat] for lon, lat in ring]]}

    try:
        with rasterio.open(raster_path) as src:
            data, transform = rio_mask(src, [geometry], crop=True, filled=True, nodata=0)
            nodata = src.nodata
    except Exception:
        logger.warning("Population zonal sum failed for %s", raster_path, exc_info=True)
        return None

    values = data[0]
    # WorldPop encodes "no data" as a large negative sentinel; summing it
    # blindly produces a hugely negative headcount.
    mask = values > 0
    if nodata is not None:
        mask &= values != nodata

    total = float(values[mask].sum()) if mask.any() else 0.0

    if raster_is_density:
        total *= polygon_area_km2(ring)

    return int(round(total))


def exposure_summary(
    population: Optional[int], area_km2: float, hazard_level: str = "unknown"
) -> Dict[str, object]:
    """Package exposure for the dashboard/alerting layer."""
    return {
        "population_exposed": population,
        "population_known": population is not None,
        "area_km2": round(area_km2, 3),
        "population_density_per_km2": (
            round(population / area_km2, 1) if population is not None and area_km2 > 0 else None
        ),
        "hazard_level": hazard_level,
    }
