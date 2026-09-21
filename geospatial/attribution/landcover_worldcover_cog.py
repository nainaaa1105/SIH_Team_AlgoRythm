"""ESA WorldCover 10 m land cover via the public AWS Open Data bucket —
a real, keyless replacement for the Google Earth Engine zonal-stats path
in `landcover_zonal.py`.

Why this exists: `zonal_landcover_gee()` needs a GCP service account
(`gee_service_account` / `gee_private_key_file`), which this project does
not have and cannot get quickly. ESA WorldCover v200 is mirrored as
public Cloud-Optimized GeoTIFFs on S3 with anonymous HTTPS access — no
AWS account, no GCP account, no API key of any kind:

    https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/
        ESA_WorldCover_10m_2021_v200_{tile}_Map.tif

Tiles are 3x3 degree, named by their south-west corner, e.g. "S48E036"
covers 36-39 E, 48-45 S (confirmed against the WorldCover product user
manual). `worldcover_tile_id()` below reproduces that scheme.

A windowed read via GDAL's `/vsicurl/` virtual filesystem pulls only the
bytes covering the sampling buffer (a few dozen KB), not the whole tile
(~200-600 MB), because the file is a Cloud-Optimized GeoTIFF with
internal tiling + overviews. Same `rasterio` optional dependency this
package already carries for population exposure — no new dependency.

Limitation stated plainly rather than hidden: a sampling buffer that
straddles a 3x3 degree tile boundary only reads the tile containing the
centroid, so context near a boundary can miss coverage on the far side.
At the buffer sizes this project uses (500 m - 20 km, see
`landcover_zonal.buffer_radius_m`) that only matters within ~20 km of a
tile edge, and it fails safe: the missed area is simply absent from the
histogram rather than silently wrong.
"""
import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)

WORLDCOVER_BUCKET_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_VERSION = "v200"
WORLDCOVER_YEAR = 2021  # v200's only vintage; v100/2020 exists but is superseded
TILE_SIZE_DEG = 3
PIXEL_SIZE_DEG = 8.333333333333333e-05  # 10 m at the equator, ~1/3 arc-second


def worldcover_tile_id(lat: float, lon: float) -> str:
    """3x3 degree tile name for a point, e.g. (28.6, 77.2) -> "N27E075".

    Tiles are named by their south-west corner. Matches the scheme WVCC
    documents: "S48E036 covers 36E-39E and 48S-45S".
    """
    tile_lat = int(math.floor(lat / TILE_SIZE_DEG)) * TILE_SIZE_DEG
    tile_lon = int(math.floor(lon / TILE_SIZE_DEG)) * TILE_SIZE_DEG
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    return f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"


def worldcover_cog_url(lat: float, lon: float) -> str:
    tile = worldcover_tile_id(lat, lon)
    filename = f"ESA_WorldCover_10m_{WORLDCOVER_YEAR}_{WORLDCOVER_VERSION}_{tile}_Map.tif"
    return f"{WORLDCOVER_BUCKET_URL}/{WORLDCOVER_VERSION}/{WORLDCOVER_YEAR}/map/{filename}"


def zonal_landcover_cog(lon: float, lat: float, radius_m: float) -> Optional[Dict[str, float]]:
    """Zonal land-cover class histogram from the public WorldCover COG.

    Returns None (never raises) when rasterio is missing, the point's
    tile doesn't exist (open ocean far from any land tile), or the
    network read fails — so the caller can fall back to GEE (if ever
    configured) or M1's local MODIS raster.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        logger.info("rasterio not installed — WorldCover COG sampling unavailable")
        return None

    # Cached on disk: each miss is a windowed HTTPS read of a remote COG
    # (~2-5 s). WorldCover v200 is a fixed 2021 product, so a cached
    # histogram stays correct — only the cache's own TTL expires it.
    from app import geo_cache

    cache_key = geo_cache.make_key("landcover", lon, lat, int(radius_m))
    cached = geo_cache.get(cache_key)
    if cached is not geo_cache.MISS:
        return cached

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat)) or 1e-6
    half_lat = radius_m / m_per_deg_lat
    half_lon = radius_m / m_per_deg_lon

    url = f"/vsicurl/{worldcover_cog_url(lat, lon)}"
    try:
        with rasterio.open(url) as src:
            window = from_bounds(
                lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat,
                transform=src.transform,
            )
            data = src.read(1, window=window, boundless=True, fill_value=0)
    except Exception:
        logger.warning(
            "WorldCover COG read failed for (%s, %s) via %s — tile may not exist "
            "(open ocean) or the network read timed out", lon, lat, url, exc_info=True,
        )
        # Deliberately NOT cached: a network failure is transient, and
        # remembering it would freeze a recoverable error into an answer.
        return None

    values, counts = np.unique(data[data != 0], return_counts=True)
    if values.size == 0:
        # A real result ("this window is entirely no-data"), so cache it.
        geo_cache.put(cache_key, None)
        return None

    from geospatial.attribution.landcover_zonal import summarise_worldcover

    summary = summarise_worldcover({int(v): int(c) for v, c in zip(values, counts)})
    geo_cache.put(cache_key, summary)
    return summary


def nearest_builtup_pixel_cog(
    lon: float, lat: float, search_radius_m: float, target_class: int = 50
) -> Optional[Dict[str, float]]:
    """Distance and bearing to the nearest built-up (WorldCover class 50,
    "urban" in `landcover_zonal.WORLDCOVER_CLASSES`) pixel within
    `search_radius_m` -- the wildland-urban interface boundary as it
    actually exists in a real 2021 10 m satellite land-cover product,
    not an assumed or looked-up settlement point.

    Same COG, same windowed-read machinery as `zonal_landcover_cog`
    above; this reads the raw class array instead of a histogram and
    finds the closest matching pixel by real haversine distance. Returns
    None (never raises) on the same conditions: rasterio missing, no
    tile at this point (open ocean), a network failure, or genuinely no
    built-up pixel within the search radius -- the last of those is a
    real result and is cached as such, same as the histogram path.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        logger.info("rasterio not installed — WUI built-up search unavailable")
        return None

    from app import geo_cache

    cache_key = geo_cache.make_key("wui_builtup", lon, lat, int(search_radius_m))
    cached = geo_cache.get(cache_key)
    if cached is not geo_cache.MISS:
        return cached

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat)) or 1e-6

    from geospatial.geometry import EARTH_RADIUS_M, bearing_deg

    def _read_window(radius_m: float):
        """One windowed read, decimated so the pixel count stays bounded
        regardless of how built-up the area is. A dense city block has
        as many class-50 pixels in a 1 km box as a sparse village has in
        10 km, so what actually needs bounding is *pixel count*, not
        just the geographic radius — decimating by the window's own
        linear size keeps both cases fast.
        """
        half_lat = radius_m / m_per_deg_lat
        half_lon = radius_m / m_per_deg_lon
        url = f"/vsicurl/{worldcover_cog_url(lat, lon)}"
        with rasterio.open(url) as src:
            window = from_bounds(
                lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat,
                transform=src.transform,
            )
            # Full 10 m resolution up to ~1500x1500 px (~1.5 km radius);
            # beyond that, read decimated so a 10 km radius (which would
            # be ~2000x2000 px at full res) still comes back as at most
            # ~1500x1500 — worst case a dense city block, still sub-second.
            out_side = min(int(window.width), int(window.height), 1500)
            out_side = max(out_side, 32)
            data = src.read(
                1, window=window, boundless=True, fill_value=0,
                out_shape=(out_side, out_side),
            )
            # window_transform describes the FULL-resolution window; the
            # decimated read needs its own transform scaled to match, or
            # every pixel centre computed below would be wrong by the
            # decimation factor.
            base_transform = src.window_transform(window)
            scale_x = window.width / out_side
            scale_y = window.height / out_side
            read_transform = base_transform * base_transform.scale(scale_x, scale_y)
        return data, read_transform

    try:
        # Start close in: most detections that matter for a WUI check
        # are already near *something* built-up (India's rural landscape
        # is dense with villages), so the common case resolves on the
        # first, cheapest read. Only a genuinely remote fire pays for the
        # larger radii.
        data = None
        read_transform = None
        for radius_m in (min(800.0, search_radius_m), search_radius_m):
            data, read_transform = _read_window(radius_m)
            if np.any(data == target_class):
                break
            if radius_m >= search_radius_m:
                break
    except Exception:
        logger.warning(
            "WorldCover COG built-up search failed for (%s, %s) — tile may "
            "not exist (open ocean) or the network read timed out", lon, lat,
            exc_info=True,
        )
        # Not cached, same reasoning as zonal_landcover_cog: a network
        # failure is transient and must not freeze into a permanent
        # "nothing nearby" answer.
        return None

    rows, cols = np.where(data == target_class)
    if rows.size == 0:
        geo_cache.put(cache_key, None)
        return None

    xs, ys = rasterio.transform.xy(read_transform, rows, cols)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    # Vectorised haversine (same formula/constants as geometry.haversine_m,
    # just over the whole pixel array at once) — a dense urban tile can
    # still match well over a thousand decimated pixels, and a
    # Python-level loop over each one would be the slow part of an
    # otherwise sub-second read.
    phi1 = math.radians(lat)
    phi2 = np.radians(ys)
    dphi = phi2 - phi1
    dlambda = np.radians(xs) - math.radians(lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    distances_m = EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    nearest_idx = int(np.argmin(distances_m))
    nearest_lon, nearest_lat = float(xs[nearest_idx]), float(ys[nearest_idx])
    distance_m = float(distances_m[nearest_idx])

    result = {
        "distance_m": distance_m,
        "lon": nearest_lon,
        "lat": nearest_lat,
        "bearing_deg": bearing_deg(lon, lat, nearest_lon, nearest_lat),
        "source": "ESA_WorldCover_10m_COG",
    }
    geo_cache.put(cache_key, result)
    return result


def landcover_with_fallback_cog(
    lon: float, lat: float, spatial_extent_km: Optional[float] = None
) -> Dict[str, object]:
    """Drop-in replacement for `landcover_zonal.landcover_with_fallback`
    that tries the keyless COG path before GEE and the local MODIS raster.

    Kept as a separate function (rather than editing the GEE tier in
    place) so `landcover_zonal.zonal_landcover_gee` still works untouched
    if a GEE service account is configured later — no functionality was
    removed, only reordered so the working path runs first.
    """
    from geospatial.attribution.landcover_zonal import buffer_radius_m, zonal_landcover_gee

    radius_m = buffer_radius_m(spatial_extent_km)

    cog = zonal_landcover_cog(lon, lat, radius_m)
    if cog is not None:
        return {**cog, "source": "ESA_WorldCover_10m_COG", "radius_m": radius_m}

    gee = zonal_landcover_gee(lon, lat, radius_m)
    if gee is not None:
        return {**gee, "source": "ESA_WorldCover_10m_GEE", "radius_m": radius_m}

    try:
        from app.enrichment.landcover import sample_landcover_at_point

        sampled = sample_landcover_at_point("data/landcover/mcd12q1_india.tif", lon, lat)
        if sampled:
            return {**sampled, "source": "MODIS_MCD12Q1_500m", "radius_m": radius_m}
    except Exception:
        logger.warning("MODIS fallback sampling failed for (%s, %s)", lon, lat, exc_info=True)

    return {"source": "unavailable", "radius_m": radius_m}
