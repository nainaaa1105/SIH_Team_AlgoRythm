"""Population-density enrichment from WorldPop rasters (people/pixel at
~100m resolution for India). One-time/annual bulk download, sampled
per-cluster the same way as land cover.
"""
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Open datasets are cached per (path, thread). Opening a GeoTIFF parses
# its header and builds the block index; doing that once per cluster,
# across thousands of clusters, is pure overhead. rasterio datasets are
# not thread-safe, so the cache is thread-local rather than shared.
_local = threading.local()


def _dataset(raster_path: str):
    """Thread-local open dataset handle for `raster_path`."""
    import rasterio

    cache: Dict[str, object] = getattr(_local, "datasets", None)
    if cache is None:
        cache = {}
        _local.datasets = cache

    handle = cache.get(raster_path)
    if handle is None:
        handle = rasterio.open(raster_path)
        cache[raster_path] = handle
    return handle


def sample_population_at_point(raster_path: str, lon: float, lat: float) -> Optional[float]:
    """Return the WorldPop population-count value at the pixel containing
    (lon, lat), or None if unavailable.

    Reads a single-pixel window rather than the whole band. The previous
    implementation was `src.read(1)[row, col]`, which decompresses the
    ENTIRE raster to index one cell — for India's WorldPop grid that is
    35075 x 34497 float32 cells, about 4.8 GB, and it measured at 90-150
    seconds per call. It was the single slowest step in the pipeline by
    two orders of magnitude. A windowed read returns the same number in
    under a millisecond.
    """
    try:
        import rasterio  # noqa: F401  (import guarded here, used via _dataset)
        from rasterio.windows import Window
    except ImportError:
        logger.warning("rasterio not installed — skipping population sampling")
        return None

    try:
        src = _dataset(raster_path)
        row, col = src.index(lon, lat)
        # boundless so a point just outside the grid yields fill rather
        # than raising — the caller wants a value or None, not an error.
        data = src.read(1, window=Window(col, row, 1, 1), boundless=True, fill_value=0)
        value = data[0][0]
    except Exception:
        logger.exception("Population sampling failed for (%s, %s)", lon, lat)
        return None

    # WorldPop marks no-coverage cells with a large negative sentinel
    # (-99999). Anything negative is treated as "no people here" rather
    # than propagated, matching the original behaviour.
    return float(value) if value is not None and value >= 0 else 0.0
