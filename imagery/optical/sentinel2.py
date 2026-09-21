"""Sentinel-2 patch fetch.

This is the gap in the pipeline: M1's plan had a triggered optical fetch
for Day 5 but it was never implemented (`app/ingestion/` has FIRMS,
INSAT-3DS, Sentinel-3 and Himawari, no Sentinel-2), and M2's evidence
engine has been treating every imagery feature as unavailable ever since.
M4 owns it.

Primary path is M1's CDSE (Sentinel Hub) client
(`app/ingestion/cdse_client.py`) — real, verified-working credentials,
no GCP account needed. Earth Engine is kept as a second-tier fallback
(untouched below) purely in case a GEE service account is ever
configured later; it is not required and not currently used.

Patch size is taken from the cluster's measured footprint (M2's
`spatial_extent_km`) rather than a fixed box — the same trick M3 uses to
size its land-cover buffer. A flare stack and a 30 km fire front should
not get identically-framed imagery.
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from imagery.config import get_m4_settings
from imagery.optical.indices import SENTINEL2_BANDS

logger = logging.getLogger(__name__)

_METRES_PER_DEGREE_LAT = 111320.0


@dataclass
class PatchResult:
    bands: Dict[str, np.ndarray] = field(default_factory=dict)
    band_order: List[str] = field(default_factory=list)
    acquired_at: Optional[datetime] = None
    cloud_percentage: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # w, s, e, n
    width_px: int = 0
    height_px: int = 0
    source: str = ""
    available: bool = False
    reason: str = ""
    # Filled in by save_patch / save_thumbnail once written to disk.
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    @property
    def stack(self) -> Optional[np.ndarray]:
        """Bands as a (n_bands, H, W) array in `band_order`."""
        if not self.available or not self.band_order:
            return None
        return np.stack([self.bands[b] for b in self.band_order], axis=0)


def halfwidth_for_cluster(spatial_extent_km: Optional[float]) -> float:
    """Patch half-width in metres, sized to the event.

    Half the footprint plus a margin, so the surroundings are visible
    rather than the frame being filled edge-to-edge by the fire itself.
    """
    settings = get_m4_settings()
    if not spatial_extent_km or spatial_extent_km <= 0:
        return settings.s2_default_halfwidth_m
    radius_m = (spatial_extent_km * 1000.0) / 2.0
    return float(min(max(radius_m * 1.5, settings.s2_default_halfwidth_m), settings.s2_max_halfwidth_m))


def bbox_around(lon: float, lat: float, halfwidth_m: float) -> Tuple[float, float, float, float]:
    """(west, south, east, north) box of the given half-width."""
    d_lat = halfwidth_m / _METRES_PER_DEGREE_LAT
    cos_lat = math.cos(math.radians(lat))
    d_lon = halfwidth_m / (_METRES_PER_DEGREE_LAT * cos_lat) if abs(cos_lat) > 1e-9 else d_lat
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def _initialise_earth_engine() -> bool:
    """Authenticate to GEE using M1's configured service account."""
    try:
        import ee
    except ImportError:
        logger.info("earthengine-api not installed — Sentinel-2 fetch unavailable")
        return False

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
        return True
    except Exception:
        logger.warning("Earth Engine initialisation failed", exc_info=True)
        return False


def _fetch_patch_via_cdse(
    lon: float, lat: float, moment: datetime, box: Tuple[float, float, float, float], settings,
) -> PatchResult:
    """Real, verified-working path: M1's CDSE (Sentinel Hub) client.

    Two calls: a catalog search to learn the actual cloud cover and
    acquisition timestamp of the scene that will be used (Process API's
    mosaicking doesn't report this back directly), then Process API for
    the real band pixel data. CDSE's FLOAT32 output is already true
    reflectance (0-1) — unlike GEE's scaled-integer product, it must NOT
    be divided by `reflectance_scale`.
    """
    from app.ingestion import cdse_client

    if not cdse_client.is_configured():
        return PatchResult(available=False, bbox=box, reason="CDSE not configured")

    start = moment - timedelta(days=settings.s2_search_days)
    end = moment + timedelta(days=settings.s2_search_days)

    scene = cdse_client.search_least_cloudy_scene(
        list(box), start, end, collection=settings.s2_cdse_collection,
    )
    if scene is None:
        return PatchResult(
            available=False, bbox=box,
            reason=f"No Sentinel-2 scene found within +/-{settings.s2_search_days} days (CDSE)",
        )
    cloud_cover = scene.get("cloud_cover")
    if cloud_cover is not None and cloud_cover > settings.s2_max_cloud_percentage:
        return PatchResult(
            available=False, bbox=box,
            reason=f"Least-cloudy scene is {cloud_cover:.0f}% cloud, over the "
                   f"{settings.s2_max_cloud_percentage:.0f}% threshold",
        )

    width_px = height_px = 256  # fixed sample grid; halfwidth already sizes the bbox to the event
    stack = cdse_client.fetch_bands(
        bbox=list(box), bands=list(SENTINEL2_BANDS), start=start, end=end,
        width=width_px, height=height_px, collection=settings.s2_cdse_collection,
        max_cloud_coverage=settings.s2_max_cloud_percentage,
    )
    if stack is None:
        return PatchResult(available=False, bbox=box, reason="CDSE Process API fetch failed")

    bands: Dict[str, np.ndarray] = {
        band: stack[i] for i, band in enumerate(SENTINEL2_BANDS) if i < stack.shape[0]
    }
    if not bands:
        return PatchResult(available=False, bbox=box, reason="CDSE response had no band data")

    acquired_at = None
    if scene.get("datetime"):
        try:
            acquired_at = datetime.fromisoformat(scene["datetime"].replace("Z", "+00:00"))
        except ValueError:
            pass

    return PatchResult(
        bands=bands,
        band_order=[b for b in SENTINEL2_BANDS if b in bands],
        acquired_at=acquired_at,
        cloud_percentage=cloud_cover,
        bbox=box,
        height_px=height_px,
        width_px=width_px,
        source=f"CDSE_{settings.s2_cdse_collection}",
        available=True,
        reason="ok",
    )


def fetch_patch(
    lon: float,
    lat: float,
    at: Optional[datetime] = None,
    spatial_extent_km: Optional[float] = None,
) -> PatchResult:
    """Fetch the least-cloudy Sentinel-2 patch around a cluster.

    Never raises: an unavailable patch is a normal outcome (cloud, no
    overpass, no credentials) that M2's evidence-weighting engine already
    handles by discounting confidence. A crash here would take down the
    whole per-cluster task instead.

    Tries CDSE first (real, verified-working credentials — see module
    docstring), then Earth Engine only if a service account is ever
    configured.
    """
    settings = get_m4_settings()
    moment = at or datetime.now(timezone.utc)
    halfwidth = halfwidth_for_cluster(spatial_extent_km)
    box = bbox_around(lon, lat, halfwidth)

    cdse_result = _fetch_patch_via_cdse(lon, lat, moment, box, settings)
    if cdse_result.available:
        return cdse_result

    if not _initialise_earth_engine():
        return PatchResult(
            available=False, bbox=box,
            reason=f"CDSE unavailable ({cdse_result.reason}); Earth Engine unavailable or unconfigured",
        )

    try:
        import ee

        region = ee.Geometry.Rectangle([box[0], box[1], box[2], box[3]])
        start = (moment - timedelta(days=settings.s2_search_days)).strftime("%Y-%m-%d")
        end = (moment + timedelta(days=settings.s2_search_days)).strftime("%Y-%m-%d")

        collection = (
            ee.ImageCollection(settings.s2_collection)
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", settings.s2_max_cloud_percentage))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        if collection.size().getInfo() == 0:
            return PatchResult(
                available=False, bbox=box,
                reason=(
                    f"No Sentinel-2 scene under {settings.s2_max_cloud_percentage:.0f}% cloud "
                    f"within +/-{settings.s2_search_days} days"
                ),
            )

        image = collection.first()
        properties = image.toDictionary().getInfo() or {}
        cloud_percentage = properties.get("CLOUDY_PIXEL_PERCENTAGE")

        acquired_at = None
        timestamp_ms = properties.get("system:time_start")
        if timestamp_ms:
            acquired_at = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

        sample = image.select(list(SENTINEL2_BANDS)).sampleRectangle(
            region=region, defaultValue=0
        ).getInfo()
        arrays = sample.get("properties", {})

        bands: Dict[str, np.ndarray] = {}
        for band in SENTINEL2_BANDS:
            raw = arrays.get(band)
            if raw is None:
                continue
            # GEE ships L2A surface reflectance as scaled integers; divide
            # to real reflectance or every index and threshold is wrong by
            # four orders of magnitude.
            bands[band] = np.asarray(raw, dtype=float) / settings.reflectance_scale

        if not bands:
            return PatchResult(available=False, bbox=box, reason="Scene returned no band data")

        first = next(iter(bands.values()))
        return PatchResult(
            bands=bands,
            band_order=[b for b in SENTINEL2_BANDS if b in bands],
            acquired_at=acquired_at,
            cloud_percentage=cloud_percentage,
            bbox=box,
            height_px=int(first.shape[0]),
            width_px=int(first.shape[1]),
            source=f"GEE_{settings.s2_collection.split('/')[-1]}",
            available=True,
            reason="ok",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentinel-2 fetch failed at (%s, %s)", lon, lat, exc_info=True)
        return PatchResult(available=False, bbox=box, reason=f"fetch failed: {exc}")


def save_patch(patch: PatchResult, cluster_id: int, directory: Optional[str] = None) -> Optional[str]:
    """Persist the band stack as a compressed .npz. Returns the path."""
    if not patch.available:
        return None

    settings = get_m4_settings()
    target_dir = Path(directory or settings.s2_patch_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = (patch.acquired_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    path = target_dir / f"cluster_{cluster_id}_{stamp}.npz"
    np.savez_compressed(path, **patch.bands)
    return str(path)


def load_patch(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_thumbnail(
    patch: PatchResult, cluster_id: int, directory: Optional[str] = None, stretch_percentile: float = 98.0
) -> Optional[str]:
    """Write an RGB PNG preview for M6's event card.

    Surface reflectance over land rarely exceeds ~0.3, so a raw 0-1 scale
    renders almost black. A percentile stretch is what makes the preview
    actually legible to a human.
    """
    if not patch.available:
        return None
    if not all(band in patch.bands for band in ("B4", "B3", "B2")):
        return None

    try:
        from PIL import Image
    except ImportError:
        logger.info("Pillow not installed — skipping thumbnail")
        return None

    settings = get_m4_settings()
    target_dir = Path(directory or settings.s2_thumbnail_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.stack([patch.bands["B4"], patch.bands["B3"], patch.bands["B2"]], axis=-1)
    ceiling = np.percentile(rgb, stretch_percentile)
    if ceiling <= 0:
        ceiling = 1.0
    scaled = np.clip(rgb / ceiling, 0.0, 1.0)

    stamp = (patch.acquired_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    path = target_dir / f"cluster_{cluster_id}_{stamp}.png"
    Image.fromarray((scaled * 255).astype(np.uint8)).save(path)
    return str(path)
