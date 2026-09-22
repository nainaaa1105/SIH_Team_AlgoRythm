"""Himawari-8/9 ingestion — SECONDARY / OPTIONAL source.

Coverage caveat: Himawari's geostationary full-disk view is centred near
140.7°E, so India (68-97.5°E) sits at the far western edge of its scan
with a high, distorting viewing angle. This source is only wired up for
the north-east India / Myanmar-border region (see HIMAWARI_REGION_BBOX in
.env), as a redundant cross-check there — it is deliberately NOT used as
a primary feed for the rest of India (INSAT-3DS and FIRMS cover that
better; see member1_data_ingestion_spec.md).

Product caveat: raw Himawari AHI L1 brightness-temperature data (as
mirrored on NOAA's public `noaa-himawari8` S3 bucket, no auth required)
is NOT itself a fire product — it must be run through a fire-detection
algorithm to produce hotspots. JAXA's Himawari Monitor / P-Tree system
distributes a derived experimental wildfire product for parts of
Asia-Oceania, but that requires separate JAXA registration and the exact
delivery path should be confirmed there before this becomes a real feed.

Until one of those is confirmed and wired in, `fetch_all()` below returns
an empty list rather than fabricating detections — this keeps the module
truthful about its current state rather than silently no-op-ing behind a
misleading success path.
"""
import logging
from typing import Dict, List

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def list_available_scenes(settings: Settings | None = None, prefix: str = "AHI-L1b-FLDK") -> List[str]:
    """List recent full-disk scene keys in NOAA's public Himawari bucket.

    This confirms connectivity/bucket access, useful as a smoke test, but
    listing raw L1 scenes is not the same as detecting fire in them — see
    module docstring.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        logger.warning("boto3 not installed — skipping Himawari bucket listing")
        return []

    settings = settings or get_settings()
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    try:
        resp = s3.list_objects_v2(Bucket=settings.himawari_s3_bucket, Prefix=prefix, MaxKeys=50)
    except Exception:
        logger.exception("Failed to list Himawari S3 bucket %s", settings.himawari_s3_bucket)
        return []
    return [obj["Key"] for obj in resp.get("Contents", [])]


def fetch_all(settings: Settings | None = None) -> List[Dict]:
    """Himawari fire-detection ingestion. Returns [] until a real
    fire-detection product/algorithm is wired in (see module docstring).
    Kept as a distinct pipeline stage (rather than deleted) so the
    scheduler wiring and NE-India bbox filtering are ready to go the
    moment a real product is confirmed.
    """
    settings = settings or get_settings()
    scenes = list_available_scenes(settings)
    if scenes:
        logger.info(
            "Himawari bucket reachable (%d recent scenes found) but no fire-detection "
            "algorithm is wired in yet — yielding 0 records this cycle", len(scenes)
        )
    else:
        logger.info("Himawari source not currently reachable/configured — yielding 0 records")
    return []
