"""Reads of M1's tables that M4 needs.

The important one is `load_dual_band_rows`. Dozier needs both the
mid-infrared and thermal-infrared brightness temperatures, but M1's
`hotspots` table has a single `brightness` column — it stores
`bright_ti4` (VIIRS) or `brightness` (MODIS) there.

The second band is not lost, though: M1's normaliser keeps the entire
original FIRMS CSV row in `hotspots.raw_payload`, so `bright_ti5` /
`bright_t31` is available from there. That means Dozier needs no schema
change to M1's busiest table — which is why this reads from raw_payload
rather than proposing a migration.
"""
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Per-sensor field names for the thermal (long-wave) band inside raw_payload.
_THERMAL_BAND_KEYS = ("bright_ti5", "bright_t31")


def thermal_band_from_payload(raw_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    """Pull the long-wave brightness temperature out of a FIRMS row."""
    if not raw_payload:
        return None
    for key in _THERMAL_BAND_KEYS:
        value = raw_payload.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_dual_band_rows(session: Session, cluster_id: int) -> List[Dict[str, Any]]:
    """A cluster's detections with both brightness bands where available.

    `brightness_mir` is M1's `brightness` column (bright_ti4 / MODIS
    band-21); `brightness_tir` is recovered from raw_payload.
    """
    from geoalchemy2.shape import to_shape

    from app.db.models import Hotspot

    hotspots = session.execute(
        select(Hotspot).where(Hotspot.cluster_id == cluster_id).order_by(Hotspot.acq_datetime)
    ).scalars().all()

    rows = []
    for h in hotspots:
        point = to_shape(h.geom)
        rows.append({
            "lon": point.x,
            "lat": point.y,
            "acq_datetime": h.acq_datetime,
            "frp": h.frp,
            "brightness_mir": h.brightness,
            "brightness_tir": thermal_band_from_payload(h.raw_payload),
            "confidence": h.confidence,
            "daynight": h.daynight,
            "scan": h.scan,
            "track": h.track,
            "source": h.source,
        })
    return rows


def dual_band_rows_only(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detections usable for Dozier: both bands present, from a dual-band sensor.

    INSAT-3DS and Sentinel-3 FRP detections are excluded — those feeds
    carry an FRP or a fire flag, not the two brightness temperatures the
    method inverts.
    """
    usable = []
    for row in rows:
        source = (row.get("source") or "").upper()
        if not ("VIIRS" in source or "MODIS" in source):
            continue
        if row.get("brightness_mir") is None or row.get("brightness_tir") is None:
            continue
        usable.append(row)
    return usable


def read_spatial_extent_km(session: Session, cluster_id: int) -> Optional[float]:
    """M2's footprint measurement, used to size the Sentinel-2 patch.

    Returns None if M2 isn't installed or hasn't run for this cluster;
    the fetcher then uses its default half-width.
    """
    try:
        from classifier.db.models import ClusterFeatures
    except ImportError:
        return None

    row = session.get(ClusterFeatures, cluster_id)
    return getattr(row, "spatial_extent_km", None) if row else None


def read_modelled_plume_bearing(session: Session, cluster_id: int) -> Optional[float]:
    """M3's modelled downwind bearing, for the smoke cross-check.

    Returns None if M3 isn't installed or hasn't modelled this cluster —
    the cross-check then reports "not comparable" rather than a
    disagreement.
    """
    try:
        from geospatial.db.models import Plume
    except ImportError:
        return None

    plume = session.get(Plume, cluster_id)
    return getattr(plume, "downwind_bearing_deg", None) if plume else None
