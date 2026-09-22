"""Reads of M1's tables that M5 needs.

Like M3 and M4, M5 does not import another member's internal loaders —
the query is short, and reaching into `classifier` or `imagery` internals
would couple the packages for no benefit.
"""
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session


def load_detection_rows(session: Session, cluster_id: int) -> List[Dict[str, Any]]:
    """A cluster's detections, oldest first.

    `lon` is needed for local-solar-hour computation in the overpass
    slot model; `source` and `daynight` define the slots themselves.
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
            "brightness": h.brightness,
            "confidence": h.confidence,
            "daynight": h.daynight,
            "source": h.source,
        })
    return rows


def hours_from_first(rows: List[Dict[str, Any]]) -> List[tuple]:
    """Convert detections into `(hours_since_first, frp)` pairs for the filter.

    The Kalman filter works in elapsed hours; only differences matter, so
    the origin is arbitrary. Rows without an FRP are dropped — the filter
    measures FRP, and a detection with no FRP carries no information for
    it (it still counts for the rhythm fingerprint, which is why the
    filtering happens here rather than in the loader).
    """
    usable = [
        r for r in rows
        if r.get("acq_datetime") is not None and r.get("frp") is not None
    ]
    if not usable:
        return []

    origin = min(r["acq_datetime"] for r in usable)
    return [
        ((r["acq_datetime"] - origin).total_seconds() / 3600.0, float(r["frp"]))
        for r in sorted(usable, key=lambda r: r["acq_datetime"])
    ]


def read_predicted_class(session: Session, cluster_id: int) -> Optional[str]:
    """M2's verdict, if it has classified this cluster yet.

    Used only to colour the PTSI summary; M5 works fine without it, and
    returns None when M2 is not installed.
    """
    try:
        from classifier.db.models import Classification
    except ImportError:
        return None

    row = session.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    return row.predicted_class if row else None
