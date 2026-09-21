"""Reads of M1's tables that M3 needs.

M2 has a similar `load_hotspot_rows` in `classifier/features/assemble.py`,
but M3 deliberately does not import it: reaching into another member's
internals couples the two packages for no benefit, and M3 must keep
working if M2 isn't installed. The query is four lines.
"""
from typing import Any, Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session


def load_hotspot_rows(session: Session, cluster_id: int) -> List[Dict[str, Any]]:
    """A cluster's detections as plain dicts, for the pure geometry code."""
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
