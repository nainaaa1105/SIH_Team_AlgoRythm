from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import HotspotList, HotspotOut
from app.db.models import Hotspot
from app.db.session import get_db

router = APIRouter(prefix="/hotspots", tags=["hotspots"])


@router.get("", response_model=HotspotList)
def list_hotspots(
    min_lon: Optional[float] = Query(None),
    min_lat: Optional[float] = Query(None),
    max_lon: Optional[float] = Query(None),
    max_lat: Optional[float] = Query(None),
    since: Optional[datetime] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(500, le=5000),
    db: Session = Depends(get_db),
):
    stmt = select(Hotspot)
    if since is not None:
        stmt = stmt.where(Hotspot.acq_datetime >= since)
    if source is not None:
        stmt = stmt.where(Hotspot.source == source)
    if None not in (min_lon, min_lat, max_lon, max_lat):
        stmt = stmt.where(
            Hotspot.geom.ST_Within(
                f"SRID=4326;POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
                f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
            )
        )
    stmt = stmt.order_by(Hotspot.acq_datetime.desc()).limit(limit)

    rows = db.execute(stmt).scalars().all()
    items = []
    for row in rows:
        point = to_shape(row.geom)
        items.append(
            HotspotOut(
                id=row.id, source=row.source, lon=point.x, lat=point.y,
                acq_datetime=row.acq_datetime, frp=row.frp, brightness=row.brightness,
                confidence=row.confidence, daynight=row.daynight, cluster_id=row.cluster_id,
            )
        )
    return HotspotList(count=len(items), items=items)
