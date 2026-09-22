from typing import Optional

from fastapi import APIRouter, Depends, Query
from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import FacilityList, FacilityOut
from app.db.models import Facility
from app.db.session import get_db

router = APIRouter(prefix="/facilities", tags=["facilities"])


@router.get("", response_model=FacilityList)
def list_facilities(
    facility_type: Optional[str] = Query(None),
    limit: int = Query(1000, le=10000),
    db: Session = Depends(get_db),
):
    stmt = select(Facility)
    if facility_type is not None:
        stmt = stmt.where(Facility.facility_type == facility_type)
    stmt = stmt.limit(limit)
    rows = db.execute(stmt).scalars().all()

    items = []
    for row in rows:
        shape = to_shape(row.geom)
        centroid = shape.centroid if shape.geom_type != "Point" else shape
        items.append(
            FacilityOut(
                id=row.id, name=row.name, facility_type=row.facility_type, source=row.source,
                lon=centroid.x, lat=centroid.y, prior_weight=row.prior_weight,
            )
        )
    return FacilityList(count=len(items), items=items)
