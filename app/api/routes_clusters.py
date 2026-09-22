from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import ClusterList, ClusterOut
from app.db.models import Cluster
from app.db.session import get_db

router = APIRouter(prefix="/clusters", tags=["clusters"])


def _to_cluster_out(row: Cluster) -> ClusterOut:
    point = to_shape(row.centroid)
    return ClusterOut(
        id=row.id, lon=point.x, lat=point.y, first_seen=row.first_seen, last_seen=row.last_seen,
        n_detections=row.n_detections, cloud_fraction=row.cloud_fraction,
        optical_available=row.optical_available, status=row.status,
        wui_threat=bool(getattr(row, "wui_threat", False)),
        wui_distance_m=getattr(row, "wui_distance_m", None),
        wui_eta_hours=getattr(row, "wui_eta_hours", None),
        wui_bearing_deg=getattr(row, "wui_bearing_deg", None),
        wui_threatened_asset=getattr(row, "wui_threatened_asset", None),
        is_crown_fire=bool(getattr(row, "is_crown_fire", False)),
        rdi_score=getattr(row, "rdi_score", None),
        coa_type=getattr(row, "coa_type", None),
    )


@router.get("", response_model=ClusterList)
def list_clusters(
    status: str | None = Query(None),
    limit: int = Query(500, le=5000),
    db: Session = Depends(get_db),
):
    stmt = select(Cluster)
    if status is not None:
        stmt = stmt.where(Cluster.status == status)
    stmt = stmt.order_by(Cluster.last_seen.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    items = [_to_cluster_out(r) for r in rows]
    return ClusterList(count=len(items), items=items)


@router.get("/{cluster_id}", response_model=ClusterOut)
def get_cluster(cluster_id: int, db: Session = Depends(get_db)):
    row = db.get(Cluster, cluster_id)
    if row is None:
        raise HTTPException(status_code=404, detail="cluster not found")
    return _to_cluster_out(row)
