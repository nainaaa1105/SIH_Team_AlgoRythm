"""Imagery read API.

Mounts onto M1's FastAPI app alongside M2's and M3's routers:

    # app/main.py
    from imagery.api.routes_imagery import router as imagery_router
    app.include_router(imagery_router)

Serves the event card's image evidence: the sub-pixel fire temperature,
what the image model alone concluded, and whether the observed smoke
direction agrees with M3's wind-driven plume model.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(prefix="/imagery", tags=["imagery"])


class ThermalOut(BaseModel):
    cluster_id: int
    fire_temperature_k: Optional[float]
    fire_temperature_c: Optional[float]
    fire_fraction: Optional[float]
    fire_area_m2: Optional[float]
    background_temperature_k: Optional[float]
    background_source: Optional[str]
    sensor: Optional[str]
    converged: bool
    reason: Optional[str]
    temperature_low_k: Optional[float]
    temperature_high_k: Optional[float]
    well_constrained: Optional[bool]


class PatchOut(BaseModel):
    id: int
    acquired_at: Optional[str]
    cloud_percentage: Optional[float]
    thumbnail_path: Optional[str]
    bands: Optional[List[str]]
    width_px: Optional[int]
    height_px: Optional[int]
    source: Optional[str]


class ImageryOut(BaseModel):
    cluster_id: int
    thermal: Optional[ThermalOut]
    image_predicted_class: Optional[str]
    image_class_probabilities: Optional[Dict[str, float]]
    image_confidence: Optional[float]
    image_model_version: Optional[str]
    smoke_detected: bool
    smoke_bearing_deg: Optional[float]
    smoke_coverage: Optional[float]
    plume_agreement: Optional[Dict[str, Any]]
    latest_patch: Optional[PatchOut]


def _thermal_out(row) -> Optional[ThermalOut]:
    if row is None:
        return None
    return ThermalOut(
        cluster_id=row.cluster_id,
        fire_temperature_k=row.fire_temperature_k,
        # Responders think in Celsius; converting here keeps the
        # arithmetic out of the frontend.
        fire_temperature_c=(
            round(row.fire_temperature_k - 273.15, 1)
            if row.fire_temperature_k is not None else None
        ),
        fire_fraction=row.fire_fraction,
        fire_area_m2=row.fire_area_m2,
        background_temperature_k=row.background_temperature_k,
        background_source=row.background_source,
        sensor=row.sensor,
        converged=bool(row.converged),
        reason=row.reason,
        # Surfaced, not buried: for a small hot source the bracket can be
        # hundreds of kelvin wide, and the UI must be able to say so
        # rather than showing a falsely precise single figure.
        temperature_low_k=row.temperature_low_k,
        temperature_high_k=row.temperature_high_k,
        well_constrained=row.well_constrained,
    )


@router.get("/{cluster_id}", response_model=ImageryOut)
def get_imagery(cluster_id: int, db: Session = Depends(get_db)):
    from imagery.db.models import ImagePrediction, Sentinel2Patch, ThermalRetrieval

    thermal = db.get(ThermalRetrieval, cluster_id)
    prediction = db.get(ImagePrediction, cluster_id)

    patch_row = db.execute(
        select(Sentinel2Patch)
        .where(Sentinel2Patch.cluster_id == cluster_id)
        .order_by(Sentinel2Patch.acquired_at.desc())
        .limit(1)
    ).scalars().first()

    if thermal is None and prediction is None and patch_row is None:
        raise HTTPException(status_code=404, detail="no imagery analysis for this cluster")

    latest_patch = None
    if patch_row is not None:
        latest_patch = PatchOut(
            id=patch_row.id,
            acquired_at=patch_row.acquired_at.isoformat() if patch_row.acquired_at else None,
            cloud_percentage=patch_row.cloud_percentage,
            thumbnail_path=patch_row.thumbnail_path,
            bands=patch_row.bands,
            width_px=patch_row.width_px,
            height_px=patch_row.height_px,
            source=patch_row.source,
        )

    return ImageryOut(
        cluster_id=cluster_id,
        thermal=_thermal_out(thermal),
        image_predicted_class=prediction.predicted_class if prediction else None,
        image_class_probabilities=prediction.class_probabilities if prediction else None,
        image_confidence=prediction.confidence if prediction else None,
        image_model_version=prediction.model_version if prediction else None,
        smoke_detected=bool(prediction.smoke_detected) if prediction else False,
        smoke_bearing_deg=prediction.smoke_bearing_deg if prediction else None,
        smoke_coverage=prediction.smoke_coverage if prediction else None,
        plume_agreement=prediction.plume_agreement if prediction else None,
        latest_patch=latest_patch,
    )


@router.get("/{cluster_id}/patches", response_model=List[PatchOut])
def list_patches(cluster_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """Every patch fetched for a cluster, newest first.

    Sentinel-2 revisits roughly every 5 days, so a persistent source
    accumulates a time series that makes before/after comparison possible.
    """
    from imagery.db.models import Sentinel2Patch

    rows = db.execute(
        select(Sentinel2Patch)
        .where(Sentinel2Patch.cluster_id == cluster_id)
        .order_by(Sentinel2Patch.acquired_at.desc())
        .limit(min(limit, 100))
    ).scalars().all()

    return [
        PatchOut(
            id=r.id,
            acquired_at=r.acquired_at.isoformat() if r.acquired_at else None,
            cloud_percentage=r.cloud_percentage,
            thumbnail_path=r.thumbnail_path,
            bands=r.bands,
            width_px=r.width_px,
            height_px=r.height_px,
            source=r.source,
        )
        for r in rows
    ]
