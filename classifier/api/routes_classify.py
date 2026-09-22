"""Classification read API.

Mounts onto M1's existing FastAPI app rather than standing up a second
service:

    # app/main.py
    from classifier.api.routes_classify import router as classify_router
    app.include_router(classify_router)

Shapes are flat and pre-sorted so M6's event-card panel can render them
without post-processing.
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(prefix="/classify", tags=["classification"])


class ShapContribution(BaseModel):
    feature: str
    contribution: float


class ClassificationOut(BaseModel):
    cluster_id: int
    predicted_class: Optional[str]
    confidence_score: Optional[float]
    class_probabilities: Optional[Dict[str, float]]
    xgb_probabilities: Optional[Dict[str, float]]
    image_probabilities: Optional[Dict[str, float]]
    fusion_weight_xgb: Optional[float]
    had_image_corroboration: bool
    top_factors: List[ShapContribution]
    explanation_method: Optional[str]
    reasons: List[str]
    evidence: Optional[dict]
    model_version: Optional[str]


class ClassificationList(BaseModel):
    count: int
    items: List[ClassificationOut]


def _to_out(row) -> ClassificationOut:
    shap_values = row.shap_values or {}
    top_factors = [
        ShapContribution(feature=k, contribution=v)
        for k, v in sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ]
    return ClassificationOut(
        cluster_id=row.cluster_id,
        predicted_class=row.predicted_class,
        confidence_score=row.confidence_score,
        class_probabilities=row.class_probabilities,
        xgb_probabilities=row.xgb_probabilities,
        image_probabilities=row.image_probabilities,
        fusion_weight_xgb=row.fusion_weight_xgb,
        # A verdict reached without a usable Sentinel-2 patch is weaker
        # evidence; the UI surfaces this rather than burying it.
        had_image_corroboration=row.image_probabilities is not None,
        top_factors=top_factors,
        explanation_method=row.explanation_method,
        reasons=row.reasons or [],
        evidence=row.evidence_weight_notes,
        model_version=row.model_version,
    )


@router.get("/{cluster_id}", response_model=ClassificationOut)
def get_classification(cluster_id: int, db: Session = Depends(get_db)):
    from classifier.db.models import Classification

    row = db.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="no classification for this cluster")
    return _to_out(row)


@router.get("", response_model=ClassificationList)
def list_classifications(
    predicted_class: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(200, le=2000),
    db: Session = Depends(get_db),
):
    from classifier.db.models import Classification

    stmt = select(Classification)
    if predicted_class is not None:
        stmt = stmt.where(Classification.predicted_class == predicted_class)
    if min_confidence is not None:
        stmt = stmt.where(Classification.confidence_score >= min_confidence)
    stmt = stmt.order_by(Classification.confidence_score.desc()).limit(limit)

    rows = db.execute(stmt).scalars().all()
    return ClassificationList(count=len(rows), items=[_to_out(r) for r in rows])
