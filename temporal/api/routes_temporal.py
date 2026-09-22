"""Temporal-analysis read API.

Mounts onto M1's FastAPI app alongside M2's, M3's and M4's routers:

    # app/main.py
    from temporal.api.routes_temporal import router as temporal_router
    app.include_router(temporal_router)

Serves the event card's escalation chart, confidence badge and the
normal-vs-abnormal verdict that the project treats as its differentiator.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(tags=["temporal"])


class PTSIOut(BaseModel):
    cluster_id: int
    ptsi_score: Optional[float]
    source_class: Optional[str]
    longevity_score: Optional[float]
    reliability_score: Optional[float]
    stability_score: Optional[float]
    baseline_frp_mean: Optional[float]
    baseline_frp_std: Optional[float]
    baseline_n_observations: Optional[int]
    observation_span_days: Optional[float]
    deviation_sigma: Optional[float]
    deviation_multiple: Optional[float]
    is_behaving_normally: Optional[bool]
    baseline_known: bool
    summary: Optional[str]


class ForecastOut(BaseModel):
    cluster_id: int
    frp_estimate: Optional[float]
    frp_rate_estimate: Optional[float]
    frp_std: Optional[float]
    rate_std: Optional[float]
    escalating: bool
    already_critical: bool
    critical_threshold_frp: Optional[float]
    time_to_critical_hours: Optional[float]
    ttc_p50: List[Optional[float]]
    ttc_p90: List[Optional[float]]
    probability_reaches_critical: Optional[float]
    freshness: Optional[str]
    age_hours: Optional[float]
    forecast_is_actionable: bool
    mean_nis: Optional[float]
    reason: Optional[str]


class RhythmOut(BaseModel):
    cluster_id: int
    shift_sharpness: Optional[float]
    weekend_suppression: Optional[float]
    day_night_contrast: Optional[float]
    slot_detection_rates: Optional[Dict[str, Any]]
    day_of_week_rates: Optional[Dict[str, float]]
    dominant_period_hours: Optional[float]
    is_weekly: Optional[bool]
    schedule_anomaly: bool
    n_observations: Optional[int]
    observation_span_days: Optional[float]
    slots_consistent: Optional[bool]
    reason: Optional[str]


@router.get("/ptsi/{cluster_id}", response_model=PTSIOut)
def get_ptsi(cluster_id: int, db: Session = Depends(get_db)):
    from temporal.db.models import PTSIRegistry

    row = db.get(PTSIRegistry, cluster_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no PTSI entry for this cluster")

    return PTSIOut(
        cluster_id=row.cluster_id,
        ptsi_score=row.ptsi_score,
        source_class=row.source_class,
        longevity_score=row.longevity_score,
        reliability_score=row.reliability_score,
        stability_score=row.stability_score,
        baseline_frp_mean=row.baseline_frp_mean,
        baseline_frp_std=row.baseline_frp_std,
        baseline_n_observations=row.baseline_n_observations,
        observation_span_days=row.observation_span_days,
        deviation_sigma=row.deviation_sigma,
        deviation_multiple=row.deviation_multiple,
        is_behaving_normally=row.is_behaving_normally,
        # Surfaced explicitly: "we don't know what normal is here" must
        # not render the same as "this is behaving normally".
        baseline_known=row.baseline_frp_mean is not None,
        summary=row.summary,
    )


@router.get("/forecast/{cluster_id}", response_model=ForecastOut)
def get_forecast(cluster_id: int, db: Session = Depends(get_db)):
    from temporal.db.models import KalmanStateRow

    row = db.get(KalmanStateRow, cluster_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no forecast for this cluster")

    return ForecastOut(
        cluster_id=row.cluster_id,
        frp_estimate=row.frp_estimate,
        frp_rate_estimate=row.frp_rate_estimate,
        frp_std=row.frp_std,
        rate_std=row.rate_std,
        escalating=bool(row.escalating),
        already_critical=bool(row.already_critical),
        critical_threshold_frp=row.critical_threshold_frp,
        time_to_critical_hours=row.time_to_critical_hours,
        ttc_p50=[row.ttc_p50_low, row.ttc_p50_high],
        ttc_p90=[row.ttc_p90_low, row.ttc_p90_high],
        probability_reaches_critical=row.probability_reaches_critical,
        freshness=row.freshness,
        age_hours=row.age_hours,
        # A projection from a three-day-old observation must not be
        # presented beside a fresh one without qualification.
        forecast_is_actionable=row.freshness in ("FRESH", "MODERATE"),
        mean_nis=row.mean_nis,
        reason=row.reason,
    )


@router.get("/rhythm/{cluster_id}", response_model=RhythmOut)
def get_rhythm(cluster_id: int, db: Session = Depends(get_db)):
    from temporal.db.models import RhythmFingerprintRow

    row = db.get(RhythmFingerprintRow, cluster_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no rhythm fingerprint for this cluster")

    return RhythmOut(
        cluster_id=row.cluster_id,
        shift_sharpness=row.shift_sharpness,
        weekend_suppression=row.weekend_suppression,
        day_night_contrast=row.day_night_contrast,
        slot_detection_rates=row.slot_detection_rates,
        day_of_week_rates=row.day_of_week_rates,
        dominant_period_hours=row.dominant_period_hours,
        is_weekly=row.is_weekly,
        schedule_anomaly=bool(row.schedule_anomaly),
        n_observations=row.n_observations,
        observation_span_days=row.observation_span_days,
        slots_consistent=row.slots_consistent,
        reason=row.reason,
    )


@router.get("/escalating", response_model=List[ForecastOut])
def list_escalating(
    limit: int = Query(50, le=500),
    actionable_only: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Sources currently projected to reach their critical threshold.

    This is the risk-ranked feed the alert console needs. Sorted by
    soonest first, and by default excludes STALE forecasts — a
    three-day-old projection should not outrank a fresh one on an
    operations screen.
    """
    from temporal.db.models import KalmanStateRow

    stmt = select(KalmanStateRow).where(KalmanStateRow.escalating.is_(True))
    if actionable_only:
        stmt = stmt.where(KalmanStateRow.freshness.in_(("FRESH", "MODERATE")))
    stmt = stmt.order_by(KalmanStateRow.time_to_critical_hours.asc()).limit(limit)

    rows = db.execute(stmt).scalars().all()
    return [
        ForecastOut(
            cluster_id=r.cluster_id,
            frp_estimate=r.frp_estimate,
            frp_rate_estimate=r.frp_rate_estimate,
            frp_std=r.frp_std,
            rate_std=r.rate_std,
            escalating=bool(r.escalating),
            already_critical=bool(r.already_critical),
            critical_threshold_frp=r.critical_threshold_frp,
            time_to_critical_hours=r.time_to_critical_hours,
            ttc_p50=[r.ttc_p50_low, r.ttc_p50_high],
            ttc_p90=[r.ttc_p90_low, r.ttc_p90_high],
            probability_reaches_critical=r.probability_reaches_critical,
            freshness=r.freshness,
            age_hours=r.age_hours,
            forecast_is_actionable=r.freshness in ("FRESH", "MODERATE"),
            mean_nis=r.mean_nis,
            reason=r.reason,
        )
        for r in rows
    ]
