"""M5's Celery task, registered on M1's Celery app.

`tasks.m5_rhythm_and_kalman` is the name M1 already dispatches
(`app/orchestration/tasks.py`), so importing this module in the worker is
what makes that dispatch land somewhere.

The task does four things in order, because each depends on the last:

    1. rhythm fingerprint   -> bias-corrected overpass slots
    2. PTSI baseline+index  -> uses the slot detection rate from (1)
    3. Kalman filter        -> critical threshold comes from (2)'s baseline
    4. escalation forecast  -> projects (3) to (2)'s threshold

Then it writes M2's three RHYTHM columns and re-triggers M2's feature
build, the same pattern M3's Phase A uses — otherwise M5's features would
not reach the classifier until the next ingestion cycle, and for a
first-time cluster they never would.

Run the worker with all five packages importable:
    celery -A app.orchestration.queue.celery_app worker \
           -I classifier.tasks,geospatial.tasks,imagery.tasks,temporal.tasks
"""
import logging
from typing import Any, Dict, Optional

from app.orchestration.queue import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.m5_rhythm_and_kalman")
def rhythm_and_kalman(cluster_id: int) -> Dict[str, Any]:
    """Rhythm fingerprint, PTSI, Kalman state and escalation forecast."""
    from app.db.session import session_scope

    from temporal.config import get_m5_settings
    from temporal.features_io import hours_from_first, load_detection_rows

    settings = get_m5_settings()

    # --- read phase (short transaction) ---
    with session_scope() as session:
        rows = load_detection_rows(session, cluster_id)

    if not rows:
        logger.warning("Cluster %s has no detections — nothing to analyse", cluster_id)
        return {"cluster_id": cluster_id, "status": "no_detections"}

    # --- compute phase (no transaction held, no network) ---
    fingerprint = _build_rhythm(rows, settings)
    periodicity = _build_periodicity(rows)
    ptsi, baseline = _build_ptsi(rows, fingerprint)
    kalman_state, freshness = _build_kalman(rows, settings)
    escalation = _build_escalation(kalman_state, baseline, settings)

    # --- write phase (short transaction) ---
    with session_scope() as session:
        _store_rhythm(session, cluster_id, fingerprint, periodicity)
        _store_ptsi(session, cluster_id, ptsi, baseline)
        _store_kalman(session, cluster_id, kalman_state, freshness, escalation)
        _update_cluster_features(session, cluster_id, fingerprint, escalation)

    # M5's features are part of M2's matrix, so ask M2 to rebuild.
    # Idempotent upsert, same as M3's Phase A.
    celery_app.send_task("tasks.m2_build_feature_vector", args=[cluster_id])

    logger.info(
        "Temporal analysis for cluster %s: PTSI=%.2f (%s), escalating=%s, freshness=%s",
        cluster_id,
        ptsi.score if ptsi else 0.0,
        ptsi.source_class if ptsi else "unknown",
        escalation.escalating if escalation else False,
        freshness.label if freshness else "unknown",
    )
    return {
        "cluster_id": cluster_id,
        "status": "analysed",
        "ptsi_score": ptsi.score if ptsi else None,
        "source_class": ptsi.source_class if ptsi else None,
        "escalating": escalation.escalating if escalation else False,
        "time_to_critical_hours": escalation.time_to_critical_hours if escalation else None,
        "freshness": freshness.label if freshness else None,
    }


# --------------------------------------------------------------------------
# compute helpers
# --------------------------------------------------------------------------

def _build_rhythm(rows, settings):
    from temporal.rhythm.fingerprint import build_fingerprint

    return build_fingerprint(rows, weekend_days=settings.weekend_day_indices)


def _build_periodicity(rows):
    from temporal.rhythm.periodicity import detect_periodicity

    return detect_periodicity(rows)


def _build_ptsi(rows, fingerprint):
    """Returns (PTSIResult, Baseline) — the baseline travels with the
    index because the escalation threshold is derived from it."""
    from temporal.ptsi.baseline import Baseline, build_baseline
    from temporal.ptsi.index import compute_ptsi

    dated = [r for r in rows if r.get("acq_datetime") is not None]
    if not dated:
        return None, Baseline()

    latest = max(dated, key=lambda r: r["acq_datetime"])

    # Exclude the current observation from its own baseline, or a spike
    # inflates the mean it is being judged against and partly hides.
    baseline = build_baseline(rows, before=latest["acq_datetime"])

    # Best available slot rate: a persistent source is reliably detected
    # in at least one slot, so the maximum is the right summary rather
    # than an average dragged down by slots it never appears in.
    rates = [
        slot["detection_rate"]
        for slot in (fingerprint.slot_detection_rates or {}).values()
        if isinstance(slot, dict) and slot.get("detection_rate") is not None
    ]
    detection_rate = max(rates) if rates else None

    ptsi = compute_ptsi(
        baseline=baseline,
        detection_rate=detection_rate,
        observation_span_days=fingerprint.observation_span_days,
        current_frp=latest.get("frp"),
    )
    return ptsi, baseline


def _build_kalman(rows, settings):
    from temporal.features_io import hours_from_first
    from temporal.forecast.freshness import assess
    from temporal.forecast.kalman import run_filter

    observations = hours_from_first(rows)
    state = run_filter(
        observations,
        intensity=settings.process_noise,
        relative_error=settings.relative_measurement_error,
        adaptive=settings.adaptive_process_noise,
    )

    timestamps = [r["acq_datetime"] for r in rows if r.get("acq_datetime") is not None]
    last_seen = max(timestamps) if timestamps else None
    freshness = assess(last_seen, timestamps)

    return state, freshness


def _build_escalation(state, baseline, settings):
    """Project the filtered state to a threshold set by the source's own
    baseline — the normal-vs-abnormal thesis in one function."""
    from temporal.forecast.escalation import critical_threshold, forecast

    if state is None:
        return None

    threshold = critical_threshold(
        baseline.mean_frp, baseline.std_frp, settings.critical_sigma_multiple
    )
    return forecast(
        state,
        threshold,
        n_samples=settings.escalation_samples,
        max_horizon_hours=settings.max_horizon_hours,
        seed=settings.escalation_seed,
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def _store_rhythm(session, cluster_id: int, fingerprint, periodicity) -> None:
    from temporal.db.models import RhythmFingerprintRow

    payload = {
        "shift_sharpness": fingerprint.shift_sharpness,
        "weekend_suppression": fingerprint.weekend_suppression,
        "day_night_contrast": fingerprint.day_night_contrast,
        "slot_detection_rates": fingerprint.slot_detection_rates,
        "day_of_week_rates": fingerprint.day_of_week_rates,
        "dominant_period_hours": periodicity.dominant_period_hours,
        "periodicity_confidence": periodicity.confidence,
        "is_weekly": periodicity.is_weekly,
        "schedule_anomaly": fingerprint.schedule_anomaly,
        "n_observations": fingerprint.n_observations,
        "n_expected_passes": fingerprint.n_expected_passes,
        "observation_span_days": fingerprint.observation_span_days,
        "slots_consistent": fingerprint.slots_consistent,
        "reason": fingerprint.reason,
    }
    _upsert(session, RhythmFingerprintRow, cluster_id, payload)


def _store_ptsi(session, cluster_id: int, ptsi, baseline) -> None:
    from temporal.db.models import PTSIRegistry

    if ptsi is None:
        return

    payload = {
        "ptsi_score": ptsi.score,
        "source_class": ptsi.source_class,
        "longevity_score": ptsi.longevity,
        "reliability_score": ptsi.reliability,
        "stability_score": ptsi.stability,
        "detection_rate": ptsi.detection_rate,
        "observation_span_days": ptsi.observation_span_days,
        "deviation_sigma": ptsi.deviation_sigma,
        "deviation_multiple": ptsi.deviation_multiple,
        "is_behaving_normally": ptsi.behaving_normally,
        "summary": ptsi.summary,
        # The baseline itself — "its normal". Stored so a deviation can
        # always be traced back to what it deviated from.
        "baseline_frp_mean": baseline.mean_frp,
        "baseline_frp_std": baseline.std_frp,
        "baseline_frp_median": baseline.median_frp,
        "baseline_frp_p95": baseline.p95_frp,
        "baseline_n_observations": baseline.n_observations,
        "first_seen": baseline.first_seen,
    }
    _upsert(session, PTSIRegistry, cluster_id, payload)


def _store_kalman(session, cluster_id: int, state, freshness, escalation) -> None:
    from temporal.db.models import KalmanStateRow
    from temporal.forecast.kalman import normalised_innovation_squared

    if state is None:
        return

    payload = {
        "frp_estimate": state.frp,
        "frp_rate_estimate": state.rate,
        "frp_std": state.frp_std,
        "rate_std": state.rate_std,
        "covariance": state.covariance,
        "n_updates": state.n_updates,
        "mean_nis": normalised_innovation_squared(state),
        "last_observation_at": freshness.last_observation_at if freshness else None,
        "freshness": freshness.label if freshness else None,
        "age_hours": freshness.age_hours if freshness else None,
        "cadence_hours": freshness.cadence_hours if freshness else None,
    }

    if escalation is not None:
        payload.update({
            "critical_threshold_frp": escalation.critical_threshold_frp,
            "time_to_critical_hours": escalation.time_to_critical_hours,
            "ttc_p50_low": escalation.p50_low,
            "ttc_p50_high": escalation.p50_high,
            "ttc_p90_low": escalation.p90_low,
            "ttc_p90_high": escalation.p90_high,
            "probability_reaches_critical": escalation.probability_reaches_critical,
            "escalating": escalation.escalating,
            "already_critical": escalation.already_critical,
            "reason": escalation.reason,
        })

    _upsert(session, KalmanStateRow, cluster_id, payload)


def _update_cluster_features(session, cluster_id: int, fingerprint, escalation) -> None:
    """Write ONLY the three columns M2 reserved for M5.

    M2's tests assert that no member clobbers another's columns, and
    M3/M4 write to this same row concurrently.
    """
    try:
        from classifier.db.models import ClusterFeatures
    except ImportError:
        return

    row = session.get(ClusterFeatures, cluster_id)
    if row is None:
        return

    if fingerprint.shift_sharpness is not None:
        row.shift_sharpness = fingerprint.shift_sharpness
    if fingerprint.weekend_suppression is not None:
        row.weekend_suppression = fingerprint.weekend_suppression
    if escalation is not None and escalation.time_to_critical_hours is not None:
        row.kalman_time_to_critical = escalation.time_to_critical_hours


def _upsert(session, model, cluster_id: int, payload: Dict[str, Any]) -> None:
    existing = session.get(model, cluster_id)
    if existing is None:
        session.add(model(cluster_id=cluster_id, **payload))
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
