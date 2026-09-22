"""DB-backed feature assembly.

Reads M1's `clusters` / `hotspots` / `fingerprints` / `facilities` rows
for one cluster, computes M2's thermal/temporal/spatial features, and
upserts the result into `cluster_features`.

Critical invariant: this function only ever writes M2-owned and
M1-snapshot columns. It must never null out M3/M4/M5's columns, because
their tasks run concurrently with this one and a blind full-row overwrite
would erase whichever of them happened to finish first. The upsert below
is column-scoped for exactly that reason.
"""
import logging
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Columns this module owns. Anything outside this set is left untouched.
M2_OWNED_COLUMNS = (
    "pct_cropland", "pct_forest", "pct_urban", "facility_distance_m",
    "facility_prior_weight", "facility_type", "population_density", "near_facility",
    "frp_mean", "frp_max", "frp_std", "frp_zscore", "brightness_mean",
    "brightness_max", "confidence_mean", "persistence_days", "n_detections",
    "detections_per_day", "night_fraction", "source_diversity",
    "spatial_extent_km", "spatial_growth_rate",
)

BASELINE_WINDOW_DAYS = 90  # matches M1's TimescaleDB retention for FRP history


def load_hotspot_rows(session: Session, cluster_id: int) -> List[Dict[str, Any]]:
    """Fetch a cluster's detections as plain dicts for the pure feature code."""
    from geoalchemy2.shape import to_shape

    from app.db.models import Hotspot

    hotspots = session.execute(
        select(Hotspot).where(Hotspot.cluster_id == cluster_id)
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


def compute_frp_baseline(
    session: Session, cluster_id: int, before_datetime=None
) -> Tuple[Optional[float], Optional[float]]:
    """Mean/stdev of this cluster's historical FRP, for the z-score.

    `before_datetime` excludes the current burst from its own baseline —
    without it, a spike inflates the mean it is being compared against
    and the anomaly partly hides itself. This is also the point-in-time
    correctness the research notes insist on: temporal features must only
    use information available before the moment being predicted.
    """
    from app.db.models import Hotspot

    stmt = select(Hotspot.frp).where(
        Hotspot.cluster_id == cluster_id, Hotspot.frp.isnot(None)
    )
    if before_datetime is not None:
        stmt = stmt.where(
            Hotspot.acq_datetime < before_datetime,
            Hotspot.acq_datetime >= before_datetime - timedelta(days=BASELINE_WINDOW_DAYS),
        )

    values = [row[0] for row in session.execute(stmt).all() if row[0] is not None]
    if len(values) < 2:
        return None, None

    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    return mean_value, math.sqrt(variance)


def load_context_features(session: Session, cluster_id: int) -> Dict[str, Any]:
    """Snapshot M1's enrichment output (fingerprints + facility type)."""
    from app.db.models import Facility, Fingerprint

    fingerprint = session.get(Fingerprint, cluster_id)
    if fingerprint is None:
        logger.info("No fingerprint row yet for cluster %s — context features unavailable", cluster_id)
        return {}

    facility_type = None
    if fingerprint.nearest_facility_id is not None:
        facility = session.get(Facility, fingerprint.nearest_facility_id)
        facility_type = facility.facility_type if facility else None

    distance = fingerprint.facility_distance_m
    return {
        "pct_cropland": fingerprint.pct_cropland,
        "pct_forest": fingerprint.pct_forest,
        "pct_urban": fingerprint.pct_urban,
        "facility_distance_m": distance,
        # Same exponential decay M1 uses for facility attribution, recomputed
        # here so the model sees a smooth proximity signal rather than only a
        # raw distance with a hard cut-off.
        "facility_prior_weight": (
            math.exp(-distance / 500.0) if distance is not None else None
        ),
        "facility_type": facility_type,
        "population_density": fingerprint.population_density,
        "near_facility": (
            None if distance is None else (1.0 if distance < 1000.0 else 0.0)
        ),
    }


def assemble_features(session: Session, cluster_id: int) -> Dict[str, Any]:
    """Build the full M2-owned feature dict for a cluster."""
    from classifier.features.thermal import compute_thermal_features

    rows = load_hotspot_rows(session, cluster_id)
    if not rows:
        logger.warning("Cluster %s has no hotspot rows — nothing to assemble", cluster_id)
        return {}

    latest = max(r["acq_datetime"] for r in rows if r.get("acq_datetime"))
    baseline_mean, baseline_std = compute_frp_baseline(session, cluster_id, before_datetime=latest)

    features: Dict[str, Any] = {}
    features.update(load_context_features(session, cluster_id))
    features.update(compute_thermal_features(rows, baseline_mean, baseline_std))
    return features


def upsert_cluster_features(session: Session, cluster_id: int, features: Dict[str, Any]) -> None:
    """Write only M2-owned columns, preserving M3/M4/M5's concurrent writes."""
    from classifier.db.models import ClusterFeatures

    payload = {k: v for k, v in features.items() if k in M2_OWNED_COLUMNS}

    existing = session.get(ClusterFeatures, cluster_id)
    if existing is None:
        session.add(ClusterFeatures(cluster_id=cluster_id, **payload))
        return

    for column, value in payload.items():
        setattr(existing, column, value)


def feature_group_availability(features: Dict[str, Any]) -> Dict[str, bool]:
    """Which groups have landed — used to decide whether to classify yet."""
    from classifier.features.evidence_weighting import group_is_present
    from classifier.features.schema import FeatureGroup

    return {g.value: group_is_present(features, g) for g in FeatureGroup}
