"""Dashboard aggregation endpoints.

The event card needs the classification (M2), the plume and attribution
(M3), the imagery and sub-pixel temperature (M4), and the PTSI and
escalation forecast (M5). Making the browser issue five requests per
click — and handle five different 404s — pushes orchestration into the
client. These endpoints do it server-side and return one shaped record.

Every downstream read is individually guarded: a member whose package is
not installed, or whose analysis has not run for a cluster yet, yields a
null section rather than failing the whole response. That is the normal
state early in the pipeline, not an error.
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Risk is a presentation concern — the dashboard ranks by it, but no
# member produces it. Composed here from the signals that do exist.
_RISK_FRP_CEILING = 100.0
_RISK_BRIGHTNESS_FLOOR = 295.0
_RISK_BRIGHTNESS_CEILING = 360.0

# The dashboard shows a rolling recent window, not the DB's full history
# — old detections stay in Postgres (nothing here deletes them) but drop
# out of what the map/Live Summary render once they age past this. Keyed
# off the newest last_seen actually in the data rather than wall-clock
# "now": FIRMS NRT can lag by up to a day, so anchoring to "now" would
# make the window silently shrink (or empty out entirely) whenever
# ingestion is a bit behind, the same reasoning already applied to the
# "FIRMS NRT runs up to a day behind" acquisition-window note below.
DASHBOARD_DETECTIONS_WINDOW_DAYS = 3


def _safe(fn, default=None):
    """Run a downstream read, swallowing anything it throws.

    A missing member package or an un-analysed cluster must degrade the
    response, not break it.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _risk_score(frp: Optional[float], brightness: Optional[float],
                predicted_class: Optional[str], escalating: bool) -> int:
    """0-100 composite used for ranking and colour-coding.

    Weighted toward radiative power because that is the measurement
    least dependent on any model having run yet.
    """
    frp_part = min((frp or 0.0) / _RISK_FRP_CEILING, 1.0) * 55.0

    span = _RISK_BRIGHTNESS_CEILING - _RISK_BRIGHTNESS_FLOOR
    bright_part = max(0.0, min(((brightness or 0.0) - _RISK_BRIGHTNESS_FLOOR) / span, 1.0)) * 30.0

    class_part = 10.0 if predicted_class == "industrial_fire" else 0.0
    escalation_part = 5.0 if escalating else 0.0

    return int(round(min(frp_part + bright_part + class_part + escalation_part, 100.0)))


# FIRMS reports detection confidence on two different native scales
# (VIIRS categorical l/n/h, MODIS numeric 0-100); `normalize.py` already
# collapsed both to 0-1 on the way in. These thresholds put the band
# boundaries back where the original VIIRS categories sat (0.3 low,
# 0.7 nominal, 0.95 high) so the dashboard's filter means the same thing
# it means in the FIRMS product.
_CONFIDENCE_HIGH = 0.8
_CONFIDENCE_NOMINAL = 0.5


def _as_utc(moment):
    """Render a stored timestamp in UTC, whatever the DB session's zone.

    `hotspots.acq_datetime` is TIMESTAMPTZ and FIRMS acquisition times are
    UTC, but psycopg hands back a datetime already converted into the
    server session's TimeZone (Asia/Calcutta on a default Indian install).
    Formatting that object directly produced strings like "12:22 UTC" for
    an acquisition that FIRMS reports at 06:52 UTC — the clock had been
    shifted by +05:30 while the label still said UTC, which is worse than
    showing no time at all.

    A naive datetime is assumed to already be UTC, matching what
    `normalize.firms_row_to_canonical` attaches on ingest.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _confidence_band(value: Optional[float]) -> Optional[str]:
    """0-1 detection confidence -> the band the UI filters on."""
    if value is None:
        return None
    if value >= _CONFIDENCE_HIGH:
        return "HIGH"
    if value >= _CONFIDENCE_NOMINAL:
        return "NOMINAL"
    return "LOW"


def _sensor_family(source: Optional[str]) -> Optional[str]:
    """'VIIRS_NOAA21_NRT' -> 'VIIRS'. The dashboard groups by instrument
    family, not by individual product feed."""
    if not source:
        return None
    upper = source.upper()
    if "VIIRS" in upper:
        return "VIIRS"
    if "MODIS" in upper:
        return "MODIS"
    if "INSAT" in upper:
        return "INSAT"
    if "HIMAWARI" in upper:
        return "HIMAWARI"
    if "SENTINEL" in upper:
        return "SENTINEL"
    return upper.split("_")[0]


def _cluster_rows(session: Session, limit: int):
    """Clusters with their centroid decoded, newest activity first.

    Restricted to DASHBOARD_DETECTIONS_WINDOW_DAYS — older clusters stay
    in Postgres untouched, they just don't come back from this query.
    """
    from geoalchemy2.shape import to_shape

    from app.db.models import Cluster

    query = select(Cluster).order_by(Cluster.last_seen.desc().nullslast())

    latest_seen = session.execute(select(func.max(Cluster.last_seen))).scalar()
    if latest_seen is not None:
        cutoff = latest_seen - timedelta(days=DASHBOARD_DETECTIONS_WINDOW_DAYS)
        query = query.where(Cluster.last_seen >= cutoff)

    clusters = session.execute(query.limit(limit)).scalars().all()

    rows = []
    for cluster in clusters:
        point = to_shape(cluster.centroid)
        rows.append((cluster, point.x, point.y))
    return rows


def _hotspot_summary(session: Session, cluster_id: int) -> Dict[str, Any]:
    """Peak thermal values and the most recent acquisition for a cluster."""
    from app.db.models import Hotspot

    rows = session.execute(
        select(
            Hotspot.frp, Hotspot.brightness, Hotspot.acq_datetime,
            Hotspot.source, Hotspot.confidence,
        )
        .where(Hotspot.cluster_id == cluster_id)
        .order_by(Hotspot.acq_datetime.desc())
        .limit(200)
    ).all()

    if not rows:
        return {}

    frps = [r.frp for r in rows if r.frp is not None]
    brights = [r.brightness for r in rows if r.brightness is not None]
    latest = rows[0]

    confidences = [r.confidence for r in rows if r.confidence is not None]

    return {
        "frp": max(frps) if frps else None,
        "brightness": max(brights) if brights else None,
        "sensor": latest.source,
        "acq_datetime": latest.acq_datetime,
        "n_detections": len(rows),
        "confidence": (sum(confidences) / len(confidences)) if confidences else None,
    }


def _batch_hotspot_summaries(session: Session, cluster_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """One query for all clusters — eliminates the N+1 that killed the
    detections endpoint when there were hundreds of active clusters."""
    from sqlalchemy import func

    from app.db.models import Hotspot

    if not cluster_ids:
        return {}

    rows = session.execute(
        select(
            Hotspot.cluster_id,
            func.max(Hotspot.frp).label("max_frp"),
            func.max(Hotspot.brightness).label("max_brightness"),
            func.max(Hotspot.acq_datetime).label("latest_acq"),
            func.count(Hotspot.id).label("n_detections"),
            func.avg(Hotspot.confidence).label("mean_confidence"),
        )
        .where(Hotspot.cluster_id.in_(cluster_ids))
        .group_by(Hotspot.cluster_id)
    ).all()

    # Sensor of the most recent detection per cluster. DISTINCT ON is
    # PostgreSQL's "first row per group" — one query for every cluster,
    # replacing a per-cluster SELECT that made this endpoint issue one
    # extra round-trip per cluster (~700 of them on a live India cycle).
    latest_sensor: Dict[int, str] = {}
    sensor_rows = session.execute(
        select(Hotspot.cluster_id, Hotspot.source)
        .where(Hotspot.cluster_id.in_(cluster_ids))
        .distinct(Hotspot.cluster_id)
        .order_by(Hotspot.cluster_id, Hotspot.acq_datetime.desc())
    ).all()
    for cid, source in sensor_rows:
        latest_sensor[cid] = source

    result: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        result[r.cluster_id] = {
            "frp": float(r.max_frp) if r.max_frp is not None else None,
            "brightness": float(r.max_brightness) if r.max_brightness is not None else None,
            "sensor": latest_sensor.get(r.cluster_id),
            "acq_datetime": r.latest_acq,
            "n_detections": r.n_detections,
            "confidence": float(r.mean_confidence) if r.mean_confidence is not None else None,
        }
    return result


# --------------------------------------------------------------------------
# Batch loaders for the detections endpoint.
#
# The per-cluster readers below (`_classification_for` and friends) are
# the right shape for the single-event endpoint, where one cluster is
# being assembled. Calling them in a loop over every cluster made
# /dashboard/detections issue roughly eight queries per cluster — about
# 16 000 round-trips for a 2000-detection page, which measured at six
# minutes and left the dashboard stuck on its loading screen.
#
# Each loader below is one query for the whole page and returns a dict
# keyed by cluster_id. All of them stay individually guarded: a member
# package that is not installed yields an empty mapping, and every
# cluster then reports that section as absent, exactly as before.
# --------------------------------------------------------------------------

def _batch_classifications(session: Session, cluster_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    from classifier.db.models import Classification, ClusterFeatures

    result: Dict[int, Dict[str, Any]] = {}

    for row in session.query(Classification).filter(
        Classification.cluster_id.in_(cluster_ids)
    ).all():
        result[row.cluster_id] = {
            "predicted_class": row.predicted_class,
            "confidence_score": row.confidence_score,
            "class_probabilities": row.class_probabilities,
            "reasons": row.reasons,
            "model_version": row.model_version,
            "dozier_temp": None,
        }

    for row in session.query(
        ClusterFeatures.cluster_id, ClusterFeatures.dozier_temp
    ).filter(ClusterFeatures.cluster_id.in_(cluster_ids)).all():
        entry = result.setdefault(row.cluster_id, {"predicted_class": None})
        entry["dozier_temp"] = row.dozier_temp

    return result


def _batch_forecasts(session: Session, cluster_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    from temporal.db.models import KalmanStateRow

    return {
        row.cluster_id: {
            "escalating": bool(row.escalating),
            "time_to_critical_hours": row.time_to_critical_hours,
            "soonest_hours": row.ttc_p90_low,
            "freshness": row.freshness,
        }
        for row in session.query(KalmanStateRow).filter(
            KalmanStateRow.cluster_id.in_(cluster_ids)
        ).all()
    }


def _batch_ptsi(session: Session, cluster_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    from temporal.db.models import PTSIRegistry

    return {
        row.cluster_id: {
            "source_class": row.source_class,
            "is_behaving_normally": row.is_behaving_normally,
            "summary": row.summary,
        }
        for row in session.query(PTSIRegistry).filter(
            PTSIRegistry.cluster_id.in_(cluster_ids)
        ).all()
    }


def _batch_top_facilities(session: Session, cluster_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Rank-1 facility attribution per cluster, with the facility joined in."""
    from app.db.models import Facility

    from geospatial.db.models import FacilityAttribution

    rows = (
        session.query(FacilityAttribution, Facility)
        .outerjoin(Facility, Facility.id == FacilityAttribution.facility_id)
        .filter(
            FacilityAttribution.cluster_id.in_(cluster_ids),
            FacilityAttribution.rank == 1,
        )
        .all()
    )
    return {
        attribution.cluster_id: {
            "facility_id": attribution.facility_id,
            "name": getattr(facility, "name", None),
            "facility_type": getattr(facility, "facility_type", None),
            "distance_m": attribution.distance_m,
        }
        for attribution, facility in rows
    }


@router.get("/detections")
def list_detections(
    limit: int = Query(500, le=2000),
    india_only: bool = Query(
        True,
        description="Drop detections outside India's land boundaries. FIRMS is "
                    "queried over a rectangular bbox that also covers parts of "
                    "Pakistan, Nepal, China, Bangladesh and Myanmar.",
    ),
    db: Session = Depends(get_db),
):
    """Flattened detections for the globe.

    One row per cluster, already carrying the fields the map renders, so
    the client does not join anything itself.
    """
    # Over-fetch before the India filter so `limit` counts rows the client
    # will actually receive. Without this, asking for 500 detections
    # returned ~350 because roughly a third of the bbox's clusters fall
    # outside India and were dropped after the SQL LIMIT had already
    # applied. The ceiling bounds the worst case.
    fetch_limit = min(limit * 3, 6000) if india_only else limit
    cluster_rows = _cluster_rows(db, fetch_limit)

    if india_only:
        try:
            from geospatial.admin_boundaries import available, resolve as resolve_admin

            if available():
                cluster_rows = [
                    row for row in cluster_rows if resolve_admin(row[1], row[2])[0] is not None
                ]
        except Exception:  # noqa: BLE001
            # Boundaries unavailable: show everything rather than silently
            # returning an empty globe.
            logger.warning("India filter unavailable — returning unfiltered", exc_info=True)

    cluster_rows = cluster_rows[:limit]
    cluster_ids = [cluster.id for cluster, _, _ in cluster_rows]

    # One query per section for the whole page, instead of ~8 per cluster.
    thermal_map = _batch_hotspot_summaries(db, cluster_ids)
    classification_map = _safe(lambda: _batch_classifications(db, cluster_ids), {}) or {}
    forecast_map = _safe(lambda: _batch_forecasts(db, cluster_ids), {}) or {}
    ptsi_map = _safe(lambda: _batch_ptsi(db, cluster_ids), {}) or {}
    attribution_map = _safe(lambda: _batch_top_facilities(db, cluster_ids), {}) or {}

    items: List[Dict[str, Any]] = []

    # Point-in-polygon against published district boundaries. Loaded
    # once per process and indexed, so this is microseconds per cluster.
    try:
        from geospatial.admin_boundaries import resolve as resolve_admin
    except Exception:  # noqa: BLE001
        def resolve_admin(lon, lat):
            return None, None

    for cluster, lon, lat in cluster_rows:
        thermal = thermal_map.get(cluster.id, {})
        state, district = resolve_admin(lon, lat)

        classification = classification_map.get(cluster.id)
        forecast = forecast_map.get(cluster.id)
        ptsi = ptsi_map.get(cluster.id)
        attribution = attribution_map.get(cluster.id)

        predicted = (classification or {}).get("predicted_class")
        escalating = bool((forecast or {}).get("escalating"))
        acq = _as_utc(thermal.get("acq_datetime"))

        items.append({
            "cluster_id": cluster.id,
            "name": (attribution or {}).get("name") or f"Cluster {cluster.id}",
            "lat": lat, "lon": lon,
            "type": _display_type(predicted, (attribution or {}).get("facility_type")),
            "frp": thermal.get("frp"),
            "brightness": thermal.get("brightness"),
            "temp": (classification or {}).get("dozier_temp") or thermal.get("brightness"),
            "risk": _risk_score(thermal.get("frp"), thermal.get("brightness"), predicted, escalating),
            "sensor": _sensor_family(thermal.get("sensor")),
            "sensor_product": thermal.get("sensor"),
            "acq_date": acq.date().isoformat() if acq else None,
            "acq_time": acq.strftime("%H:%M UTC") if acq else None,
            "detected_at": acq.isoformat() if acq else None,
            "status": _status_for(cluster, escalating, ptsi),
            "state": state, "district": district,
            "circle": None, "division": None, "range": None, "block": None, "beat": None,
            "area_ha": None, "perimeter_km": None,
            "summary": (ptsi or {}).get("summary"),
            # Persistent vs. transient thermal source (temporal.ptsi) —
            # None until M5 has built a baseline for this cluster, not a
            # guess in the meantime.
            "source_class": (ptsi or {}).get("source_class"),
            "predicted_class": predicted,
            "confidence": _confidence_band(thermal.get("confidence")),
            "confidence_value": thermal.get("confidence"),
            "model_confidence": (classification or {}).get("confidence_score"),
            "n_detections": thermal.get("n_detections"),
            "escalating": escalating,
            # WUI (wildland-urban interface) proximity — computed by
            # geospatial.tasks.wui_threat, read straight off the Cluster
            # row already in hand (no extra join needed for this one).
            "wui_threat": bool(getattr(cluster, "wui_threat", False)),
            "wui_distance_m": getattr(cluster, "wui_distance_m", None),
            "wui_eta_hours": getattr(cluster, "wui_eta_hours", None),
            "wui_bearing_deg": getattr(cluster, "wui_bearing_deg", None),
            "wui_threatened_asset": getattr(cluster, "wui_threatened_asset", None),
            # Crown-fire FRP threshold — computed by
            # geospatial.tasks.crown_fire, independent of WUI.
            "is_crown_fire": bool(getattr(cluster, "is_crown_fire", False)),
            # WFDSS-inspired Resource Demand Index — computed by
            # geospatial.tasks.resource_demand off the two fields above.
            "rdi_score": getattr(cluster, "rdi_score", None),
            "coa_type": getattr(cluster, "coa_type", None),
            "area_ha": getattr(cluster, "area_ha", None),
        })

    return {"count": len(items), "items": items}


def _display_type(predicted_class: Optional[str], facility_type: Optional[str]) -> str:
    """Map the classifier's five classes onto the four marker styles the
    existing globe already knows how to draw."""
    if predicted_class == "agricultural_burning":
        return "agri"
    if predicted_class == "mining":
        return "coal"
    if predicted_class == "gas_flare":
        return "petroleum"
    if facility_type in ("refinery", "oil_well", "flare", "LNG"):
        return "petroleum"
    if facility_type == "mine":
        return "coal"
    return "industrial"


def _status_for(cluster, escalating: bool, ptsi: Optional[Dict]) -> str:
    from temporal.status import is_active_from_signals

    if is_active_from_signals(escalating, ptsi):
        return "Active"
    if cluster.status == "resolved":
        return "Extinguished"
    if ptsi and ptsi.get("source_class") == "persistent":
        return "Verified"
    return "Under Control"


# --------------------------------------------------------------------------
# per-member readers, each degrading to None
# --------------------------------------------------------------------------

def _classification_for(db: Session, cluster_id: int) -> Optional[Dict[str, Any]]:
    from classifier.db.models import Classification, ClusterFeatures

    row = db.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    features = db.get(ClusterFeatures, cluster_id)
    if row is None and features is None:
        return None
    return {
        "predicted_class": getattr(row, "predicted_class", None),
        "confidence_score": getattr(row, "confidence_score", None),
        "class_probabilities": getattr(row, "class_probabilities", None),
        "shap_values": getattr(row, "shap_values", None),
        "reasons": getattr(row, "reasons", None),
        "explanation_method": getattr(row, "explanation_method", None),
        "evidence": getattr(row, "evidence_weight_notes", None),
        "model_version": getattr(row, "model_version", None),
        "dozier_temp": getattr(features, "dozier_temp", None),
    }


def _forecast_for(db: Session, cluster_id: int) -> Optional[Dict[str, Any]]:
    from temporal.db.models import KalmanStateRow

    row = db.get(KalmanStateRow, cluster_id)
    if row is None:
        return None
    return {
        "escalating": bool(row.escalating),
        "already_critical": bool(row.already_critical),
        "frp_estimate": row.frp_estimate,
        "frp_rate_estimate": row.frp_rate_estimate,
        "time_to_critical_hours": row.time_to_critical_hours,
        # p90_low is the soonest plausible arrival — the number M5's
        # changelog says responders should plan against.
        "soonest_hours": row.ttc_p90_low,
        "ttc_p90": [row.ttc_p90_low, row.ttc_p90_high],
        "probability_reaches_critical": row.probability_reaches_critical,
        "freshness": row.freshness,
        "actionable": row.freshness in ("FRESH", "MODERATE"),
    }


def _ptsi_for(db: Session, cluster_id: int) -> Optional[Dict[str, Any]]:
    from temporal.db.models import PTSIRegistry

    row = db.get(PTSIRegistry, cluster_id)
    if row is None:
        return None
    return {
        "ptsi_score": row.ptsi_score,
        "source_class": row.source_class,
        "baseline_frp_mean": row.baseline_frp_mean,
        "deviation_multiple": row.deviation_multiple,
        "deviation_sigma": row.deviation_sigma,
        "is_behaving_normally": row.is_behaving_normally,
        "baseline_known": row.baseline_frp_mean is not None,
        "summary": row.summary,
    }


# BCCL/ECL's real administrative naming convention for the Jharia/Raniganj
# coalfields ("Cluster 8 and Cluster 9 Coal Mines", "Cluster 6 (BCCL) Coal
# Mines") -- genuine OSM names, not a bug, but an administrative zone label
# reads as a much less useful LOCATION than an actual company/site name.
# Matched the same way an absent name already is: try one more live,
# narrow lookup for a real, specifically-named neighbour (see
# nearest_named_industrial_feature's own "never invents a name" docstring)
# rather than replace it with anything invented.
_GENERIC_FACILITY_NAME = re.compile(r"^Cluster \d+", re.IGNORECASE)


def _top_facility_for(
    db: Session, cluster_id: int, lon: Optional[float] = None, lat: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    from app.db.models import Facility

    from geospatial.db.models import FacilityAttribution

    row = (
        db.query(FacilityAttribution)
        .filter_by(cluster_id=cluster_id)
        .order_by(FacilityAttribution.rank)
        .first()
    )
    if row is None:
        return None
    facility = db.get(Facility, row.facility_id)
    name = getattr(facility, "name", None)
    name_is_generic = bool(name and _GENERIC_FACILITY_NAME.match(name))

    # The bulk OSM ingestion's tag list (app/enrichment/osm_facilities.py)
    # commonly matches a bare landuse=industrial polygon with no `name`
    # tag at all — the actual named company is a separate OSM element it
    # never queried for. Rather than leave LOCATION blank (or an
    # administrative zone label like "Cluster 8 and Cluster 9 Coal
    # Mines" — a real name, just not a specific site), one live, narrow
    # Overpass lookup for the nearest genuinely named industrial feature
    # — real OSM data, never fabricated, and only used if it's close
    # enough to plausibly be the same site.
    nearby_named = None
    if (not name or name_is_generic) and lon is not None and lat is not None:
        from app.enrichment.osm_facilities import (
            named_feature_is_same_site,
            nearest_named_industrial_feature,
        )

        candidate = _safe(lambda: nearest_named_industrial_feature(lon, lat), None)
        if named_feature_is_same_site(candidate, row.distance_m):
            nearby_named = candidate["name"]

    # Only actually swap the generic name out once a real, specific
    # replacement was found — a generic-but-real name still beats
    # showing nothing when the live lookup comes up empty.
    display_name = None if (name_is_generic and nearby_named) else name

    return {
        "facility_id": row.facility_id,
        "name": display_name,
        "nearby_named_feature": nearby_named,
        "facility_type": getattr(facility, "facility_type", None),
        "distance_m": row.distance_m,
        "probability": row.attribution_probability,
        "inside_polygon": bool(row.inside_polygon),
    }


def _plume_for(db: Session, cluster_id: int) -> Optional[Dict[str, Any]]:
    from geoalchemy2.shape import to_shape

    from geospatial.db.models import Plume

    row = db.get(Plume, cluster_id)
    if row is None:
        return None
    geometry = None
    if row.geom is not None:
        shape = to_shape(row.geom)
        geometry = {"type": "Polygon", "coordinates": [[list(c) for c in shape.exterior.coords]]}
    return {
        "geometry": geometry,
        "wind_speed_ms": row.wind_speed_ms,
        "downwind_bearing_deg": row.downwind_bearing_deg,
        "wind_is_assumed": row.wind_source == "fallback",
        "stability_class": row.stability_class,
        "chemical_profile": row.chemical_profile,
        "population_exposed": row.population_exposed,
        "population_known": bool(row.population_known),
        "area_km2": row.area_km2,
    }


def _imagery_for(db: Session, cluster_id: int) -> Optional[Dict[str, Any]]:
    from imagery.db.models import ImagePrediction, ThermalRetrieval

    thermal = db.get(ThermalRetrieval, cluster_id)
    prediction = db.get(ImagePrediction, cluster_id)
    if thermal is None and prediction is None:
        return None
    return {
        "fire_temperature_k": getattr(thermal, "fire_temperature_k", None),
        # M4's changelog is explicit that a bare Dozier temperature
        # overstates precision for small hot sources; carry the flag.
        "well_constrained": getattr(thermal, "well_constrained", None),
        "temperature_range_k": [
            getattr(thermal, "temperature_low_k", None),
            getattr(thermal, "temperature_high_k", None),
        ],
        "fire_area_m2": getattr(thermal, "fire_area_m2", None),
        "image_predicted_class": getattr(prediction, "predicted_class", None),
        "image_confidence": getattr(prediction, "confidence", None),
        "smoke_detected": bool(getattr(prediction, "smoke_detected", False)),
        "smoke_bearing_deg": getattr(prediction, "smoke_bearing_deg", None),
        "plume_agreement": getattr(prediction, "plume_agreement", None),
        "thumbnail_path": None,
    }


@router.get("/event/{cluster_id}")
def get_event(cluster_id: int, db: Session = Depends(get_db)):
    """Everything the event card shows, in one request."""
    from app.db.models import Cluster

    cluster = db.get(Cluster, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="cluster not found")

    from geoalchemy2.shape import to_shape

    point = to_shape(cluster.centroid)
    thermal = _hotspot_summary(db, cluster_id)

    try:
        from geospatial.admin_boundaries import resolve as resolve_admin
        state, district = resolve_admin(point.x, point.y)
    except Exception:  # noqa: BLE001
        state, district = None, None

    thermal["acq_datetime"] = _as_utc(thermal.get("acq_datetime"))

    return {
        "cluster_id": cluster_id,
        "lat": point.y, "lon": point.x,
        "state": state, "district": district,
        "sensor": _sensor_family(thermal.get("sensor")),
        "confidence": _confidence_band(thermal.get("confidence")),
        "wui_threat": bool(getattr(cluster, "wui_threat", False)),
        "wui_distance_m": getattr(cluster, "wui_distance_m", None),
        "wui_eta_hours": getattr(cluster, "wui_eta_hours", None),
        "wui_bearing_deg": getattr(cluster, "wui_bearing_deg", None),
        "wui_threatened_asset": getattr(cluster, "wui_threatened_asset", None),
        "is_crown_fire": bool(getattr(cluster, "is_crown_fire", False)),
        "rdi_score": getattr(cluster, "rdi_score", None),
        "coa_type": getattr(cluster, "coa_type", None),
        "area_ha": getattr(cluster, "area_ha", None),
        "first_seen": _as_utc(cluster.first_seen),
        "last_seen": _as_utc(cluster.last_seen),
        "n_detections": cluster.n_detections,
        "cloud_fraction": cluster.cloud_fraction,
        "optical_available": bool(cluster.optical_available),
        "thermal": thermal,
        "classification": _safe(lambda: _classification_for(db, cluster_id)),
        "attribution": _safe(lambda: _top_facility_for(db, cluster_id, point.x, point.y)),
        "plume": _safe(lambda: _plume_for(db, cluster_id)),
        "imagery": _safe(lambda: _imagery_for(db, cluster_id)),
        "ptsi": _safe(lambda: _ptsi_for(db, cluster_id)),
        "forecast": _safe(lambda: _forecast_for(db, cluster_id)),
    }


@router.post("/event/{cluster_id}/sms-dispatch")
def dispatch_sms(cluster_id: int, db: Session = Depends(get_db)):
    """Emergency SMS dispatch — TEST MODE ONLY (see app/notifications/).

    Always sends to the single configured TEST_SMS_RECIPIENT, never a
    real fire station — see app/notifications/sms_recipient.py for what
    changes when nearest-fire-station routing is built later. A second
    call within 2 minutes re-reports the earlier attempt instead of
    sending again (app/notifications/sms_dispatch.py::_recent_dispatch).
    """
    from app.notifications.sms_dispatch import ClusterNotFound, dispatch_emergency_sms

    try:
        result = dispatch_emergency_sms(db, cluster_id)
    except ClusterNotFound:
        raise HTTPException(status_code=404, detail="cluster not found")
    db.commit()
    return result


@router.get("/alerts")
def list_alerts(limit: int = Query(40, le=200), db: Session = Depends(get_db)):
    """Risk-ranked alert feed.

    Prefers M5's escalating sources (a forecast that something is about
    to get worse is more actionable than a record of something that
    already happened), then falls back to M1's persisted alert rows.
    """
    items: List[Dict[str, Any]] = []

    def _from_wui():
        """Ember-jump-range clusters — ranked ahead of everything else
        in this feed. A fire that is about to reach a settlement by
        ember-cast is a different, more immediate kind of urgent than a
        forecast that FRP is trending upward, and an operator scanning
        this list top-to-bottom should see it first.
        """
        from app.db.models import Cluster

        rows = (
            db.query(Cluster)
            .filter(Cluster.wui_threat.is_(True))
            .order_by(Cluster.wui_eta_hours.asc().nullslast())
            .limit(limit)
            .all()
        )
        for row in rows:
            eta = row.wui_eta_hours
            distance_km = (row.wui_distance_m / 1000.0) if row.wui_distance_m is not None else None
            message = "Ember-jump range of " + (row.wui_threatened_asset or "an area")
            if distance_km is not None:
                message += f" ({distance_km:.1f} km"
                message += f", ETA {eta:.1f} h)" if eta is not None else ")"
            items.append({
                "cluster_id": row.id,
                "severity": "CRITICAL",
                "message": message,
                "freshness": None,
                "actionable": True,
                "wui": True,
            })

    _safe(_from_wui)

    def _from_forecasts():
        from temporal.db.models import KalmanStateRow

        rows = (
            db.query(KalmanStateRow)
            .filter(KalmanStateRow.escalating.is_(True))
            .order_by(KalmanStateRow.time_to_critical_hours.asc())
            .limit(limit)
            .all()
        )
        for row in rows:
            hours = row.ttc_p90_low if row.ttc_p90_low is not None else row.time_to_critical_hours
            items.append({
                "cluster_id": row.cluster_id,
                "severity": "CRITICAL" if (hours is not None and hours < 12) else "HIGH",
                "message": (
                    f"Projected to reach its critical threshold in "
                    f"{hours:.1f} h (soonest plausible)" if hours is not None
                    else "Escalating above its own baseline"
                ),
                "freshness": row.freshness,
                "actionable": row.freshness in ("FRESH", "MODERATE"),
            })

    _safe(_from_forecasts)

    # Only reached if NEITHER WUI nor the Kalman forecasts produced
    # anything — the true last resort.
    if not items:
        def _from_alert_table():
            from app.db.models import Alert

            rows = db.query(Alert).order_by(Alert.id.desc()).limit(limit).all()
            for row in rows:
                items.append({
                    "cluster_id": row.cluster_id,
                    "severity": (row.severity or "HIGH").upper(),
                    "message": row.message or "Alert raised",
                    "freshness": None, "actionable": True,
                })

        _safe(_from_alert_table)

    # WUI and the Kalman forecasts each queried up to `limit` rows
    # independently, so the combined list can run past it; trim here
    # rather than lowering either source's own limit, since a page with
    # room for only a few more items should still prefer showing the
    # ember-jump-critical ones first (already sorted to the front above).
    items = items[:limit]

    return {"count": len(items), "items": items}


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    """Counts for the statistics strip and any ops console."""
    from app.db.models import Cluster, Facility, Hotspot

    def _count(model):
        return db.query(model).count()

    payload = {
        "clusters": _safe(lambda: _count(Cluster), 0),
        "hotspots": _safe(lambda: _count(Hotspot), 0),
        "facilities": _safe(lambda: _count(Facility), 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _classified():
        from classifier.db.models import Classification

        return db.query(Classification).count()

    def _escalating():
        from temporal.db.models import KalmanStateRow

        return db.query(KalmanStateRow).filter(KalmanStateRow.escalating.is_(True)).count()

    payload["classified"] = _safe(_classified, 0)
    payload["escalating"] = _safe(_escalating, 0)
    return payload


@router.get("/states")
def list_states():
    """States/UTs the boundary resolver can actually attribute a
    detection to.

    The dashboard's state filter is built from this rather than from a
    hardcoded list in the page, so the dropdown cannot offer a state that
    would always match zero detections, and cannot omit one that would.
    """
    try:
        from geospatial.admin_boundaries import available, state_names

        return {"available": available(), "items": state_names()}
    except Exception:  # noqa: BLE001
        logger.warning("State list unavailable", exc_info=True)
        return {"available": False, "items": []}
