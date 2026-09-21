"""M3's Celery tasks, registered on M1's Celery app.

Two phases, for a reason:

  Phase A - `tasks.m3_spatial_attribution` (M1 already dispatches this
  name, in parallel with M2's feature build). Produces better facility
  attribution and land-cover context, writes them to M1's `fingerprints`
  row, then **re-triggers M2's feature build** so M2 doesn't train on the
  coarser values it may already have snapshotted. M2's assemble is an
  idempotent upsert, so re-running it is safe.

  Phase B - `tasks.m3_threat_and_plume`, dispatched by M2 *after*
  classification. Plume and corridor geometry both depend on knowing what
  the source is: you don't draw an advance corridor for a gas flare, and
  the chemical profile depends on the facility/class. This is also why
  `threat_corridor_present` was excluded from M2's model matrix — it is
  computed downstream of the prediction it would otherwise leak into.

Run the worker with all three packages importable:
    celery -A app.orchestration.queue.celery_app worker \
           -I classifier.tasks,geospatial.tasks
"""
import logging
from typing import Any, Dict, List, Optional

from app.orchestration.queue import celery_app

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Phase A: attribution + land cover (pre-classification)
# --------------------------------------------------------------------------

@celery_app.task(name="tasks.m3_spatial_attribution")
def spatial_attribution(cluster_id: int, lon: float, lat: float) -> Dict[str, Any]:
    """Probabilistic facility attribution + zonal land cover for a cluster."""
    from sqlalchemy import text

    from app.db.session import session_scope

    from geospatial.attribution.facility_match import (
        CANDIDATE_FACILITIES_SQL,
        attribute,
        attribution_is_confident,
    )
    from geospatial.attribution.landcover_worldcover_cog import landcover_with_fallback_cog
    from geospatial.config import get_m3_settings

    settings = get_m3_settings()

    # Read first, in a short transaction.
    with session_scope() as session:
        rows = [
            dict(row)
            for row in session.execute(
                text(CANDIDATE_FACILITIES_SQL),
                {"cluster_id": cluster_id, "search_radius_m": settings.facility_search_radius_m},
            ).mappings().all()
        ]
        spatial_extent_km = _read_spatial_extent(session, cluster_id)

    # Network round-trip (WorldCover COG windowed read, or Earth Engine
    # if ever configured) happens with NO transaction open. It takes
    # seconds; holding a Postgres transaction across it pins a connection
    # and its locks for the duration, which is how a worker pool starves
    # under load.
    landcover = landcover_with_fallback_cog(lon, lat, spatial_extent_km)

    candidates = attribute(lon, lat, rows, search_radius_m=settings.facility_search_radius_m)

    # Write in a second short transaction.
    with session_scope() as session:
        _persist_attributions(session, cluster_id, candidates)
        _update_fingerprint(session, cluster_id, candidates, landcover)

    confident = attribution_is_confident(candidates, settings.attribution_confidence_margin)
    top = candidates[0] if candidates else None

    # M2 may already have snapshotted the coarser values M1 wrote, so ask
    # it to rebuild. Safe to call repeatedly: it is an upsert.
    celery_app.send_task("tasks.m2_build_feature_vector", args=[cluster_id])

    logger.info(
        "Attributed cluster %s: %d candidates, top=%s (p=%.2f, confident=%s), landcover=%s",
        cluster_id, len(candidates),
        top.facility_id if top else None,
        top.probability if top else 0.0,
        confident, landcover.get("source"),
    )
    return {
        "cluster_id": cluster_id,
        "status": "attributed",
        "n_candidates": len(candidates),
        "top_facility_id": top.facility_id if top else None,
        "confident": confident,
        "landcover_source": landcover.get("source"),
    }


def _read_spatial_extent(session, cluster_id: int) -> Optional[float]:
    """M2's footprint measurement, used to size the land-cover buffer.

    Returns None if M2's package isn't installed or hasn't run yet —
    land cover then falls back to a default radius rather than failing.
    """
    try:
        from classifier.db.models import ClusterFeatures
    except ImportError:
        return None

    row = session.get(ClusterFeatures, cluster_id)
    return getattr(row, "spatial_extent_km", None) if row else None


def _persist_attributions(session, cluster_id: int, candidates) -> None:
    from geospatial.db.models import FacilityAttribution

    existing = {
        row.facility_id: row
        for row in session.query(FacilityAttribution).filter_by(cluster_id=cluster_id).all()
    }

    for candidate in candidates:
        row = existing.pop(candidate.facility_id, None)
        payload = {
            "distance_m": candidate.distance_m,
            "inside_polygon": candidate.inside_polygon,
            "attribution_probability": candidate.probability,
            "rank": candidate.rank,
        }
        if row is None:
            session.add(FacilityAttribution(
                cluster_id=cluster_id, facility_id=candidate.facility_id, **payload
            ))
        else:
            for key, value in payload.items():
                setattr(row, key, value)

    # Facilities that are no longer candidates (the cluster moved, or the
    # registry changed) must not linger with a stale probability.
    for stale in existing.values():
        session.delete(stale)


def _update_fingerprint(session, cluster_id: int, candidates, landcover: Dict[str, Any]) -> None:
    """Write M3's better values into M1's fingerprints row.

    M2 snapshots this row into `cluster_features`, so improving it here
    is what actually gets the better context into the model.
    """
    from app.db.models import Fingerprint

    fingerprint = session.get(Fingerprint, cluster_id)
    if fingerprint is None:
        fingerprint = Fingerprint(cluster_id=cluster_id)
        session.add(fingerprint)

    if candidates:
        top = candidates[0]
        fingerprint.nearest_facility_id = top.facility_id
        fingerprint.facility_distance_m = top.distance_m

    for column in ("pct_cropland", "pct_forest", "pct_urban"):
        value = landcover.get(column)
        if value is not None:
            setattr(fingerprint, column, value)


# --------------------------------------------------------------------------
# Phase B: plume + threat corridor (post-classification)
# --------------------------------------------------------------------------

@celery_app.task(name="tasks.m3_threat_and_plume")
def threat_and_plume(cluster_id: int, predicted_class: Optional[str] = None) -> Dict[str, Any]:
    """Plume footprint, and a threat corridor if the fire is spreading."""
    from app.db.models import Cluster
    from app.db.session import session_scope

    from geospatial.features_io import load_hotspot_rows
    from geospatial.plume.wind import fetch_wind

    # Read first, in a short transaction.
    with session_scope() as session:
        cluster = session.get(Cluster, cluster_id)
        if cluster is None:
            logger.warning("Cluster %s not found — nothing to model", cluster_id)
            return {"cluster_id": cluster_id, "status": "cluster_not_found"}

        from geoalchemy2.shape import to_shape

        centroid = to_shape(cluster.centroid)
        lon, lat = centroid.x, centroid.y
        last_seen = cluster.last_seen
        rows = load_hotspot_rows(session, cluster_id)

    # Open-Meteo round-trip with no transaction open — same reasoning as
    # Phase A: never hold a DB connection across a network call.
    wind = fetch_wind(lon, lat, last_seen)

    # Write in a second short transaction.
    with session_scope() as session:
        plume_result = _build_and_store_plume(
            session, cluster_id, lon, lat, rows, wind, predicted_class
        )
        corridor_result = _build_and_store_corridor(
            session, cluster_id, lon, lat, rows, wind, predicted_class
        )

    logger.info(
        "Modelled cluster %s: plume=%s corridor=%s (wind %.1f m/s from %.0f deg, %s)",
        cluster_id, plume_result.get("status"), corridor_result.get("status"),
        wind.speed_ms, wind.direction_deg, wind.source,
    )

    # WUI proximity is its own task rather than folded into this one:
    # `local_pipeline.process_cluster` needs to invoke it as a separate,
    # individually-guarded stage the same way it does everything else
    # (a WUI evaluation failing must not be reported as a plume/corridor
    # failure), and the Celery deployment chains it the same way M2
    # chains M3's Phase A back into itself. Dispatched unconditionally —
    # `wui_threat` itself gates on SPREADING_CLASSES before doing any
    # network call, so a non-spreading class returns "not applicable"
    # almost instantly rather than needing to be filtered out here too.
    celery_app.send_task("tasks.m3_wui_threat", args=[cluster_id, predicted_class])

    return {
        "cluster_id": cluster_id,
        "status": "modelled",
        "plume": plume_result,
        "corridor": corridor_result,
        "wind_source": wind.source,
    }


def _build_and_store_plume(session, cluster_id, lon, lat, rows, wind, predicted_class) -> Dict[str, Any]:
    from geoalchemy2.shape import from_shape
    from shapely.geometry import Polygon

    from app.db.models import Fingerprint, Facility

    from geospatial.config import get_m3_settings
    from geospatial.db.models import Plume
    from geospatial.plume.chemicals import chemical_profile
    from geospatial.plume.exposure import polygon_area_km2, population_in_polygon
    from geospatial.plume.gaussian import plume_polygon, stability_class, suggested_length_m

    settings = get_m3_settings()

    frp_values = [r["frp"] for r in rows if r.get("frp") is not None]
    peak_frp = max(frp_values) if frp_values else None

    stability = stability_class(wind.speed_ms, wind.is_daytime, wind.cloud_cover_fraction)
    length_m = suggested_length_m(wind.speed_ms, stability, peak_frp)

    geometry = plume_polygon(
        lon, lat,
        wind_direction_deg=wind.direction_deg,
        wind_speed_ms=wind.speed_ms,
        frp_mw=peak_frp,
        length_m=length_m,
        steps=settings.plume_polygon_steps,
        stability=stability,
    )

    facility_type = None
    fingerprint = session.get(Fingerprint, cluster_id)
    if fingerprint is not None and fingerprint.nearest_facility_id is not None:
        facility = session.get(Facility, fingerprint.nearest_facility_id)
        facility_type = facility.facility_type if facility else None

    profile = chemical_profile(facility_type, predicted_class)
    area_km2 = polygon_area_km2(geometry.polygon)
    
    from app.db.models import Cluster
    from pathlib import Path
    cluster = session.get(Cluster, cluster_id)
    year = cluster.first_seen.year if cluster and cluster.first_seen else 2020
    year = max(2018, min(2020, year))
    pop_raster_path = f"data/population/worldpop_india_{year}.tif"
    
    # fallback if dynamically picked one is missing
    if not Path(pop_raster_path).exists():
        pop_raster_path = settings.population_raster
        
    population = population_in_polygon(
        pop_raster_path, geometry.polygon, settings.population_raster_is_density
    )

    payload = {
        "geom": from_shape(Polygon(geometry.polygon), srid=4326),
        "wind_speed_ms": wind.speed_ms,
        "wind_direction_deg": wind.direction_deg,
        "downwind_bearing_deg": geometry.downwind_bearing_deg,
        "wind_source": wind.source,
        "stability_class": geometry.stability_class,
        "plume_rise_m": geometry.plume_rise_m,
        "length_m": geometry.length_m,
        "area_km2": area_km2,
        "chemical_profile": profile,
        "population_exposed": population,
        "population_known": population is not None,
    }

    existing = session.get(Plume, cluster_id)
    if existing is None:
        session.add(Plume(cluster_id=cluster_id, **payload))
    else:
        for key, value in payload.items():
            setattr(existing, key, value)

    return {
        "status": "built",
        "stability_class": geometry.stability_class,
        "area_km2": round(area_km2, 2),
        "population_exposed": population,
    }


def _build_and_store_corridor(session, cluster_id, lon, lat, rows, wind, predicted_class) -> Dict[str, Any]:
    from geoalchemy2.shape import from_shape
    from shapely.geometry import Polygon

    from geospatial.config import get_m3_settings
    from geospatial.db.models import ThreatCorridor
    from geospatial.geometry import downwind_bearing
    from geospatial.threat.corridor import (
        build_corridor,
        facilities_in_corridor,
        should_build_corridor,
    )
    from geospatial.threat.trajectory import (
        compute_trajectory,
        conservative_spread_rate,
        effective_spread_bearing,
    )

    settings = get_m3_settings()

    trajectory = compute_trajectory(rows)
    footprint_growth = _read_growth_rate(session, cluster_id)
    spread_rate = conservative_spread_rate(trajectory, footprint_growth)

    if not should_build_corridor(predicted_class, spread_rate):
        _clear_corridor(session, cluster_id)
        return {
            "status": "not_applicable",
            "reason": f"class={predicted_class}, spread_rate={spread_rate:.3f} km/day",
        }

    bearing = effective_spread_bearing(trajectory, downwind_bearing(wind.direction_deg))
    if bearing is None:
        _clear_corridor(session, cluster_id)
        return {"status": "no_direction"}

    corridor = build_corridor(
        lon, lat, bearing, spread_rate,
        confidence=trajectory.confidence,
        projection_hours=settings.corridor_projection_hours,
    )

    facilities = _load_nearby_facilities(session, lon, lat, corridor.length_km)
    threatened = facilities_in_corridor(
        lon, lat, corridor.bearing_deg, corridor.half_angle_deg,
        corridor.length_km, spread_rate, facilities,
    )

    threatened_payload = [
        {
            "facility_id": f.facility_id,
            "name": f.name,
            "facility_type": f.facility_type,
            "distance_km": round(f.distance_km, 3),
            "time_to_impact_hours": (
                round(f.time_to_impact_hours, 2) if f.time_to_impact_hours is not None else None
            ),
        }
        for f in threatened
    ]
    eta_values = [f.time_to_impact_hours for f in threatened if f.time_to_impact_hours is not None]

    payload = {
        "geom": from_shape(Polygon(corridor.polygon), srid=4326),
        "bearing_deg": corridor.bearing_deg,
        "half_angle_deg": corridor.half_angle_deg,
        "length_km": corridor.length_km,
        "spread_rate_km_day": spread_rate,
        "trajectory_confidence": trajectory.confidence,
        "projection_hours": corridor.projection_hours,
        "threatened_facilities": threatened_payload,
        "min_time_to_impact_hours": min(eta_values) if eta_values else None,
    }

    existing = session.get(ThreatCorridor, cluster_id)
    if existing is None:
        session.add(ThreatCorridor(cluster_id=cluster_id, **payload))
    else:
        for key, value in payload.items():
            setattr(existing, key, value)

    _set_threat_flag(session, cluster_id, bool(threatened))

    return {
        "status": "built",
        "n_threatened": len(threatened),
        "min_time_to_impact_hours": payload["min_time_to_impact_hours"],
    }


def _load_nearby_facilities(session, lon: float, lat: float, radius_km: float) -> List[Dict]:
    from sqlalchemy import text

    sql = """
    SELECT f.id, f.name, f.facility_type,
           ST_X(ST_Centroid(f.geom)) AS lon,
           ST_Y(ST_Centroid(f.geom)) AS lat
    FROM facilities f
    WHERE ST_DWithin(
        f.geom::geography,
        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
        :radius_m
    )
    LIMIT 500;
    """
    rows = session.execute(
        text(sql), {"lon": lon, "lat": lat, "radius_m": radius_km * 1000.0}
    ).mappings().all()
    return [dict(row) for row in rows]


def _read_growth_rate(session, cluster_id: int) -> Optional[float]:
    """M2's `spatial_growth_rate` feature, if M2 has run for this cluster."""
    try:
        from classifier.db.models import ClusterFeatures
    except ImportError:
        return None

    row = session.get(ClusterFeatures, cluster_id)
    return getattr(row, "spatial_growth_rate", None) if row else None


def _set_threat_flag(session, cluster_id: int, present: bool) -> None:
    """Write the single `cluster_features` column M3 owns.

    M2's tests assert that no member clobbers another's columns, so this
    touches `threat_corridor_present` and nothing else. Skipped entirely
    if M2's package isn't installed or hasn't created the row yet.
    """
    try:
        from classifier.db.models import ClusterFeatures
    except ImportError:
        return

    row = session.get(ClusterFeatures, cluster_id)
    if row is None:
        return
    row.threat_corridor_present = present


def _clear_corridor(session, cluster_id: int) -> None:
    """Remove a corridor that no longer applies.

    A cluster reclassified from wildfire to industrial_fire must not keep
    a stale advance corridor on the operations map.
    """
    from geospatial.db.models import ThreatCorridor

    existing = session.get(ThreatCorridor, cluster_id)
    if existing is not None:
        session.delete(existing)
    _set_threat_flag(session, cluster_id, False)



# --------------------------------------------------------------------------
# WUI (wildland-urban interface) proximity — dispatched by threat_and_plume
# --------------------------------------------------------------------------

@celery_app.task(name="tasks.m3_wui_threat")
def wui_threat(cluster_id: int, predicted_class: Optional[str] = None) -> Dict[str, Any]:
    """Wildland-Urban Interface proximity check: how close is this
    (already-classified, already-advancing) fire to the nearest built-up
    land, and does that put it inside ember-jump range or a 12-hour
    projected burn window of it.

    Two short transactions around one network-bound computation, same
    shape as `threat_and_plume` above: read the cluster's context, do
    every network call (WorldCover COG, Overpass, admin boundary lookup)
    with no session open, then write. The actual physics and lookups all
    live in `geospatial.wui_analysis` — this task is the DB-owning
    wrapper around it, not a second implementation.
    """
    from app.db.session import session_scope

    from geospatial.threat.corridor import SPREADING_CLASSES
    from geospatial.wui_analysis import _compute_threat, _read_context

    # Fast path on the caller-supplied class: skips the network/physics
    # work entirely for the common non-spreading case, which is most of
    # them. `_read_context` still re-reads `Classification.predicted_class`
    # itself further down and that value is what actually decides the
    # verdict — the classification is already committed by the time this
    # task runs (classify_cluster persists before dispatching
    # threat_and_plume, which dispatches this), so the two never disagree
    # in the real chained-dispatch path; this check is purely an
    # optimisation, not a second source of truth.
    #
    # It still writes, though a cheap one: a cluster that WAS a WUI
    # threat and got reclassified away from a spreading class (e.g. via
    # `--reclassify`) must not keep a stale CRITICAL flag on the map just
    # because the fast path returned before reaching `_persist_wui`
    # below — same reasoning `_clear_corridor` already applies to a
    # stale industrial threat corridor.
    if predicted_class and predicted_class.lower() not in SPREADING_CLASSES:
        with session_scope() as session:
            _persist_wui(session, cluster_id, {
                "wui_threat": False, "distance_m": None, "eta_hours": None,
                "bearing_deg": None, "threatened_asset": None,
            })
        celery_app.send_task("tasks.m3_resource_demand", args=[cluster_id])
        return {
            "cluster_id": cluster_id, "status": "not_applicable",
            "wui_threat": False, "priority": "NOT_APPLICABLE",
        }

    with session_scope() as session:
        context = _read_context(session, cluster_id)

    if context is None:
        logger.warning("WUI check: cluster %s not found", cluster_id)
        return {"cluster_id": cluster_id, "status": "cluster_not_found"}

    result = _compute_threat(context)

    with session_scope() as session:
        _persist_wui(session, cluster_id, result)

    celery_app.send_task("tasks.m3_resource_demand", args=[cluster_id])

    if result.get("wui_threat"):
        logger.warning(
            "WUI %s: cluster %s is %.0f m from %s, ETA %s h",
            result.get("priority"), cluster_id, result.get("distance_m") or -1.0,
            result.get("threatened_asset"), result.get("eta_hours"),
        )
        from gateway.live import broadcast_sync

        broadcast_sync({
            "type": "WUI_THREAT_ALERT",
            "cluster_id": cluster_id,
            "priority": result.get("priority"),
            "distance_m": result.get("distance_m"),
            "eta_hours": result.get("eta_hours"),
            "bearing_deg": result.get("bearing_deg"),
            "threatened_asset": result.get("threatened_asset"),
        })

    logger.info(
        "WUI check cluster %s: threat=%s priority=%s reason=%s",
        cluster_id, result.get("wui_threat"), result.get("priority"), result.get("reason"),
    )
    return {
        "cluster_id": cluster_id,
        "status": "evaluated",
        "wui_threat": result.get("wui_threat"),
        "priority": result.get("priority"),
        "distance_m": result.get("distance_m"),
        "eta_hours": result.get("eta_hours"),
    }


def _persist_wui(session, cluster_id: int, result: Dict[str, Any]) -> None:
    """Write the five columns this feature owns on `clusters`.

    A cluster reclassified away from a spreading class must not keep a
    stale CRITICAL flag on the map — `_compute_threat` already returns
    `wui_threat=False` / `priority="NOT_APPLICABLE"` for that case, and
    writing those values here clears whatever was there before, the same
    way `_clear_corridor` clears a stale industrial threat corridor.
    """
    from app.db.models import Cluster

    cluster = session.get(Cluster, cluster_id)
    if cluster is None:
        return
    cluster.wui_threat = bool(result.get("wui_threat"))
    cluster.wui_distance_m = result.get("distance_m")
    cluster.wui_eta_hours = result.get("eta_hours")
    cluster.wui_bearing_deg = result.get("bearing_deg")
    cluster.wui_threatened_asset = result.get("threatened_asset")



# --------------------------------------------------------------------------
# Crown-fire FRP threshold detection — independent of WUI, runs for
# every classified cluster regardless of class or proximity to anything.
# --------------------------------------------------------------------------

@celery_app.task(name="tasks.m3_crown_fire")
def crown_fire(cluster_id: int) -> Dict[str, Any]:
    """Is this cluster radiating at (or accelerating toward) crown-fire
    intensity, from its own real detection history — no network calls,
    so this is one short transaction rather than the read/network/write
    split `threat_and_plume` and `wui_threat` need.
    """
    from app.db.session import session_scope

    from geospatial.crown_fire import evaluate_crown_fire
    from geospatial.features_io import load_hotspot_rows

    with session_scope() as session:
        rows = load_hotspot_rows(session, cluster_id)
        if not rows:
            return {"cluster_id": cluster_id, "status": "no_detections"}

        frp_values = [r["frp"] for r in rows if r.get("frp") is not None]
        frp_max = max(frp_values) if frp_values else None

        result = evaluate_crown_fire(frp_max, rows)

        from app.db.models import Cluster

        cluster = session.get(Cluster, cluster_id)
        if cluster is None:
            return {"cluster_id": cluster_id, "status": "cluster_not_found"}
        cluster.is_crown_fire = bool(result["is_crown_fire"])

    if result["is_crown_fire"]:
        logger.warning(
            "CROWN FIRE: cluster %s — %s", cluster_id, result.get("reason"),
        )
        from gateway.live import broadcast_sync

        broadcast_sync({
            "type": "CROWN_FIRE_ALERT",
            "cluster_id": cluster_id,
            "frp_max_mw": result.get("frp_max_mw"),
            "frp_acceleration_mw_per_hour": result.get("frp_acceleration_mw_per_hour"),
            "reason": result.get("reason"),
        })

    return {
        "cluster_id": cluster_id, "status": "evaluated",
        "is_crown_fire": result["is_crown_fire"],
        "frp_max_mw": result.get("frp_max_mw"),
        "frp_acceleration_mw_per_hour": result.get("frp_acceleration_mw_per_hour"),
    }


# --------------------------------------------------------------------------
# WFDSS-inspired Resource Demand Index — dispatched by wui_threat, since
# it needs that task's freshly persisted wui_distance_m. Ranks this
# cluster against every other active one for air-tanker / crew
# allocation, the way WFDSS weighs a small WUI-adjacent fire above a
# large remote one.
# --------------------------------------------------------------------------

@celery_app.task(name="tasks.m3_resource_demand")
def resource_demand(cluster_id: int) -> Dict[str, Any]:
    """Score this cluster's resource demand from data already on
    record — this cluster's real hotspot history plus whatever
    `wui_threat` just persisted. No network calls, one read, one write.
    """
    from app.db.session import session_scope

    from geospatial.decision_engine import (
        RDI_FULL_SUPPRESSION_THRESHOLD,
        evaluate_decision_support,
    )

    with session_scope() as session:
        result = evaluate_decision_support(cluster_id, session)

        from app.db.models import Cluster

        cluster = session.get(Cluster, cluster_id)
        if cluster is None:
            return {"cluster_id": cluster_id, "status": "cluster_not_found"}
        cluster.rdi_score = result["rdi_score"]
        cluster.coa_type = result["coa_type"]
        cluster.area_ha = result.get("area_ha")

    if result["rdi_score"] >= RDI_FULL_SUPPRESSION_THRESHOLD or result.get("wui_threat"):
        logger.warning(
            "RESOURCE DEMAND: cluster %s RDI=%.1f coa=%s",
            cluster_id, result["rdi_score"], result["coa_type"],
        )
        from gateway.live import broadcast_sync

        broadcast_sync({
            "type": "RESOURCE_DEMAND_ALERT",
            "cluster_id": cluster_id,
            "rdi_score": result["rdi_score"],
            "coa_type": result["coa_type"],
            "wui_threat": result.get("wui_threat"),
            "is_crown_fire": result.get("is_crown_fire"),
            "threatened_asset": result.get("wui_threatened_asset"),
        })

    return {
        "cluster_id": cluster_id, "status": "evaluated",
        "rdi_score": result["rdi_score"],
        "coa_type": result["coa_type"],
    }
