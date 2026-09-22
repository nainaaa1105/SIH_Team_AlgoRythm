"""Geospatial read API.

Mounts onto M1's existing FastAPI app, same as M2's router:

    # app/main.py
    from geospatial.api.routes_geo import router as geo_router
    app.include_router(geo_router)

Geometries come back as GeoJSON so M6's Leaflet layers can consume them
directly with no client-side conversion.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(tags=["geospatial"])


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: List[List[List[float]]]


class PlumeOut(BaseModel):
    cluster_id: int
    geometry: Optional[GeoJSONPolygon]
    wind_speed_ms: Optional[float]
    wind_direction_deg: Optional[float]
    downwind_bearing_deg: Optional[float]
    wind_source: Optional[str]
    wind_is_assumed: bool
    stability_class: Optional[str]
    plume_rise_m: Optional[float]
    area_km2: Optional[float]
    chemical_profile: Optional[Dict[str, Any]]
    population_exposed: Optional[int]
    population_known: bool


class ThreatenedFacilityOut(BaseModel):
    facility_id: int
    name: Optional[str] = None
    facility_type: Optional[str] = None
    distance_km: Optional[float] = None
    time_to_impact_hours: Optional[float] = None


class ThreatOut(BaseModel):
    cluster_id: int
    geometry: Optional[GeoJSONPolygon]
    bearing_deg: Optional[float]
    half_angle_deg: Optional[float]
    length_km: Optional[float]
    spread_rate_km_day: Optional[float]
    trajectory_confidence: Optional[float]
    projection_hours: Optional[float]
    threatened_facilities: List[ThreatenedFacilityOut]
    min_time_to_impact_hours: Optional[float]


class AttributionCandidateOut(BaseModel):
    facility_id: int
    name: Optional[str]
    facility_type: Optional[str]
    distance_m: Optional[float]
    inside_polygon: bool
    attribution_probability: Optional[float]
    rank: Optional[int]


class AttributionOut(BaseModel):
    cluster_id: int
    confident: bool
    candidates: List[AttributionCandidateOut]


class WuiThreatOut(BaseModel):
    cluster_id: int
    wui_threat: bool
    priority: str
    distance_m: Optional[float] = None
    eta_hours: Optional[float] = None
    bearing_deg: Optional[float] = None
    threatened_asset: Optional[str] = None
    settlement_name: Optional[str] = None
    spread_rate_km_day: Optional[float] = None
    corridor_geojson: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class ResourceDemandOut(BaseModel):
    cluster_id: int
    rdi_score: float
    coa_type: str
    frp_component: float
    proximity_component: float
    area_component: float
    frp_max_mw: Optional[float] = None
    spatial_extent_km: Optional[float] = None
    area_ha: Optional[float] = None
    wui_distance_m: Optional[float] = None
    wui_threat: bool = False
    wui_eta_hours: Optional[float] = None
    wui_threatened_asset: Optional[str] = None
    is_crown_fire: bool = False


class SuppressionOut(BaseModel):
    cluster_id: int
    predicted_class: Optional[str] = None
    area_ha: float
    measured_area_ha: float
    intensity_multiplier: float = 1.0
    priority: Optional[str] = None
    priority_label: str
    priority_level: str = "LOW"
    resource_kind: str
    resource_label: str
    primary_volume_l: Optional[float] = None
    primary_volume_m3: Optional[float] = None
    tanker_trips: int = 0
    air_drop_trips: int = 0
    truck_loads: int = 0
    well_control_units: int = 0
    air_support_recommended: bool = False
    is_estimate: bool = True
    notes: List[str] = []


def _to_geojson(geom) -> Optional[GeoJSONPolygon]:
    if geom is None:
        return None
    from geoalchemy2.shape import to_shape

    shape = to_shape(geom)
    return GeoJSONPolygon(coordinates=[[list(c) for c in shape.exterior.coords]])


@router.get("/plume/{cluster_id}", response_model=PlumeOut)
def get_plume(cluster_id: int, db: Session = Depends(get_db)):
    from geospatial.db.models import Plume

    row = db.get(Plume, cluster_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no plume modelled for this cluster")

    return PlumeOut(
        cluster_id=row.cluster_id,
        geometry=_to_geojson(row.geom),
        wind_speed_ms=row.wind_speed_ms,
        wind_direction_deg=row.wind_direction_deg,
        downwind_bearing_deg=row.downwind_bearing_deg,
        wind_source=row.wind_source,
        # Surfaced rather than buried: a plume drawn from assumed wind is
        # a much weaker claim than one drawn from an observation.
        wind_is_assumed=(row.wind_source == "fallback"),
        stability_class=row.stability_class,
        plume_rise_m=row.plume_rise_m,
        area_km2=row.area_km2,
        chemical_profile=row.chemical_profile,
        population_exposed=row.population_exposed,
        population_known=bool(row.population_known),
    )


@router.get("/threat/{cluster_id}", response_model=ThreatOut)
def get_threat(cluster_id: int, db: Session = Depends(get_db)):
    from geospatial.db.models import ThreatCorridor

    row = db.get(ThreatCorridor, cluster_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="no threat corridor for this cluster (not a spreading source, or not advancing)",
        )

    return ThreatOut(
        cluster_id=row.cluster_id,
        geometry=_to_geojson(row.geom),
        bearing_deg=row.bearing_deg,
        half_angle_deg=row.half_angle_deg,
        length_km=row.length_km,
        spread_rate_km_day=row.spread_rate_km_day,
        trajectory_confidence=row.trajectory_confidence,
        projection_hours=row.projection_hours,
        threatened_facilities=[
            ThreatenedFacilityOut(**f) for f in (row.threatened_facilities or [])
        ],
        min_time_to_impact_hours=row.min_time_to_impact_hours,
    )


@router.get("/attribution/{cluster_id}", response_model=AttributionOut)
def get_attribution(cluster_id: int, db: Session = Depends(get_db)):
    from app.db.models import Facility

    from geospatial.config import get_m3_settings
    from geospatial.db.models import FacilityAttribution

    rows = (
        db.query(FacilityAttribution)
        .filter_by(cluster_id=cluster_id)
        .order_by(FacilityAttribution.rank)
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="no attribution for this cluster")

    candidates = []
    for row in rows:
        facility = db.get(Facility, row.facility_id)
        candidates.append(AttributionCandidateOut(
            facility_id=row.facility_id,
            name=facility.name if facility else None,
            facility_type=facility.facility_type if facility else None,
            distance_m=row.distance_m,
            inside_polygon=bool(row.inside_polygon),
            attribution_probability=row.attribution_probability,
            rank=row.rank,
        ))

    margin = get_m3_settings().attribution_confidence_margin
    confident = len(candidates) == 1 or (
        len(candidates) >= 2
        and (candidates[0].attribution_probability or 0)
        - (candidates[1].attribution_probability or 0)
        >= margin
    )

    return AttributionOut(cluster_id=cluster_id, confident=confident, candidates=candidates)


@router.get("/geospatial/wui/{cluster_id}", response_model=WuiThreatOut)
def get_wui_threat(cluster_id: int, db: Session = Depends(get_db)):
    """Live Wildland-Urban Interface proximity evaluation for one
    cluster: distance to the nearest ESA WorldCover built-up pixel, the
    12-hour projected-arrival ETA from the same spread-rate physics the
    industrial threat corridor uses, and the resulting dispatch
    priority.

    Recomputed on every call ("live PostGIS calculations", per spec)
    rather than only returning the cached wui_* columns on `clusters` —
    those cached columns are what `/dashboard/detections` and
    `/dashboard/alerts` render between requests; this endpoint is the
    "evaluate it right now" one.
    """
    from app.db.models import Cluster

    if db.get(Cluster, cluster_id) is None:
        raise HTTPException(status_code=404, detail="cluster not found")

    from geospatial.wui_analysis import evaluate_wui_threat

    result = evaluate_wui_threat(cluster_id, db)

    return WuiThreatOut(
        cluster_id=result.get("cluster_id", cluster_id),
        wui_threat=bool(result.get("wui_threat")),
        priority=result.get("priority", "NOT_APPLICABLE"),
        distance_m=result.get("distance_m"),
        eta_hours=result.get("eta_hours"),
        bearing_deg=result.get("bearing_deg"),
        threatened_asset=result.get("threatened_asset"),
        settlement_name=result.get("settlement_name"),
        spread_rate_km_day=result.get("spread_rate_km_day"),
        corridor_geojson=result.get("corridor_geojson"),
        reason=result.get("reason"),
    )


@router.get("/geospatial/resource-demand/{cluster_id}", response_model=ResourceDemandOut)
def get_resource_demand(cluster_id: int, db: Session = Depends(get_db)):
    """Live WFDSS-inspired Resource Demand Index for one cluster: ranks
    it against every other active cluster for air-tanker/crew
    allocation, from this cluster's own real FRP history, footprint,
    and its persisted WUI proximity verdict.

    Recomputed on every call, same reasoning as `/geospatial/wui/{id}`
    above — this is the "evaluate it right now" endpoint; the cached
    `rdi_score`/`coa_type` columns are what the dashboard list renders
    between requests.
    """
    from app.db.models import Cluster

    if db.get(Cluster, cluster_id) is None:
        raise HTTPException(status_code=404, detail="cluster not found")

    from geospatial.decision_engine import evaluate_decision_support

    result = evaluate_decision_support(cluster_id, db)

    return ResourceDemandOut(
        cluster_id=result.get("cluster_id", cluster_id),
        rdi_score=result["rdi_score"],
        coa_type=result["coa_type"],
        frp_component=result["frp_component"],
        proximity_component=result["proximity_component"],
        area_component=result["area_component"],
        frp_max_mw=result.get("frp_max_mw"),
        spatial_extent_km=result.get("spatial_extent_km"),
        area_ha=result.get("area_ha"),
        wui_distance_m=result.get("wui_distance_m"),
        wui_threat=bool(result.get("wui_threat")),
        wui_eta_hours=result.get("wui_eta_hours"),
        wui_threatened_asset=result.get("wui_threatened_asset"),
        is_crown_fire=bool(result.get("is_crown_fire")),
    )


@router.get("/geospatial/suppression/{cluster_id}", response_model=SuppressionOut)
def get_suppression_estimate(cluster_id: int, db: Session = Depends(get_db)):
    """Live suppression-resource estimate for one cluster: real class,
    real FRP, real footprint area in, a doctrine-based material/fleet
    estimate out. See geospatial/suppression.py for what's real
    (the inputs) versus estimated (everything derived from them,
    always flagged `is_estimate: True`).

    Computed for every cluster regardless of class or size — an
    unclassified or zero-footprint fire still gets an honest result
    (geospatial/suppression.py handles those explicitly), not a 404.
    """
    from app.db.models import Cluster

    cluster = db.get(Cluster, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="cluster not found")

    from classifier.db.models import Classification
    from geospatial.decision_engine import evaluate_decision_support
    from geospatial.features_io import load_hotspot_rows
    from geospatial.suppression import estimate_suppression

    classification = db.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    predicted_class = classification.predicted_class if classification else None

    # Real resolving instrument for this cluster's own detections — the
    # sensor-pixel-footprint floor in estimate_suppression() needs to
    # know whether it's VIIRS (375 m) or MODIS (1 km), not a guess.
    hotspot_rows = load_hotspot_rows(db, cluster_id)
    sensor = None
    for row in hotspot_rows:
        source = (row.get("source") or "").upper()
        if "VIIRS" in source:
            sensor = "VIIRS"
            break
        if "MODIS" in source:
            sensor = "MODIS"
            break

    decision = evaluate_decision_support(cluster_id, db)
    result = estimate_suppression(
        predicted_class=predicted_class,
        frp_max_mw=decision.get("frp_max_mw"),
        area_ha=decision.get("area_ha"),
        coa_type=decision.get("coa_type"),
        sensor=sensor,
    )

    return SuppressionOut(cluster_id=cluster_id, **result)
