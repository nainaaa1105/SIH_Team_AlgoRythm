"""WFDSS-inspired Resource Demand Index (RDI).

A single 0-100 score that ranks clusters against each other for
resource allocation the way the WFDSS-style example this feature is
modelled on does: a small, WUI-adjacent, high-FRP fire outranks a much
larger, remote, low-FRP one.

This module deliberately does NOT re-derive proximity or thermal
history from scratch — it reuses signals this codebase already
computes for real, from real sources, elsewhere:

- Proximity: the cluster's persisted `wui_distance_m`, from
  `wui_analysis.py`'s real WorldCover/Overpass nearest-settlement
  lookup. There is no `admin_boundaries` table with settlement
  name/geometry columns in this schema to query directly, and even if
  there were, computing distance a second, independent way here could
  silently disagree with the number already driving the WUI alert for
  the same fire.
- FRP: this cluster's own real hotspot history, the same source
  `crown_fire.py` reads.
- Footprint: `spatial_extent_km`, the real max-pairwise-distance
  between this cluster's actual detections
  (`classifier/features/thermal.py`) — not a fabricated acreage
  figure. Converted to a resource-planning area proxy, not a survey
  acreage.
"""
import math
from typing import Any, Dict, Optional

from geospatial.wui_analysis import EMBER_JUMP_THRESHOLD_M

RDI_FULL_SUPPRESSION_THRESHOLD = 80.0
RDI_POINT_ZONE_THRESHOLD = 50.0

_FRP_SCORE_CAP_MW = 150.0
_FRP_SCORE_WEIGHT = 35.0

_PROXIMITY_SCORE_WEIGHT = 40.0
_PROXIMITY_TAPER_KM = 10.0
_PROXIMITY_FAR_SCORE = 5.0

_AREA_SCORE_CAP_HA = 500.0
_AREA_SCORE_WEIGHT = 25.0
_KM2_TO_HA = 100.0

COA_FULL_SUPPRESSION = "FULL_SUPPRESSION_AIR_TANKERS"
COA_POINT_ZONE = "POINT_ZONE_PROTECTION"
COA_MONITOR = "MONITOR_ECOLOGICAL_BENEFIT"


def frp_score(frp_max_mw: Optional[float]) -> float:
    if not frp_max_mw or frp_max_mw <= 0:
        return 0.0
    return min((frp_max_mw / _FRP_SCORE_CAP_MW) * _FRP_SCORE_WEIGHT, _FRP_SCORE_WEIGHT)


def proximity_score(distance_m: Optional[float]) -> float:
    """None means the WUI check found no built-up land within its search
    radius at all — a confirmed-far result, not missing data — so it
    scores the same as a fire that measured far away, rather than
    guessing a worst case.
    """
    if distance_m is None:
        return _PROXIMITY_FAR_SCORE
    distance_km = distance_m / 1000.0
    ember_km = EMBER_JUMP_THRESHOLD_M / 1000.0
    if distance_km <= ember_km:
        return _PROXIMITY_SCORE_WEIGHT
    if distance_km <= _PROXIMITY_TAPER_KM:
        # Linear taper from the max score at the ember-jump boundary
        # down to the far-score floor at the 10 km boundary — not down
        # to 0, which would undershoot the floor right at 10 km.
        span = _PROXIMITY_TAPER_KM - ember_km
        fraction = (distance_km - ember_km) / span
        return _PROXIMITY_SCORE_WEIGHT - (_PROXIMITY_SCORE_WEIGHT - _PROXIMITY_FAR_SCORE) * fraction
    return _PROXIMITY_FAR_SCORE


def area_score(area_ha: Optional[float]) -> float:
    if not area_ha or area_ha <= 0:
        return 0.0
    return min((area_ha / _AREA_SCORE_CAP_HA) * _AREA_SCORE_WEIGHT, _AREA_SCORE_WEIGHT)


def footprint_area_ha(spatial_extent_km: Optional[float]) -> float:
    """Circle-of-diameter approximation from the cluster's real detection
    footprint. A resource-planning proxy for fire size, not a survey
    acreage figure — the system has no ground-truth perimeter polygon.
    """
    if not spatial_extent_km or spatial_extent_km <= 0:
        return 0.0
    radius_km = spatial_extent_km / 2.0
    return math.pi * radius_km * radius_km * _KM2_TO_HA


def course_of_action(rdi_score: float) -> str:
    if rdi_score >= RDI_FULL_SUPPRESSION_THRESHOLD:
        return COA_FULL_SUPPRESSION
    if rdi_score >= RDI_POINT_ZONE_THRESHOLD:
        return COA_POINT_ZONE
    return COA_MONITOR


def compute_rdi(
    frp_max_mw: Optional[float],
    distance_m: Optional[float],
    area_ha: Optional[float],
) -> Dict[str, Any]:
    """Pure scoring function — no I/O. Exposed separately from
    `evaluate_decision_support` so it can be unit-tested against known
    inputs without a database.
    """
    f = frp_score(frp_max_mw)
    p = proximity_score(distance_m)
    a = area_score(area_ha)
    score = round(min(f + p + a, 100.0), 1)
    return {
        "rdi_score": score,
        "coa_type": course_of_action(score),
        "frp_component": round(f, 1),
        "proximity_component": round(p, 1),
        "area_component": round(a, 1),
    }


def evaluate_decision_support(cluster_id: int, db_session) -> Dict[str, Any]:
    """Entry point: compute the RDI for one cluster from data already on
    record — this cluster's own real hotspot history plus whatever
    `wui_threat` last persisted for it. No network calls, one read.
    """
    from app.db.models import Cluster
    from classifier.features.thermal import spatial_extent_km as compute_spatial_extent_km
    from geospatial.features_io import load_hotspot_rows

    rows = load_hotspot_rows(db_session, cluster_id)
    frp_values = [r["frp"] for r in rows if r.get("frp") is not None]
    frp_max = max(frp_values) if frp_values else None
    extent_km = compute_spatial_extent_km(rows) if rows else 0.0
    area_ha = footprint_area_ha(extent_km)

    cluster = db_session.get(Cluster, cluster_id)
    distance_m = cluster.wui_distance_m if cluster else None
    wui_threat = bool(cluster.wui_threat) if cluster else False
    is_crown_fire = bool(cluster.is_crown_fire) if cluster else False
    threatened_asset = cluster.wui_threatened_asset if cluster else None
    eta_hours = cluster.wui_eta_hours if cluster else None

    result = compute_rdi(frp_max, distance_m, area_ha)
    result.update({
        "cluster_id": cluster_id,
        "frp_max_mw": frp_max,
        "spatial_extent_km": round(extent_km, 3) if rows else None,
        "area_ha": round(area_ha, 1),
        "wui_distance_m": distance_m,
        "wui_threat": wui_threat,
        "wui_eta_hours": eta_hours,
        "wui_threatened_asset": threatened_asset,
        "is_crown_fire": is_crown_fire,
    })
    return result
