"""Threat corridor: the wedge a spreading fire is likely to advance into,
and which facilities sit inside it.

Only meaningful once the source has been classified as a spreading fire
— which is exactly why M2 excluded `threat_corridor_present` from the
model's feature matrix (a corridor is computed downstream of
classification, so feeding it back would be target leakage) and why this
runs in M3's Phase B rather than alongside attribution.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from geospatial.geometry import (
    angular_difference,
    close_ring,
    destination_point,
    haversine_m,
    normalise_bearing,
)

# Classes worth projecting a corridor for. A gas flare or a fixed
# industrial source does not advance across the landscape, so drawing a
# corridor for one would be actively misleading on an operations map.
SPREADING_CLASSES = ("wildfire", "agricultural_burning")

MIN_HALF_ANGLE_DEG = 15.0
MAX_HALF_ANGLE_DEG = 60.0
DEFAULT_PROJECTION_HOURS = 24.0


@dataclass
class Corridor:
    polygon: List[Tuple[float, float]]
    bearing_deg: float
    half_angle_deg: float
    length_km: float
    spread_rate_km_day: float
    projection_hours: float


@dataclass
class ThreatenedFacility:
    facility_id: int
    name: Optional[str]
    facility_type: Optional[str]
    distance_km: float
    bearing_deg: float
    time_to_impact_hours: Optional[float]


def half_angle_for_confidence(confidence: float) -> float:
    """Uncertain direction => wider wedge.

    A confidently-tracked fire front gets a narrow corridor; one whose
    observed movement is erratic gets a wide one, because we genuinely
    don't know which way it will run.
    """
    clamped = max(0.0, min(1.0, confidence))
    return MAX_HALF_ANGLE_DEG - (MAX_HALF_ANGLE_DEG - MIN_HALF_ANGLE_DEG) * clamped


def corridor_polygon(
    lon: float,
    lat: float,
    bearing: float,
    length_km: float,
    half_angle_deg: float,
    arc_steps: int = 12,
) -> List[Tuple[float, float]]:
    """A wedge rooted at the fire, opening along `bearing`.

    The far edge is drawn as an arc rather than a straight chord so every
    point on the boundary is the same distance from the source — the
    boundary is "how far it can get", which is a radius, not a line.
    """
    if length_km <= 0:
        raise ValueError("length_km must be positive")
    if not 0 < half_angle_deg < 180:
        raise ValueError("half_angle_deg must be between 0 and 180 exclusive")

    length_m = length_km * 1000.0
    start = normalise_bearing(bearing - half_angle_deg)
    sweep = 2 * half_angle_deg

    ring: List[Tuple[float, float]] = [(lon, lat)]
    for i in range(arc_steps + 1):
        edge_bearing = normalise_bearing(start + sweep * i / arc_steps)
        ring.append(destination_point(lon, lat, edge_bearing, length_m))

    return close_ring(ring)


def time_to_impact_hours(distance_km: float, spread_rate_km_day: float) -> Optional[float]:
    """Hours for a front advancing at `spread_rate_km_day` to cover a distance.

    Returns None for a non-advancing fire rather than infinity, so the
    caller stores a NULL the dashboard can render as "not advancing"
    instead of an absurd number.
    """
    if spread_rate_km_day <= 0:
        return None
    if distance_km <= 0:
        return 0.0
    return (distance_km / spread_rate_km_day) * 24.0


def facilities_in_corridor(
    lon: float,
    lat: float,
    bearing: float,
    half_angle_deg: float,
    length_km: float,
    spread_rate_km_day: float,
    facilities: Sequence[Dict],
) -> List[ThreatenedFacility]:
    """Which facilities fall inside the wedge, nearest (soonest) first.

    Pure-Python containment test on (lon, lat) dicts so the geometry is
    unit-testable; the DB path in tasks.py uses PostGIS ST_Intersects
    against the stored polygon for the same result at scale.
    """
    threatened: List[ThreatenedFacility] = []

    for facility in facilities:
        f_lon, f_lat = facility.get("lon"), facility.get("lat")
        if f_lon is None or f_lat is None:
            continue

        distance_km = haversine_m(lon, lat, f_lon, f_lat) / 1000.0
        if distance_km > length_km:
            continue

        from geospatial.geometry import bearing_deg as _bearing

        facility_bearing = _bearing(lon, lat, f_lon, f_lat)
        if angular_difference(facility_bearing, bearing) > half_angle_deg:
            continue

        threatened.append(ThreatenedFacility(
            facility_id=facility["id"],
            name=facility.get("name"),
            facility_type=facility.get("facility_type"),
            distance_km=distance_km,
            bearing_deg=facility_bearing,
            time_to_impact_hours=time_to_impact_hours(distance_km, spread_rate_km_day),
        ))

    threatened.sort(key=lambda f: f.distance_km)
    return threatened


def build_corridor(
    lon: float,
    lat: float,
    bearing: float,
    spread_rate_km_day: float,
    confidence: float = 0.5,
    projection_hours: float = DEFAULT_PROJECTION_HOURS,
    min_length_km: float = 1.0,
) -> Corridor:
    """Project the fire forward for `projection_hours` at its spread rate."""
    length_km = max(min_length_km, spread_rate_km_day * projection_hours / 24.0)
    half_angle = half_angle_for_confidence(confidence)

    return Corridor(
        polygon=corridor_polygon(lon, lat, bearing, length_km, half_angle),
        bearing_deg=normalise_bearing(bearing),
        half_angle_deg=half_angle,
        length_km=length_km,
        spread_rate_km_day=spread_rate_km_day,
        projection_hours=projection_hours,
    )


def should_build_corridor(predicted_class: Optional[str], spread_rate_km_day: float) -> bool:
    """Gate: only spreading classes that are actually spreading."""
    if predicted_class is None:
        return False
    if predicted_class.lower() not in SPREADING_CLASSES:
        return False
    return spread_rate_km_day > 0
