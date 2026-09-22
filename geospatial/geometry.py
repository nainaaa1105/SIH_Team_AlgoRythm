"""Geodesy helpers shared by the plume and threat modules.

Everything here is pure and unit-tested, because every polygon M3
produces is built on these three operations and a sign error in any of
them puts a toxic-plume warning on the wrong side of a town.

Conventions used throughout M3 (stated once, here):
  * Bearings are degrees clockwise from true north: 0=N, 90=E, 180=S, 270=W.
  * Coordinates are (lon, lat) in that order, matching PostGIS/GeoJSON
    and M1's existing `to_shape(...).x / .y` usage — NOT the (lat, lon)
    order that slips in from mapping libraries.
  * Meteorological wind direction is the direction wind blows FROM.
    See `downwind_bearing`.
"""
import math
from typing import List, Tuple

EARTH_RADIUS_M = 6371000.0
_METRES_PER_DEGREE_LAT = 111320.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two (lon, lat) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def destination_point(lon: float, lat: float, bearing: float, distance_m: float) -> Tuple[float, float]:
    """Point reached by travelling `distance_m` along `bearing` from (lon, lat).

    Uses a local equirectangular approximation rather than the full
    spherical formula. At the distances M3 works with (plume cones and
    threat corridors of a few km to a few tens of km) the error is well
    under a metre, and it keeps the inverse relationship with
    `haversine_m`/`bearing_deg` numerically clean for testing.
    """
    bearing_rad = math.radians(bearing)
    dlat = (distance_m * math.cos(bearing_rad)) / _METRES_PER_DEGREE_LAT

    # Longitude degrees shrink toward the poles. Guard the degenerate
    # case at the poles where cos(lat) -> 0 and dlon would explode.
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-9:
        dlon = 0.0
    else:
        dlon = (distance_m * math.sin(bearing_rad)) / (_METRES_PER_DEGREE_LAT * cos_lat)

    return lon + dlon, lat + dlat


def downwind_bearing(wind_direction_deg: float) -> float:
    """Convert a meteorological wind direction into a travel bearing.

    Meteorological convention: wind direction is where the wind comes
    FROM. A "westerly" wind of 270 deg blows TOWARD the east (90 deg), so
    a plume from that wind travels on bearing 90. Getting this backwards
    puts the hazard zone on precisely the wrong side of the source, which
    is why it is a named function with its own tests rather than an
    inline `+ 180`.
    """
    return (wind_direction_deg + 180.0) % 360.0


def normalise_bearing(bearing: float) -> float:
    return bearing % 360.0


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute angle between two bearings, in [0, 180]."""
    diff = abs(normalise_bearing(a) - normalise_bearing(b)) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def circular_mean(bearings: List[float], weights: List[float] | None = None) -> float:
    """Weighted mean of bearings, done on the unit circle.

    A plain arithmetic mean of 350 deg and 10 deg gives 180 deg — exactly
    backwards. Averaging the unit vectors gives 0 deg, which is correct.
    """
    if not bearings:
        raise ValueError("cannot average an empty list of bearings")
    if weights is None:
        weights = [1.0] * len(bearings)
    if len(weights) != len(bearings):
        raise ValueError("bearings and weights must be the same length")

    x = sum(w * math.sin(math.radians(b)) for b, w in zip(bearings, weights))
    y = sum(w * math.cos(math.radians(b)) for b, w in zip(bearings, weights))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        # Diametrically opposed bearings cancel out; no meaningful mean.
        return normalise_bearing(bearings[0])
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def close_ring(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ensure a coordinate ring is explicitly closed, as PostGIS requires."""
    if not points:
        return points
    if points[0] != points[-1]:
        return [*points, points[0]]
    return points
