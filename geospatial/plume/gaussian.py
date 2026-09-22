"""Gaussian plume dispersion.

Implemented directly from published atmospheric-dispersion formulae
rather than via a package. The team task-division doc names "pyELDQM"
for this; I could not verify that such a library exists, and building a
safety-relevant calculation on an unverifiable dependency is a bad
trade when the underlying physics is this well documented and compact.

References for the constants used below:
  * Pasquill-Gifford stability classification (wind speed x insolation).
  * Briggs (1973) open-country/rural dispersion coefficients for
    sigma_y and sigma_z.
  * Briggs final plume rise for buoyant sources.

Scope and honesty about it: this is a steady-state, flat-terrain,
single-source Gaussian model. It is appropriate for a screening-level
hazard footprint on a dashboard. It is NOT a regulatory dispersion model
— it ignores terrain, building downwash, plume depletion, chemistry and
wind shear, and it assumes the wind is constant over the whole domain.
The output is a decision-support cone, not a compliance product.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from geospatial.geometry import close_ring, destination_point, downwind_bearing

# Briggs (1973) rural / open-country dispersion coefficients.
# sigma_y = a*x / sqrt(1 + b*x); sigma_z per-class form below.
_SIGMA_Y_COEFF = {"A": 0.22, "B": 0.16, "C": 0.11, "D": 0.08, "E": 0.06, "F": 0.04}

# Radiative fraction of a fire's total heat release. FIRMS reports the
# radiative component (FRP), so total heat release ~= FRP / this.
RADIATIVE_FRACTION = 0.17

# Ambient constants for the buoyancy-flux calculation.
_G = 9.81            # m/s^2
_AIR_DENSITY = 1.2   # kg/m^3
_AIR_CP = 1004.0     # J/(kg.K)
_AMBIENT_T = 293.0   # K

# Lateral extent of the drawn polygon, in multiples of sigma_y.
# At y = 2.146 * sigma_y the concentration is 10% of the centreline
# value, which is the contour we render as the visible hazard edge.
LATERAL_SIGMA_MULTIPLE = 2.146

MIN_WIND_SPEED_MS = 0.5  # calm-wind floor; the model degenerates at u -> 0


@dataclass
class PlumeGeometry:
    polygon: List[Tuple[float, float]]   # (lon, lat) ring, explicitly closed
    centreline: List[Tuple[float, float]]
    downwind_bearing_deg: float
    stability_class: str
    plume_rise_m: float
    length_m: float
    max_half_width_m: float


def stability_class(
    wind_speed_ms: float, is_daytime: bool, cloud_cover_fraction: Optional[float] = None
) -> str:
    """Pasquill-Gifford stability class from wind speed and insolation.

    Daytime insolation is inferred from cloud cover (we have no direct
    solar-radiation feed): little cloud implies strong incoming
    radiation, heavy cloud implies slight. Night-time is split at the
    conventional 4/8 cloud-cover boundary.
    """
    u = max(wind_speed_ms, 0.0)
    cloud = 0.5 if cloud_cover_fraction is None else max(0.0, min(1.0, cloud_cover_fraction))

    if is_daytime:
        if cloud < 0.35:
            insolation = "strong"
        elif cloud < 0.75:
            insolation = "moderate"
        else:
            insolation = "slight"

        if u < 2:
            return {"strong": "A", "moderate": "A", "slight": "B"}[insolation]
        if u < 3:
            return {"strong": "A", "moderate": "B", "slight": "C"}[insolation]
        if u < 5:
            return {"strong": "B", "moderate": "B", "slight": "C"}[insolation]
        if u < 6:
            return {"strong": "C", "moderate": "C", "slight": "D"}[insolation]
        return {"strong": "C", "moderate": "D", "slight": "D"}[insolation]

    # Night. Thin/low cloud cools the surface faster => more stable.
    if u < 2:
        return "F"
    if u < 3:
        return "F" if cloud < 0.5 else "E"
    if u < 5:
        return "E" if cloud < 0.5 else "D"
    return "D"


def sigma_y(distance_m: float, stability: str) -> float:
    """Crosswind (horizontal) dispersion coefficient, metres."""
    if distance_m <= 0:
        return 0.0
    a = _SIGMA_Y_COEFF.get(stability.upper())
    if a is None:
        raise ValueError(f"unknown stability class {stability!r}; expected A-F")
    return a * distance_m / math.sqrt(1.0 + 0.0001 * distance_m)


def sigma_z(distance_m: float, stability: str) -> float:
    """Vertical dispersion coefficient, metres (Briggs rural)."""
    if distance_m <= 0:
        return 0.0
    x = distance_m
    cls = stability.upper()
    if cls == "A":
        return 0.20 * x
    if cls == "B":
        return 0.12 * x
    if cls == "C":
        return 0.08 * x / math.sqrt(1.0 + 0.0002 * x)
    if cls == "D":
        return 0.06 * x / math.sqrt(1.0 + 0.0015 * x)
    if cls == "E":
        return 0.03 * x / (1.0 + 0.0003 * x)
    if cls == "F":
        return 0.016 * x / (1.0 + 0.0003 * x)
    raise ValueError(f"unknown stability class {stability!r}; expected A-F")


def buoyancy_flux(frp_mw: float) -> float:
    """Briggs buoyancy flux parameter F (m^4/s^3) from fire radiative power.

    FRP is the radiated component only, so total heat release is scaled
    up by the radiative fraction before converting to a buoyancy flux.
    """
    if frp_mw is None or frp_mw <= 0:
        return 0.0
    total_heat_w = (frp_mw * 1e6) / RADIATIVE_FRACTION
    return (_G * total_heat_w) / (math.pi * _AIR_DENSITY * _AIR_CP * _AMBIENT_T)


def plume_rise(frp_mw: float, wind_speed_ms: float) -> float:
    """Briggs final rise for a buoyant plume, metres above the source."""
    flux = buoyancy_flux(frp_mw)
    if flux <= 0:
        return 0.0
    u = max(wind_speed_ms, MIN_WIND_SPEED_MS)
    if flux < 55.0:
        return 21.4 * (flux ** 0.75) / u
    return 38.7 * (flux ** 0.6) / u


def concentration(
    downwind_m: float,
    crosswind_m: float,
    receptor_height_m: float,
    emission_rate_g_s: float,
    wind_speed_ms: float,
    effective_height_m: float,
    stability: str,
) -> float:
    """Steady-state Gaussian concentration (g/m^3) with ground reflection.

    Provided so the plume footprint can be validated against an actual
    concentration field rather than only trusting the drawn polygon.
    """
    if downwind_m <= 0:
        return 0.0
    sy = sigma_y(downwind_m, stability)
    sz = sigma_z(downwind_m, stability)
    if sy <= 0 or sz <= 0:
        return 0.0

    u = max(wind_speed_ms, MIN_WIND_SPEED_MS)
    lateral = math.exp(-(crosswind_m ** 2) / (2 * sy ** 2))

    z, h = receptor_height_m, effective_height_m
    vertical = math.exp(-((z - h) ** 2) / (2 * sz ** 2)) + math.exp(-((z + h) ** 2) / (2 * sz ** 2))

    return (emission_rate_g_s / (2 * math.pi * u * sy * sz)) * lateral * vertical


def plume_polygon(
    lon: float,
    lat: float,
    wind_direction_deg: float,
    wind_speed_ms: float,
    frp_mw: Optional[float] = None,
    length_m: float = 10000.0,
    steps: int = 24,
    is_daytime: bool = True,
    cloud_cover_fraction: Optional[float] = None,
    stability: Optional[str] = None,
) -> PlumeGeometry:
    """Build the downwind hazard footprint as a (lon, lat) polygon ring.

    The shape is the 10%-of-centreline concentration contour: at each
    downwind step the lateral half-width is LATERAL_SIGMA_MULTIPLE *
    sigma_y. The ring runs out along one flank and back along the other,
    so the result is a closed, non-self-intersecting teardrop rooted at
    the source.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    resolved_stability = stability or stability_class(
        wind_speed_ms, is_daytime, cloud_cover_fraction
    )
    travel_bearing = downwind_bearing(wind_direction_deg)
    left_bearing = (travel_bearing - 90.0) % 360.0
    right_bearing = (travel_bearing + 90.0) % 360.0

    centreline: List[Tuple[float, float]] = []
    left_edge: List[Tuple[float, float]] = []
    right_edge: List[Tuple[float, float]] = []
    max_half_width = 0.0

    for i in range(steps + 1):
        distance = length_m * i / steps
        centre = destination_point(lon, lat, travel_bearing, distance)
        centreline.append(centre)

        half_width = LATERAL_SIGMA_MULTIPLE * sigma_y(distance, resolved_stability)
        max_half_width = max(max_half_width, half_width)

        left_edge.append(destination_point(centre[0], centre[1], left_bearing, half_width))
        right_edge.append(destination_point(centre[0], centre[1], right_bearing, half_width))

    # Out along the right flank, back along the left, so the ring closes
    # cleanly at the source without crossing itself.
    ring = close_ring(right_edge + list(reversed(left_edge)))

    return PlumeGeometry(
        polygon=ring,
        centreline=centreline,
        downwind_bearing_deg=travel_bearing,
        stability_class=resolved_stability,
        plume_rise_m=plume_rise(frp_mw or 0.0, wind_speed_ms),
        length_m=length_m,
        max_half_width_m=max_half_width,
    )


def suggested_length_m(wind_speed_ms: float, stability: str, frp_mw: Optional[float] = None) -> float:
    """Pick a sensible plume length instead of always drawing 10 km.

    Stronger winds carry material further before it disperses, bigger
    fires inject more material, and stable air keeps a plume coherent for
    longer. Clamped to 2-50 km so the dashboard never gets a degenerate
    or absurd polygon.
    """
    base = 5000.0 + 2000.0 * max(wind_speed_ms, 0.0)
    if stability.upper() in ("E", "F"):
        base *= 1.5      # stable air: less mixing, longer coherent plume
    elif stability.upper() in ("A", "B"):
        base *= 0.7      # unstable air: rapid vertical mixing, shorter reach
    if frp_mw:
        base *= 1.0 + min(frp_mw / 500.0, 1.0)
    return max(2000.0, min(base, 50000.0))
