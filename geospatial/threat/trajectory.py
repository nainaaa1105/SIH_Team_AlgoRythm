"""Fire-front trajectory from a cluster's detection history.

Takes the time-ordered hotspot rows M1 stores and works out which way
the fire is actually moving and how fast — the inputs to the threat
corridor.

Two different "speeds" matter and they are not the same quantity:
  * centroid travel rate — how fast the centre of mass is moving, i.e.
    directional advance;
  * footprint growth rate — how fast the fire's extent is expanding,
    which M2 already computes as `spatial_growth_rate`.

A fire burning outward in all directions has a large growth rate and a
near-zero travel rate. A wind-driven front has both. For a
safety-relevant time-to-impact we take the larger of the two, because
under-estimating how soon a fire reaches a facility is the dangerous
direction to be wrong in.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from geospatial.geometry import bearing_deg, circular_mean, haversine_m


@dataclass
class Trajectory:
    bearing_deg: Optional[float]        # None when the fire isn't going anywhere
    travel_rate_km_day: float
    total_displacement_km: float
    duration_days: float
    n_observations: int
    confidence: float                   # 0-1, how well-defined the direction is


def _grouped_centroids(
    rows: Sequence[Dict], bucket_hours: float = 12.0
) -> List[Tuple[datetime, float, float]]:
    """Collapse detections into time-bucketed centroids.

    Satellites revisit in bursts, so raw detections cluster in time.
    Averaging within a bucket stops a single dense overpass dominating
    the direction estimate.
    """
    dated = [
        r for r in rows
        if r.get("acq_datetime") is not None
        and r.get("lon") is not None
        and r.get("lat") is not None
    ]
    if not dated:
        return []

    ordered = sorted(dated, key=lambda r: r["acq_datetime"])
    start = ordered[0]["acq_datetime"]
    bucket_seconds = bucket_hours * 3600.0

    buckets: Dict[int, List[Dict]] = {}
    for row in ordered:
        index = int((row["acq_datetime"] - start).total_seconds() // bucket_seconds)
        buckets.setdefault(index, []).append(row)

    centroids = []
    for index in sorted(buckets):
        members = buckets[index]
        centroids.append((
            min(m["acq_datetime"] for m in members),
            sum(m["lon"] for m in members) / len(members),
            sum(m["lat"] for m in members) / len(members),
        ))
    return centroids


def compute_trajectory(
    rows: Sequence[Dict], bucket_hours: float = 12.0, max_buckets: int = 6
) -> Trajectory:
    """Direction and rate of advance from the most recent observations.

    Only the last `max_buckets` time buckets are used: a fire that ran
    north for a week and then turned east should be projected east.
    """
    centroids = _grouped_centroids(rows, bucket_hours)

    if len(centroids) < 2:
        return Trajectory(
            bearing_deg=None, travel_rate_km_day=0.0, total_displacement_km=0.0,
            duration_days=0.0, n_observations=len(centroids), confidence=0.0,
        )

    recent = centroids[-max_buckets:]

    legs_bearings: List[float] = []
    legs_weights: List[float] = []
    path_length_km = 0.0

    for (t0, lon0, lat0), (t1, lon1, lat1) in zip(recent, recent[1:]):
        leg_m = haversine_m(lon0, lat0, lon1, lat1)
        path_length_km += leg_m / 1000.0
        if leg_m < 1.0:
            continue  # stationary leg carries no direction information
        legs_bearings.append(bearing_deg(lon0, lat0, lon1, lat1))
        legs_weights.append(leg_m)

    first, last = recent[0], recent[-1]
    duration_days = (last[0] - first[0]).total_seconds() / 86400.0
    net_km = haversine_m(first[1], first[2], last[1], last[2]) / 1000.0

    if not legs_bearings or duration_days <= 0:
        return Trajectory(
            bearing_deg=None, travel_rate_km_day=0.0, total_displacement_km=net_km,
            duration_days=max(duration_days, 0.0), n_observations=len(recent), confidence=0.0,
        )

    mean_bearing = circular_mean(legs_bearings, legs_weights)

    # Straightness: net displacement over path length. A fire that
    # wandered back and forth has a low ratio and a poorly-defined
    # direction, so downstream code can widen the corridor accordingly.
    straightness = net_km / path_length_km if path_length_km > 0 else 0.0
    confidence = max(0.0, min(1.0, straightness))

    return Trajectory(
        bearing_deg=mean_bearing,
        travel_rate_km_day=net_km / duration_days,
        total_displacement_km=net_km,
        duration_days=duration_days,
        n_observations=len(recent),
        confidence=confidence,
    )


def effective_spread_bearing(
    trajectory: Trajectory, downwind_bearing_deg: Optional[float], wind_weight: float = 0.4
) -> Optional[float]:
    """Blend observed movement with the downwind direction.

    Fire spreads preferentially downwind, so when the observed trajectory
    is poorly defined the wind should dominate; when the fire is clearly
    running in one direction, the observation should. The blend weight is
    scaled by trajectory confidence for exactly that reason.
    """
    if trajectory.bearing_deg is None:
        return downwind_bearing_deg
    if downwind_bearing_deg is None:
        return trajectory.bearing_deg

    observed_weight = max(0.0, min(1.0, trajectory.confidence))
    effective_wind_weight = wind_weight * (1.0 - observed_weight) + wind_weight * 0.5 * observed_weight

    return circular_mean(
        [trajectory.bearing_deg, downwind_bearing_deg],
        [1.0 - effective_wind_weight, effective_wind_weight],
    )


def conservative_spread_rate(
    trajectory: Trajectory, footprint_growth_km_day: Optional[float] = None
) -> float:
    """The faster of centroid travel and footprint growth.

    `footprint_growth_km_day` is M2's `spatial_growth_rate` feature,
    already stored on `cluster_features`. Taking the max is deliberate:
    time-to-impact feeds an evacuation decision, and the safe error is to
    say a fire arrives sooner than it does.
    """
    rates = [max(trajectory.travel_rate_km_day, 0.0)]
    if footprint_growth_km_day is not None and footprint_growth_km_day > 0:
        rates.append(float(footprint_growth_km_day))
    return max(rates)
