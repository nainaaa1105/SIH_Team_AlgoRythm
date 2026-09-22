from datetime import datetime, timedelta, timezone

import pytest

from geospatial.threat.trajectory import (
    compute_trajectory,
    conservative_spread_rate,
    effective_spread_bearing,
)

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _row(lon, lat, hours):
    return {"lon": lon, "lat": lat, "acq_datetime": BASE + timedelta(hours=hours)}


def _northward_track(n=6, step_deg=0.05, bucket_hours=24):
    return [_row(77.0, 28.0 + i * step_deg, i * bucket_hours) for i in range(n)]


def test_no_trajectory_from_a_single_observation():
    trajectory = compute_trajectory([_row(77.0, 28.0, 0)])
    assert trajectory.bearing_deg is None
    assert trajectory.travel_rate_km_day == 0.0
    assert trajectory.confidence == 0.0


def test_no_trajectory_from_empty_input():
    trajectory = compute_trajectory([])
    assert trajectory.bearing_deg is None


def test_northward_movement_is_detected():
    trajectory = compute_trajectory(_northward_track())
    assert trajectory.bearing_deg == pytest.approx(0.0, abs=2.0)
    assert trajectory.travel_rate_km_day > 0


def test_eastward_movement_is_detected():
    rows = [_row(77.0 + i * 0.05, 28.0, i * 24) for i in range(6)]
    trajectory = compute_trajectory(rows)
    assert trajectory.bearing_deg == pytest.approx(90.0, abs=2.0)


def test_stationary_source_has_no_direction_and_no_rate():
    """A flare sitting in one place must not produce a spread vector."""
    rows = [_row(77.0, 28.0, i * 24) for i in range(6)]
    trajectory = compute_trajectory(rows)
    assert trajectory.bearing_deg is None
    assert trajectory.travel_rate_km_day == 0.0


def test_travel_rate_is_in_km_per_day():
    # 0.09 deg of latitude ~ 10 km, over one day
    rows = [_row(77.0, 28.0, 0), _row(77.0, 28.09, 24)]
    trajectory = compute_trajectory(rows, bucket_hours=6)
    assert trajectory.travel_rate_km_day == pytest.approx(10.0, rel=0.05)


def test_straight_line_scores_higher_confidence_than_a_wandering_path():
    straight = compute_trajectory(_northward_track())

    wandering = compute_trajectory([
        _row(77.0, 28.00, 0), _row(77.0, 28.05, 24), _row(77.0, 28.00, 48),
        _row(77.0, 28.05, 72), _row(77.0, 28.01, 96),
    ])
    assert straight.confidence > wandering.confidence


def test_confidence_is_bounded():
    trajectory = compute_trajectory(_northward_track())
    assert 0.0 <= trajectory.confidence <= 1.0


def test_only_recent_buckets_drive_the_direction():
    """A fire that ran north for days then turned east should be
    projected east, not averaged into the middle."""
    north_leg = [_row(77.0, 28.0 + i * 0.05, i * 24) for i in range(6)]
    east_leg = [_row(77.0 + i * 0.05, 28.25, (6 + i) * 24) for i in range(1, 6)]

    trajectory = compute_trajectory(north_leg + east_leg, max_buckets=4)
    assert trajectory.bearing_deg == pytest.approx(90.0, abs=25.0)


def test_dense_overpass_bursts_are_bucketed_not_over_counted():
    """Ten detections in one minute is one look, not ten."""
    burst = [_row(77.0 + i * 0.0001, 28.0, 0) for i in range(10)]
    later = [_row(77.0, 28.1, 24)]
    trajectory = compute_trajectory(burst + later, bucket_hours=12)
    assert trajectory.n_observations == 2


def test_rows_without_timestamps_are_ignored():
    rows = _northward_track() + [{"lon": 99.0, "lat": 9.0, "acq_datetime": None}]
    trajectory = compute_trajectory(rows)
    assert trajectory.bearing_deg == pytest.approx(0.0, abs=2.0)


# --- blending with wind -------------------------------------------------

def test_wind_is_used_when_there_is_no_observed_trajectory():
    trajectory = compute_trajectory([_row(77.0, 28.0, 0)])
    assert effective_spread_bearing(trajectory, 90.0) == 90.0


def test_observed_trajectory_is_used_when_there_is_no_wind():
    trajectory = compute_trajectory(_northward_track())
    blended = effective_spread_bearing(trajectory, None)
    assert blended == pytest.approx(trajectory.bearing_deg)


def test_blend_sits_between_observation_and_wind():
    trajectory = compute_trajectory(_northward_track())   # heading ~0 (north)
    blended = effective_spread_bearing(trajectory, 90.0)  # wind pushes east
    assert 0.0 < blended < 90.0


def test_blend_leans_toward_the_observation_when_it_is_confident():
    """A well-tracked front should not be dragged far by the wind term."""
    confident = compute_trajectory(_northward_track())
    blended = effective_spread_bearing(confident, 90.0)
    assert blended < 45.0


def test_returns_none_when_neither_source_has_a_direction():
    trajectory = compute_trajectory([_row(77.0, 28.0, 0)])
    assert effective_spread_bearing(trajectory, None) is None


# --- conservative rate --------------------------------------------------

def test_conservative_rate_takes_the_larger_of_the_two_measures():
    """Under-estimating how soon a fire arrives is the dangerous error."""
    trajectory = compute_trajectory(_northward_track())
    assert conservative_spread_rate(trajectory, footprint_growth_km_day=999.0) == 999.0
    assert conservative_spread_rate(trajectory, footprint_growth_km_day=0.001) == pytest.approx(
        trajectory.travel_rate_km_day
    )


def test_conservative_rate_ignores_a_missing_or_negative_growth_rate():
    trajectory = compute_trajectory(_northward_track())
    expected = trajectory.travel_rate_km_day
    assert conservative_spread_rate(trajectory, None) == pytest.approx(expected)
    assert conservative_spread_rate(trajectory, -5.0) == pytest.approx(expected)


def test_conservative_rate_is_never_negative():
    trajectory = compute_trajectory([_row(77.0, 28.0, 0)])
    assert conservative_spread_rate(trajectory, None) >= 0.0
