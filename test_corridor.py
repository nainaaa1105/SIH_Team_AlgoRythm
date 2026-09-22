import pytest

from geospatial.geometry import bearing_deg, haversine_m
from geospatial.threat.corridor import (
    MAX_HALF_ANGLE_DEG,
    MIN_HALF_ANGLE_DEG,
    build_corridor,
    corridor_polygon,
    facilities_in_corridor,
    half_angle_for_confidence,
    should_build_corridor,
    time_to_impact_hours,
)

SOURCE = (77.0, 28.0)


def _facility(fid, lon, lat, ftype="refinery", name="Plant"):
    return {"id": fid, "lon": lon, "lat": lat, "facility_type": ftype, "name": name}


# --- gating -------------------------------------------------------------

def test_corridor_only_for_spreading_classes():
    """A gas flare does not advance across the landscape; drawing it a
    corridor would be actively misleading on an operations map."""
    assert should_build_corridor("wildfire", 5.0) is True
    assert should_build_corridor("agricultural_burning", 5.0) is True
    assert should_build_corridor("gas_flare", 5.0) is False
    assert should_build_corridor("industrial_fire", 5.0) is False
    assert should_build_corridor("mining", 5.0) is False


def test_no_corridor_for_a_wildfire_that_is_not_advancing():
    assert should_build_corridor("wildfire", 0.0) is False


def test_no_corridor_without_a_classification():
    assert should_build_corridor(None, 5.0) is False


def test_gating_is_case_insensitive():
    assert should_build_corridor("WILDFIRE", 5.0) is True


# --- half angle ---------------------------------------------------------

def test_uncertain_direction_gives_a_wider_wedge():
    assert half_angle_for_confidence(0.0) > half_angle_for_confidence(1.0)


def test_half_angle_stays_within_bounds():
    for confidence in (-1.0, 0.0, 0.5, 1.0, 2.0):
        angle = half_angle_for_confidence(confidence)
        assert MIN_HALF_ANGLE_DEG <= angle <= MAX_HALF_ANGLE_DEG


# --- polygon ------------------------------------------------------------

def test_corridor_polygon_is_closed_and_rooted_at_the_source():
    ring = corridor_polygon(*SOURCE, bearing=90.0, length_km=10, half_angle_deg=30)
    assert ring[0] == ring[-1] == SOURCE


def test_corridor_points_lie_at_the_projected_range():
    ring = corridor_polygon(*SOURCE, bearing=0.0, length_km=10, half_angle_deg=30)
    for point in ring[1:-1]:
        assert haversine_m(*SOURCE, *point) == pytest.approx(10000, rel=0.02)


def test_corridor_opens_along_the_given_bearing():
    ring = corridor_polygon(*SOURCE, bearing=90.0, length_km=10, half_angle_deg=20)
    for point in ring[1:-1]:
        assert bearing_deg(*SOURCE, *point) == pytest.approx(90.0, abs=21.0)


def test_wider_half_angle_produces_a_wider_wedge():
    narrow = corridor_polygon(*SOURCE, bearing=0.0, length_km=10, half_angle_deg=10)
    wide = corridor_polygon(*SOURCE, bearing=0.0, length_km=10, half_angle_deg=60)
    narrow_span = haversine_m(*narrow[1], *narrow[-2])
    wide_span = haversine_m(*wide[1], *wide[-2])
    assert wide_span > narrow_span


def test_corridor_polygon_validates_inputs():
    with pytest.raises(ValueError):
        corridor_polygon(*SOURCE, bearing=0.0, length_km=0, half_angle_deg=30)
    with pytest.raises(ValueError):
        corridor_polygon(*SOURCE, bearing=0.0, length_km=10, half_angle_deg=0)
    with pytest.raises(ValueError):
        corridor_polygon(*SOURCE, bearing=0.0, length_km=10, half_angle_deg=200)


# --- time to impact -----------------------------------------------------

def test_time_to_impact_basic_arithmetic():
    assert time_to_impact_hours(distance_km=10.0, spread_rate_km_day=10.0) == pytest.approx(24.0)
    assert time_to_impact_hours(distance_km=5.0, spread_rate_km_day=10.0) == pytest.approx(12.0)


def test_time_to_impact_is_none_for_a_non_advancing_fire():
    """None, not infinity — the dashboard renders 'not advancing'."""
    assert time_to_impact_hours(10.0, 0.0) is None
    assert time_to_impact_hours(10.0, -1.0) is None


def test_time_to_impact_zero_when_already_there():
    assert time_to_impact_hours(0.0, 10.0) == 0.0


# --- facilities in corridor ---------------------------------------------

def test_facility_directly_ahead_is_threatened():
    ahead = _facility(1, 77.0, 28.05)   # ~5.5 km north
    threatened = facilities_in_corridor(
        *SOURCE, bearing=0.0, half_angle_deg=30, length_km=20,
        spread_rate_km_day=10.0, facilities=[ahead],
    )
    assert len(threatened) == 1
    assert threatened[0].facility_id == 1
    assert threatened[0].time_to_impact_hours is not None


def test_facility_behind_the_fire_is_not_threatened():
    behind = _facility(2, 77.0, 27.95)  # south, fire heading north
    threatened = facilities_in_corridor(
        *SOURCE, bearing=0.0, half_angle_deg=30, length_km=20,
        spread_rate_km_day=10.0, facilities=[behind],
    )
    assert threatened == []


def test_facility_beyond_the_projection_range_is_not_threatened():
    far = _facility(3, 77.0, 29.5)      # ~165 km north
    threatened = facilities_in_corridor(
        *SOURCE, bearing=0.0, half_angle_deg=30, length_km=20,
        spread_rate_km_day=10.0, facilities=[far],
    )
    assert threatened == []


def test_facility_outside_the_wedge_angle_is_not_threatened():
    sideways = _facility(4, 77.10, 28.0)  # due east, fire heading north
    threatened = facilities_in_corridor(
        *SOURCE, bearing=0.0, half_angle_deg=20, length_km=20,
        spread_rate_km_day=10.0, facilities=[sideways],
    )
    assert threatened == []


def test_threatened_facilities_are_ordered_nearest_first():
    facilities = [
        _facility(1, 77.0, 28.10),
        _facility(2, 77.0, 28.03),
        _facility(3, 77.0, 28.06),
    ]
    threatened = facilities_in_corridor(
        *SOURCE, bearing=0.0, half_angle_deg=40, length_km=30,
        spread_rate_km_day=10.0, facilities=facilities,
    )
    assert [f.facility_id for f in threatened] == [2, 3, 1]
    etas = [f.time_to_impact_hours for f in threatened]
    assert etas == sorted(etas)


def test_facilities_without_coordinates_are_skipped():
    broken = {"id": 9, "lon": None, "lat": None}
    threatened = facilities_in_corridor(
        *SOURCE, bearing=0.0, half_angle_deg=30, length_km=20,
        spread_rate_km_day=10.0, facilities=[broken],
    )
    assert threatened == []


# --- build_corridor -----------------------------------------------------

def test_build_corridor_projects_rate_over_the_projection_window():
    corridor = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=12.0, projection_hours=24)
    assert corridor.length_km == pytest.approx(12.0)


def test_build_corridor_halves_the_length_for_a_twelve_hour_window():
    corridor = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=12.0, projection_hours=12)
    assert corridor.length_km == pytest.approx(6.0)


def test_build_corridor_enforces_a_minimum_length():
    corridor = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=0.001, projection_hours=24)
    assert corridor.length_km >= 1.0


def test_build_corridor_widens_when_the_trajectory_is_uncertain():
    confident = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=10.0, confidence=1.0)
    uncertain = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=10.0, confidence=0.0)
    assert uncertain.half_angle_deg > confident.half_angle_deg
