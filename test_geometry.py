"""Geodesy foundations.

Every polygon M3 produces is built on these operations. A sign error
here puts a toxic-plume warning on the wrong side of a town, so the
conventions are pinned explicitly rather than assumed.
"""
import math

import pytest

from geospatial.geometry import (
    angular_difference,
    bearing_deg,
    circular_mean,
    close_ring,
    destination_point,
    downwind_bearing,
    haversine_m,
    normalise_bearing,
)

DELHI = (77.1025, 28.7041)
MUMBAI = (72.8777, 19.0760)


def test_haversine_zero_for_identical_points():
    assert haversine_m(*DELHI, *DELHI) == 0.0


def test_haversine_known_distance():
    d = haversine_m(*DELHI, *MUMBAI)
    assert 1_100_000 < d < 1_200_000


def test_bearing_cardinal_directions():
    lon, lat = 77.0, 28.0
    assert bearing_deg(lon, lat, lon, lat + 1) == pytest.approx(0.0, abs=0.5)      # north
    assert bearing_deg(lon, lat, lon + 1, lat) == pytest.approx(90.0, abs=0.5)     # east
    assert bearing_deg(lon, lat, lon, lat - 1) == pytest.approx(180.0, abs=0.5)    # south
    assert bearing_deg(lon, lat, lon - 1, lat) == pytest.approx(270.0, abs=0.5)    # west


def test_bearing_is_always_in_zero_to_360():
    for dlon, dlat in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        b = bearing_deg(77.0, 28.0, 77.0 + dlon, 28.0 + dlat)
        assert 0.0 <= b < 360.0


def test_destination_point_moves_north_for_bearing_zero():
    lon, lat = destination_point(77.0, 28.0, 0.0, 10000.0)
    assert lon == pytest.approx(77.0, abs=1e-9)
    assert lat > 28.0


def test_destination_point_moves_east_for_bearing_ninety():
    lon, lat = destination_point(77.0, 28.0, 90.0, 10000.0)
    assert lon > 77.0
    assert lat == pytest.approx(28.0, abs=1e-9)


def test_destination_point_round_trips_with_haversine():
    start = (77.0, 28.0)
    for bearing in (0, 45, 90, 135, 180, 225, 270, 315):
        end = destination_point(*start, bearing, 5000.0)
        assert haversine_m(*start, *end) == pytest.approx(5000.0, rel=0.01)
        assert bearing_deg(*start, *end) == pytest.approx(bearing, abs=0.5)


def test_destination_point_longitude_scaling_varies_with_latitude():
    """A degree of longitude is narrower near the poles, so the same
    eastward distance must produce a larger longitude change up north."""
    near_equator = destination_point(77.0, 5.0, 90.0, 10000.0)[0] - 77.0
    far_north = destination_point(77.0, 60.0, 90.0, 10000.0)[0] - 77.0
    assert far_north > near_equator


def test_destination_point_does_not_explode_at_the_pole():
    lon, lat = destination_point(77.0, 90.0, 90.0, 10000.0)
    assert math.isfinite(lon) and math.isfinite(lat)


def test_downwind_bearing_inverts_meteorological_convention():
    """A westerly wind (270, coming FROM the west) blows TOWARD the east."""
    assert downwind_bearing(270.0) == 90.0
    assert downwind_bearing(0.0) == 180.0     # northerly blows south
    assert downwind_bearing(90.0) == 270.0    # easterly blows west
    assert downwind_bearing(180.0) == 0.0     # southerly blows north


def test_downwind_bearing_stays_in_range():
    for direction in (0, 45, 179, 180, 181, 359, 360):
        assert 0.0 <= downwind_bearing(direction) < 360.0


def test_angular_difference_takes_the_short_way_round():
    assert angular_difference(350.0, 10.0) == pytest.approx(20.0)
    assert angular_difference(10.0, 350.0) == pytest.approx(20.0)
    assert angular_difference(0.0, 180.0) == pytest.approx(180.0)
    assert angular_difference(90.0, 90.0) == 0.0


def test_circular_mean_handles_the_wraparound():
    """A plain arithmetic mean of 350 and 10 gives 180 — backwards."""
    assert circular_mean([350.0, 10.0]) == pytest.approx(0.0, abs=0.01)


def test_circular_mean_simple_case():
    assert circular_mean([80.0, 100.0]) == pytest.approx(90.0, abs=0.01)


def test_circular_mean_respects_weights():
    weighted = circular_mean([0.0, 90.0], [3.0, 1.0])
    assert 0.0 < weighted < 45.0


def test_circular_mean_of_opposed_bearings_does_not_crash():
    result = circular_mean([0.0, 180.0])
    assert 0.0 <= result < 360.0


def test_circular_mean_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError):
        circular_mean([])
    with pytest.raises(ValueError):
        circular_mean([0.0, 90.0], [1.0])


def test_normalise_bearing_wraps():
    assert normalise_bearing(370.0) == pytest.approx(10.0)
    assert normalise_bearing(-10.0) == pytest.approx(350.0)


def test_close_ring_appends_the_first_point():
    ring = close_ring([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    assert ring[0] == ring[-1]
    assert len(ring) == 4


def test_close_ring_leaves_an_already_closed_ring_alone():
    closed = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    assert close_ring(closed) == closed
