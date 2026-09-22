"""Polygon validity on the production storage path.

The tasks wrap every ring in `shapely.geometry.Polygon(...)` before
handing it to GeoAlchemy2 for a PostGIS geometry column. A ring that is
self-intersecting, degenerate or zero-area passes every geometric unit
test and then fails (or silently stores garbage) at write time, so it is
worth exercising the actual conversion.
"""
import pytest

from geospatial.plume.gaussian import plume_polygon, suggested_length_m
from geospatial.threat.corridor import build_corridor, corridor_polygon

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402

SOURCE = (77.0, 28.0)


@pytest.mark.parametrize("wind_direction", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 359.0])
@pytest.mark.parametrize("stability", ["A", "D", "F"])
def test_plume_rings_form_valid_polygons(wind_direction, stability):
    geometry = plume_polygon(
        *SOURCE, wind_direction_deg=wind_direction, wind_speed_ms=5.0,
        length_m=10000, stability=stability,
    )
    polygon = Polygon(geometry.polygon)
    assert polygon.is_valid
    assert not polygon.is_empty
    assert polygon.area > 0


def test_plume_polygon_is_not_self_intersecting():
    """The ring runs out one flank and back the other; if the order were
    wrong it would produce a bow-tie that PostGIS rejects."""
    geometry = plume_polygon(*SOURCE, wind_direction_deg=270.0, wind_speed_ms=5.0, length_m=15000)
    assert Polygon(geometry.polygon).is_simple


def test_calm_wind_plume_is_still_a_valid_polygon():
    geometry = plume_polygon(*SOURCE, wind_direction_deg=0.0, wind_speed_ms=0.0, length_m=2000, steps=2)
    assert Polygon(geometry.polygon).is_valid


def test_minimum_steps_still_produces_a_valid_polygon():
    geometry = plume_polygon(*SOURCE, wind_direction_deg=90.0, wind_speed_ms=3.0, steps=2)
    assert Polygon(geometry.polygon).is_valid


def test_plume_contains_a_point_just_downwind_of_the_source():
    """Sanity check that the footprint covers the area it claims to."""
    from geospatial.geometry import destination_point
    from shapely.geometry import Point

    geometry = plume_polygon(
        *SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=10000, stability="D"
    )
    downwind_point = destination_point(*SOURCE, geometry.downwind_bearing_deg, 5000.0)
    assert Polygon(geometry.polygon).contains(Point(*downwind_point))


def test_plume_does_not_contain_a_point_upwind():
    from geospatial.geometry import destination_point
    from shapely.geometry import Point

    geometry = plume_polygon(
        *SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=10000, stability="D"
    )
    upwind_bearing = (geometry.downwind_bearing_deg + 180.0) % 360.0
    upwind_point = destination_point(*SOURCE, upwind_bearing, 5000.0)
    assert not Polygon(geometry.polygon).contains(Point(*upwind_point))


@pytest.mark.parametrize("bearing", [0.0, 90.0, 180.0, 270.0, 359.0])
@pytest.mark.parametrize("half_angle", [15.0, 30.0, 60.0])
def test_corridor_rings_form_valid_polygons(bearing, half_angle):
    ring = corridor_polygon(*SOURCE, bearing=bearing, length_km=10, half_angle_deg=half_angle)
    polygon = Polygon(ring)
    assert polygon.is_valid
    assert polygon.area > 0


def test_tiny_corridor_is_still_valid():
    corridor = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=0.0001, confidence=0.0)
    assert Polygon(corridor.polygon).is_valid
    assert corridor.length_km >= 1.0


def test_corridor_contains_a_point_directly_ahead():
    from geospatial.geometry import destination_point
    from shapely.geometry import Point

    corridor = build_corridor(*SOURCE, bearing=0.0, spread_rate_km_day=20.0, confidence=1.0)
    ahead = destination_point(*SOURCE, 0.0, corridor.length_km * 1000 * 0.5)
    assert Polygon(corridor.polygon).contains(Point(*ahead))


def test_suggested_lengths_always_produce_valid_plumes():
    """The task uses suggested_length_m rather than a fixed length, so the
    whole range it can emit must be safe to store."""
    for speed in (0.0, 2.0, 5.0, 15.0, 40.0):
        for stability in "ABCDEF":
            length = suggested_length_m(speed, stability, frp_mw=100.0)
            geometry = plume_polygon(
                *SOURCE, wind_direction_deg=180.0, wind_speed_ms=speed,
                length_m=length, stability=stability,
            )
            assert Polygon(geometry.polygon).is_valid
