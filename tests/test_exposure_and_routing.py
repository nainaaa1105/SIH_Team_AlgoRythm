import pytest

from geospatial.plume.exposure import exposure_summary, polygon_area_km2
from geospatial.threat.routing import point_in_ring, safe_destinations


def _square(lon0, lat0, size_deg):
    return [
        (lon0, lat0),
        (lon0 + size_deg, lat0),
        (lon0 + size_deg, lat0 + size_deg),
        (lon0, lat0 + size_deg),
        (lon0, lat0),
    ]


# --- area ---------------------------------------------------------------

def test_area_of_a_known_square():
    # 0.01 deg at the equator is ~1.113 km, so the square is ~1.24 km^2
    area = polygon_area_km2(_square(0.0, 0.0, 0.01))
    assert area == pytest.approx(1.24, rel=0.05)


def test_area_shrinks_with_latitude_for_the_same_degree_box():
    """Longitude degrees narrow toward the poles."""
    equator = polygon_area_km2(_square(0.0, 0.0, 0.1))
    north = polygon_area_km2(_square(0.0, 60.0, 0.1))
    assert north < equator


def test_area_is_orientation_independent():
    clockwise = _square(0.0, 0.0, 0.01)
    counter = list(reversed(clockwise))
    assert polygon_area_km2(clockwise) == pytest.approx(polygon_area_km2(counter))


def test_area_of_a_degenerate_ring_is_zero():
    assert polygon_area_km2([(0.0, 0.0), (1.0, 1.0)]) == 0.0
    assert polygon_area_km2([]) == 0.0


def test_area_handles_an_unclosed_ring():
    unclosed = _square(0.0, 0.0, 0.01)[:-1]
    assert polygon_area_km2(unclosed) == pytest.approx(polygon_area_km2(_square(0.0, 0.0, 0.01)))


# --- exposure summary ---------------------------------------------------

def test_exposure_summary_computes_density():
    summary = exposure_summary(population=2000, area_km2=4.0)
    assert summary["population_density_per_km2"] == pytest.approx(500.0)
    assert summary["population_known"] is True


def test_unknown_population_is_distinguishable_from_zero():
    """'We don't know' and 'nobody lives there' must not look the same
    to a responder."""
    unknown = exposure_summary(population=None, area_km2=4.0)
    empty = exposure_summary(population=0, area_km2=4.0)

    assert unknown["population_known"] is False
    assert unknown["population_exposed"] is None
    assert empty["population_known"] is True
    assert empty["population_exposed"] == 0


def test_zero_area_does_not_divide_by_zero():
    summary = exposure_summary(population=100, area_km2=0.0)
    assert summary["population_density_per_km2"] is None


# --- point in polygon ---------------------------------------------------

def test_point_inside_a_square():
    assert point_in_ring(0.005, 0.005, _square(0.0, 0.0, 0.01))


def test_point_outside_a_square():
    assert not point_in_ring(0.02, 0.005, _square(0.0, 0.0, 0.01))
    assert not point_in_ring(0.005, 0.02, _square(0.0, 0.0, 0.01))
    assert not point_in_ring(-0.01, 0.005, _square(0.0, 0.0, 0.01))


def test_degenerate_ring_contains_nothing():
    assert not point_in_ring(0.0, 0.0, [(0.0, 0.0), (1.0, 1.0)])


def test_point_in_ring_handles_a_concave_shape():
    # A C-shape: the notch on the right is outside the polygon.
    c_shape = [
        (0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0),
        (1.0, 2.0), (3.0, 2.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0),
    ]
    assert point_in_ring(0.5, 1.5, c_shape)       # inside the spine
    assert not point_in_ring(2.0, 1.5, c_shape)   # inside the notch


# --- safe destinations --------------------------------------------------

def test_safe_destinations_avoid_the_hazard_polygon():
    hazard = _square(76.99, 27.99, 0.05)
    destinations = safe_destinations(77.0, 28.0, hazard, distance_m=20000)

    assert destinations
    for lon, lat in destinations:
        assert not point_in_ring(lon, lat, hazard)


def test_no_safe_destinations_when_the_hazard_covers_everything():
    huge = _square(70.0, 20.0, 20.0)
    assert safe_destinations(77.0, 28.0, huge, distance_m=5000) == []


def test_custom_bearings_are_respected():
    hazard = _square(76.99, 27.99, 0.001)
    destinations = safe_destinations(77.0, 28.0, hazard, distance_m=10000, bearings=[0.0, 180.0])
    assert len(destinations) == 2
