"""Gaussian plume model.

These tests pin the physics behaviour (does a more stable atmosphere give
a narrower plume? does a bigger fire loft higher?) and the geometry
behaviour (is the polygon actually downwind, closed, and non-degenerate?).
"""
import math

import pytest

from geospatial.geometry import angular_difference, bearing_deg, haversine_m
from geospatial.plume.gaussian import (
    LATERAL_SIGMA_MULTIPLE,
    buoyancy_flux,
    concentration,
    plume_polygon,
    plume_rise,
    sigma_y,
    sigma_z,
    stability_class,
    suggested_length_m,
)

SOURCE = (77.0, 28.0)


# --- stability classification ------------------------------------------

def test_light_wind_strong_sun_is_very_unstable():
    assert stability_class(1.0, is_daytime=True, cloud_cover_fraction=0.0) == "A"


def test_strong_wind_is_neutral_regardless_of_sun():
    assert stability_class(8.0, is_daytime=True, cloud_cover_fraction=0.5) == "D"


def test_calm_clear_night_is_most_stable():
    assert stability_class(1.0, is_daytime=False, cloud_cover_fraction=0.1) == "F"


def test_windy_night_tends_neutral():
    assert stability_class(7.0, is_daytime=False, cloud_cover_fraction=0.5) == "D"


def test_stability_class_always_returns_a_valid_class():
    for speed in (0.0, 1.0, 2.5, 4.0, 5.5, 10.0):
        for day in (True, False):
            for cloud in (0.0, 0.5, 1.0, None):
                assert stability_class(speed, day, cloud) in set("ABCDEF")


def test_overcast_day_is_less_unstable_than_clear_day():
    """More cloud means weaker insolation means a more neutral atmosphere."""
    clear = stability_class(2.5, True, 0.0)
    overcast = stability_class(2.5, True, 1.0)
    assert "ABCDEF".index(overcast) > "ABCDEF".index(clear)


# --- dispersion coefficients -------------------------------------------

def test_sigma_grows_with_distance():
    assert sigma_y(2000, "D") > sigma_y(500, "D")
    assert sigma_z(2000, "D") > sigma_z(500, "D")


def test_unstable_air_disperses_faster_than_stable_air():
    """Class A (very unstable) must spread more than class F (very stable)."""
    assert sigma_y(1000, "A") > sigma_y(1000, "D") > sigma_y(1000, "F")
    assert sigma_z(1000, "A") > sigma_z(1000, "D") > sigma_z(1000, "F")


def test_sigma_zero_at_the_source():
    assert sigma_y(0, "D") == 0.0
    assert sigma_z(0, "D") == 0.0


def test_sigma_rejects_an_unknown_stability_class():
    with pytest.raises(ValueError, match="stability"):
        sigma_y(1000, "Z")
    with pytest.raises(ValueError, match="stability"):
        sigma_z(1000, "Z")


def test_sigma_accepts_lowercase_class():
    assert sigma_y(1000, "d") == pytest.approx(sigma_y(1000, "D"))


# --- plume rise ---------------------------------------------------------

def test_no_fire_no_rise():
    assert plume_rise(0.0, 5.0) == 0.0
    assert plume_rise(None, 5.0) == 0.0


def test_bigger_fire_lofts_higher():
    assert plume_rise(500.0, 5.0) > plume_rise(50.0, 5.0)


def test_stronger_wind_bends_the_plume_over():
    """Briggs rise is inversely proportional to wind speed."""
    assert plume_rise(100.0, 2.0) > plume_rise(100.0, 10.0)


def test_buoyancy_flux_scales_with_frp():
    assert buoyancy_flux(200.0) == pytest.approx(2 * buoyancy_flux(100.0), rel=1e-6)


def test_calm_wind_does_not_divide_by_zero():
    assert math.isfinite(plume_rise(100.0, 0.0))


# --- concentration field ------------------------------------------------

def test_concentration_falls_off_downwind():
    kwargs = dict(
        crosswind_m=0.0, receptor_height_m=0.0, emission_rate_g_s=100.0,
        wind_speed_ms=5.0, effective_height_m=50.0, stability="D",
    )
    assert concentration(downwind_m=500, **kwargs) > concentration(downwind_m=5000, **kwargs)


def test_concentration_falls_off_crosswind():
    kwargs = dict(
        downwind_m=1000.0, receptor_height_m=0.0, emission_rate_g_s=100.0,
        wind_speed_ms=5.0, effective_height_m=0.0, stability="D",
    )
    assert concentration(crosswind_m=0.0, **kwargs) > concentration(crosswind_m=500.0, **kwargs)


def test_concentration_is_zero_at_and_behind_the_source():
    kwargs = dict(
        crosswind_m=0.0, receptor_height_m=0.0, emission_rate_g_s=100.0,
        wind_speed_ms=5.0, effective_height_m=10.0, stability="D",
    )
    assert concentration(downwind_m=0.0, **kwargs) == 0.0
    assert concentration(downwind_m=-100.0, **kwargs) == 0.0


def test_lateral_sigma_multiple_really_is_the_ten_percent_contour():
    """The polygon edge is drawn at LATERAL_SIGMA_MULTIPLE * sigma_y; that
    constant is only meaningful if concentration there is ~10% of the
    centreline value."""
    sy = sigma_y(1000.0, "D")
    ratio = math.exp(-((LATERAL_SIGMA_MULTIPLE * sy) ** 2) / (2 * sy ** 2))
    assert ratio == pytest.approx(0.10, abs=0.005)


# --- polygon geometry ---------------------------------------------------

def test_plume_extends_downwind_not_upwind():
    """A northerly wind (from the north) must push the plume SOUTH."""
    geometry = plume_polygon(*SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=10000)

    assert geometry.downwind_bearing_deg == pytest.approx(180.0)
    tip = geometry.centreline[-1]
    assert tip[1] < SOURCE[1], "plume tip should be south of the source"


def test_plume_tip_bearing_matches_the_downwind_direction():
    for wind_from in (0.0, 45.0, 90.0, 180.0, 270.0, 315.0):
        geometry = plume_polygon(
            *SOURCE, wind_direction_deg=wind_from, wind_speed_ms=5.0, length_m=8000
        )
        tip = geometry.centreline[-1]
        assert angular_difference(
            bearing_deg(*SOURCE, *tip), geometry.downwind_bearing_deg
        ) < 1.0


def test_plume_length_matches_the_request():
    geometry = plume_polygon(*SOURCE, wind_direction_deg=270.0, wind_speed_ms=5.0, length_m=12000)
    assert haversine_m(*SOURCE, *geometry.centreline[-1]) == pytest.approx(12000, rel=0.01)


def test_plume_polygon_is_closed():
    geometry = plume_polygon(*SOURCE, wind_direction_deg=270.0, wind_speed_ms=5.0)
    assert geometry.polygon[0] == geometry.polygon[-1]


def test_plume_polygon_has_enough_points_for_a_real_polygon():
    geometry = plume_polygon(*SOURCE, wind_direction_deg=270.0, wind_speed_ms=5.0, steps=24)
    assert len(geometry.polygon) >= 4


def test_plume_widens_with_distance_from_the_source():
    """The footprint is a teardrop: narrow at the source, wide downwind."""
    geometry = plume_polygon(
        *SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=10000, steps=10
    )
    half = len(geometry.polygon) // 2
    right, left = geometry.polygon[:half], list(reversed(geometry.polygon[half:-1]))

    near_width = haversine_m(*right[0], *left[0])
    far_width = haversine_m(*right[-1], *left[-1])
    assert far_width > near_width


def test_stable_air_gives_a_narrower_plume_than_unstable_air():
    stable = plume_polygon(
        *SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=10000, stability="F"
    )
    unstable = plume_polygon(
        *SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=10000, stability="A"
    )
    assert stable.max_half_width_m < unstable.max_half_width_m


def test_explicit_stability_overrides_the_inferred_one():
    geometry = plume_polygon(
        *SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, is_daytime=True, stability="F"
    )
    assert geometry.stability_class == "F"


def test_plume_polygon_validates_its_inputs():
    with pytest.raises(ValueError):
        plume_polygon(*SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, length_m=0)
    with pytest.raises(ValueError):
        plume_polygon(*SOURCE, wind_direction_deg=0.0, wind_speed_ms=5.0, steps=1)


# --- length heuristic ---------------------------------------------------

def test_suggested_length_grows_with_wind_speed():
    assert suggested_length_m(10.0, "D") > suggested_length_m(2.0, "D")


def test_stable_air_carries_a_plume_further_than_unstable_air():
    assert suggested_length_m(5.0, "F") > suggested_length_m(5.0, "D") > suggested_length_m(5.0, "A")


def test_bigger_fire_gives_a_longer_plume():
    assert suggested_length_m(5.0, "D", frp_mw=500) > suggested_length_m(5.0, "D", frp_mw=1)


def test_suggested_length_is_clamped_to_sane_bounds():
    assert 2000.0 <= suggested_length_m(0.0, "A") <= 50000.0
    assert suggested_length_m(200.0, "F", frp_mw=100000) <= 50000.0
