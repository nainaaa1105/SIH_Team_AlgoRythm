"""Planck radiation law.

Every temperature M4 reports comes out of these two functions, so the
round-trip property is checked exhaustively: if radiance and brightness
temperature stop being exact inverses, the Dozier retrieval is silently
wrong and nothing else would catch it.
"""
import math

import pytest

from imagery.thermal.planck import (
    C1,
    C2,
    MODIS_B21_WAVELENGTH_M,
    MODIS_B31_WAVELENGTH_M,
    VIIRS_I4_WAVELENGTH_M,
    VIIRS_I5_WAVELENGTH_M,
    brightness_temperature,
    mixed_pixel_radiance,
    radiance,
)


def test_radiation_constants_match_published_values():
    """c1 = 2hc^2 = 1.1910e-16 W.m^2, c2 = hc/k = 1.4388e-2 m.K"""
    assert C1 == pytest.approx(1.191042e-16, rel=1e-5)
    assert C2 == pytest.approx(1.438777e-2, rel=1e-5)


@pytest.mark.parametrize("temperature", [250, 280, 300, 350, 500, 800, 1200, 1800])
@pytest.mark.parametrize(
    "wavelength",
    [VIIRS_I4_WAVELENGTH_M, VIIRS_I5_WAVELENGTH_M, MODIS_B21_WAVELENGTH_M, MODIS_B31_WAVELENGTH_M],
)
def test_radiance_and_brightness_temperature_round_trip(wavelength, temperature):
    """The load-bearing property of this whole package."""
    recovered = brightness_temperature(wavelength, radiance(wavelength, temperature))
    assert recovered == pytest.approx(temperature, rel=1e-9)


def test_radiance_increases_with_temperature():
    """Stefan-Boltzmann in spirit: hotter bodies radiate more at every wavelength."""
    for wavelength in (VIIRS_I4_WAVELENGTH_M, VIIRS_I5_WAVELENGTH_M):
        assert radiance(wavelength, 400) > radiance(wavelength, 300) > radiance(wavelength, 250)


def test_mid_infrared_is_far_more_fire_sensitive_than_thermal_infrared():
    """This asymmetry is *why* two-band retrieval works: going from 300 K
    to 800 K multiplies the 3.74 um signal enormously while the 11.45 um
    signal barely moves, so the two bands carry independent information."""
    mir_gain = radiance(VIIRS_I4_WAVELENGTH_M, 800) / radiance(VIIRS_I4_WAVELENGTH_M, 300)
    tir_gain = radiance(VIIRS_I5_WAVELENGTH_M, 800) / radiance(VIIRS_I5_WAVELENGTH_M, 300)
    assert mir_gain > 100 * tir_gain


def test_wien_displacement_peak_shifts_to_shorter_wavelengths_when_hotter():
    """Wien's law: peak wavelength = 2.898e-3 / T."""
    for temperature in (300, 800, 1500):
        peak = 2.898e-3 / temperature
        at_peak = radiance(peak, temperature)
        assert at_peak > radiance(peak * 0.5, temperature)
        assert at_peak > radiance(peak * 2.0, temperature)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        radiance(0, 300)
    with pytest.raises(ValueError):
        radiance(VIIRS_I4_WAVELENGTH_M, 0)
    with pytest.raises(ValueError):
        radiance(VIIRS_I4_WAVELENGTH_M, -10)
    with pytest.raises(ValueError):
        brightness_temperature(VIIRS_I4_WAVELENGTH_M, 0)
    with pytest.raises(ValueError):
        brightness_temperature(0, 1.0)


def test_very_cold_temperature_does_not_overflow():
    """The exponent blows up at low T; it must return 0, not raise."""
    assert radiance(VIIRS_I4_WAVELENGTH_M, 1.0) == 0.0
    assert math.isfinite(radiance(VIIRS_I4_WAVELENGTH_M, 5.0))


# --- mixed pixel --------------------------------------------------------

def test_mixed_pixel_with_no_fire_is_pure_background():
    value = mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, 0.0, 900.0, 300.0)
    assert value == pytest.approx(radiance(VIIRS_I4_WAVELENGTH_M, 300.0))


def test_mixed_pixel_fully_ablaze_is_pure_fire():
    value = mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, 1.0, 900.0, 300.0)
    assert value == pytest.approx(radiance(VIIRS_I4_WAVELENGTH_M, 900.0))


def test_mixed_pixel_is_a_linear_area_weighted_blend():
    hot = radiance(VIIRS_I4_WAVELENGTH_M, 900.0)
    cool = radiance(VIIRS_I4_WAVELENGTH_M, 300.0)
    mixed = mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, 0.25, 900.0, 300.0)
    assert mixed == pytest.approx(0.25 * hot + 0.75 * cool)


def test_a_tiny_hot_fraction_still_dominates_the_mid_infrared():
    """0.1% of a pixel at 900 K contributes more MIR radiance than the
    other 99.9% at 300 K — the physical reason a 375 m pixel can detect a
    small flaming front at all.

    The comparison is contribution-vs-contribution rather than an
    arbitrary multiple of the background: at 3.74 um the fire's share is
    about 5x the background's, which is decisive without being 10x.
    """
    fraction = 0.001
    fire_contribution = fraction * radiance(VIIRS_I4_WAVELENGTH_M, 900.0)
    background_contribution = (1 - fraction) * radiance(VIIRS_I4_WAVELENGTH_M, 300.0)

    assert fire_contribution > background_contribution
    assert mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, fraction, 900.0, 300.0) == pytest.approx(
        fire_contribution + background_contribution
    )


def test_the_same_tiny_fraction_barely_moves_the_thermal_infrared():
    """The mirror image of the test above, and the reason two bands can
    be inverted for two unknowns: at 11.45 um the background still
    dominates, so the thermal band measures the scene while the
    mid-infrared band measures the fire."""
    fraction = 0.001
    fire_contribution = fraction * radiance(VIIRS_I5_WAVELENGTH_M, 900.0)
    background_contribution = (1 - fraction) * radiance(VIIRS_I5_WAVELENGTH_M, 300.0)
    assert background_contribution > fire_contribution


def test_mixed_pixel_rejects_an_impossible_fraction():
    with pytest.raises(ValueError):
        mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, 1.5, 900.0, 300.0)
    with pytest.raises(ValueError):
        mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, -0.1, 900.0, 300.0)
