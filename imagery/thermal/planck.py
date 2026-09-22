"""Planck blackbody radiance and its inverse.

The whole Dozier retrieval rests on these two functions, so they are
kept separate, pure, and tested to round-trip exactly. Everything is in
SI: wavelength in metres, temperature in kelvin, radiance in
W.m^-2.sr^-1.m^-1.

Spectral radiance (Planck's law):

    B(lambda, T) = c1 / ( lambda^5 * (exp(c2 / (lambda*T)) - 1) )

    c1 = 2*h*c^2      c2 = h*c/k

Inverting for temperature:

    T = c2 / ( lambda * ln(1 + c1 / (lambda^5 * L)) )

These are exact inverses of one another, which is what
`test_planck.py::test_radiance_and_brightness_temperature_round_trip`
checks — if that ever fails, every temperature this package reports is
suspect.
"""
import math
from typing import Final

# CODATA 2018 values.
PLANCK_H: Final[float] = 6.62607015e-34      # J.s
SPEED_OF_LIGHT: Final[float] = 2.99792458e8  # m/s
BOLTZMANN_K: Final[float] = 1.380649e-23     # J/K

C1: Final[float] = 2.0 * PLANCK_H * SPEED_OF_LIGHT ** 2     # W.m^2
C2: Final[float] = PLANCK_H * SPEED_OF_LIGHT / BOLTZMANN_K  # m.K

# Central wavelengths of the bands the fire products actually use.
# VIIRS I-bands (375 m, what FIRMS reports for VIIRS detections):
VIIRS_I4_WAVELENGTH_M: Final[float] = 3.74e-6    # mid-infrared, fire-sensitive
VIIRS_I5_WAVELENGTH_M: Final[float] = 11.45e-6   # thermal infrared, background-sensitive

# MODIS equivalents (1 km), for detections FIRMS reports from Terra/Aqua:
MODIS_B21_WAVELENGTH_M: Final[float] = 3.959e-6
MODIS_B31_WAVELENGTH_M: Final[float] = 11.03e-6


def radiance(wavelength_m: float, temperature_k: float) -> float:
    """Spectral radiance of a blackbody at a wavelength and temperature."""
    if wavelength_m <= 0:
        raise ValueError("wavelength must be positive")
    if temperature_k <= 0:
        raise ValueError("temperature must be positive kelvin")

    exponent = C2 / (wavelength_m * temperature_k)

    # At mid-infrared wavelengths and terrestrial temperatures the
    # exponent is large (~13 at 3.74um/300K), so exp() is well-behaved;
    # but guard the overflow regime for very cold inputs rather than
    # letting it raise.
    if exponent > 700:
        return 0.0

    return C1 / (wavelength_m ** 5 * (math.exp(exponent) - 1.0))


def brightness_temperature(wavelength_m: float, radiance_value: float) -> float:
    """Temperature of a blackbody that would emit this radiance."""
    if wavelength_m <= 0:
        raise ValueError("wavelength must be positive")
    if radiance_value <= 0:
        raise ValueError("radiance must be positive")

    return C2 / (wavelength_m * math.log1p(C1 / (wavelength_m ** 5 * radiance_value)))


def mixed_pixel_radiance(
    wavelength_m: float, fire_fraction: float, fire_temperature_k: float, background_temperature_k: float
) -> float:
    """Radiance of a pixel that is part fire, part background.

    This is the physical model Dozier inverts: a sensor pixel is far
    larger than the flaming front inside it, so what reaches the sensor
    is an area-weighted mix of a small very hot fraction and a large
    ambient remainder.
    """
    if not 0.0 <= fire_fraction <= 1.0:
        raise ValueError("fire_fraction must be between 0 and 1")

    hot = radiance(wavelength_m, fire_temperature_k)
    cool = radiance(wavelength_m, background_temperature_k)
    return fire_fraction * hot + (1.0 - fire_fraction) * cool
