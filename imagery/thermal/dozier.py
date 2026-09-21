"""Dozier two-band sub-pixel fire retrieval.

A 375 m VIIRS pixel is enormously larger than the flaming front inside
it, so the brightness temperature the sensor reports is not the fire's
temperature — it is an area-weighted mix of a tiny very hot fraction and
a large ambient remainder. Dozier (1981) recovers the two unknowns from
two bands:

    L4_obs = p * B(l4, Tf) + (1-p) * B(l4, Tb)
    L5_obs = p * B(l5, Tf) + (1-p) * B(l5, Tb)

Two equations, two unknowns (p = fire fraction, Tf = fire temperature).
The mid-infrared band (3.74 um) is hugely sensitive to the hot fraction
while the thermal band (11.45 um) is dominated by the background, which
is exactly what makes the system solvable.

This is the "Dozier temperature" feature in M2's matrix and the
sub-pixel item in the project's Phase-4 list. Being able to say "this
burns at 1100 K over 0.02% of the pixel" separates a flare (small, very
hot, steady) from a smouldering agricultural fire (larger, cooler).

Two honest limitations, both handled explicitly rather than hidden:

  * **Background temperature is not measurable from FIRMS.** The classic
    method takes it from neighbouring non-fire pixels in the granule;
    FIRMS distributes only the fire pixels. See `background.py` for how
    it is estimated and why the estimate is stored alongside the result.
  * **VIIRS I4 saturates near 367 K.** Above that the observed radiance
    is clipped, the system is no longer physical, and the solver will
    happily converge on nonsense. Saturated inputs are rejected with a
    reason instead.
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from imagery.thermal.planck import (
    MODIS_B21_WAVELENGTH_M,
    MODIS_B31_WAVELENGTH_M,
    VIIRS_I4_WAVELENGTH_M,
    VIIRS_I5_WAVELENGTH_M,
    brightness_temperature,
    radiance,
)

logger = logging.getLogger(__name__)

# VIIRS I4 saturation brightness temperature. Above this the retrieval
# is not trustworthy no matter how nicely the solver converges.
VIIRS_I4_SATURATION_K = 367.0
MODIS_B21_SATURATION_K = 500.0

# Physically plausible bounds for a flaming/smouldering front. Anything
# outside these is a solver artefact, not a fire.
MIN_FIRE_TEMPERATURE_K = 400.0
MAX_FIRE_TEMPERATURE_K = 1800.0
MIN_FIRE_FRACTION = 1e-7
MAX_FIRE_FRACTION = 1.0


@dataclass
class DozierResult:
    fire_temperature_k: Optional[float]
    fire_fraction: Optional[float]
    fire_area_m2: Optional[float]
    background_temperature_k: float
    converged: bool
    reason: str = ""
    sensor: str = ""
    # Provenance of the background above: "cluster_percentile" or
    # "climatology" (see background.py). Declared as a real field rather
    # than attached dynamically, so the retrieval and the assumption it
    # rests on always travel together.
    background_source: Optional[str] = None
    # Populated by solve_with_uncertainty: how much the answer moves when
    # the assumed background is perturbed within its own uncertainty.
    temperature_low_k: Optional[float] = None
    temperature_high_k: Optional[float] = None
    well_constrained: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return self.converged and self.fire_temperature_k is not None

    @property
    def temperature_spread_k(self) -> Optional[float]:
        if self.temperature_low_k is None or self.temperature_high_k is None:
            return None
        return self.temperature_high_k - self.temperature_low_k


def band_wavelengths(sensor: str) -> Tuple[float, float]:
    """Central wavelengths of the (mid-IR, thermal-IR) pair for a sensor."""
    normalised = (sensor or "").upper()
    if "MODIS" in normalised:
        return MODIS_B21_WAVELENGTH_M, MODIS_B31_WAVELENGTH_M
    if "VIIRS" in normalised:
        return VIIRS_I4_WAVELENGTH_M, VIIRS_I5_WAVELENGTH_M
    raise ValueError(
        f"No known band pair for sensor {sensor!r}. Dozier needs a dual-band "
        "mid-IR/thermal-IR sensor; INSAT-3DS and Sentinel-3 FRP detections "
        "do not carry the two brightness temperatures this method needs."
    )


def saturation_limit(sensor: str) -> float:
    return MODIS_B21_SATURATION_K if "MODIS" in (sensor or "").upper() else VIIRS_I4_SATURATION_K


def solve(
    t4_observed_k: float,
    t5_observed_k: float,
    background_t4_k: float,
    background_t5_k: Optional[float] = None,
    sensor: str = "VIIRS",
    pixel_area_m2: Optional[float] = None,
) -> DozierResult:
    """Recover fire temperature and fractional area from two bands.

    `background_t5_k` defaults to `background_t4_k`: with no neighbourhood
    data we cannot distinguish the two background temperatures, and
    assuming they are equal is the standard simplification.
    """
    lambda_mir, lambda_tir = band_wavelengths(sensor)
    if background_t5_k is None:
        background_t5_k = background_t4_k

    result = DozierResult(
        fire_temperature_k=None,
        fire_fraction=None,
        fire_area_m2=None,
        background_temperature_k=background_t4_k,
        converged=False,
        sensor=sensor,
    )

    # --- physical sanity gates, before touching the solver -------------
    limit = saturation_limit(sensor)
    if t4_observed_k >= limit:
        result.reason = (
            f"mid-infrared band saturated ({t4_observed_k:.1f} K >= {limit:.1f} K) — "
            "the retrieval would be numerically fine and physically meaningless"
        )
        return result

    if t4_observed_k <= background_t4_k:
        result.reason = (
            f"no thermal excess: observed {t4_observed_k:.1f} K is not above "
            f"background {background_t4_k:.1f} K"
        )
        return result

    try:
        from scipy.optimize import fsolve
    except ImportError:
        result.reason = "scipy is required for the Dozier retrieval"
        return result

    l4_observed = radiance(lambda_mir, t4_observed_k)
    l5_observed = radiance(lambda_tir, t5_observed_k)
    l4_background = radiance(lambda_mir, background_t4_k)
    l5_background = radiance(lambda_tir, background_t5_k)

    def equations(unknowns):
        # Solve for log(p) and Tf rather than p directly: p spans several
        # orders of magnitude (1e-6 to 1e-2), and an unconstrained linear
        # p lets the solver wander negative, which is unphysical.
        log_p, fire_t = unknowns
        p = math.exp(min(log_p, 0.0))
        fire_t = max(fire_t, 1.0)
        return [
            p * radiance(lambda_mir, fire_t) + (1 - p) * l4_background - l4_observed,
            p * radiance(lambda_tir, fire_t) + (1 - p) * l5_background - l5_observed,
        ]

    # Seed from a plausible flaming front; the solution surface is smooth
    # enough that this converges reliably for real detections.
    initial_guess = [math.log(1e-3), 750.0]

    try:
        solution, _, status, message = fsolve(equations, initial_guess, full_output=True)
    except Exception as exc:  # noqa: BLE001 - solver failure must not kill the task
        result.reason = f"solver raised: {exc}"
        return result

    if status != 1:
        result.reason = f"solver did not converge: {message.strip()}"
        return result

    fire_fraction = math.exp(min(solution[0], 0.0))
    fire_temperature = float(solution[1])

    if not (MIN_FIRE_TEMPERATURE_K <= fire_temperature <= MAX_FIRE_TEMPERATURE_K):
        result.reason = (
            f"retrieved temperature {fire_temperature:.0f} K is outside the plausible "
            f"range {MIN_FIRE_TEMPERATURE_K:.0f}-{MAX_FIRE_TEMPERATURE_K:.0f} K"
        )
        return result

    if not (MIN_FIRE_FRACTION <= fire_fraction <= MAX_FIRE_FRACTION):
        result.reason = f"retrieved fire fraction {fire_fraction:.2e} is not physical"
        return result

    result.fire_temperature_k = fire_temperature
    result.fire_fraction = fire_fraction
    result.converged = True
    result.reason = "ok"
    if pixel_area_m2:
        result.fire_area_m2 = fire_fraction * pixel_area_m2
    return result


# How far the background estimate is assumed to be uncertain. The estimate
# comes from a percentile of neighbouring detections or from climatology
# (see background.py), so ~1 K is optimistic for the former and generous
# for the latter.
DEFAULT_BACKGROUND_UNCERTAINTY_K = 1.0

# Above this relative spread the retrieval is reported but flagged: the
# answer depends more on the assumed background than on the measurement.
WELL_CONSTRAINED_RELATIVE_SPREAD = 0.30


def solve_with_uncertainty(
    t4_observed_k: float,
    t5_observed_k: float,
    background_t4_k: float,
    background_uncertainty_k: float = DEFAULT_BACKGROUND_UNCERTAINTY_K,
    sensor: str = "VIIRS",
    pixel_area_m2_value: Optional[float] = None,
) -> DozierResult:
    """Retrieve, and quantify how much the answer depends on the background.

    This exists because the Dozier inversion is badly ill-conditioned for
    small, hot fires, and the background it inverts against is *estimated*
    rather than measured (FIRMS ships no non-fire neighbours). Measured on
    synthetic fires, a 1 K background error moves the answer by:

        small hot fire (950 K, p=3e-4):   648 K .. rejected
        larger cooler fire (700 K, p=5e-3): 658 K .. 764 K

    In other words the gas-flare case — small and very hot — is exactly
    the badly-conditioned one. Reporting a bare temperature would imply a
    precision the method does not have, so the answer is bracketed by
    re-solving at the edges of the background's own uncertainty and
    flagged when the bracket is wide.
    """
    nominal = solve(
        t4_observed_k, t5_observed_k, background_t4_k,
        sensor=sensor, pixel_area_m2=pixel_area_m2_value,
    )
    if not nominal.ok:
        return nominal

    bracket = []
    for offset in (-background_uncertainty_k, background_uncertainty_k):
        perturbed = solve(
            t4_observed_k, t5_observed_k, background_t4_k + offset,
            sensor=sensor, pixel_area_m2=pixel_area_m2_value,
        )
        if perturbed.ok:
            bracket.append(perturbed.fire_temperature_k)

    if not bracket:
        # Both perturbations fell outside the physical bounds, which is
        # itself evidence the retrieval sits on a knife edge.
        nominal.well_constrained = False
        nominal.reason = (
            "ok, but poorly constrained: a "
            f"{background_uncertainty_k:.1f} K change in the assumed background "
            "pushes the retrieval outside physical bounds"
        )
        return nominal

    low = min(bracket + [nominal.fire_temperature_k])
    high = max(bracket + [nominal.fire_temperature_k])

    nominal.temperature_low_k = low
    nominal.temperature_high_k = high
    relative_spread = (high - low) / nominal.fire_temperature_k
    nominal.well_constrained = relative_spread <= WELL_CONSTRAINED_RELATIVE_SPREAD

    if not nominal.well_constrained:
        nominal.reason = (
            f"ok, but poorly constrained: {low:.0f}-{high:.0f} K across a "
            f"{background_uncertainty_k:.1f} K background uncertainty "
            f"({relative_spread:.0%} spread)"
        )
    return nominal


def pixel_area_m2(scan_km: Optional[float], track_km: Optional[float]) -> Optional[float]:
    """Ground area of the detection footprint.

    FIRMS reports per-detection `scan` and `track` dimensions in km,
    which M1 stores on `hotspots`. They grow toward the swath edge, so
    using them beats assuming a nominal 375 m pixel.
    """
    if not scan_km or not track_km:
        return None
    return float(scan_km) * float(track_km) * 1_000_000.0
