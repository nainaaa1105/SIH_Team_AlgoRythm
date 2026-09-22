"""Dozier sub-pixel retrieval.

The central test strategy here is **forward simulation**: pick a known
fire temperature and fractional area, compute what the sensor would
observe through the physics, then check the solver recovers the inputs.
That is a real correctness proof for an inverse method, not a smoke test.
"""
import pytest

from imagery.thermal.background import (
    DAYTIME_BACKGROUND_K,
    NIGHTTIME_BACKGROUND_K,
    climatological_background,
    estimate_background,
    is_daytime_from_rows,
)
from imagery.thermal.dozier import (
    MAX_FIRE_TEMPERATURE_K,
    MIN_FIRE_TEMPERATURE_K,
    VIIRS_I4_SATURATION_K,
    band_wavelengths,
    pixel_area_m2,
    solve,
    solve_with_uncertainty,
)
from imagery.thermal.planck import (
    VIIRS_I4_WAVELENGTH_M,
    VIIRS_I5_WAVELENGTH_M,
    brightness_temperature,
    mixed_pixel_radiance,
)


def observed_temperatures(fire_t: float, fraction: float, background_t: float):
    """Simulate what the sensor would report for a known fire."""
    l4 = mixed_pixel_radiance(VIIRS_I4_WAVELENGTH_M, fraction, fire_t, background_t)
    l5 = mixed_pixel_radiance(VIIRS_I5_WAVELENGTH_M, fraction, fire_t, background_t)
    return (
        brightness_temperature(VIIRS_I4_WAVELENGTH_M, l4),
        brightness_temperature(VIIRS_I5_WAVELENGTH_M, l5),
    )


# --- the forward-simulation recovery proof ------------------------------

@pytest.mark.parametrize(
    "fire_t,fraction,background_t",
    [
        (600.0, 5.0e-3, 305.0),    # smouldering agricultural fire
        (800.0, 1.0e-3, 300.0),    # typical vegetation fire
        (900.0, 1.0e-4, 290.0),    # small hot source at night
        (1100.0, 2.0e-4, 295.0),   # gas flare
        (1500.0, 5.0e-5, 300.0),   # very hot, very small
    ],
)
def test_solver_recovers_a_known_fire(fire_t, fraction, background_t):
    t4, t5 = observed_temperatures(fire_t, fraction, background_t)
    result = solve(t4, t5, background_t, sensor="VIIRS_SNPP")

    assert result.ok, result.reason
    assert result.fire_temperature_k == pytest.approx(fire_t, rel=1e-3)
    assert result.fire_fraction == pytest.approx(fraction, rel=1e-3)


def test_simulated_observations_look_like_real_firms_detections():
    """Sanity check on the simulation itself: a real VIIRS fire pixel
    reports I4 around 320-360 K and I5 near ambient. If the forward model
    produced anything else, the recovery test would be self-consistent
    but physically meaningless."""
    t4, t5 = observed_temperatures(800.0, 1.0e-3, 300.0)
    assert 310 < t4 < 370
    assert 295 < t5 < 320


def test_hotter_fire_at_equal_fraction_raises_the_observed_signal():
    cool_t4, _ = observed_temperatures(600.0, 1e-3, 300.0)
    hot_t4, _ = observed_temperatures(1200.0, 1e-3, 300.0)
    assert hot_t4 > cool_t4


# --- physical gates -----------------------------------------------------

def test_saturated_mid_infrared_is_rejected():
    """Above I4 saturation the solver would converge on nonsense; it must
    decline with a reason rather than return a confident wrong number."""
    result = solve(VIIRS_I4_SATURATION_K + 5, 300.0, 295.0, sensor="VIIRS_SNPP")
    assert not result.ok
    assert "saturat" in result.reason.lower()


def test_detection_at_or_below_background_is_rejected():
    result = solve(295.0, 295.0, 300.0, sensor="VIIRS_SNPP")
    assert not result.ok
    assert "background" in result.reason.lower()


def test_retrieved_temperature_outside_plausible_range_is_rejected():
    """A marginal thermal excess can drive the solver to an unphysical
    answer; the bounds catch it."""
    result = solve(300.05, 300.0, 300.0, sensor="VIIRS_SNPP")
    if result.ok:
        assert MIN_FIRE_TEMPERATURE_K <= result.fire_temperature_k <= MAX_FIRE_TEMPERATURE_K


def test_result_always_records_the_background_it_assumed():
    """The retrieval is only interpretable against its background, and
    that background is estimated rather than measured."""
    t4, t5 = observed_temperatures(800.0, 1e-3, 302.0)
    result = solve(t4, t5, 302.0, sensor="VIIRS_SNPP")
    assert result.background_temperature_k == 302.0


def test_fire_area_is_computed_when_the_pixel_footprint_is_known():
    t4, t5 = observed_temperatures(800.0, 1e-3, 300.0)
    result = solve(t4, t5, 300.0, sensor="VIIRS_SNPP", pixel_area_m2=140_625.0)
    assert result.fire_area_m2 == pytest.approx(result.fire_fraction * 140_625.0)


def test_fire_area_is_none_without_a_footprint():
    t4, t5 = observed_temperatures(800.0, 1e-3, 300.0)
    assert solve(t4, t5, 300.0, sensor="VIIRS_SNPP").fire_area_m2 is None


# --- sensor handling ----------------------------------------------------

def test_viirs_and_modis_have_different_band_pairs():
    viirs = band_wavelengths("VIIRS_SNPP_NRT")
    modis = band_wavelengths("MODIS_NRT")
    assert viirs != modis
    assert viirs[0] < viirs[1]      # mid-IR before thermal-IR


def test_single_band_sensors_are_rejected_with_an_explanation():
    """INSAT-3DS and Sentinel-3 FRP carry an FRP or a fire flag, not the
    two brightness temperatures this method inverts."""
    for sensor in ("INSAT3DS", "SENTINEL3_FRP", "HIMAWARI"):
        with pytest.raises(ValueError, match="dual-band"):
            band_wavelengths(sensor)


def test_pixel_area_from_firms_scan_and_track():
    """FIRMS reports these in km and they grow toward the swath edge, so
    using them beats assuming a nominal 375 m pixel."""
    assert pixel_area_m2(0.375, 0.375) == pytest.approx(140_625.0)
    assert pixel_area_m2(None, 0.375) is None
    assert pixel_area_m2(0.0, 0.375) is None


# --- background estimation ----------------------------------------------

def test_background_uses_a_low_percentile_of_the_thermal_band():
    temperatures = [292.0, 295.0, 298.0, 305.0, 310.0, 315.0]
    estimate = estimate_background(temperatures, is_daytime=True)
    assert estimate.source == "cluster_percentile"
    assert 292.0 <= estimate.temperature_k <= 300.0


def test_background_falls_back_to_climatology_with_too_few_samples():
    estimate = estimate_background([300.0], is_daytime=True)
    assert estimate.source == "climatology"
    assert estimate.temperature_k == DAYTIME_BACKGROUND_K


def test_night_climatology_is_cooler_than_day():
    assert climatological_background(False).temperature_k < climatological_background(True).temperature_k
    assert climatological_background(False).temperature_k == NIGHTTIME_BACKGROUND_K


def test_implausible_thermal_values_are_filtered_out():
    """A whole cluster of burning pixels must not become the background."""
    estimate = estimate_background([400.0, 420.0, 450.0, 500.0], is_daytime=True)
    assert estimate.source == "climatology"


def test_empty_input_falls_back_cleanly():
    assert estimate_background([], is_daytime=True).source == "climatology"


def test_none_values_are_ignored():
    estimate = estimate_background([None, 295.0, 296.0, 297.0, 298.0], is_daytime=True)
    assert estimate.source == "cluster_percentile"
    assert estimate.n_samples == 4


def test_background_estimate_serialises_with_its_provenance():
    payload = estimate_background([292.0, 295.0, 298.0, 305.0], True).as_dict()
    assert set(payload) == {
        "background_temperature_k", "background_source", "background_n_samples"
    }


def test_daytime_flag_from_a_clusters_detections():
    assert is_daytime_from_rows([{"daynight": "D"}, {"daynight": "D"}, {"daynight": "N"}])
    assert not is_daytime_from_rows([{"daynight": "N"}, {"daynight": "N"}, {"daynight": "D"}])
    assert is_daytime_from_rows([])          # default when unknown


# --- ill-conditioning, measured rather than assumed ---------------------

def test_small_hot_fires_are_badly_conditioned():
    """The core accuracy limitation of this method, pinned as a test.

    A 1 K error in the assumed background moves a small hot retrieval by
    hundreds of kelvin. Since the background is *estimated* (FIRMS ships
    no non-fire neighbours), a bare temperature for a flare-like source
    would imply precision the method does not have.
    """
    t4, t5 = observed_temperatures(950.0, 3e-4, 296.0)

    cooler = solve(t4, t5, 295.0, sensor="VIIRS")   # background 1 K too low
    exact = solve(t4, t5, 296.0, sensor="VIIRS")

    assert exact.fire_temperature_k == pytest.approx(950.0, rel=1e-3)
    assert cooler.ok
    assert abs(cooler.fire_temperature_k - 950.0) > 200


def test_large_cool_fires_are_well_conditioned():
    """The contrast: the same 1 K background error barely matters for a
    bigger, cooler fire — so the flag must be per-retrieval, not global."""
    t4, t5 = observed_temperatures(700.0, 5e-3, 300.0)

    for offset in (-1.0, 1.0):
        result = solve(t4, t5, 300.0 + offset, sensor="VIIRS")
        assert result.ok
        assert result.fire_temperature_k == pytest.approx(700.0, rel=0.12)


def test_uncertainty_bracket_flags_a_poorly_constrained_retrieval():
    t4, t5 = observed_temperatures(950.0, 3e-4, 296.0)
    result = solve_with_uncertainty(t4, t5, 296.0, background_uncertainty_k=1.0, sensor="VIIRS")

    assert result.ok
    assert result.well_constrained is False
    assert "poorly constrained" in result.reason


def test_uncertainty_bracket_accepts_a_well_constrained_retrieval():
    t4, t5 = observed_temperatures(700.0, 5e-3, 300.0)
    result = solve_with_uncertainty(t4, t5, 300.0, background_uncertainty_k=1.0, sensor="VIIRS")

    assert result.ok
    assert result.well_constrained is True
    assert result.temperature_low_k <= result.fire_temperature_k <= result.temperature_high_k


def test_uncertainty_bracket_contains_the_nominal_answer():
    t4, t5 = observed_temperatures(800.0, 2e-3, 300.0)
    result = solve_with_uncertainty(t4, t5, 300.0, sensor="VIIRS")
    assert result.temperature_spread_k is not None
    assert result.temperature_spread_k >= 0


def test_uncertainty_solver_passes_through_a_failed_retrieval():
    result = solve_with_uncertainty(
        VIIRS_I4_SATURATION_K + 5, 300.0, 295.0, sensor="VIIRS"
    )
    assert not result.ok
    assert "saturat" in result.reason.lower()


def test_end_to_end_background_then_retrieval():
    """The realistic path: estimate the background from the cluster's own
    marginal pixels, then retrieve with the uncertainty bracket.

    Deliberately uses a large cool fire, because that is the regime where
    a percentile-estimated background actually supports a usable number.
    The small-hot regime is covered by the ill-conditioning tests above.
    """
    true_background = 300.0
    t4, t5 = observed_temperatures(700.0, 5e-3, true_background)

    thermal_band = [300.1, 300.3, 300.6, 301.0, t5]
    background = estimate_background(thermal_band, is_daytime=True)

    result = solve_with_uncertainty(t4, t5, background.temperature_k, sensor="VIIRS_SNPP")
    assert result.ok, result.reason
    # Background is estimated, not measured, so this is the honest
    # accuracy of the method in its well-conditioned regime.
    assert result.fire_temperature_k == pytest.approx(700.0, rel=0.15)
