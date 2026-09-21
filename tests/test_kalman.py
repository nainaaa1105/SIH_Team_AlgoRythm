"""Kalman filter over FRP.

Tested by **forward simulation**, the same strategy used for M4's Dozier
solver: generate a trajectory with known parameters, run the filter, and
check it recovers them. For a filter there is a second, stronger check —
whether its *reported uncertainty* matches its *actual error*. A filter
that tracks well but lies about its confidence is worse than useless for
a number that drives evacuation timing.
"""
import random
import statistics

import pytest

from temporal.forecast.kalman import (
    DEFAULT_PROCESS_NOISE,
    initial_state,
    measurement_variance,
    normalised_innovation_squared,
    predict,
    process_noise,
    run_filter,
    update,
)


def linear_trajectory(rate, n=30, dt=6.0, start=50.0, noise=0.0, seed=0):
    """FRP changing linearly at `rate` MW/h, sampled every `dt` hours.

    `start` must be high enough that a negative rate does not drive the
    series to zero — a clamped trajectory is no longer linear, and a test
    built on one would be asserting a rate the data no longer contains.
    """
    rng = random.Random(seed)
    final_true = start + rate * dt * (n - 1)
    assert final_true > 0, (
        f"fixture clamps: rate={rate} over {n} samples drives FRP to {final_true:.1f}. "
        "Raise `start` so the trajectory stays linear."
    )

    observations = []
    for i in range(n):
        t = i * dt
        true = start + rate * t
        measured = true * (1 + rng.gauss(0, noise)) if noise else true
        observations.append((t, max(measured, 0.5)))
    return observations


# --- noiseless recovery: the core correctness proof --------------------

@pytest.mark.parametrize(
    "rate,start",
    [(0.0, 50.0), (0.5, 50.0), (2.0, 50.0), (5.0, 50.0), (-0.5, 500.0), (-2.0, 800.0)],
)
def test_filter_recovers_a_noiseless_linear_trajectory(rate, start):
    observations = linear_trajectory(rate, n=40, start=start, noise=0.0)
    state = run_filter(observations, relative_error=0.01)

    assert state is not None
    assert state.rate == pytest.approx(rate, abs=0.02)
    assert state.frp == pytest.approx(observations[-1][1], rel=0.02)


def test_a_decaying_source_is_reported_as_decaying():
    """A fire burning out must show a negative rate, so the escalation
    forecast can say it is not going to reach anything."""
    state = run_filter(linear_trajectory(-2.0, n=30, start=800.0, noise=0.0), relative_error=0.01)
    assert state.rate < 0


def test_filter_tracks_a_noisy_trajectory():
    observations = linear_trajectory(2.0, n=40, noise=0.15, seed=1)
    state = run_filter(observations)
    assert state.rate == pytest.approx(2.0, abs=1.5)
    assert state.frp > observations[0][1]


# --- calibration: does it tell the truth about its own uncertainty? ----

def test_reported_uncertainty_matches_actual_error():
    """The claim the Day-10 deliverable rests on.

    Over many runs the spread of actual rate errors must be close to the
    filter's own reported rate_std. A ratio near 1 means calibrated;
    below 1 means conservative (wider intervals than needed), which is
    the safe direction; well above 1 means overconfident, which would
    make a stated 90% CI meaningless.
    """
    errors, reported = [], []
    for seed in range(120):
        observations = linear_trajectory(2.0, n=30, noise=0.30, seed=seed)
        state = run_filter(observations)
        errors.append(state.rate - 2.0)
        reported.append(state.rate_std)

    empirical_std = statistics.pstdev(errors)
    mean_reported = sum(reported) / len(reported)
    ratio = empirical_std / mean_reported

    assert 0.5 <= ratio <= 1.3, (
        f"filter is mis-calibrated: empirical/reported = {ratio:.2f} "
        f"(empirical {empirical_std:.3f}, reported {mean_reported:.3f})"
    )


def test_mean_nis_is_near_one_for_a_well_specified_filter():
    """Normalised innovation squared has expectation 1 for a scalar
    measurement when the filter's noise model is right."""
    observations = linear_trajectory(1.0, n=50, noise=0.30, seed=7)
    state = run_filter(observations)
    nis = normalised_innovation_squared(state)

    assert nis is not None
    assert 0.3 <= nis <= 3.0, f"NIS {nis:.2f} indicates a mis-specified noise model"


# --- gap handling -------------------------------------------------------

def test_longer_gaps_widen_the_covariance():
    """A projection from stale data must not look as confident as one
    from fresh data — this is what makes the freshness labels honest."""
    state = initial_state(50.0)
    stds = [predict(state, gap).frp_std for gap in (6, 24, 60)]
    assert stds[0] < stds[1] < stds[2]


def test_process_noise_scales_with_the_cube_of_the_gap_in_position():
    """The dt^3 term is what makes a long gap genuinely uncertain in FRP
    rather than merely uncertain in rate."""
    short = process_noise(1.0, frp_scale=100.0)
    long = process_noise(10.0, frp_scale=100.0)
    assert long[0][0] / short[0][0] == pytest.approx(1000.0, rel=1e-6)
    assert long[1][1] / short[1][1] == pytest.approx(10.0, rel=1e-6)


def test_process_noise_scales_with_the_square_of_frp():
    """Regression: process noise used to be a fixed absolute value while
    measurement noise scaled with FRP. At high FRP the measurement term
    dominated completely, the filter stopped correcting its rate, and a
    rising source was tracked with a +159 MW lag and only 75% coverage on
    a stated 90% interval. Making both scale with the signal fixes it."""
    small = process_noise(12.0, frp_scale=50.0)
    large = process_noise(12.0, frp_scale=500.0)
    assert large[1][1] / small[1][1] == pytest.approx(100.0, rel=1e-6)


def test_process_noise_has_an_frp_floor():
    """A briefly-tiny source must not freeze the filter's rate estimate."""
    from temporal.forecast.kalman import MIN_FRP_SCALE_MW

    assert process_noise(6.0, frp_scale=0.0) == process_noise(6.0, frp_scale=MIN_FRP_SCALE_MW)


def test_filter_tracks_large_and_small_sources_equally_well():
    """Scale invariance: a 20 MW source and a 600 MW source should be
    tracked with comparable *relative* accuracy."""
    small = run_filter(linear_trajectory(0.2, n=40, start=20.0, noise=0.0), relative_error=0.01)
    large = run_filter(linear_trajectory(6.0, n=40, start=600.0, noise=0.0), relative_error=0.01)

    assert small.rate == pytest.approx(0.2, rel=0.15)
    assert large.rate == pytest.approx(6.0, rel=0.15)


def test_prediction_moves_frp_along_the_estimated_rate():
    state = initial_state(50.0)
    state.rate = 2.0
    assert predict(state, 10.0).frp == pytest.approx(70.0)


def test_zero_gap_prediction_is_a_no_op_in_the_mean():
    state = initial_state(50.0)
    state.rate = 3.0
    assert predict(state, 0.0).frp == pytest.approx(50.0)


# --- update behaviour ---------------------------------------------------

def test_update_moves_the_estimate_toward_the_measurement():
    state = initial_state(50.0)
    assert update(state, 80.0).frp > 50.0
    assert update(state, 20.0).frp < 50.0


def test_update_reduces_positional_uncertainty():
    state = initial_state(50.0)
    assert update(state, 55.0).frp_variance < state.frp_variance


def test_covariance_stays_symmetric_over_many_updates():
    """Rounding drives the off-diagonal terms apart; left unchecked that
    eventually produces a negative variance and a crash far from the
    real cause."""
    observations = linear_trajectory(1.0, n=200, noise=0.3, seed=3)
    state = run_filter(observations)
    assert state.covariance[0][1] == pytest.approx(state.covariance[1][0], abs=1e-12)


def test_variances_stay_non_negative():
    observations = linear_trajectory(3.0, n=150, noise=0.4, seed=5)
    state = run_filter(observations)
    assert state.covariance[0][0] >= 0
    assert state.covariance[1][1] >= 0


# --- measurement noise model -------------------------------------------

def test_measurement_noise_scales_with_frp():
    """FIRMS FRP error is roughly proportional, not additive."""
    assert measurement_variance(100.0) > measurement_variance(10.0)


def test_measurement_noise_has_a_floor():
    """A near-zero reading must not claim near-infinite precision."""
    assert measurement_variance(0.0) >= 1.0


# --- edge cases ---------------------------------------------------------

def test_empty_input_returns_none():
    assert run_filter([]) is None


def test_single_observation_seeds_without_a_rate_claim():
    state = run_filter([(0.0, 42.0)])
    assert state.frp == pytest.approx(42.0)
    assert state.rate == 0.0
    # Rate variance must stay loose: one point says nothing about growth.
    assert state.rate_variance > 0.5


def test_out_of_order_observations_are_sorted():
    ordered = run_filter(linear_trajectory(1.0, n=20, noise=0.0), relative_error=0.01)
    shuffled_input = linear_trajectory(1.0, n=20, noise=0.0)
    random.Random(0).shuffle(shuffled_input)
    shuffled = run_filter(shuffled_input, relative_error=0.01)
    assert shuffled.rate == pytest.approx(ordered.rate, abs=1e-6)


def test_negative_and_null_frp_values_are_dropped():
    state = run_filter([(0.0, 10.0), (6.0, None), (12.0, -5.0), (18.0, 20.0)])
    assert state is not None
    assert state.n_updates == 2


def test_adaptive_noise_inflates_when_observations_keep_surprising():
    from temporal.forecast.kalman import adaptive_intensity

    calm = initial_state(50.0)
    calm.nis_history = [1.0, 1.0, 1.0, 1.0]
    surprising = initial_state(50.0)
    surprising.nis_history = [8.0, 9.0, 10.0, 7.0]

    assert adaptive_intensity(surprising) > adaptive_intensity(calm)
    assert adaptive_intensity(calm) == pytest.approx(DEFAULT_PROCESS_NOISE)


def test_adaptive_noise_is_bounded():
    from temporal.forecast.kalman import MAX_ADAPTIVE_SCALE, adaptive_intensity

    extreme = initial_state(50.0)
    extreme.nis_history = [1000.0] * 5
    assert adaptive_intensity(extreme) <= DEFAULT_PROCESS_NOISE * MAX_ADAPTIVE_SCALE
