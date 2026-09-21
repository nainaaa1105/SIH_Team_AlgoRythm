"""Walk-forward calibration.

This is the Day-10 deliverable expressed as a test: a falsifiable claim
about the forecast's own honesty. If the stated 90% interval stops
holding ~90% of the time, these fail.
"""
import random

import pytest

from temporal.validation.calibration import (
    MIN_HISTORY_POINTS,
    Z_50,
    Z_90,
    Prediction,
    assess_predictions,
    calibrate_dataset,
    walk_forward,
)


def synthetic_series(rate, start, n=40, noise=0.30, seed=0):
    """Irregular sampling on purpose — FIRMS gaps are 6 h to 24 h+."""
    rng = random.Random(seed)
    observations, t = [], 0.0
    for _ in range(n):
        t += rng.choice([6.0, 12.0, 12.0, 24.0])
        true = max(start + rate * t, 2.0)
        observations.append((t, max(true * (1 + rng.gauss(0, noise)), 0.5)))
    return observations


def mixed_dataset(n_clusters=60, seed=11):
    random.seed(seed)
    profiles = [(0.0, 20.0), (1.5, 40.0), (-0.8, 300.0)]   # steady, growing, decaying
    return {
        cid: synthetic_series(*profiles[cid % 3], seed=cid)
        for cid in range(n_clusters)
    }


# --- the headline claim -------------------------------------------------

def test_ninety_percent_interval_holds_about_ninety_percent_of_the_time():
    """The claim the whole escalation forecast rests on."""
    result = calibrate_dataset(mixed_dataset())

    assert result.n_predictions > 1000
    assert 0.80 <= result.coverage_90 <= 0.97, (
        f"90% interval covered {result.coverage_90:.1%} — "
        f"{'overconfident' if result.coverage_90 < 0.80 else 'over-conservative'}"
    )
    assert "CALIBRATED" in result.verdict


def test_fifty_percent_interval_is_conservative_and_we_say_so():
    """The 90% interval is calibrated; the 50% one over-covers.

    That is an honest consequence of correcting a heavy-tailed, biased
    error distribution with a single variance multiplier: inflating
    enough to fix the tails necessarily over-inflates the centre. The
    90% interval is the one safety decisions are made against, so it is
    the one tuned to be right — but the 50% figure must not be presented
    as if it were equally calibrated.
    """
    result = calibrate_dataset(mixed_dataset())
    assert result.coverage_50 > 0.50, "the inner interval should not under-cover"
    assert result.coverage_50 <= 0.85


def test_calibration_holds_across_different_source_behaviours():
    """A filter tuned only for growing fires would fail on steady ones."""
    for rate, start in ((0.0, 20.0), (1.5, 40.0), (-0.8, 300.0)):
        dataset = {i: synthetic_series(rate, start, seed=i) for i in range(30)}
        result = calibrate_dataset(dataset)
        assert 0.75 <= result.coverage_90 <= 0.99, (
            f"rate={rate}: coverage {result.coverage_90:.1%}"
        )


# --- walk-forward mechanics --------------------------------------------

def test_walk_forward_uses_only_prior_observations():
    """The strict cut is the point of the exercise. If the target leaked
    into its own history this would become a smoothing test and produce
    excellent, meaningless numbers."""
    observations = synthetic_series(2.0, 50.0, n=20, noise=0.0)
    predictions = walk_forward(observations)

    assert len(predictions) == len(observations) - MIN_HISTORY_POINTS
    # With a noiseless linear series and no leakage, a constant-velocity
    # filter should predict very well but never perfectly from the start.
    assert all(p.std > 0 for p in predictions)


def test_walk_forward_needs_a_minimum_history():
    assert walk_forward([(0.0, 10.0), (6.0, 12.0)]) == []


def test_walk_forward_handles_an_empty_series():
    assert walk_forward([]) == []


def test_prediction_intervals_widen_with_the_gap():
    """A 24-hour-ahead prediction must be less certain than a 6-hour one."""
    short_gaps = [(i * 6.0, 50.0 + i) for i in range(15)]
    long_gaps = [(i * 48.0, 50.0 + i) for i in range(15)]

    short_std = sum(p.std for p in walk_forward(short_gaps)) / len(walk_forward(short_gaps))
    long_std = sum(p.std for p in walk_forward(long_gaps)) / len(walk_forward(long_gaps))
    assert long_std > short_std


def test_interval_includes_measurement_noise():
    """We predict an *observation*, not the latent true FRP, so the
    interval must carry the observation's own error too. Without it,
    coverage would fall short no matter how good the filter is."""
    observations = synthetic_series(1.0, 50.0, n=20, noise=0.0)
    predictions = walk_forward(observations, relative_error=0.30)

    from temporal.forecast.kalman import measurement_variance

    for prediction in predictions[:5]:
        floor = measurement_variance(max(prediction.predicted, 0.0), 0.30) ** 0.5
        assert prediction.std >= floor


# --- assessment ---------------------------------------------------------

def test_perfect_predictions_are_fully_covered():
    predictions = [Prediction(predicted=10.0, actual=10.0, std=1.0, horizon_hours=6.0)] * 20
    result = assess_predictions(predictions)
    assert result.coverage_90 == 1.0
    assert result.mean_absolute_error == 0.0


def test_an_overconfident_filter_is_called_out():
    """Tiny intervals, big errors — the dangerous failure mode."""
    predictions = [
        Prediction(predicted=10.0, actual=100.0, std=0.1, horizon_hours=6.0)
    ] * 20
    result = assess_predictions(predictions)
    assert result.coverage_90 == 0.0
    assert "OVERCONFIDENT" in result.verdict


def test_an_over_conservative_filter_is_called_out():
    """Safe but useless: intervals so wide every alert looks uncertain."""
    predictions = [
        Prediction(predicted=10.0, actual=11.0, std=500.0, horizon_hours=6.0)
    ] * 20
    result = assess_predictions(predictions)
    assert result.coverage_90 == 1.0
    assert "OVER-CONSERVATIVE" in result.verdict


def test_empty_assessment_is_handled():
    assert assess_predictions([]).n_predictions == 0


def test_per_cluster_coverage_is_reported():
    """Pooled coverage can hide one pathological cluster."""
    result = calibrate_dataset(mixed_dataset(n_clusters=12))
    assert len(result.per_cluster) == 12
    assert all(0.0 <= v <= 1.0 for v in result.per_cluster.values())


def test_prediction_interval_membership():
    inside = Prediction(predicted=10.0, actual=11.0, std=2.0, horizon_hours=6.0)
    outside = Prediction(predicted=10.0, actual=30.0, std=2.0, horizon_hours=6.0)
    assert inside.inside(Z_90)
    assert not outside.inside(Z_90)
    assert inside.normalised_error == pytest.approx(0.5)
