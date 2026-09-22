"""Walk-forward calibration of the forecast.

The Day-10 deliverable, and the only honest way to judge a probabilistic
forecast: for each observation in a cluster's history, filter everything
strictly before it, predict the next value with a stated interval, then
check whether the actual landed inside. Repeat across every cluster and
count.

A 90% interval that contains the truth 90% of the time is calibrated. One
that contains it 99% of the time is over-conservative — technically safe
but it makes every alert look uncertain and people stop reading them. One
that contains it 60% of the time is dangerous, because a stated
time-to-critical would be trusted more than it deserves.

The strict `< t` cut is the point of the exercise. Including the
observation being predicted, or any later one, turns this from a forecast
test into a smoothing test and would produce excellent, meaningless
numbers — the same point-in-time discipline the project's research notes
demand of temporal features generally.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Normal-distribution multipliers for two-sided intervals.
Z_50 = 0.6745
Z_90 = 1.6449

# Need some history before a prediction is meaningful.
MIN_HISTORY_POINTS = 4


@dataclass
class CalibrationResult:
    n_predictions: int = 0
    coverage_50: Optional[float] = None
    coverage_90: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    median_absolute_error: Optional[float] = None
    mean_normalised_error: Optional[float] = None
    n_clusters: int = 0
    verdict: str = ""
    per_cluster: Dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_predictions": self.n_predictions,
            "n_clusters": self.n_clusters,
            "coverage_50": round(self.coverage_50, 4) if self.coverage_50 is not None else None,
            "coverage_90": round(self.coverage_90, 4) if self.coverage_90 is not None else None,
            "mean_absolute_error_mw": (
                round(self.mean_absolute_error, 3) if self.mean_absolute_error is not None else None
            ),
            "median_absolute_error_mw": (
                round(self.median_absolute_error, 3)
                if self.median_absolute_error is not None else None
            ),
            "verdict": self.verdict,
        }


@dataclass
class Prediction:
    predicted: float
    actual: float
    std: float
    horizon_hours: float

    @property
    def error(self) -> float:
        return self.actual - self.predicted

    @property
    def normalised_error(self) -> Optional[float]:
        if self.std <= 0:
            return None
        return self.error / self.std

    def inside(self, z: float) -> bool:
        if self.std <= 0:
            return self.actual == self.predicted
        return abs(self.error) <= z * self.std


def walk_forward(
    observations: Sequence[Tuple[float, float]],
    intensity: Optional[float] = None,
    relative_error: Optional[float] = None,
) -> List[Prediction]:
    """One-step-ahead predictions over a single cluster's history.

    `observations` are `(hours_since_first, frp)` pairs. For each index i
    beyond the minimum history, the filter sees only points `< i` and
    predicts point `i`.
    """
    from temporal.config import get_m5_settings
    from temporal.forecast.kalman import predict, run_filter

    settings = get_m5_settings()
    intensity = settings.process_noise if intensity is None else intensity
    relative_error = (
        settings.relative_measurement_error if relative_error is None else relative_error
    )

    usable = sorted(
        (float(t), float(f)) for t, f in observations
        if t is not None and f is not None and f >= 0
    )
    if len(usable) <= MIN_HISTORY_POINTS:
        return []

    predictions: List[Prediction] = []
    for i in range(MIN_HISTORY_POINTS, len(usable)):
        history = usable[:i]                       # strictly before the target
        target_time, actual = usable[i]

        state = run_filter(
            history, intensity=intensity, relative_error=relative_error, adaptive=True
        )
        if state is None:
            continue

        horizon = target_time - history[-1][0]
        future = predict(state, horizon, intensity)

        # The interval must include measurement noise as well as state
        # uncertainty: we are predicting an *observation*, not the latent
        # true FRP, and the observation carries its own error.
        from temporal.forecast.kalman import CALIBRATED_STD_MULTIPLIER, measurement_variance

        total_variance = future.frp_variance + measurement_variance(
            max(future.frp, 0.0), relative_error
        )
        predictions.append(Prediction(
            predicted=future.frp,
            actual=actual,
            # Empirical variance correction — the raw covariance is
            # optimistic because a constant-velocity model lags a real
            # accelerating source. See kalman.CALIBRATED_STD_MULTIPLIER.
            std=math.sqrt(max(total_variance, 0.0)) * CALIBRATED_STD_MULTIPLIER,
            horizon_hours=horizon,
        ))

    return predictions


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _verdict(coverage_90: Optional[float]) -> str:
    if coverage_90 is None:
        return "no predictions to assess"
    if coverage_90 < 0.75:
        return (
            f"OVERCONFIDENT: the stated 90% interval held only {coverage_90:.0%} of the "
            "time. A time-to-critical from this filter would be trusted more than it "
            "deserves — raise the process noise."
        )
    if coverage_90 > 0.98:
        return (
            f"OVER-CONSERVATIVE: the 90% interval held {coverage_90:.0%} of the time. "
            "Safe, but intervals this wide make every alert look uncertain — lower the "
            "process noise."
        )
    return f"CALIBRATED: the stated 90% interval held {coverage_90:.0%} of the time."


def assess_predictions(predictions: Sequence[Prediction], n_clusters: int = 0) -> CalibrationResult:
    """Summarise coverage and error across a set of walk-forward predictions."""
    if not predictions:
        return CalibrationResult(verdict="no predictions to assess")

    errors = [abs(p.error) for p in predictions]
    normalised = [p.normalised_error for p in predictions if p.normalised_error is not None]

    return CalibrationResult(
        n_predictions=len(predictions),
        coverage_50=sum(p.inside(Z_50) for p in predictions) / len(predictions),
        coverage_90=sum(p.inside(Z_90) for p in predictions) / len(predictions),
        mean_absolute_error=sum(errors) / len(errors),
        median_absolute_error=_median(errors),
        mean_normalised_error=(sum(normalised) / len(normalised)) if normalised else None,
        n_clusters=n_clusters,
        verdict=_verdict(sum(p.inside(Z_90) for p in predictions) / len(predictions)),
    )


def calibrate_dataset(
    series_by_cluster: Dict[int, Sequence[Tuple[float, float]]]
) -> CalibrationResult:
    """Run the walk-forward test across many clusters and pool the result."""
    all_predictions: List[Prediction] = []
    per_cluster: Dict[int, float] = {}

    for cluster_id, observations in series_by_cluster.items():
        predictions = walk_forward(observations)
        if not predictions:
            continue
        all_predictions.extend(predictions)
        per_cluster[cluster_id] = (
            sum(p.inside(Z_90) for p in predictions) / len(predictions)
        )

    result = assess_predictions(all_predictions, n_clusters=len(per_cluster))
    result.per_cluster = per_cluster
    return result
