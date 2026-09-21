"""Time-to-critical, with honest confidence intervals.

The projection itself is one line:

    t = (FRP_critical - FRP_now) / rate

The interesting part is its uncertainty. Both FRP and rate are uncertain,
and `t` is a **ratio** of two correlated Gaussians. Its distribution is
heavy-tailed and asymmetric — when the rate's uncertainty straddles zero
the ratio's variance is formally infinite, because "the source might not
be growing at all" means "it might never get there". Propagating variance
through the division with a first-order Taylor expansion, the usual
shortcut, produces a neat symmetric interval that is simply wrong in
exactly the cases that matter most.

So this samples instead: draw states from the filter's covariance,
compute `t` for each, and read percentiles off the empirical
distribution. Slower, but it tells the truth — including the most useful
output here, `probability_reaches_critical`, which a variance-propagation
approach cannot express at all. "60% chance of reaching critical, and if
it does, most likely in 6-14 hours" is a far more actionable statement
for a responder than a single bare ETA.

"Critical" is defined relative to the source's **own** PTSI baseline, not
an absolute FRP. A steel furnace at 80 MW is normal; a 20 MW rise at a
site that has never exceeded 5 MW is not. That is the whole normal-vs-
abnormal thesis of the project.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from temporal.forecast.kalman import KalmanState

DEFAULT_SAMPLES = 4000
DEFAULT_SIGMA_MULTIPLE = 3.0        # critical = baseline mean + 3 sigma

# Beyond this horizon a projection is meaningless — fires change regime,
# weather turns, crews intervene.
MAX_HORIZON_HOURS = 240.0           # 10 days

# Below this the source is not meaningfully escalating.
MIN_ESCALATION_RATE_MW_PER_HOUR = 1e-3

# A handful of sampled trajectories reaching the threshold is not an
# escalation. With a widened covariance even a clearly *falling* source
# has a thin positive tail — a rate of -3 MW/h with 1.25 std still puts
# ~0.8% of draws above zero, which was enough to flag a dying fire as
# escalating and would have produced false alerts on exactly the events
# an operator most wants filtered out. The probability is still reported
# in full; this only gates the boolean.
MIN_ESCALATION_PROBABILITY = 0.10


@dataclass
class EscalationForecast:
    escalating: bool
    time_to_critical_hours: Optional[float]         # median of the reaching samples
    p50_low: Optional[float] = None
    p50_high: Optional[float] = None
    p90_low: Optional[float] = None
    p90_high: Optional[float] = None
    probability_reaches_critical: float = 0.0
    critical_threshold_frp: Optional[float] = None
    current_frp: Optional[float] = None
    rate_mw_per_hour: Optional[float] = None
    already_critical: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "escalating": self.escalating,
            "time_to_critical_hours": self.time_to_critical_hours,
            "p50": [self.p50_low, self.p50_high],
            "p90": [self.p90_low, self.p90_high],
            "probability_reaches_critical": round(self.probability_reaches_critical, 4),
            "critical_threshold_frp": self.critical_threshold_frp,
            "already_critical": self.already_critical,
            "reason": self.reason,
        }


def critical_threshold(
    baseline_mean: Optional[float],
    baseline_std: Optional[float],
    sigma_multiple: float = DEFAULT_SIGMA_MULTIPLE,
    absolute_floor_mw: float = 10.0,
) -> Optional[float]:
    """The FRP at which this particular source counts as abnormal.

    Relative to its own history, with a floor so a source whose baseline
    is essentially zero does not get a critical threshold of nearly zero
    and alarm on its first real detection.
    """
    if baseline_mean is None:
        return None
    spread = baseline_std if baseline_std and baseline_std > 0 else max(baseline_mean * 0.5, 1.0)
    return max(baseline_mean + sigma_multiple * spread, absolute_floor_mw)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _sample_states(
    state: KalmanState, n_samples: int, seed: Optional[int], std_multiplier: float = 1.0
) -> List[tuple]:
    """Draw (frp, rate) pairs from the filter's covariance.

    Uses a Cholesky factor of the 2x2 covariance so the correlation
    between FRP and rate is preserved — sampling them independently would
    understate the uncertainty in their ratio, which is the one quantity
    we care about here.

    `std_multiplier` applies the empirical variance correction from
    `kalman.CALIBRATED_STD_MULTIPLIER`. It matters here in a way that is
    directly safety-relevant: the filter systematically *underestimates*
    growth (a true 1.5 MW/h rise is estimated at ~1.2), which on its own
    would push every time-to-critical later than reality — the dangerous
    direction. Widening the sampled rate distribution pulls the early
    percentiles back in, so `p90_low`, the "soonest plausible arrival"
    a responder should actually plan against, stays honest.
    """
    import random

    rng = random.Random(seed)
    variance_scale = std_multiplier ** 2

    a = max(state.covariance[0][0], 0.0) * variance_scale
    b = state.covariance[0][1] * variance_scale
    d = max(state.covariance[1][1], 0.0) * variance_scale

    l11 = math.sqrt(a) if a > 0 else 0.0
    l21 = (b / l11) if l11 > 0 else 0.0
    under_root = d - l21 * l21
    l22 = math.sqrt(under_root) if under_root > 0 else 0.0

    samples = []
    for _ in range(n_samples):
        z1, z2 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        samples.append((
            state.frp + l11 * z1,
            state.rate + l21 * z1 + l22 * z2,
        ))
    return samples


def forecast(
    state: KalmanState,
    threshold_frp: Optional[float],
    n_samples: int = DEFAULT_SAMPLES,
    max_horizon_hours: float = MAX_HORIZON_HOURS,
    seed: Optional[int] = 42,
) -> EscalationForecast:
    """Project the filtered state forward to a critical threshold."""
    if threshold_frp is None:
        return EscalationForecast(
            escalating=False, time_to_critical_hours=None,
            current_frp=state.frp, rate_mw_per_hour=state.rate,
            reason="no baseline for this source, so no critical threshold can be set",
        )

    if state.frp >= threshold_frp:
        return EscalationForecast(
            escalating=True, time_to_critical_hours=0.0,
            probability_reaches_critical=1.0,
            critical_threshold_frp=threshold_frp,
            current_frp=state.frp, rate_mw_per_hour=state.rate,
            already_critical=True,
            reason="already at or above its critical threshold",
        )

    from temporal.forecast.kalman import CALIBRATED_STD_MULTIPLIER

    samples = _sample_states(state, n_samples, seed, CALIBRATED_STD_MULTIPLIER)

    times: List[float] = []
    for frp, rate in samples:
        if rate <= MIN_ESCALATION_RATE_MW_PER_HOUR:
            continue                       # this draw never gets there
        if frp >= threshold_frp:
            times.append(0.0)
            continue
        t = (threshold_frp - frp) / rate
        if 0 <= t <= max_horizon_hours:
            times.append(t)

    probability = len(times) / n_samples if n_samples else 0.0

    if not times or probability < MIN_ESCALATION_PROBABILITY:
        return EscalationForecast(
            escalating=False, time_to_critical_hours=None,
            probability_reaches_critical=probability,
            critical_threshold_frp=threshold_frp,
            current_frp=state.frp, rate_mw_per_hour=state.rate,
            reason=(
                "not escalating: no sampled trajectory reaches the threshold within "
                f"{max_horizon_hours:.0f} h"
                if not times else
                f"not escalating: only {probability:.1%} of sampled trajectories reach "
                f"{threshold_frp:.1f} MW, below the {MIN_ESCALATION_PROBABILITY:.0%} "
                "threshold for calling this an escalation"
            ),
        )

    return EscalationForecast(
        escalating=True,
        time_to_critical_hours=_percentile(times, 50.0),
        p50_low=_percentile(times, 25.0),
        p50_high=_percentile(times, 75.0),
        p90_low=_percentile(times, 5.0),
        p90_high=_percentile(times, 95.0),
        probability_reaches_critical=probability,
        critical_threshold_frp=threshold_frp,
        current_frp=state.frp,
        rate_mw_per_hour=state.rate,
        reason=(
            f"{probability:.0%} of sampled trajectories reach {threshold_frp:.1f} MW "
            f"within {max_horizon_hours:.0f} h"
        ),
    )


def predict_frp_at(state: KalmanState, horizon_hours: float) -> tuple:
    """Predicted FRP and 1-sigma at a future time.

    Used by the walk-forward calibration harness, which is the honest
    test of whether any of this works: predict the next observation,
    then check how often the actual landed inside the stated interval.
    """
    from temporal.forecast.kalman import predict

    future = predict(state, horizon_hours)
    return future.frp, future.frp_std
