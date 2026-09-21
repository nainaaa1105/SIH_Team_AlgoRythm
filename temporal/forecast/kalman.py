"""2-D Kalman filter over fire radiative power.

State:      x = [FRP, dFRP/dt]^T      (megawatts, megawatts per hour)
Transition: F = [[1, dt], [0, 1]]      constant-velocity
Measurement: z = FRP                   H = [1, 0]

Why a filter rather than a regression over recent FRP values: FIRMS
observations arrive at irregular intervals (a 6-hour gap and a 60-hour
gap are both normal), each carries its own uncertainty, and the quantity
we actually want — the *rate* of change — is never measured directly. A
Kalman filter handles all three properly: it propagates uncertainty
across arbitrary gaps, weights each observation by its own noise, and
estimates the unobserved rate as part of the state.

Two details that matter for this data in particular:

  * **Process noise scales with the gap.** Q uses the continuous
    white-noise-acceleration form, so a 60-hour gap widens the covariance
    far more than a 6-hour one. Using a fixed Q would make a projection
    from stale data look as confident as one from fresh data, which is
    exactly the failure the FRESH/MODERATE/STALE labels exist to prevent.
  * **Measurement noise scales with FRP.** FIRMS FRP error is roughly
    proportional rather than additive, so R = (relative_error * FRP)^2
    with a floor so a near-zero reading does not claim infinite precision.

`normalised_innovation_squared` is exposed because it is the standard way
to check a filter is honest: if the reported covariance is right, NIS has
mean 1 for a scalar measurement. `test_kalman.py` checks exactly that.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# RELATIVE process-noise intensity: the spectral density of acceleration
# noise per unit FRP-squared (see `process_noise` for why it is relative
# rather than absolute).
#
# Tuning this exposed a genuine tension that a single objective hides.
# Raising q widens the predictive interval and improves coverage, but it
# also makes the filter distrust its own model and stop tracking the
# trend — and the trend *is* the product here, because time-to-critical
# is computed from the rate. Measured across steady, growing and decaying
# synthetic sources:
#
#     q_rel     cover90    rate estimate (true 1.5 MW/h)
#     1e-7        0.709      1.04     <- rate usable
#     1e-6        0.775      0.46
#     5e-6        0.836      0.20
#     2e-5        0.882     -0.72     <- rate destroyed; a rising fire
#     5e-5        0.904     -2.76        reported as shrinking
#
# So q is set low enough to keep the rate meaningful, and the *interval*
# is corrected separately by CALIBRATED_STD_MULTIPLIER below. Chasing
# coverage through q alone would have produced a beautifully calibrated
# filter that reports every escalating fire as stable.
DEFAULT_PROCESS_NOISE = 1e-7

# Empirical variance correction, measured by walk-forward calibration.
#
# The filter's own covariance is optimistic: its errors are not just
# under-dispersed but *biased*, because a constant-velocity model lags a
# genuinely accelerating source. A pure variance inflation is the
# pragmatic correction. Coverage of the stated 90% interval at q=1e-7:
#
#     multiplier   cover90
#     1.0            0.678
#     2.0            0.867
#     2.5            0.902   <- calibrated
#     3.0            0.930
#
# Applied wherever a predictive interval is reported or sampled, so the
# stated intervals mean what they say. `test_calibration.py` pins it.
CALIBRATED_STD_MULTIPLIER = 2.5

# Floor on the FRP used to scale process noise, so a very small or
# briefly-zero source still admits some change rather than freezing the
# filter's rate estimate at whatever it last held.
MIN_FRP_SCALE_MW = 5.0

# FIRMS FRP uncertainty, as a fraction of the reported value.
DEFAULT_RELATIVE_MEASUREMENT_ERROR = 0.30
MIN_MEASUREMENT_STD_MW = 1.0

# Adaptive-Q bounds: how far the filter may inflate its own process noise
# when observations keep surprising it.
MIN_ADAPTIVE_SCALE = 1.0
MAX_ADAPTIVE_SCALE = 25.0


@dataclass
class KalmanState:
    frp: float
    rate: float                       # MW per hour
    covariance: List[List[float]]     # 2x2
    n_updates: int = 0
    innovations: List[float] = field(default_factory=list)
    nis_history: List[float] = field(default_factory=list)

    @property
    def frp_variance(self) -> float:
        return self.covariance[0][0]

    @property
    def rate_variance(self) -> float:
        return self.covariance[1][1]

    @property
    def frp_std(self) -> float:
        return math.sqrt(max(self.frp_variance, 0.0))

    @property
    def rate_std(self) -> float:
        return math.sqrt(max(self.rate_variance, 0.0))

    def as_dict(self) -> dict:
        return {
            "frp": self.frp,
            "rate": self.rate,
            "covariance": self.covariance,
            "n_updates": self.n_updates,
        }


def _matmul2(a, b):
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def _transpose2(a):
    return [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]


def process_noise(
    dt_hours: float,
    intensity: float = DEFAULT_PROCESS_NOISE,
    frp_scale: Optional[float] = None,
) -> List[List[float]]:
    """Continuous white-noise-acceleration Q for a gap of `dt_hours`.

        Q = q * [[dt^3/3, dt^2/2],
                 [dt^2/2, dt    ]]

    The dt^3 term is what makes a long gap genuinely uncertain in
    position rather than merely uncertain in rate.

    `frp_scale` makes q **proportional to the signal**, and that matters
    more than it looks. FIRMS FRP spans roughly 1 to 1000 MW, and the
    measurement noise is already proportional (30% of the reading). If
    the process noise stayed a fixed absolute value, then at 500 MW the
    measurement term would dwarf it, the filter would stop trusting
    observations, and a rising source would be tracked by a model whose
    rate it can no longer correct — it lags, and because the noise
    interval is computed from the lagging prediction it is also too
    narrow. Measured on a rising synthetic series that produced a +159 MW
    mean lag and 75% coverage on a stated 90% interval.

    Scaling q with FRP^2 makes the whole filter scale-invariant, so a
    20 MW source and a 600 MW source are tracked equally well.
    """
    dt = max(dt_hours, 0.0)
    scale = max(abs(frp_scale) if frp_scale is not None else 1.0, MIN_FRP_SCALE_MW)
    q = intensity * scale ** 2

    return [
        [q * dt ** 3 / 3.0, q * dt ** 2 / 2.0],
        [q * dt ** 2 / 2.0, q * dt],
    ]


def measurement_variance(
    frp: float, relative_error: float = DEFAULT_RELATIVE_MEASUREMENT_ERROR
) -> float:
    std = max(relative_error * abs(frp), MIN_MEASUREMENT_STD_MW)
    return std ** 2


def initial_state(frp: float, frp_variance: Optional[float] = None) -> KalmanState:
    """Seed from a single observation, with a diffuse rate prior.

    The rate starts at zero, but its variance must be **scale-appropriate
    and loose**. A hardcoded variance of 1.0 (rate std 1 MW/h) says we are
    fairly sure a 600 MW fire will not change by more than about 1 MW an
    hour, which is nonsense — and because the measurement noise at that
    magnitude is enormous, a tight prior means the Kalman gain for the
    rate stays near zero and the filter never learns the trend. Measured
    on a rising synthetic series, that produced a persistent ~30%
    underestimate of the true growth rate.

    The prior used instead: a source could plausibly change by roughly
    its own current magnitude over a day, so the rate scale is FRP/24 and
    the variance its square. That is deliberately vague — which is the
    point of a prior you want the data to overwrite quickly.
    """
    variance = measurement_variance(frp) if frp_variance is None else frp_variance
    rate_scale = max(abs(float(frp)), MIN_FRP_SCALE_MW) / 24.0
    return KalmanState(
        frp=float(frp),
        rate=0.0,
        covariance=[[variance, 0.0], [0.0, rate_scale ** 2]],
    )


def predict(
    state: KalmanState, dt_hours: float, intensity: float = DEFAULT_PROCESS_NOISE
) -> KalmanState:
    """Propagate the state forward across a time gap."""
    dt = max(dt_hours, 0.0)
    f = [[1.0, dt], [0.0, 1.0]]

    predicted_frp = state.frp + state.rate * dt
    predicted_rate = state.rate

    fp = _matmul2(f, state.covariance)
    fpf = _matmul2(fp, _transpose2(f))
    # Scale process noise by the source's own magnitude — see the note in
    # `process_noise`. The predicted FRP is used rather than the prior so
    # a fast-rising source admits proportionally more change.
    q = process_noise(dt, intensity, frp_scale=max(predicted_frp, state.frp))
    covariance = [[fpf[i][j] + q[i][j] for j in range(2)] for i in range(2)]

    return KalmanState(
        frp=predicted_frp,
        rate=predicted_rate,
        covariance=covariance,
        n_updates=state.n_updates,
        innovations=list(state.innovations),
        nis_history=list(state.nis_history),
    )


def update(
    state: KalmanState,
    measurement: float,
    relative_error: float = DEFAULT_RELATIVE_MEASUREMENT_ERROR,
) -> KalmanState:
    """Fold in an FRP observation."""
    r = measurement_variance(measurement, relative_error)

    innovation = measurement - state.frp            # H = [1, 0]
    innovation_variance = state.covariance[0][0] + r

    if innovation_variance <= 0:
        return state

    gain = [state.covariance[0][0] / innovation_variance,
            state.covariance[1][0] / innovation_variance]

    frp = state.frp + gain[0] * innovation
    rate = state.rate + gain[1] * innovation

    # P = (I - K H) P, with H = [1, 0].
    covariance = [
        [
            state.covariance[0][0] - gain[0] * state.covariance[0][0],
            state.covariance[0][1] - gain[0] * state.covariance[0][1],
        ],
        [
            state.covariance[1][0] - gain[1] * state.covariance[0][0],
            state.covariance[1][1] - gain[1] * state.covariance[0][1],
        ],
    ]

    # Keep the covariance symmetric. Rounding drives the two off-diagonal
    # terms apart over many updates, and an asymmetric P eventually
    # produces a negative variance and a crash far from the real cause.
    off_diagonal = 0.5 * (covariance[0][1] + covariance[1][0])
    covariance[0][1] = covariance[1][0] = off_diagonal

    return KalmanState(
        frp=frp,
        rate=rate,
        covariance=covariance,
        n_updates=state.n_updates + 1,
        innovations=[*state.innovations, innovation],
        nis_history=[*state.nis_history, innovation ** 2 / innovation_variance],
    )


def adaptive_intensity(
    state: KalmanState, base: float = DEFAULT_PROCESS_NOISE, window: int = 5
) -> float:
    """Inflate process noise when recent observations keep surprising the filter.

    If the mean NIS over the recent window is well above 1, the model is
    under-describing how fast the source changes, and the honest response
    is to admit more process noise rather than keep reporting a tight
    covariance that the data contradicts.
    """
    recent = state.nis_history[-window:]
    if len(recent) < 2:
        return base

    mean_nis = sum(recent) / len(recent)
    scale = max(MIN_ADAPTIVE_SCALE, min(mean_nis, MAX_ADAPTIVE_SCALE))
    return base * scale


def run_filter(
    observations: Sequence[Tuple[float, float]],
    intensity: float = DEFAULT_PROCESS_NOISE,
    relative_error: float = DEFAULT_RELATIVE_MEASUREMENT_ERROR,
    adaptive: bool = False,
) -> Optional[KalmanState]:
    # `adaptive` defaults off. It inflates q from recent innovations,
    # which made sense when q was a fixed absolute value, but q now scales
    # with FRP^2 and already responds to the source's magnitude. Stacking
    # the two multiplied the process noise enough to destroy the rate
    # estimate: on a rising series it turned an estimate of 1.04 MW/h
    # (true 1.5) into -1.64. Kept available for experimentation, off by
    # default.
    """Run the filter over `(time_hours, frp)` pairs, oldest first.

    Returns None for an empty series. Times are hours from an arbitrary
    origin; only the differences matter.
    """
    usable = [
        (float(t), float(f)) for t, f in observations
        if t is not None and f is not None and f >= 0
    ]
    if not usable:
        return None

    usable.sort(key=lambda pair: pair[0])

    state = initial_state(usable[0][1])
    state.n_updates = 1
    previous_time = usable[0][0]

    for time_hours, frp in usable[1:]:
        dt = time_hours - previous_time
        effective = adaptive_intensity(state, intensity) if adaptive else intensity
        state = predict(state, dt, effective)
        state = update(state, frp, relative_error)
        previous_time = time_hours

    return state


def normalised_innovation_squared(state: KalmanState) -> Optional[float]:
    """Mean NIS — the filter's own consistency check.

    For a scalar measurement a correctly-specified filter gives mean NIS
    near 1. Much above 1 means the reported uncertainty is too tight
    (overconfident); much below means it is too loose.
    """
    if not state.nis_history:
        return None
    return sum(state.nis_history) / len(state.nis_history)
