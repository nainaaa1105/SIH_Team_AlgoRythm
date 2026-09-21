"""Per-source baseline: what "normal" looks like for this thermal source.

This is the foundation of the Persistent Thermal Source Index. The
project brief's own example of a good output —

    "Thermal anomaly -> likely industrial source -> refinery ->
     persistent for 27 days -> currently 4.2x its normal FRP"

— is only expressible if something knows what "its normal" *is*. Nothing
in M1-M4 did: M2 computes an `frp_zscore` per feature-assembly, but no
component maintains a durable baseline per source. That is what this
builds.

The one rule that matters here: **a baseline must never include the event
being judged against it.** A spike folded into its own mean inflates the
thing it is being compared to and partially hides itself. Every function
takes an explicit cut-off and excludes everything at or after it — the
same point-in-time correctness the research notes demand of temporal
features generally, and which M2 and M4 already apply.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

# A baseline from one or two observations is not a baseline.
MIN_OBSERVATIONS_FOR_BASELINE = 5

# How far back a baseline looks. Matches M1's TimescaleDB retention.
BASELINE_WINDOW_DAYS = 90

# Guard against a degenerate spread making every deviation enormous.
MIN_BASELINE_STD_MW = 0.5


@dataclass
class Baseline:
    mean_frp: Optional[float] = None
    std_frp: Optional[float] = None
    median_frp: Optional[float] = None
    p95_frp: Optional[float] = None
    max_frp: Optional[float] = None
    n_observations: int = 0
    window_days: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.mean_frp is not None

    @property
    def coefficient_of_variation(self) -> Optional[float]:
        """Relative variability — the signature that separates a steady
        flare from an erratic fire."""
        if not self.ok or not self.mean_frp or self.mean_frp <= 0:
            return None
        return (self.std_frp or 0.0) / self.mean_frp

    def as_dict(self) -> Dict[str, object]:
        return {
            "mean_frp": self.mean_frp,
            "std_frp": self.std_frp,
            "median_frp": self.median_frp,
            "p95_frp": self.p95_frp,
            "n_observations": self.n_observations,
            "window_days": round(self.window_days, 2),
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def build_baseline(
    rows: Sequence[Dict],
    before: Optional[datetime] = None,
    window_days: float = BASELINE_WINDOW_DAYS,
) -> Baseline:
    """Summarise a source's normal FRP from its history.

    `before` excludes the current burst from its own baseline. Pass the
    timestamp of the observation you are about to judge.
    """
    usable = [
        r for r in rows
        if r.get("frp") is not None and r.get("acq_datetime") is not None
    ]

    if before is not None:
        cutoff = before - timedelta(days=window_days)
        usable = [r for r in usable if cutoff <= r["acq_datetime"] < before]

    if len(usable) < MIN_OBSERVATIONS_FOR_BASELINE:
        return Baseline(
            n_observations=len(usable),
            reason=(
                f"only {len(usable)} prior observations; need "
                f"{MIN_OBSERVATIONS_FOR_BASELINE} before 'normal' means anything"
            ),
        )

    values = [float(r["frp"]) for r in usable]
    times = [r["acq_datetime"] for r in usable]

    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    span = (max(times) - min(times)).total_seconds() / 86400.0

    return Baseline(
        mean_frp=mean_value,
        std_frp=max(math.sqrt(variance), MIN_BASELINE_STD_MW),
        median_frp=_percentile(values, 50.0),
        p95_frp=_percentile(values, 95.0),
        max_frp=max(values),
        n_observations=len(values),
        window_days=span,
        first_seen=min(times),
        last_seen=max(times),
        reason="ok",
    )


def deviation_sigma(current_frp: Optional[float], baseline: Baseline) -> Optional[float]:
    """How many standard deviations the current FRP sits from normal."""
    if current_frp is None or not baseline.ok or not baseline.std_frp:
        return None
    return (float(current_frp) - baseline.mean_frp) / baseline.std_frp


def deviation_multiple(current_frp: Optional[float], baseline: Baseline) -> Optional[float]:
    """Current FRP as a multiple of normal — the brief's "4.2x" figure.

    Reported alongside the sigma because they say different things: 4.2x
    is intuitive, while sigma accounts for how variable the source
    normally is. A flare that always burns at 10 +/- 0.2 MW hitting 42 MW
    is far more alarming than a wildfire that swings between 5 and 40 MW
    doing the same, and only the sigma captures that.
    """
    if current_frp is None or not baseline.ok or not baseline.mean_frp:
        return None
    if baseline.mean_frp <= 0:
        return None
    return float(current_frp) / baseline.mean_frp


def is_behaving_normally(
    current_frp: Optional[float], baseline: Baseline, sigma_limit: float = 3.0
) -> Optional[bool]:
    """Is this source currently within its own normal range?

    Returns None — not False — when there is no baseline. "We don't know
    what normal is here" and "this is abnormal" are different statements
    and must not look the same on an operations dashboard.
    """
    sigma = deviation_sigma(current_frp, baseline)
    if sigma is None:
        return None
    return abs(sigma) < sigma_limit
