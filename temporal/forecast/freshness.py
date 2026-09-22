"""FRESH / MODERATE / STALE data-currency labels.

A Kalman projection made from a three-day-old observation and one made
from a two-hour-old observation look identical on a dashboard unless
something says otherwise. The covariance does widen with the gap — that
is handled properly in `kalman.py` — but a responder reading a
time-to-critical needs the staleness stated plainly, not inferred from an
error bar.

Freshness is judged against **each source's own observed cadence** rather
than a fixed clock. A flare detected on nearly every overpass is stale
after 12 hours of silence; a marginal source seen twice a week is not.
Using one absolute threshold would either cry stale constantly for the
second or never for the first.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

FRESH = "FRESH"
MODERATE = "MODERATE"
STALE = "STALE"

# Multiples of the source's own typical gap.
FRESH_CADENCE_MULTIPLE = 1.5
MODERATE_CADENCE_MULTIPLE = 3.0

# Fallbacks when the cadence is unknown (too few detections to measure).
FALLBACK_FRESH_HOURS = 12.0
FALLBACK_MODERATE_HOURS = 36.0

# However chatty a source is, a gap this long is stale in absolute terms:
# operations change, crews intervene, and a projection stops meaning much.
ABSOLUTE_STALE_HOURS = 72.0


@dataclass
class Freshness:
    label: str
    age_hours: Optional[float]
    cadence_hours: Optional[float]
    last_observation_at: Optional[datetime]
    reason: str = ""

    @property
    def is_usable(self) -> bool:
        """Should a forecast built on this be shown as actionable?"""
        return self.label in (FRESH, MODERATE)


def median_cadence_hours(timestamps: Sequence[datetime]) -> Optional[float]:
    """Typical gap between consecutive detections.

    Median rather than mean: revisit gaps are heavily skewed by the
    occasional multi-day cloud outage, and one long gap would drag a mean
    far enough to make a genuinely stale source look freshly observed.
    """
    usable = sorted(t for t in timestamps if t is not None)
    if len(usable) < 3:
        return None

    gaps = [
        (later - earlier).total_seconds() / 3600.0
        for earlier, later in zip(usable, usable[1:])
        if (later - earlier).total_seconds() > 0
    ]
    if not gaps:
        return None

    gaps.sort()
    middle = len(gaps) // 2
    if len(gaps) % 2:
        return gaps[middle]
    return (gaps[middle - 1] + gaps[middle]) / 2.0


def assess(
    last_observation_at: Optional[datetime],
    timestamps: Optional[Sequence[datetime]] = None,
    now: Optional[datetime] = None,
) -> Freshness:
    """Label how current a cluster's most recent observation is."""
    if last_observation_at is None:
        return Freshness(
            label=STALE, age_hours=None, cadence_hours=None, last_observation_at=None,
            reason="no observations on record",
        )

    moment = now or datetime.now(timezone.utc)

    # M1 stores tz-aware UTC, but a naive datetime reaching here would
    # raise on subtraction; normalise both sides rather than crash.
    reference = last_observation_at
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    age_hours = (moment - reference).total_seconds() / 3600.0
    cadence = median_cadence_hours(timestamps) if timestamps else None

    if age_hours >= ABSOLUTE_STALE_HOURS:
        return Freshness(
            label=STALE, age_hours=age_hours, cadence_hours=cadence,
            last_observation_at=reference,
            reason=f"{age_hours:.0f} h since the last detection, beyond the absolute limit",
        )

    if cadence and cadence > 0:
        fresh_limit = cadence * FRESH_CADENCE_MULTIPLE
        moderate_limit = cadence * MODERATE_CADENCE_MULTIPLE
        basis = f"against this source's own {cadence:.1f} h typical gap"
    else:
        fresh_limit = FALLBACK_FRESH_HOURS
        moderate_limit = FALLBACK_MODERATE_HOURS
        basis = "against the default cadence (too few detections to measure its own)"

    if age_hours <= fresh_limit:
        label, description = FRESH, "recently observed"
    elif age_hours <= moderate_limit:
        label, description = MODERATE, "a little overdue"
    else:
        label, description = STALE, "overdue"

    return Freshness(
        label=label, age_hours=age_hours, cadence_hours=cadence,
        last_observation_at=reference,
        reason=f"{age_hours:.1f} h old, {description} {basis}",
    )
