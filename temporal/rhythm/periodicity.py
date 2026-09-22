"""Periodicity detection, applied only where it is defensible.

The task division asks for "FFT periodicity". An FFT is only meaningful
on a uniformly-sampled series, and FIRMS detections are not uniformly
sampled in time — they arrive in bursts at fixed overpass hours (see
`overpass.py`). Running a transform over raw detection times recovers the
satellite's revisit rhythm and presents it as the facility's.

What *is* uniformly sampled is the **daily** series: one value per
calendar day, the number of detections that day. Every day gets the same
overpass opportunities, so day-to-day variation is a real property of the
source. A transform over that series can legitimately find a weekly
(7-day) cycle, which is the industrially interesting one — a plant that
idles every Sunday.

So: daily resampling, then a periodogram, and periods shorter than two
days are not reported at all because the sampling cannot support them.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

# Nyquist on a daily series: nothing below a 2-day period is recoverable.
MIN_RESOLVABLE_PERIOD_DAYS = 2.0
MAX_REPORTABLE_PERIOD_DAYS = 30.0

# A weekly cycle is the one with an industrial interpretation.
WEEKLY_PERIOD_DAYS = 7.0
WEEKLY_TOLERANCE_DAYS = 0.75

MIN_DAYS_FOR_PERIODOGRAM = 14


@dataclass
class PeriodicityResult:
    dominant_period_days: Optional[float] = None
    dominant_period_hours: Optional[float] = None
    confidence: float = 0.0
    is_weekly: bool = False
    n_days: int = 0
    reason: str = ""


def daily_series(rows: Sequence[Dict]) -> Tuple[List[datetime], List[float]]:
    """Resample detections onto a gap-filled daily grid.

    Days with no detection become 0 rather than being skipped — an
    absence is information (the source was quiet), and omitting those
    days would silently make the series non-uniform again, reintroducing
    the very problem this module exists to avoid.
    """
    dated = [r["acq_datetime"] for r in rows if r.get("acq_datetime") is not None]
    if not dated:
        return [], []

    start = min(dated).date()
    end = max(dated).date()
    n_days = (end - start).days + 1

    counts: Dict[object, int] = {}
    for moment in dated:
        counts[moment.date()] = counts.get(moment.date(), 0) + 1

    days = [start + timedelta(days=i) for i in range(n_days)]
    values = [float(counts.get(day, 0)) for day in days]
    return [datetime(d.year, d.month, d.day) for d in days], values


def periodogram(values: Sequence[float]) -> Dict[float, float]:
    """Power at each candidate period, in days.

    Uses numpy's real FFT on the mean-removed series. The mean must go —
    otherwise the zero-frequency term dominates everything and the
    "dominant period" is just the series length.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    n = array.size
    if n < MIN_DAYS_FOR_PERIODOGRAM:
        return {}

    centred = array - array.mean()
    if not np.any(centred):
        return {}

    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    frequencies = np.fft.rfftfreq(n, d=1.0)   # cycles per day

    power: Dict[float, float] = {}
    for frequency, value in zip(frequencies[1:], spectrum[1:]):   # skip DC
        if frequency <= 0:
            continue
        period = 1.0 / frequency
        if MIN_RESOLVABLE_PERIOD_DAYS <= period <= MAX_REPORTABLE_PERIOD_DAYS:
            power[float(period)] = float(value)
    return power


def detect_periodicity(rows: Sequence[Dict]) -> PeriodicityResult:
    """Dominant cycle in a cluster's daily activity, if there is one."""
    days, values = daily_series(rows)

    if len(days) < MIN_DAYS_FOR_PERIODOGRAM:
        return PeriodicityResult(
            n_days=len(days),
            reason=(
                f"only {len(days)} days of history; need "
                f"{MIN_DAYS_FOR_PERIODOGRAM} before a cycle can be claimed"
            ),
        )

    power = periodogram(values)
    if not power:
        return PeriodicityResult(
            n_days=len(days),
            reason="no variation in daily activity — nothing to find a cycle in",
        )

    total_power = sum(power.values())
    dominant_period, dominant_power = max(power.items(), key=lambda kv: kv[1])
    confidence = dominant_power / total_power if total_power > 0 else 0.0

    return PeriodicityResult(
        dominant_period_days=dominant_period,
        dominant_period_hours=dominant_period * 24.0,
        confidence=confidence,
        is_weekly=abs(dominant_period - WEEKLY_PERIOD_DAYS) <= WEEKLY_TOLERANCE_DAYS,
        n_days=len(days),
        reason="ok",
    )


def sub_daily_periodicity_is_not_recoverable() -> str:
    """Documentation as code, because this will be asked in review.

    Kept as a callable so the reasoning travels with the module rather
    than living only in a design document nobody opens.
    """
    return (
        "Sub-daily periodicity (a two-shift or three-shift pattern) cannot be "
        "recovered from FIRMS. The satellites sample a given point only a few "
        "times a day, at fixed local solar hours, so the Nyquist limit on a "
        "daily series is a 2-day period. Detecting shift structure would need "
        "a geostationary sensor sampling every 10-30 minutes — INSAT-3DS is "
        "exactly that, which is why M1 wired it in, but its fire product is "
        "still stubbed pending MOSDAC access."
    )
