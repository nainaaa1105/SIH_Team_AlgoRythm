"""Shift-rhythm fingerprint.

Produces two of M2's three RHYTHM features — `shift_sharpness` and
`weekend_suppression` — plus a schedule-anomaly flag for the event card.

Both are built on `overpass.py`'s slot rates rather than raw detection
counts, for the sampling-bias reasons set out there. The short version:
raw hour histograms measure the satellite's schedule, not the plant's.

India-specific detail worth knowing: the weekly off in Indian industry
is **Sunday**, with Saturday commonly a working day. Treating
Saturday+Sunday as "the weekend" the way a Western calendar would
dilutes a real signal with a working day, so the default here is Sunday
only, and it is configurable.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from temporal.rhythm.overpass import SlotSummary, build_slots

# Python's weekday(): Monday=0 ... Sunday=6.
INDIA_WEEKEND_DAYS = (6,)              # Sunday
WESTERN_WEEKEND_DAYS = (5, 6)

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Below this many detections the rates are too noisy to characterise.
MIN_DETECTIONS_FOR_RHYTHM = 8
MIN_SPAN_DAYS_FOR_WEEKLY = 14.0        # need a couple of weeks to compare weekdays


@dataclass
class RhythmFingerprint:
    shift_sharpness: Optional[float] = None
    weekend_suppression: Optional[float] = None
    day_night_contrast: Optional[float] = None
    slot_detection_rates: Dict[str, object] = field(default_factory=dict)
    day_of_week_rates: Dict[str, float] = field(default_factory=dict)
    schedule_anomaly: bool = False
    n_observations: int = 0
    n_expected_passes: float = 0.0
    observation_span_days: float = 0.0
    slots_consistent: Optional[bool] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Did we compute anything usable?

        Deliberately *not* tied to `shift_sharpness` alone. That metric
        needs at least two overpass slots, so a source covered by a
        single platform in daylight only — a common case — would have
        been marked unusable despite yielding a perfectly good
        `weekend_suppression`, which is one of the two rhythm features M2
        actually consumes.
        """
        return self.shift_sharpness is not None or self.weekend_suppression is not None


def _normalised_entropy(values: Sequence[float]) -> Optional[float]:
    """Shannon entropy of a distribution, scaled to [0, 1].

    1.0 means the source is equally likely to be seen in every slot
    (continuous burning); 0.0 means it only ever appears in one
    (sharply scheduled).
    """
    positive = [v for v in values if v > 0]
    if len(values) < 2 or not positive:
        return None

    total = sum(positive)
    probabilities = [v / total for v in positive]
    entropy = -sum(p * math.log(p) for p in probabilities)
    return entropy / math.log(len(values))


def shift_sharpness(summary: SlotSummary) -> Optional[float]:
    """How unevenly the source is detected across overpass slots.

    0 = seen just as often on every pass (a continuously-burning flare or
    furnace). 1 = only ever seen in one slot (something that runs on a
    schedule, or only during daylight operations).

    Returns None with fewer than two slots, and that is the honest
    answer rather than a missing feature: with a single slot we cannot
    tell "this source only burns during the day" apart from "only one
    satellite happens to cover this location". `weekend_suppression`
    still works in that case, which is why `RhythmFingerprint.ok` does
    not depend on this metric.
    """
    rates = summary.rates()
    entropy = _normalised_entropy(rates)
    if entropy is None:
        return None
    return max(0.0, min(1.0, 1.0 - entropy))


def day_night_contrast(summary: SlotSummary) -> Optional[float]:
    """Signed day-vs-night detection contrast, in [-1, 1].

    Positive means more often detected by day. A steel furnace or flare
    runs around the clock and sits near zero; a source only detected in
    daylight is either a daytime operation or — importantly — an
    artefact of the mid-infrared band being easier to trigger at night,
    which is why this is reported alongside rather than inside
    `shift_sharpness`.
    """
    day = [s.detection_rate for s in summary.slots if s.daynight == "D"]
    night = [s.detection_rate for s in summary.slots if s.daynight == "N"]
    if not day or not night:
        return None

    day_rate = sum(day) / len(day)
    night_rate = sum(night) / len(night)
    total = day_rate + night_rate
    if total <= 0:
        return None
    return (day_rate - night_rate) / total


def day_of_week_rates(
    rows: Sequence[Dict], span_days: float
) -> Dict[str, float]:
    """Detections per day-of-week, normalised by how many of each
    weekday the observation window actually contained.

    The normalisation matters: a 10-day window holds two Mondays but
    maybe only one Sunday, and comparing raw counts across them would
    manufacture a weekend effect out of the calendar.
    """
    if span_days <= 0:
        return {}

    counts = {name: 0 for name in DAY_NAMES}
    for row in rows:
        moment: Optional[datetime] = row.get("acq_datetime")
        if moment is None:
            continue
        counts[DAY_NAMES[moment.weekday()]] += 1

    weeks = max(span_days / 7.0, 1.0 / 7.0)
    return {name: counts[name] / weeks for name in DAY_NAMES}


def weekend_suppression(
    dow_rates: Dict[str, float], weekend_days: Sequence[int] = INDIA_WEEKEND_DAYS
) -> Optional[float]:
    """Relative drop in activity on the weekly off, in [-1, 1].

    Positive => quieter at the weekend, the signature of a scheduled
    industrial process. Near zero => a continuous process, a wildfire, or
    anything else that does not read a calendar. Negative => busier at
    the weekend, which for agricultural burning is a real pattern.
    """
    if not dow_rates:
        return None

    weekend_names = {DAY_NAMES[d] for d in weekend_days}
    weekday_values = [v for name, v in dow_rates.items() if name not in weekend_names]
    weekend_values = [v for name, v in dow_rates.items() if name in weekend_names]

    if not weekday_values or not weekend_values:
        return None

    weekday_mean = sum(weekday_values) / len(weekday_values)
    weekend_mean = sum(weekend_values) / len(weekend_values)

    if weekday_mean <= 0:
        # No weekday activity at all. If there is weekend activity, that
        # is maximal *negative* suppression — a real and strong signal
        # (agricultural burning does cluster at weekends in places), so
        # return the bound rather than discarding it as undefined.
        if weekend_mean > 0:
            return -1.0
        return None

    return max(-1.0, min(1.0, (weekday_mean - weekend_mean) / weekday_mean))


def detect_schedule_anomaly(
    summary: SlotSummary, latest_slot_key: Optional[str], quiet_rate: float = 0.15
) -> bool:
    """Did the most recent detection land in a slot this source is
    normally quiet in?

    A refinery that has only ever been seen on daytime passes suddenly
    appearing at 01:30 is exactly the kind of off-schedule event the
    project brief cares about. It is a flag for the analyst, not a
    verdict.
    """
    if latest_slot_key is None:
        return False

    for slot in summary.slots:
        if slot.key != latest_slot_key:
            continue
        # Discount the current detection itself before judging the slot's
        # historical rate, or a first-ever detection always looks normal.
        historical = max(slot.n_detections - 1, 0)
        historical_rate = (
            min(historical / slot.expected_passes, 1.0) if slot.expected_passes > 0 else 0.0
        )
        return historical_rate < quiet_rate
    return False


def build_fingerprint(
    rows: Sequence[Dict], weekend_days: Sequence[int] = INDIA_WEEKEND_DAYS
) -> RhythmFingerprint:
    """Full rhythm fingerprint for one cluster's detection history."""
    dated = [r for r in rows if r.get("acq_datetime") is not None]

    if len(dated) < MIN_DETECTIONS_FOR_RHYTHM:
        return RhythmFingerprint(
            n_observations=len(dated),
            reason=(
                f"only {len(dated)} detections; need at least "
                f"{MIN_DETECTIONS_FOR_RHYTHM} before a rhythm means anything"
            ),
        )

    summary = build_slots(dated)
    ordered = sorted(dated, key=lambda r: r["acq_datetime"])
    latest = ordered[-1]

    from temporal.rhythm.overpass import normalise_platform

    latest_key = f"{normalise_platform(latest.get('source'))}:{latest.get('daynight') or 'U'}"

    dow_rates = day_of_week_rates(dated, summary.observation_span_days)
    # A weekly comparison needs more than a week of history, or it is
    # comparing one Sunday against one Monday.
    suppression = (
        weekend_suppression(dow_rates, weekend_days)
        if summary.observation_span_days >= MIN_SPAN_DAYS_FOR_WEEKLY else None
    )

    from temporal.rhythm.overpass import slot_consistency

    return RhythmFingerprint(
        shift_sharpness=shift_sharpness(summary),
        weekend_suppression=suppression,
        day_night_contrast=day_night_contrast(summary),
        slot_detection_rates=summary.as_dict(),
        day_of_week_rates={k: round(v, 4) for k, v in dow_rates.items()},
        schedule_anomaly=detect_schedule_anomaly(summary, latest_key),
        n_observations=len(dated),
        n_expected_passes=sum(s.expected_passes for s in summary.slots),
        observation_span_days=summary.observation_span_days,
        slots_consistent=slot_consistency(summary),
        reason=(
            "ok" if summary.observation_span_days >= MIN_SPAN_DAYS_FOR_WEEKLY
            else f"span {summary.observation_span_days:.1f}d too short for a weekly comparison"
        ),
    )
