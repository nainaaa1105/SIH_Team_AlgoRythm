"""Satellite overpass slots — the sampling-bias correction.

**Read this before touching anything else in M5.**

The obvious way to look for a shift schedule is a histogram of detection
hours. It does not work, and it fails in a way that produces
confident-looking nonsense.

Polar-orbiting satellites cross a given point at fixed local solar
times — roughly 01:30/13:30 for VIIRS S-NPP and MODIS Aqua, 10:30/22:30
for MODIS Terra, and so on. A histogram of detection hours therefore
shows a handful of spikes *at the overpass times*, no matter what the
facility below actually does. Run an FFT over that and it will report a
clean 12- or 24-hour period for a flare that burns continuously. You
would be measuring the satellite's schedule, not the plant's.

What survives the bias:

  * **Day-of-week comparisons.** The overpass time is the same every
    day, so Monday-13:30 against Sunday-13:30 is a fair comparison. This
    is why `weekend_suppression` is a sound feature.
  * **Detection *rate* per overpass slot.** Of all the S-NPP daytime
    passes over this cluster, what fraction saw a fire? A flare detected
    on ~100% of passes is continuous; one detected on 60% of day passes
    and 5% of night passes has a daytime-operation signature. That is a
    real statement about the source.

What does not survive: any sub-daily periodicity. We cannot recover a
two-shift pattern from four samples a day pinned to fixed hours, and
this module does not pretend to.

Slots are keyed on `(platform, daynight)` — both already stored by M1 on
`hotspots.source` and `hotspots.daynight` — rather than on hardcoded
equator-crossing times, which vary by satellite, drift over a mission,
and would be a guess. The observed local solar time of each slot is
computed as a descriptive statistic so the grouping can be checked
against reality.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

# Wide-swath instruments see a given point roughly once per slot per day.
# Not exactly true at low latitudes where swath gaps open up, which is
# why detection rates are clamped to 1.0 rather than trusted blindly.
EXPECTED_PASSES_PER_DAY_PER_SLOT = 1.0

MIN_SPAN_DAYS_FOR_RATE = 1.0


@dataclass
class OverpassSlot:
    platform: str
    daynight: str
    n_detections: int = 0
    expected_passes: float = 0.0
    detection_rate: float = 0.0
    mean_local_solar_hour: Optional[float] = None
    local_solar_hour_spread: Optional[float] = None

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.daynight}"


@dataclass
class SlotSummary:
    slots: List[OverpassSlot] = field(default_factory=list)
    observation_span_days: float = 0.0
    n_detections: int = 0

    def as_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            slot.key: {
                "n_detections": slot.n_detections,
                "expected_passes": round(slot.expected_passes, 2),
                "detection_rate": round(slot.detection_rate, 4),
                "mean_local_solar_hour": (
                    round(slot.mean_local_solar_hour, 2)
                    if slot.mean_local_solar_hour is not None else None
                ),
            }
            for slot in self.slots
        }

    def rates(self) -> List[float]:
        return [slot.detection_rate for slot in self.slots]


def normalise_platform(source: Optional[str]) -> str:
    """Collapse a FIRMS source string to the physical platform.

    `VIIRS_SNPP_NRT` and `VIIRS_SNPP` are the same satellite on the same
    orbit — they must land in one slot, or the same pass gets counted as
    two independent looks.
    """
    if not source:
        return "UNKNOWN"
    return source.upper().replace("_NRT", "").replace("_RT", "")


def local_solar_hour(utc_time: datetime, longitude: float) -> float:
    """Local solar time in hours, from UTC and longitude.

    Solar time rather than civil/zone time on purpose: a satellite's
    orbit is fixed with respect to the sun, not to India Standard Time,
    so solar hour is the coordinate in which overpasses are actually
    constant.
    """
    utc_hours = utc_time.hour + utc_time.minute / 60.0 + utc_time.second / 3600.0
    return (utc_hours + longitude / 15.0) % 24.0


def _circular_mean_hours(hours: Sequence[float]) -> Tuple[float, float]:
    """Mean and spread of times-of-day, done on the circle.

    A plain mean of 23:30 and 00:30 gives noon, which would make a
    night-time slot look like a daytime one.
    """
    import math

    if not hours:
        return 0.0, 0.0

    angles = [h / 24.0 * 2 * math.pi for h in hours]
    x = sum(math.cos(a) for a in angles) / len(angles)
    y = sum(math.sin(a) for a in angles) / len(angles)

    mean_angle = math.atan2(y, x)
    mean_hour = (mean_angle / (2 * math.pi) * 24.0) % 24.0

    # Circular standard deviation, in hours.
    resultant = math.hypot(x, y)
    if resultant >= 1.0:
        spread = 0.0
    else:
        spread = math.sqrt(-2.0 * math.log(max(resultant, 1e-12))) / (2 * math.pi) * 24.0

    return mean_hour, spread


def observation_span_days(rows: Sequence[Dict]) -> float:
    times = [r["acq_datetime"] for r in rows if r.get("acq_datetime") is not None]
    if len(times) < 2:
        return 0.0
    return (max(times) - min(times)).total_seconds() / 86400.0


def build_slots(rows: Sequence[Dict], span_days: Optional[float] = None) -> SlotSummary:
    """Group a cluster's detections into overpass slots and rate them.

    Each row needs `acq_datetime`, `source`, `daynight` and `lon`.
    """
    dated = [r for r in rows if r.get("acq_datetime") is not None]
    if not dated:
        return SlotSummary()

    span = observation_span_days(dated) if span_days is None else span_days
    effective_span = max(span, MIN_SPAN_DAYS_FOR_RATE)

    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in dated:
        platform = normalise_platform(row.get("source"))
        daynight = row.get("daynight") or "U"
        grouped[(platform, daynight)].append(row)

    slots: List[OverpassSlot] = []
    for (platform, daynight), members in sorted(grouped.items()):
        expected = EXPECTED_PASSES_PER_DAY_PER_SLOT * effective_span

        hours = [
            local_solar_hour(m["acq_datetime"], m["lon"])
            for m in members
            if m.get("lon") is not None
        ]
        mean_hour, spread = _circular_mean_hours(hours) if hours else (None, None)

        slots.append(OverpassSlot(
            platform=platform,
            daynight=daynight,
            n_detections=len(members),
            expected_passes=expected,
            # A source can be detected more than once per pass (adjacent
            # pixels of one cluster), so the ratio can exceed 1 and is
            # clamped: "seen on every pass" is the ceiling of the claim.
            detection_rate=min(len(members) / expected, 1.0) if expected > 0 else 0.0,
            mean_local_solar_hour=mean_hour,
            local_solar_hour_spread=spread,
        ))

    return SlotSummary(slots=slots, observation_span_days=span, n_detections=len(dated))


def slot_consistency(summary: SlotSummary, tolerance_hours: float = 2.0) -> bool:
    """Do the slots actually sit at consistent local solar times?

    A sanity check on the premise of this whole module. If a slot's
    detections are scattered across the day rather than clustered near
    one local hour, the (platform, daynight) grouping is not capturing a
    real overpass and the rates built on it mean less.
    """
    for slot in summary.slots:
        if slot.n_detections < 3 or slot.local_solar_hour_spread is None:
            continue
        if slot.local_solar_hour_spread > tolerance_hours:
            return False
    return True
