"""Persistent Thermal Source Index.

Phase-1 deliverable #2 asks for a "Persistent vs. Transient Anomaly
Segregator": something that separates planned, recurring operational heat
(refinery stacks, smelters, flares) from sudden accidental events (plant
explosions, runaway fires). This is it.

The index combines three things a persistent source has and a transient
one does not:

  * **Longevity** — it has been around a while. A flare burns for months;
    an accidental fire is over in hours.
  * **Reliability** — it shows up on most overpasses. This uses the
    bias-corrected detection rate from `rhythm/overpass.py`, not a raw
    detection count, because a raw count mostly measures how often the
    satellite looked.
  * **Stability** — its FRP does not swing wildly. A furnace is steady;
    a wildfire is not.

Deliberately *not* in the index: land cover, facility proximity, or
anything else about *what* the source is. Those are M1/M2/M3's job, and
mixing them in would make the index a second, worse classifier instead of
what it should be — a clean statement about temporal behaviour that the
classifier can then use as evidence.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from temporal.ptsi.baseline import Baseline

PERSISTENT = "persistent"
INTERMITTENT = "intermittent"
TRANSIENT = "transient"

# Weights sum to 1.0 — asserted below.
LONGEVITY_WEIGHT = 0.40
RELIABILITY_WEIGHT = 0.35
STABILITY_WEIGHT = 0.25

assert abs(LONGEVITY_WEIGHT + RELIABILITY_WEIGHT + STABILITY_WEIGHT - 1.0) < 1e-9

# Days of activity at which longevity saturates. A month of continuous
# presence is as persistent as the classification needs; beyond that the
# distinction stops being useful.
LONGEVITY_SATURATION_DAYS = 30.0

PERSISTENT_THRESHOLD = 0.60
INTERMITTENT_THRESHOLD = 0.30


@dataclass
class PTSIResult:
    score: float
    source_class: str
    longevity: float
    reliability: float
    stability: float
    observation_span_days: float
    detection_rate: Optional[float]
    deviation_sigma: Optional[float] = None
    deviation_multiple: Optional[float] = None
    behaving_normally: Optional[bool] = None
    summary: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "ptsi_score": round(self.score, 4),
            "source_class": self.source_class,
            "components": {
                "longevity": round(self.longevity, 4),
                "reliability": round(self.reliability, 4),
                "stability": round(self.stability, 4),
            },
            "observation_span_days": round(self.observation_span_days, 2),
            "detection_rate": (
                round(self.detection_rate, 4) if self.detection_rate is not None else None
            ),
            "deviation_sigma": (
                round(self.deviation_sigma, 2) if self.deviation_sigma is not None else None
            ),
            "deviation_multiple": (
                round(self.deviation_multiple, 2) if self.deviation_multiple is not None else None
            ),
            "behaving_normally": self.behaving_normally,
            "summary": self.summary,
        }


def longevity_score(observation_span_days: float) -> float:
    """How long the source has been present, saturating at a month."""
    if observation_span_days <= 0:
        return 0.0
    return min(observation_span_days / LONGEVITY_SATURATION_DAYS, 1.0)


def reliability_score(detection_rate: Optional[float]) -> float:
    """How consistently it appears, from the bias-corrected slot rate."""
    if detection_rate is None:
        return 0.0
    return max(0.0, min(float(detection_rate), 1.0))


def stability_score(baseline: Baseline) -> float:
    """How steady its output is: 1 for constant, falling as it varies.

    A coefficient of variation at or above 1 (standard deviation as large
    as the mean) is treated as fully unstable — that is wildfire
    behaviour, not plant behaviour.
    """
    cv = baseline.coefficient_of_variation
    if cv is None:
        return 0.0
    return max(0.0, 1.0 - min(cv, 1.0))


def classify_source(score: float) -> str:
    if score >= PERSISTENT_THRESHOLD:
        return PERSISTENT
    if score >= INTERMITTENT_THRESHOLD:
        return INTERMITTENT
    return TRANSIENT


def _summarise(
    source_class: str,
    span_days: float,
    multiple: Optional[float],
    sigma: Optional[float],
    normal: Optional[bool],
) -> str:
    """The sentence the event card shows, in the brief's own idiom."""
    if source_class == PERSISTENT:
        opening = f"Established persistent source, active across {span_days:.0f} days"
    elif source_class == INTERMITTENT:
        opening = f"Intermittent source, seen on and off across {span_days:.0f} days"
    else:
        opening = f"Transient event, only {span_days:.1f} days of history"

    if normal is None:
        return f"{opening}. No baseline yet, so normal behaviour is unknown."
    if normal:
        return f"{opening}. Currently within its normal range."

    parts = []
    if multiple is not None and multiple > 0:
        parts.append(f"{multiple:.1f}x its normal output")
    if sigma is not None:
        parts.append(f"{sigma:+.1f} sigma from its baseline")
    detail = ", ".join(parts) if parts else "outside its normal range"
    return f"{opening}. Currently {detail} — potentially abnormal."


def compute_ptsi(
    baseline: Baseline,
    detection_rate: Optional[float],
    observation_span_days: float,
    current_frp: Optional[float] = None,
) -> PTSIResult:
    """Score a source's persistence and judge whether it is behaving normally."""
    from temporal.ptsi.baseline import (
        deviation_multiple,
        deviation_sigma,
        is_behaving_normally,
    )

    longevity = longevity_score(observation_span_days)
    reliability = reliability_score(detection_rate)
    stability = stability_score(baseline)

    score = (
        LONGEVITY_WEIGHT * longevity
        + RELIABILITY_WEIGHT * reliability
        + STABILITY_WEIGHT * stability
    )
    source_class = classify_source(score)

    sigma = deviation_sigma(current_frp, baseline)
    multiple = deviation_multiple(current_frp, baseline)
    normal = is_behaving_normally(current_frp, baseline)

    return PTSIResult(
        score=score,
        source_class=source_class,
        longevity=longevity,
        reliability=reliability,
        stability=stability,
        observation_span_days=observation_span_days,
        detection_rate=detection_rate,
        deviation_sigma=sigma,
        deviation_multiple=multiple,
        behaving_normally=normal,
        summary=_summarise(source_class, observation_span_days, multiple, sigma, normal),
    )
