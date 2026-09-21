"""Background temperature estimation for the Dozier retrieval.

This module exists because of a genuine data limitation, and it is worth
being explicit about it rather than burying an assumption in a constant.

The textbook Dozier method takes the background radiance from
neighbouring non-fire pixels in the same satellite granule. NASA FIRMS
does not distribute granules — it distributes *only the pixels that
triggered the fire detection*. There are no neighbouring non-fire pixels
in the data we have.

So the background is estimated, in descending order of preference:

  1. **Coolest detections in the same cluster.** A cluster usually
     contains marginal edge pixels whose thermal-band temperature is
     close to ambient. Taking a low percentile of the I5/T31 values is
     the closest thing to a local measurement available.
  2. **Day/night climatological fallback.** Crude, but bounded and
     honest, and adequate because the retrieval is far more sensitive to
     the mid-infrared signal than to the background.

Whichever is used, the value and its provenance are returned together
and stored with the result, so a temperature can always be traced back
to the background it assumed.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Typical Indian land-surface skin temperatures. Deliberately conservative:
# over-estimating the background slightly under-estimates the fire
# fraction, which is the safer direction for a headline "how hot" number.
DAYTIME_BACKGROUND_K = 305.0
NIGHTTIME_BACKGROUND_K = 290.0

# Percentile of the thermal band taken as ambient.
BACKGROUND_PERCENTILE = 10.0

# Need a few detections before a percentile means anything.
MIN_SAMPLES_FOR_PERCENTILE = 4

# A background hotter than this is itself a fire signal, not ambient.
MAX_PLAUSIBLE_BACKGROUND_K = 330.0
MIN_PLAUSIBLE_BACKGROUND_K = 260.0


@dataclass
class BackgroundEstimate:
    temperature_k: float
    source: str          # "cluster_percentile" | "climatology"
    n_samples: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "background_temperature_k": round(self.temperature_k, 2),
            "background_source": self.source,
            "background_n_samples": self.n_samples,
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile.

    Implemented here rather than pulled from numpy so this module stays
    dependency-free and the behaviour at tiny sample sizes is explicit.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def climatological_background(is_daytime: bool) -> BackgroundEstimate:
    return BackgroundEstimate(
        temperature_k=DAYTIME_BACKGROUND_K if is_daytime else NIGHTTIME_BACKGROUND_K,
        source="climatology",
        n_samples=0,
    )


def estimate_background(
    thermal_band_temperatures: Sequence[float], is_daytime: bool = True
) -> BackgroundEstimate:
    """Ambient temperature under a cluster's detections.

    `thermal_band_temperatures` are the long-wave (I5 / band-31) values
    for the cluster — the mid-infrared band must NOT be used here, since
    that is the band the fire dominates.
    """
    usable = [
        float(t) for t in thermal_band_temperatures
        if t is not None and MIN_PLAUSIBLE_BACKGROUND_K <= float(t) <= MAX_PLAUSIBLE_BACKGROUND_K
    ]

    if len(usable) < MIN_SAMPLES_FOR_PERCENTILE:
        return climatological_background(is_daytime)

    value = _percentile(usable, BACKGROUND_PERCENTILE)

    # A percentile can still land somewhere implausible if the whole
    # cluster is unusually hot; fall back rather than feed the solver a
    # background that is itself burning.
    if not MIN_PLAUSIBLE_BACKGROUND_K <= value <= MAX_PLAUSIBLE_BACKGROUND_K:
        return climatological_background(is_daytime)

    return BackgroundEstimate(
        temperature_k=value, source="cluster_percentile", n_samples=len(usable)
    )


def is_daytime_from_rows(rows: Sequence[Dict]) -> bool:
    """Majority day/night flag across a cluster's detections."""
    flags = [r.get("daynight") for r in rows if r.get("daynight") in ("D", "N")]
    if not flags:
        return True
    return flags.count("D") >= flags.count("N")
