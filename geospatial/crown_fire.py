"""Crown-fire FRP threshold detection.

The physical distinction this answers: a *surface* fire burns fuel on
the ground and can usually be held by ground crews. A *crown* fire has
climbed into the canopy — it moves faster than a person can run, throws
embers well ahead of the front, and generates its own local weather.
Fire Radiative Power (FRP), which every hotspot detection already
carries in megawatts, is a real proxy for that transition: a crown fire
radiates far more energy per unit time than a surface fire in the same
fuel type, and a sudden *increase* in FRP between consecutive
overpasses of the same cluster is itself diagnostic of a fire actively
transitioning into the canopy, independent of the absolute value.

Two independent triggers, matching how `wui_analysis.priority_for` is
structured for the same reason — an absolute-level signal and a
rate-of-change signal can each be decisive on their own:

  * `frp_max >= CROWN_FIRE_FRP_THRESHOLD_MW` — the fire is currently
    radiating at crown-fire intensity, however it got there;
  * `frp_acceleration_mw_per_hour(...) >= CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR`
    — FRP is climbing fast enough, right now, that a currently-moderate
    fire is in the middle of that transition.

Both numbers come from the cluster's own real detection history
(`hotspots.frp`, `hotspots.acq_datetime`) — nothing here is estimated or
assumed. A cluster with only one detection has no rate to compute and is
judged on the threshold alone, which is the conservative direction to be
wrong in (a lone very-hot pixel still trips the absolute threshold).

Applies to every classified cluster, not gated by class the way WUI is:
an FRP spike is diagnostic of fire behaviour regardless of what started
it, and an industrial fire radiating at crown-fire intensity is exactly
as urgent to a commander deciding where to send air resources as a
wildfire would be.
"""
import logging
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

# Cited in the brief as the "critical structural-threat threshold" — a
# fire radiating at or above this intensity is behaving like a crown
# fire regardless of how it started.
CROWN_FIRE_FRP_THRESHOLD_MW = 100.0

# "FRP acceleration rate ... >= 40 MW/hr" — the second, independent
# trigger: a fire caught in the middle of climbing into the canopy,
# before its absolute FRP has necessarily crossed the threshold above.
CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR = 40.0


def frp_acceleration_mw_per_hour(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """The steepest FRP increase between any two time-ordered detections
    in this cluster's real history, in MW per hour.

    Returns None (not zero — "unknown", not "not accelerating") when
    there are fewer than two dated, FRP-bearing detections to compare.
    Only positive-going legs count: a fire that flared and then cooled
    is not "decelerating" in any sense that matters here, so the
    steepest *rise* is what is reported, and a monotonically cooling
    series correctly reports None (found rows, but no rising leg).
    """
    dated = sorted(
        (r for r in rows if r.get("acq_datetime") is not None and r.get("frp") is not None),
        key=lambda r: r["acq_datetime"],
    )
    if len(dated) < 2:
        return None

    steepest = None
    for earlier, later in zip(dated, dated[1:]):
        hours = (later["acq_datetime"] - earlier["acq_datetime"]).total_seconds() / 3600.0
        if hours <= 0:
            continue
        rate = (later["frp"] - earlier["frp"]) / hours
        if rate > 0 and (steepest is None or rate > steepest):
            steepest = rate

    return steepest


def evaluate_crown_fire(
    frp_max: Optional[float], hotspot_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pure evaluation — no DB, no network — matching the shape
    `wui_analysis._compute_threat` uses for the same reason: testable
    without a session, and safe to call from anywhere that already has
    the two inputs in hand.
    """
    rate = frp_acceleration_mw_per_hour(hotspot_rows)

    over_threshold = frp_max is not None and frp_max >= CROWN_FIRE_FRP_THRESHOLD_MW
    accelerating = rate is not None and rate >= CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR
    is_crown = over_threshold or accelerating

    if over_threshold and accelerating:
        reason = f"FRP {frp_max:.1f} MW at {rate:.1f} MW/h"
    elif over_threshold:
        reason = f"FRP {frp_max:.1f} MW >= {CROWN_FIRE_FRP_THRESHOLD_MW:.0f} MW threshold"
    elif accelerating:
        reason = f"FRP rising at {rate:.1f} MW/h >= {CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR:.0f} MW/h"
    else:
        reason = None

    return {
        "is_crown_fire": is_crown,
        "frp_max_mw": frp_max,
        "frp_acceleration_mw_per_hour": round(rate, 2) if rate is not None else None,
        "reason": reason,
    }
