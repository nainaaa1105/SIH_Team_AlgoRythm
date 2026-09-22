"""Observed smoke plume direction, and cross-validation against M3.

This is the piece that closes a loop the project's research notes call
out explicitly: wind is *supporting evidence*, not a classifier. If the
imagery shows smoke drifting north-east and M3's dispersion model — built
from Open-Meteo wind — also points north-east, the two independent
sources agree and the event is corroborated. If they disagree sharply,
something is wrong: stale wind data, a misattributed detection, or a
plume that is not from this source at all.

M3 stores its modelled travel direction on `plumes.downwind_bearing_deg`
(bearing the plume travels TOWARD, already converted out of the
meteorological from-convention). This module produces the same quantity
from pixels, so the two are directly comparable.

Array-to-bearing convention, stated once because it is the easy thing to
get backwards: in a north-up raster, row index increases *southward*
while column index increases *eastward*.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Below this, the flagged pixels are noise rather than a plume.
MIN_SMOKE_PIXELS = 12
MIN_DISPLACEMENT_PX = 1.5

# Agreement threshold between observed and modelled bearing.
BEARING_AGREEMENT_DEG = 45.0


@dataclass
class SmokeObservation:
    detected: bool
    bearing_deg: Optional[float]
    coverage: float
    n_pixels: int
    displacement_px: float
    reason: str = ""


def array_offset_to_bearing(d_col: float, d_row: float) -> float:
    """Convert a pixel offset into a compass bearing.

    d_col is eastward; d_row is southward (raster convention), so the
    northward component is its negation. Bearing is degrees clockwise
    from north, matching M1/M3's convention throughout the project.
    """
    east = d_col
    north = -d_row
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def smoke_bearing(
    mask: np.ndarray, source_row: Optional[float] = None, source_col: Optional[float] = None
) -> SmokeObservation:
    """Direction from the fire to the centroid of the smoke-flagged pixels.

    The source defaults to the patch centre, which is where the fetcher
    centres the patch on the cluster centroid.
    """
    mask = np.asarray(mask, dtype=bool)
    n_pixels = int(mask.sum())
    coverage = float(n_pixels) / mask.size if mask.size else 0.0

    if source_row is None:
        source_row = (mask.shape[0] - 1) / 2.0
    if source_col is None:
        source_col = (mask.shape[1] - 1) / 2.0

    if n_pixels < MIN_SMOKE_PIXELS:
        return SmokeObservation(
            detected=False, bearing_deg=None, coverage=coverage, n_pixels=n_pixels,
            displacement_px=0.0,
            reason=f"only {n_pixels} smoke pixels, below the {MIN_SMOKE_PIXELS}-pixel floor",
        )

    rows, cols = np.nonzero(mask)
    d_row = float(rows.mean()) - float(source_row)
    d_col = float(cols.mean()) - float(source_col)
    displacement = math.hypot(d_row, d_col)

    if displacement < MIN_DISPLACEMENT_PX:
        # Smoke symmetric about the source: either calm air or the plume
        # is not resolved. A bearing here would be meaningless noise.
        return SmokeObservation(
            detected=True, bearing_deg=None, coverage=coverage, n_pixels=n_pixels,
            displacement_px=displacement,
            reason="smoke is symmetric about the source; no usable direction",
        )

    return SmokeObservation(
        detected=True,
        bearing_deg=array_offset_to_bearing(d_col, d_row),
        coverage=coverage,
        n_pixels=n_pixels,
        displacement_px=displacement,
        reason="ok",
    )


def bearing_difference(a: float, b: float) -> float:
    """Smallest absolute angle between two bearings, in [0, 180]."""
    diff = abs((a % 360.0) - (b % 360.0)) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def cross_validate_with_model(
    observed_bearing_deg: Optional[float],
    modelled_downwind_bearing_deg: Optional[float],
    tolerance_deg: float = BEARING_AGREEMENT_DEG,
) -> Dict[str, object]:
    """Compare the observed plume direction with M3's modelled one.

    Returns a verdict rather than a bare boolean so the event card can
    explain *why* there is (or isn't) corroboration — an unavailable
    comparison must not read as a disagreement.
    """
    if observed_bearing_deg is None or modelled_downwind_bearing_deg is None:
        return {
            "comparable": False,
            "agrees": None,
            "difference_deg": None,
            "note": (
                "No comparison possible: "
                + ("no smoke direction observed in the imagery"
                   if observed_bearing_deg is None
                   else "no modelled plume direction available for this cluster")
            ),
        }

    difference = bearing_difference(observed_bearing_deg, modelled_downwind_bearing_deg)
    agrees = difference <= tolerance_deg

    return {
        "comparable": True,
        "agrees": agrees,
        "difference_deg": round(difference, 1),
        "note": (
            f"Observed smoke drifts {observed_bearing_deg:.0f}deg; wind model predicts "
            f"{modelled_downwind_bearing_deg:.0f}deg ({difference:.0f}deg apart) — "
            + ("consistent, the plume corroborates the wind-driven model."
               if agrees else
               "inconsistent; check wind currency, attribution, or whether the "
               "smoke belongs to a different source.")
        ),
    }
