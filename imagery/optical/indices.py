"""Sentinel-2 spectral indices and smoke features.

Band mapping used throughout (Sentinel-2 L2A surface reflectance):

    B2  490 nm   blue
    B3  560 nm   green
    B4  665 nm   red
    B8  842 nm   NIR
    B11 1610 nm  SWIR-1
    B12 2190 nm  SWIR-2

Three of these feed M2's feature matrix directly (`ndvi`, `ndbi`,
`smoke_red_blue_ratio`); the rest support the smoke mask and the CNN.

On the smoke indices specifically: these are documented *heuristics*,
not validated retrievals. Smoke aerosol scatters strongly at short
wavelengths and is largely transparent in the SWIR, so a smoke-covered
pixel looks bright and flat in the visible, shows suppressed NDVI
(vegetation is obscured), and stays relatively dark in SWIR — which is
also what separates smoke from cloud, since water/ice cloud is bright in
SWIR too. That physical reasoning is sound; the specific thresholds are
tuned guesses until validated against the case studies.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Reflectance below this is sensor noise or shadow; excluded from ratios
# so a near-zero denominator cannot manufacture a huge ratio.
MIN_VALID_REFLECTANCE = 1e-4

SENTINEL2_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")


@dataclass
class SmokeMaskResult:
    mask: np.ndarray          # bool array, True where smoke is suspected
    coverage: float           # fraction of valid pixels flagged
    mean_red_blue_ratio: float


def _safe_normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), with 0 where the sum is ~0 rather than NaN/inf."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = a + b
    result = np.zeros_like(denominator, dtype=float)
    valid = np.abs(denominator) > MIN_VALID_REFLECTANCE
    np.divide(a - b, denominator, out=result, where=valid)
    return result


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Vegetation index. High over healthy vegetation, near zero over
    built-up/barren, negative over water."""
    return _safe_normalised_difference(nir, red)


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Built-up index. Positive over urban/industrial surfaces."""
    return _safe_normalised_difference(swir1, nir)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalised burn ratio. Drops sharply over freshly burnt ground,
    which distinguishes a fire that has already swept through from one
    that is only smoking."""
    return _safe_normalised_difference(nir, swir2)


def red_blue_ratio(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Red over blue reflectance.

    Smoke scatters short wavelengths preferentially, lifting blue
    relative to red, so smoke-covered pixels sit *below* 1 while bare
    soil and most surfaces sit above it.
    """
    red = np.asarray(red, dtype=float)
    blue = np.asarray(blue, dtype=float)
    result = np.zeros_like(blue, dtype=float)
    valid = blue > MIN_VALID_REFLECTANCE
    np.divide(red, blue, out=result, where=valid)
    return result


def visible_brightness(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (np.asarray(blue, float) + np.asarray(green, float) + np.asarray(red, float)) / 3.0


def smoke_mask(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    brightness_threshold: float = 0.12,
    ratio_threshold: float = 1.0,
    swir_threshold: float = 0.25,
    ndvi_threshold: float = 0.35,
) -> SmokeMaskResult:
    """Flag pixels that look like smoke rather than cloud or clear ground.

    Four conditions together, because no single one is specific:
      * bright in the visible          (smoke is not dark)
      * red/blue ratio below 1         (blue lifted by scattering)
      * SWIR relatively dark           (this is what excludes cloud)
      * NDVI suppressed                (surface vegetation is obscured)
    """
    brightness = visible_brightness(blue, green, red)
    ratio = red_blue_ratio(red, blue)
    vegetation = ndvi(nir, red)
    swir = np.asarray(swir1, dtype=float)

    mask = (
        (brightness > brightness_threshold)
        & (ratio < ratio_threshold)
        & (swir < swir_threshold)
        & (vegetation < ndvi_threshold)
    )

    total = mask.size
    coverage = float(mask.sum()) / total if total else 0.0
    flagged_ratio = float(ratio[mask].mean()) if mask.any() else float("nan")

    return SmokeMaskResult(mask=mask, coverage=coverage, mean_red_blue_ratio=flagged_ratio)


def summarise_patch(bands: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    """Scene-level features for one Sentinel-2 patch.

    Returns the three columns M2 reserved for M4 (`ndvi`, `ndbi`,
    `smoke_red_blue_ratio`) plus supporting values for the event card.
    Missing bands yield None rather than raising, so a partial fetch
    still contributes what it can and M2's evidence weighting handles the
    rest.
    """
    def _band(name: str) -> Optional[np.ndarray]:
        value = bands.get(name)
        return None if value is None else np.asarray(value, dtype=float)

    blue, green, red = _band("B2"), _band("B3"), _band("B4")
    nir, swir1, swir2 = _band("B8"), _band("B11"), _band("B12")

    features: Dict[str, Optional[float]] = {
        "ndvi": None, "ndbi": None, "nbr": None,
        "smoke_red_blue_ratio": None, "visible_brightness": None,
        "smoke_coverage": None,
    }

    if nir is not None and red is not None:
        features["ndvi"] = float(np.nanmean(ndvi(nir, red)))
    if swir1 is not None and nir is not None:
        features["ndbi"] = float(np.nanmean(ndbi(swir1, nir)))
    if nir is not None and swir2 is not None:
        features["nbr"] = float(np.nanmean(nbr(nir, swir2)))
    if red is not None and blue is not None:
        features["smoke_red_blue_ratio"] = float(np.nanmean(red_blue_ratio(red, blue)))
    # `None not in (array, ...)` would compare element-wise and raise
    # "truth value of an array is ambiguous" — use explicit identity tests.
    have_visible = all(b is not None for b in (blue, green, red))
    if have_visible:
        features["visible_brightness"] = float(np.nanmean(visible_brightness(blue, green, red)))
    if have_visible and nir is not None and swir1 is not None:
        features["smoke_coverage"] = smoke_mask(blue, green, red, nir, swir1).coverage

    return features
