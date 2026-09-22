import math

import numpy as np
import pytest

from imagery.optical.indices import (
    SENTINEL2_BANDS,
    ndbi,
    ndvi,
    nbr,
    red_blue_ratio,
    smoke_mask,
    summarise_patch,
    visible_brightness,
)
from imagery.optical.smoke import (
    array_offset_to_bearing,
    bearing_difference,
    cross_validate_with_model,
    smoke_bearing,
)


def _uniform(value, shape=(8, 8)):
    return np.full(shape, value, dtype=float)


# --- spectral indices ---------------------------------------------------

def test_ndvi_is_high_over_vegetation_and_low_over_bare_ground():
    vegetated = ndvi(_uniform(0.40), _uniform(0.05))   # NIR high, red low
    bare = ndvi(_uniform(0.20), _uniform(0.18))
    assert vegetated.mean() > 0.7
    assert bare.mean() < 0.2


def test_ndvi_is_negative_over_water():
    """Water absorbs NIR strongly, so NDVI goes negative — a useful
    discriminator when a 'hotspot' sits on a river or reservoir."""
    assert ndvi(_uniform(0.02), _uniform(0.06)).mean() < 0


def test_ndbi_is_positive_over_built_up_surfaces():
    assert ndbi(_uniform(0.30), _uniform(0.20)).mean() > 0     # SWIR > NIR
    assert ndbi(_uniform(0.15), _uniform(0.40)).mean() < 0     # vegetation


def test_nbr_drops_over_burnt_ground():
    """Fresh burn scars are dark in NIR and bright in SWIR-2."""
    unburnt = nbr(_uniform(0.40), _uniform(0.10))
    burnt = nbr(_uniform(0.10), _uniform(0.35))
    assert unburnt.mean() > burnt.mean()


def test_indices_do_not_divide_by_zero():
    """A zero-reflectance pixel (shadow, nodata) must give 0, not NaN."""
    for result in (
        ndvi(_uniform(0.0), _uniform(0.0)),
        ndbi(_uniform(0.0), _uniform(0.0)),
        nbr(_uniform(0.0), _uniform(0.0)),
        red_blue_ratio(_uniform(0.2), _uniform(0.0)),
    ):
        assert np.all(np.isfinite(result))


def test_red_blue_ratio_drops_below_one_under_smoke():
    """Smoke scatters short wavelengths, lifting blue relative to red."""
    smoky = red_blue_ratio(_uniform(0.18), _uniform(0.25))
    clear_soil = red_blue_ratio(_uniform(0.28), _uniform(0.12))
    assert smoky.mean() < 1.0 < clear_soil.mean()


def test_visible_brightness_averages_the_three_visible_bands():
    assert visible_brightness(_uniform(0.1), _uniform(0.2), _uniform(0.3)).mean() == pytest.approx(0.2)


# --- smoke mask ---------------------------------------------------------

def test_smoke_mask_flags_a_bright_blue_low_swir_scene():
    result = smoke_mask(
        blue=_uniform(0.30), green=_uniform(0.28), red=_uniform(0.26),
        nir=_uniform(0.22), swir1=_uniform(0.08),
    )
    assert result.coverage == pytest.approx(1.0)


def test_smoke_mask_rejects_cloud():
    """Cloud is bright in the visible AND bright in SWIR; the SWIR test is
    precisely what separates the two."""
    result = smoke_mask(
        blue=_uniform(0.60), green=_uniform(0.60), red=_uniform(0.58),
        nir=_uniform(0.55), swir1=_uniform(0.50),   # bright SWIR => cloud
    )
    assert result.coverage == 0.0


def test_smoke_mask_rejects_healthy_vegetation():
    result = smoke_mask(
        blue=_uniform(0.03), green=_uniform(0.06), red=_uniform(0.04),
        nir=_uniform(0.45), swir1=_uniform(0.20),
    )
    assert result.coverage == 0.0


def test_smoke_mask_rejects_dark_surfaces():
    result = smoke_mask(
        blue=_uniform(0.02), green=_uniform(0.02), red=_uniform(0.01),
        nir=_uniform(0.03), swir1=_uniform(0.02),
    )
    assert result.coverage == 0.0


# --- patch summary ------------------------------------------------------

def test_summarise_patch_returns_the_columns_m2_reserved():
    bands = {b: _uniform(0.2) for b in SENTINEL2_BANDS}
    features = summarise_patch(bands)
    for column in ("ndvi", "ndbi", "smoke_red_blue_ratio"):
        assert column in features and features[column] is not None


def test_summarise_patch_degrades_with_missing_bands():
    """A partial fetch should still contribute what it can; M2's evidence
    weighting handles the gaps."""
    features = summarise_patch({"B8": _uniform(0.4), "B4": _uniform(0.1)})
    assert features["ndvi"] is not None
    assert features["ndbi"] is None
    assert features["smoke_red_blue_ratio"] is None


def test_summarise_patch_on_no_bands_returns_all_none():
    assert all(v is None for v in summarise_patch({}).values())


# --- smoke bearing ------------------------------------------------------

def test_array_offset_to_bearing_cardinal_directions():
    """Raster convention: row increases southward, column eastward."""
    assert array_offset_to_bearing(d_col=0, d_row=-1) == pytest.approx(0.0)     # north
    assert array_offset_to_bearing(d_col=1, d_row=0) == pytest.approx(90.0)     # east
    assert array_offset_to_bearing(d_col=0, d_row=1) == pytest.approx(180.0)    # south
    assert array_offset_to_bearing(d_col=-1, d_row=0) == pytest.approx(270.0)   # west


def test_smoke_drifting_north_is_reported_as_north():
    mask = np.zeros((21, 21), dtype=bool)
    mask[2:8, 8:13] = True      # smoke above the centre => northward
    observation = smoke_bearing(mask)
    assert observation.detected
    assert observation.bearing_deg == pytest.approx(0.0, abs=15.0)


def test_smoke_drifting_east_is_reported_as_east():
    mask = np.zeros((21, 21), dtype=bool)
    mask[8:13, 14:20] = True
    assert smoke_bearing(mask).bearing_deg == pytest.approx(90.0, abs=15.0)


def test_too_few_smoke_pixels_is_not_a_detection():
    mask = np.zeros((21, 21), dtype=bool)
    mask[0, 0] = True
    observation = smoke_bearing(mask)
    assert not observation.detected
    assert observation.bearing_deg is None


def test_symmetric_smoke_gives_no_direction():
    """Smoke evenly surrounding the source carries no directional
    information; a bearing here would be noise."""
    mask = np.zeros((21, 21), dtype=bool)
    mask[6:15, 6:15] = True
    observation = smoke_bearing(mask)
    assert observation.detected
    assert observation.bearing_deg is None
    assert "symmetric" in observation.reason


def test_empty_mask_is_handled():
    observation = smoke_bearing(np.zeros((10, 10), dtype=bool))
    assert not observation.detected
    assert observation.coverage == 0.0


# --- cross-validation against M3 ---------------------------------------

def test_bearing_difference_wraps_correctly():
    assert bearing_difference(350.0, 10.0) == pytest.approx(20.0)
    assert bearing_difference(10.0, 350.0) == pytest.approx(20.0)
    assert bearing_difference(0.0, 180.0) == pytest.approx(180.0)


def test_agreement_when_observed_matches_the_model():
    verdict = cross_validate_with_model(observed_bearing_deg=45.0, modelled_downwind_bearing_deg=50.0)
    assert verdict["comparable"] and verdict["agrees"]
    assert verdict["difference_deg"] == pytest.approx(5.0)


def test_disagreement_when_the_plume_points_elsewhere():
    verdict = cross_validate_with_model(45.0, 200.0)
    assert verdict["comparable"] and not verdict["agrees"]
    assert "inconsistent" in verdict["note"]


def test_missing_data_is_not_reported_as_disagreement():
    """'We could not compare' must not read as 'the model is wrong'."""
    for observed, modelled in ((None, 90.0), (90.0, None), (None, None)):
        verdict = cross_validate_with_model(observed, modelled)
        assert verdict["comparable"] is False
        assert verdict["agrees"] is None


def test_agreement_respects_the_wraparound():
    verdict = cross_validate_with_model(355.0, 5.0)
    assert verdict["agrees"] is True
    assert verdict["difference_deg"] == pytest.approx(10.0)
