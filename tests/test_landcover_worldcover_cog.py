"""ESA WorldCover COG path — the keyless replacement for the GEE zonal
stats call, since this project has no GCP service account.

Network reads are mocked: these tests must pass with no internet access,
same as every other test in this project. Live-fetch correctness was
checked manually against the public bucket's actual tile listing.
"""
from unittest.mock import patch

import pytest

from geospatial.attribution.landcover_worldcover_cog import (
    worldcover_cog_url,
    worldcover_tile_id,
    zonal_landcover_cog,
    landcover_with_fallback_cog,
)


def test_tile_id_matches_the_documented_example():
    # WorldCover's product user manual: "S48E036 covers 36E-39E, 48S-45S".
    assert worldcover_tile_id(lat=-46.0, lon=37.0) == "S48E036"


def test_tile_id_for_new_delhi():
    # 28.6°N, 77.2°E -> floor(28.6/3)*3=27, floor(77.2/3)*3=75
    assert worldcover_tile_id(lat=28.6, lon=77.2) == "N27E075"


def test_tile_id_handles_the_equator_and_prime_meridian():
    assert worldcover_tile_id(lat=0.5, lon=0.5) == "N00E000"
    assert worldcover_tile_id(lat=-0.5, lon=-0.5) == "S03W003"


def test_cog_url_is_well_formed():
    url = worldcover_cog_url(lat=28.6, lon=77.2)
    assert url == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_N27E075_Map.tif"
    )


def test_zonal_cog_returns_none_without_rasterio():
    with patch.dict("sys.modules", {"rasterio": None}):
        assert zonal_landcover_cog(77.2, 28.6, 1000.0) is None


def test_zonal_cog_returns_none_on_read_failure():
    """A dead tile (open ocean) or network timeout must degrade to None,
    not raise, so the caller can fall through to the next tier.

    Skipped when rasterio isn't installed — it's an optional dependency
    this whole module degrades without, same as population exposure."""
    pytest.importorskip("rasterio")
    with patch("rasterio.open", side_effect=OSError("no such tile")):
        assert zonal_landcover_cog(77.2, 28.6, 1000.0) is None


def test_fallback_prefers_cog_over_gee_and_modis():
    with patch(
        "geospatial.attribution.landcover_worldcover_cog.zonal_landcover_cog",
        return_value={"pct_forest": 0.4, "pct_cropland": 0.3, "pct_urban": 0.1,
                      "pct_water": 0.1, "pct_barren": 0.1},
    ) as cog, patch(
        "geospatial.attribution.landcover_zonal.zonal_landcover_gee"
    ) as gee:
        result = landcover_with_fallback_cog(77.2, 28.6, spatial_extent_km=2.0)

    assert result["source"] == "ESA_WorldCover_10m_COG"
    cog.assert_called_once()
    gee.assert_not_called()  # the working path must not pay for the dead one


def test_fallback_tries_gee_when_cog_tile_is_unavailable():
    with patch(
        "geospatial.attribution.landcover_worldcover_cog.zonal_landcover_cog",
        return_value=None,
    ), patch(
        "geospatial.attribution.landcover_zonal.zonal_landcover_gee",
        return_value={"pct_forest": 0.2, "pct_cropland": 0.2, "pct_urban": 0.2,
                      "pct_water": 0.2, "pct_barren": 0.2},
    ):
        result = landcover_with_fallback_cog(77.2, 28.6, spatial_extent_km=2.0)

    assert result["source"] == "ESA_WorldCover_10m_GEE"


def test_fallback_reaches_modis_when_both_network_tiers_fail():
    with patch(
        "geospatial.attribution.landcover_worldcover_cog.zonal_landcover_cog",
        return_value=None,
    ), patch(
        "geospatial.attribution.landcover_zonal.zonal_landcover_gee",
        return_value=None,
    ), patch(
        "app.enrichment.landcover.sample_landcover_at_point",
        return_value={"pct_forest": 0.5, "pct_cropland": 0.5, "pct_urban": 0.0},
    ):
        result = landcover_with_fallback_cog(77.2, 28.6, spatial_extent_km=2.0)

    assert result["source"] == "MODIS_MCD12Q1_500m"
