import numpy as np

from app.enrichment.landcover import classify_pixel, summarize_landcover_window


def test_classify_pixel_forest_codes():
    for code in (1, 2, 3, 4, 5, 8, 9):
        assert classify_pixel(code) == "forest"


def test_classify_pixel_cropland_codes():
    assert classify_pixel(12) == "cropland"
    assert classify_pixel(14) == "cropland"


def test_classify_pixel_urban_code():
    assert classify_pixel(13) == "urban"


def test_classify_pixel_other_for_unmapped_code():
    assert classify_pixel(0) == "other"
    assert classify_pixel(17) == "other"  # water body


def test_summarize_landcover_window_percentages_sum_correctly():
    # 3x3 window: 3 forest, 3 cropland, 2 urban, 1 other
    window = np.array([
        [1, 1, 1],
        [12, 12, 12],
        [13, 13, 0],
    ])
    result = summarize_landcover_window(window)
    assert abs(result["pct_forest"] - 3 / 9) < 1e-9
    assert abs(result["pct_cropland"] - 3 / 9) < 1e-9
    assert abs(result["pct_urban"] - 2 / 9) < 1e-9


def test_summarize_landcover_window_handles_empty_array():
    result = summarize_landcover_window(np.array([]))
    assert result == {"pct_forest": 0.0, "pct_cropland": 0.0, "pct_urban": 0.0}
