from datetime import datetime, timezone

from app.ingestion.normalize import (
    firms_row_to_canonical,
    generic_pixel_to_canonical,
    normalize_modis_confidence,
    normalize_viirs_confidence,
)


def test_normalize_viirs_confidence_categorical():
    assert normalize_viirs_confidence("h") == 0.95
    assert normalize_viirs_confidence("nominal") == 0.7
    assert normalize_viirs_confidence("l") == 0.3
    assert normalize_viirs_confidence("garbage") is None
    assert normalize_viirs_confidence(None) is None


def test_normalize_modis_confidence_numeric_scaled_to_unit_interval():
    assert normalize_modis_confidence(100) == 1.0
    assert normalize_modis_confidence(0) == 0.0
    assert normalize_modis_confidence(50) == 0.5
    assert normalize_modis_confidence(150) == 1.0  # clamped
    assert normalize_modis_confidence(None) is None


def test_firms_row_to_canonical_viirs():
    row = {
        "latitude": "28.7041",
        "longitude": "77.1025",
        "acq_date": "2026-06-01",
        "acq_time": "530",  # note: not zero-padded, as FIRMS sometimes emits
        "confidence": "h",
        "frp": "12.4",
        "daynight": "D",
        "scan": "0.4",
        "track": "0.4",
        "bright_ti4": "330.2",
    }
    result = firms_row_to_canonical(row, "VIIRS_SNPP_NRT")

    assert result["source"] == "VIIRS_SNPP_NRT"
    assert result["lon"] == 77.1025
    assert result["lat"] == 28.7041
    assert result["acq_datetime"] == datetime(2026, 6, 1, 5, 30, tzinfo=timezone.utc)
    assert result["frp"] == 12.4
    assert result["brightness"] == 330.2
    assert result["confidence"] == 0.95
    assert result["daynight"] == "D"


def test_firms_row_to_canonical_modis_uses_numeric_confidence_and_brightness_field():
    row = {
        "latitude": "21.0",
        "longitude": "82.0",
        "acq_date": "2026-06-01",
        "acq_time": "0000",
        "confidence": "80",
        "frp": "5.0",
        "daynight": "N",
        "scan": "1.0",
        "track": "1.0",
        "brightness": "305.0",
    }
    result = firms_row_to_canonical(row, "MODIS_NRT")

    assert result["brightness"] == 305.0
    assert result["confidence"] == 0.8


def test_generic_pixel_to_canonical_adds_utc_tz_when_missing():
    naive_dt = datetime(2026, 1, 1, 0, 0, 0)
    result = generic_pixel_to_canonical(
        source="INSAT3DS", lon=80.0, lat=20.0, acq_datetime=naive_dt, frp=3.2,
    )
    assert result["acq_datetime"].tzinfo is not None
    assert result["source"] == "INSAT3DS"
