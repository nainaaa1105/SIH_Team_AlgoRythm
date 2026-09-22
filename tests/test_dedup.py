from datetime import datetime, timezone

from app.ingestion.dedup import (
    _exact_key,
    dedup_against_existing,
    dedup_exact,
    dedup_near,
    exact_key_for_db_row,
    full_dedup,
)


def _record(source="VIIRS_SNPP_NRT", lon=77.1, lat=28.7, minute=0, **overrides):
    base = {
        "source": source,
        "lon": lon,
        "lat": lat,
        "acq_datetime": datetime(2026, 6, 1, 5, minute, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_dedup_exact_removes_byte_identical_records():
    r1 = _record()
    r2 = _record()  # identical
    r3 = _record(lon=77.2)  # different location

    result = dedup_exact([r1, r2, r3])
    assert len(result) == 2


def test_dedup_near_collapses_float_jitter_within_same_minute():
    r1 = _record(lon=77.10001, lat=28.70001)
    r2 = _record(lon=77.10002, lat=28.70002)  # jitter within ~11m tolerance, same minute
    result = dedup_near([r1, r2])
    assert len(result) == 1


def test_dedup_near_keeps_records_in_different_minutes():
    r1 = _record(minute=0)
    r2 = _record(minute=1)
    result = dedup_near([r1, r2])
    assert len(result) == 2


def test_dedup_against_existing_filters_previously_ingested_rows():
    r1 = _record()
    r2 = _record(lon=77.3)
    existing_keys = {_exact_key(r1)}
    result = dedup_against_existing([r1, r2], existing_keys)
    assert result == [r2]


def test_full_dedup_pipeline_combines_all_passes():
    r1 = _record()
    r2 = _record()  # exact dup
    r3 = _record(lon=77.10001)  # near dup of r1 within tolerance
    r4 = _record(lon=79.0)  # genuinely distinct

    result = full_dedup([r1, r2, r3, r4])
    assert len(result) == 2
    lons = sorted(r["lon"] for r in result)
    assert lons == [77.1, 79.0]


def test_exact_key_for_db_row_matches_shape_of_exact_key_from_canonical_record():
    """Regression test: pipeline.py builds existing_keys from DB rows via
    exact_key_for_db_row and compares them against _exact_key(record) for
    freshly-fetched records — the two must produce identical tuples for
    the same (source, lon, lat, acq_datetime), or dedup_against_existing
    silently never matches anything.
    """
    record = _record(lon=77.123456789, lat=28.987654321)
    from_record = _exact_key(record)
    from_db_row = exact_key_for_db_row(record["source"], record["lon"], record["lat"], record["acq_datetime"])
    assert from_record == from_db_row


def test_cross_sensor_records_are_not_collapsed_by_dedup():
    """Dedup must preserve provenance across sensors — the same physical
    point seen by two satellites is NOT a duplicate at this stage,
    clustering handles cross-sensor merging instead.
    """
    r1 = _record(source="VIIRS_SNPP_NRT")
    r2 = _record(source="SENTINEL3_FRP")
    result = full_dedup([r1, r2])
    assert len(result) == 2
