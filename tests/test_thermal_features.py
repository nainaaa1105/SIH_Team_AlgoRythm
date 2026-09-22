from datetime import datetime, timedelta, timezone

import pytest

from classifier.features.thermal import (
    compute_thermal_features,
    frp_zscore,
    haversine_km,
    night_fraction,
    persistence_days,
    source_diversity,
    spatial_extent_km,
    spatial_growth_rate,
)

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _row(lon=77.1, lat=28.7, hours=0, frp=10.0, brightness=320.0,
         confidence=0.7, daynight="D", source="VIIRS_SNPP_NRT"):
    return {
        "lon": lon, "lat": lat,
        "acq_datetime": BASE + timedelta(hours=hours),
        "frp": frp, "brightness": brightness, "confidence": confidence,
        "daynight": daynight, "source": source,
    }


def test_haversine_km_matches_known_distance():
    # Delhi to Mumbai is ~1150 km
    assert 1100 < haversine_km(77.1025, 28.7041, 72.8777, 19.0760) < 1200


def test_spatial_extent_is_zero_for_a_single_point():
    assert spatial_extent_km([_row()]) == 0.0


def test_spatial_extent_measures_the_widest_separation():
    rows = [_row(lon=77.10), _row(lon=77.11), _row(lon=77.20)]
    extent = spatial_extent_km(rows)
    # 0.10 degrees of longitude at 28.7N is roughly 9.8 km
    assert 9.0 < extent < 11.0


def test_spatial_growth_rate_positive_when_footprint_expands():
    """A spreading wildfire should score above a stationary flare —
    this is the spatial-behaviour discriminator from the research notes."""
    spreading = [
        _row(lon=77.10, hours=0), _row(lon=77.101, hours=1),
        _row(lon=77.15, hours=48), _row(lon=77.25, hours=72),
    ]
    assert spatial_growth_rate(spreading) > 0


def test_spatial_growth_rate_zero_for_a_stationary_source():
    stationary = [_row(lon=77.10, hours=h) for h in (0, 24, 48, 72)]
    assert spatial_growth_rate(stationary) == pytest.approx(0.0, abs=1e-9)


def test_spatial_growth_rate_handles_too_few_points():
    assert spatial_growth_rate([_row(), _row(hours=1)]) == 0.0


def test_persistence_days_spans_first_to_last_detection():
    rows = [_row(hours=0), _row(hours=48)]
    assert persistence_days(rows) == pytest.approx(2.0)


def test_night_fraction_counts_only_valid_flags():
    rows = [_row(daynight="N"), _row(daynight="D"), _row(daynight=None)]
    assert night_fraction(rows) == pytest.approx(0.5)


def test_night_fraction_none_when_no_flags():
    assert night_fraction([_row(daynight=None)]) is None


def test_source_diversity_counts_distinct_platforms_not_feed_variants():
    """VIIRS_SNPP and VIIRS_SNPP_NRT are the same platform seen twice,
    not two independent corroborating looks."""
    rows = [_row(source="VIIRS_SNPP_NRT"), _row(source="VIIRS_SNPP")]
    assert source_diversity(rows) == 1

    rows = [_row(source="VIIRS_SNPP_NRT"), _row(source="MODIS_NRT"), _row(source="INSAT3DS")]
    assert source_diversity(rows) == 3


def test_frp_zscore_measures_departure_from_baseline():
    rows = [_row(frp=50.0)]
    assert frp_zscore(rows, baseline_mean=10.0, baseline_std=5.0) == pytest.approx(8.0)


def test_frp_zscore_zero_variance_baseline_does_not_divide_by_zero():
    """A perfectly steady flare has no variance; a departure from it must
    not produce inf, which XGBoost cannot split on sensibly."""
    steady = frp_zscore([_row(frp=10.0)], baseline_mean=10.0, baseline_std=0.0)
    assert steady == 0.0

    spike = frp_zscore([_row(frp=80.0)], baseline_mean=10.0, baseline_std=0.0)
    assert spike == 3.0

    drop = frp_zscore([_row(frp=1.0)], baseline_mean=10.0, baseline_std=0.0)
    assert drop == -3.0


def test_frp_zscore_none_without_a_baseline():
    assert frp_zscore([_row(frp=10.0)], baseline_mean=None, baseline_std=None) is None


def test_compute_thermal_features_returns_every_m2_column():
    rows = [_row(hours=0, frp=5.0), _row(hours=24, frp=15.0)]
    features = compute_thermal_features(rows, baseline_mean=5.0, baseline_std=2.0)

    for column in (
        "frp_mean", "frp_max", "frp_std", "frp_zscore", "brightness_mean",
        "brightness_max", "confidence_mean", "persistence_days", "n_detections",
        "detections_per_day", "night_fraction", "source_diversity",
        "spatial_extent_km", "spatial_growth_rate",
    ):
        assert column in features

    assert features["frp_mean"] == pytest.approx(10.0)
    assert features["frp_max"] == 15.0
    assert features["n_detections"] == 2.0


def test_detections_per_day_does_not_explode_for_sub_day_bursts():
    """Ten detections within an hour must not divide by a near-zero span."""
    rows = [_row(hours=h / 60) for h in range(10)]
    features = compute_thermal_features(rows)
    assert features["detections_per_day"] == 10.0


def test_missing_frp_values_are_skipped_not_treated_as_zero():
    rows = [_row(frp=None), _row(frp=20.0)]
    features = compute_thermal_features(rows)
    assert features["frp_mean"] == 20.0


def test_all_none_frp_yields_none_not_a_crash():
    features = compute_thermal_features([_row(frp=None)])
    assert features["frp_mean"] is None
    assert features["frp_max"] is None


def test_std_of_single_observation_is_zero_not_none():
    features = compute_thermal_features([_row(frp=10.0)])
    assert features["frp_std"] == 0.0
