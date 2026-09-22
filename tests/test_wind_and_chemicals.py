from datetime import datetime, timezone

import pytest

from datetime import timedelta

from geospatial.plume.chemicals import chemical_profile, known_facility_types
from geospatial.plume.wind import choose_endpoint, fallback_wind, parse_open_meteo

AT = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _payload(**overrides):
    payload = {
        "hourly": {
            "time": ["2026-06-01T11:00", "2026-06-01T12:00", "2026-06-01T13:00"],
            "wind_speed_10m": [3.6, 18.0, 7.2],       # km/h
            "wind_direction_10m": [10.0, 270.0, 100.0],
            "cloud_cover": [10, 50, 90],
            "is_day": [1, 1, 1],
        }
    }
    payload["hourly"].update(overrides)
    return payload


def test_wind_speed_is_converted_from_kmh_to_ms():
    """Open-Meteo reports km/h by default; the dispersion model is all
    in m/s. Getting this wrong scales the whole plume by 3.6."""
    wind = parse_open_meteo(_payload(), AT)
    assert wind.speed_ms == pytest.approx(5.0)   # 18 km/h


def test_the_hour_nearest_the_detection_is_selected():
    wind = parse_open_meteo(_payload(), AT)
    assert wind.direction_deg == 270.0
    assert wind.observed_at == "2026-06-01T12:00"


def test_an_off_hour_detection_snaps_to_the_closest_sample():
    wind = parse_open_meteo(_payload(), datetime(2026, 6, 1, 12, 50, tzinfo=timezone.utc))
    assert wind.observed_at == "2026-06-01T13:00"


def test_timezone_aware_target_against_naive_samples_does_not_raise():
    """Open-Meteo returns naive strings; M1's acq_datetime is tz-aware."""
    wind = parse_open_meteo(_payload(), AT)
    assert wind.source == "open-meteo"


def test_cloud_cover_is_converted_to_a_fraction():
    wind = parse_open_meteo(_payload(), AT)
    assert wind.cloud_cover_fraction == pytest.approx(0.5)


def test_legacy_field_names_are_accepted():
    payload = {
        "hourly": {
            "time": ["2026-06-01T12:00"],
            "windspeed_10m": [18.0],
            "winddirection_10m": [180.0],
            "cloudcover": [20],
        }
    }
    wind = parse_open_meteo(payload, AT)
    assert wind.speed_ms == pytest.approx(5.0)
    assert wind.direction_deg == 180.0


def test_daytime_is_inferred_from_the_hour_when_is_day_is_absent():
    payload = _payload()
    payload["hourly"].pop("is_day")
    assert parse_open_meteo(payload, AT).is_daytime is True

    payload["hourly"]["time"] = ["2026-06-01T23:00"]
    payload["hourly"]["wind_speed_10m"] = [18.0]
    payload["hourly"]["wind_direction_10m"] = [270.0]
    payload["hourly"]["cloud_cover"] = [50]
    night = parse_open_meteo(payload, datetime(2026, 6, 1, 23, 0, tzinfo=timezone.utc))
    assert night.is_daytime is False


def test_a_response_without_wind_fields_is_rejected():
    with pytest.raises(ValueError, match="missing hourly wind"):
        parse_open_meteo({"hourly": {"time": ["2026-06-01T12:00"]}}, AT)


def test_fallback_wind_is_labelled_as_assumed():
    """A plume from assumed wind is a weaker claim; the flag is what lets
    the UI say so instead of presenting it as observed."""
    wind = fallback_wind(AT)
    assert wind.source == "fallback"
    assert wind.is_fallback is True
    assert wind.speed_ms > 0


def test_real_observation_is_not_flagged_as_fallback():
    assert parse_open_meteo(_payload(), AT).is_fallback is False


# --- endpoint selection -------------------------------------------------

FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def test_recent_detections_use_the_forecast_endpoint():
    recent = datetime.now(timezone.utc) - timedelta(hours=2)
    assert choose_endpoint(recent, FORECAST, ARCHIVE, cutoff_days=5) == FORECAST


def test_historical_detections_use_the_archive_endpoint():
    """Without this the 90-day replay silently falls back to assumed wind
    for every cluster, producing meteorologically meaningless plumes that
    still look like they worked."""
    old = datetime.now(timezone.utc) - timedelta(days=60)
    assert choose_endpoint(old, FORECAST, ARCHIVE, cutoff_days=5) == ARCHIVE


def test_the_cutoff_is_respected_on_both_sides():
    just_inside = datetime.now(timezone.utc) - timedelta(days=4)
    just_outside = datetime.now(timezone.utc) - timedelta(days=6)
    assert choose_endpoint(just_inside, FORECAST, ARCHIVE, 5) == FORECAST
    assert choose_endpoint(just_outside, FORECAST, ARCHIVE, 5) == ARCHIVE


# --- chemical profiles --------------------------------------------------

def test_refinery_profile_flags_hydrogen_sulphide():
    profile = chemical_profile("refinery")
    assert "H2S" in profile["primary"]
    assert profile["hazard"] == "high"


def test_facility_type_takes_precedence_over_the_predicted_class():
    """A located industrial footprint says more about the inventory than
    the broad source class does."""
    profile = chemical_profile("refinery", "wildfire")
    assert "H2S" in profile["primary"]
    assert profile["basis"] == "facility_type=refinery"


def test_class_profile_used_when_there_is_no_facility():
    profile = chemical_profile(None, "wildfire")
    assert "PM2.5" in profile["primary"]
    assert profile["basis"] == "predicted_class=wildfire"


def test_industrial_class_without_a_facility_still_gets_an_industrial_profile():
    profile = chemical_profile(None, "industrial_fire")
    assert profile["basis"] == "predicted_class=industrial_fire"
    assert profile["hazard"] != "unknown"


def test_unknown_everything_falls_back_to_generic_combustion():
    profile = chemical_profile(None, None)
    assert profile["hazard"] == "unknown"
    assert profile["primary"]


def test_facility_type_lookup_is_case_insensitive():
    assert chemical_profile("REFINERY")["hazard"] == "high"


def test_every_known_facility_type_has_a_complete_profile():
    for facility_type in known_facility_types():
        profile = chemical_profile(facility_type)
        assert profile["primary"] and profile["hazard"] and profile["note"]


def test_profiles_cover_the_types_m1_actually_writes():
    """M1's OSM/GEM loaders emit these facility_type values; a missing
    one silently degrades to the generic profile."""
    for facility_type in ("refinery", "industrial", "power", "oil_well", "mine", "flare"):
        assert chemical_profile(facility_type)["basis"] == f"facility_type={facility_type}"
