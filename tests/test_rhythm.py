"""Overpass slots, rhythm fingerprint and periodicity.

The tests that matter most here are the ones demonstrating *why* the
slot model exists: a naive hour-of-day histogram measures the satellite's
schedule, not the facility's.
"""
from datetime import datetime, timedelta, timezone

import pytest

from temporal.rhythm.fingerprint import (
    DAY_NAMES,
    INDIA_WEEKEND_DAYS,
    WESTERN_WEEKEND_DAYS,
    build_fingerprint,
    day_night_contrast,
    day_of_week_rates,
    detect_schedule_anomaly,
    shift_sharpness,
    weekend_suppression,
)
from temporal.rhythm.overpass import (
    build_slots,
    local_solar_hour,
    normalise_platform,
    observation_span_days,
    slot_consistency,
)
from temporal.rhythm.periodicity import (
    daily_series,
    detect_periodicity,
    sub_daily_periodicity_is_not_recoverable,
)

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)
DELHI_LON = 77.1


def _row(days=0, hour_utc=8, source="VIIRS_SNPP_NRT", daynight="D", lon=DELHI_LON, frp=10.0):
    return {
        "acq_datetime": BASE + timedelta(days=days, hours=hour_utc - 8),
        "source": source,
        "daynight": daynight,
        "lon": lon,
        "frp": frp,
    }


# --- platform normalisation --------------------------------------------

def test_nrt_suffix_does_not_create_a_second_platform():
    """VIIRS_SNPP and VIIRS_SNPP_NRT are the same satellite on the same
    orbit — counting them separately would double-count one pass."""
    assert normalise_platform("VIIRS_SNPP_NRT") == normalise_platform("VIIRS_SNPP")


def test_distinct_platforms_stay_distinct():
    assert normalise_platform("VIIRS_SNPP") != normalise_platform("VIIRS_NOAA20")
    assert normalise_platform(None) == "UNKNOWN"


# --- local solar time ---------------------------------------------------

def test_local_solar_hour_shifts_with_longitude():
    """India is ~+5.1 h of solar time from Greenwich at 77 E."""
    moment = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    assert local_solar_hour(moment, 77.1) == pytest.approx(8 + 77.1 / 15.0, abs=0.01)


def test_local_solar_hour_wraps_past_midnight():
    moment = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)
    assert 0 <= local_solar_hour(moment, 77.1) < 24


def test_greenwich_local_solar_hour_equals_utc():
    moment = datetime(2026, 6, 1, 13, 30, tzinfo=timezone.utc)
    assert local_solar_hour(moment, 0.0) == pytest.approx(13.5)


# --- slot construction --------------------------------------------------

def test_slots_are_keyed_on_platform_and_daynight():
    rows = [
        _row(days=d, source="VIIRS_SNPP_NRT", daynight="D") for d in range(5)
    ] + [
        _row(days=d, source="VIIRS_SNPP_NRT", daynight="N") for d in range(5)
    ] + [
        _row(days=d, source="MODIS_NRT", daynight="D") for d in range(5)
    ]
    summary = build_slots(rows)
    assert {s.key for s in summary.slots} == {
        "VIIRS_SNPP:D", "VIIRS_SNPP:N", "MODIS:D"
    }


def test_detection_rate_is_normalised_by_expected_passes():
    """The point of the whole module: a rate, not a raw count. A source
    seen on every one of 10 days has rate 1.0 whether the window is 10
    days or 100 detections long."""
    rows = [_row(days=d) for d in range(11)]     # 11 days, one per day
    summary = build_slots(rows)
    assert summary.slots[0].detection_rate == pytest.approx(1.0, abs=0.15)


def test_a_sporadically_detected_source_has_a_low_rate():
    rows = [_row(days=d) for d in (0, 10, 20)]   # 3 detections over 20 days
    summary = build_slots(rows)
    assert summary.slots[0].detection_rate < 0.3


def test_detection_rate_is_capped_at_one():
    """A cluster can produce several detections in a single pass; that is
    still only one look, so the rate cannot exceed 1."""
    rows = [_row(days=0, hour_utc=8) for _ in range(50)]
    summary = build_slots(rows)
    assert all(s.detection_rate <= 1.0 for s in summary.slots)


def test_slot_local_solar_hours_are_consistent_for_a_real_overpass():
    """A genuine overpass slot sits at the same local solar hour daily."""
    rows = [_row(days=d, hour_utc=8) for d in range(10)]
    summary = build_slots(rows)
    assert slot_consistency(summary)
    assert summary.slots[0].local_solar_hour_spread == pytest.approx(0.0, abs=0.1)


def test_scattered_times_fail_the_consistency_check():
    """If detections are spread across the day the (platform, daynight)
    grouping is not capturing a real overpass, and rates built on it mean
    less. The check exists to notice that."""
    rows = [_row(days=d, hour_utc=h) for d, h in enumerate([0, 4, 9, 14, 19, 22])]
    assert not slot_consistency(build_slots(rows))


def test_empty_input_gives_an_empty_summary():
    assert build_slots([]).slots == []


def test_observation_span_needs_two_points():
    assert observation_span_days([_row(days=0)]) == 0.0
    assert observation_span_days([_row(days=0), _row(days=7)]) == pytest.approx(7.0, abs=0.01)


# --- shift sharpness ----------------------------------------------------

def test_a_continuous_source_has_low_shift_sharpness():
    """A flare burning around the clock is detected in every slot."""
    rows = [
        _row(days=d, daynight=dn, hour_utc=8 if dn == "D" else 20)
        for d in range(20) for dn in ("D", "N")
    ]
    assert shift_sharpness(build_slots(rows)) < 0.15


def test_a_daytime_only_source_has_high_shift_sharpness():
    rows = [_row(days=d, daynight="D") for d in range(20)]
    rows += [_row(days=0, daynight="N", hour_utc=20)]   # one lone night hit
    assert shift_sharpness(build_slots(rows)) > 0.4


def test_shift_sharpness_needs_at_least_two_slots():
    """With one slot we cannot distinguish 'only burns by day' from 'only
    one satellite covers this place'."""
    assert shift_sharpness(build_slots([_row(days=d) for d in range(5)])) is None


def test_a_single_slot_source_is_still_a_usable_fingerprint():
    """Regression: `ok` used to be tied to shift_sharpness alone, so a
    single-platform daytime source was marked unusable even though it
    produced a perfectly good weekend_suppression — one of the two
    rhythm features M2 actually consumes."""
    rows = [
        _row(days=d, daynight="D") for d in range(60)
        if (BASE + timedelta(days=d)).weekday() != 6
    ]
    result = build_fingerprint(rows)

    assert result.shift_sharpness is None      # only one slot
    assert result.weekend_suppression is not None
    assert result.ok                            # but still usable


# --- day/night contrast -------------------------------------------------

def test_day_night_contrast_is_positive_for_a_daytime_source():
    rows = [_row(days=d, daynight="D") for d in range(20)]
    rows += [_row(days=d, daynight="N", hour_utc=20) for d in (0, 5)]
    assert day_night_contrast(build_slots(rows)) > 0


def test_day_night_contrast_near_zero_for_a_round_the_clock_source():
    rows = [
        _row(days=d, daynight=dn, hour_utc=8 if dn == "D" else 20)
        for d in range(20) for dn in ("D", "N")
    ]
    assert abs(day_night_contrast(build_slots(rows))) < 0.1


def test_day_night_contrast_is_none_without_both():
    rows = [_row(days=d, daynight="D") for d in range(10)]
    assert day_night_contrast(build_slots(rows)) is None


# --- weekend suppression ------------------------------------------------

def test_weekday_counts_are_normalised_by_calendar_occurrences():
    """A 10-day window holds two Mondays but perhaps one Sunday; raw
    counts would manufacture a weekend effect from the calendar alone."""
    rows = [_row(days=d) for d in range(28)]      # every day for 4 weeks
    rates = day_of_week_rates(rows, span_days=27.0)
    assert len(set(round(v, 2) for v in rates.values())) == 1


def test_weekend_suppression_detects_an_industrial_pattern():
    """Active every day except Sunday — the Indian weekly off."""
    rows = [
        _row(days=d) for d in range(28)
        if (BASE + timedelta(days=d)).weekday() != 6
    ]
    suppression = weekend_suppression(day_of_week_rates(rows, 27.0), INDIA_WEEKEND_DAYS)
    assert suppression == pytest.approx(1.0, abs=0.05)


def test_weekend_suppression_near_zero_for_a_continuous_source():
    rows = [_row(days=d) for d in range(28)]
    assert abs(weekend_suppression(day_of_week_rates(rows, 27.0), INDIA_WEEKEND_DAYS)) < 0.05


def test_weekend_suppression_can_be_negative():
    """Agricultural burning is genuinely more active at weekends in some
    regions; the sign carries information and must not be clipped away."""
    rows = [
        _row(days=d) for d in range(28)
        if (BASE + timedelta(days=d)).weekday() == 6
    ]
    assert weekend_suppression(day_of_week_rates(rows, 27.0), INDIA_WEEKEND_DAYS) < 0


def test_india_and_western_weekends_differ():
    """Saturday is commonly a working day in Indian industry, so counting
    it as weekend dilutes the signal with a working day."""
    rows = [
        _row(days=d) for d in range(28)
        if (BASE + timedelta(days=d)).weekday() != 6      # Sundays off only
    ]
    rates = day_of_week_rates(rows, 27.0)
    india = weekend_suppression(rates, INDIA_WEEKEND_DAYS)
    western = weekend_suppression(rates, WESTERN_WEEKEND_DAYS)
    assert india > western


def test_weekend_suppression_handles_empty_rates():
    assert weekend_suppression({}) is None


# --- schedule anomaly ---------------------------------------------------

def test_detection_in_a_normally_quiet_slot_is_flagged():
    rows = [_row(days=d, daynight="D") for d in range(30)]
    rows.append(_row(days=30, daynight="N", hour_utc=20))    # first ever night hit
    summary = build_slots(rows)
    assert detect_schedule_anomaly(summary, "VIIRS_SNPP:N")


def test_detection_in_a_normal_slot_is_not_flagged():
    rows = [_row(days=d, daynight="D") for d in range(30)]
    assert not detect_schedule_anomaly(build_slots(rows), "VIIRS_SNPP:D")


def test_unknown_slot_is_not_an_anomaly():
    assert not detect_schedule_anomaly(build_slots([_row()]), None)


# --- full fingerprint ---------------------------------------------------

def test_fingerprint_declines_on_too_few_detections():
    result = build_fingerprint([_row(days=d) for d in range(3)])
    assert not result.ok
    assert "detections" in result.reason


def test_fingerprint_of_a_scheduled_industrial_source():
    rows = [
        _row(days=d, daynight="D") for d in range(60)
        if (BASE + timedelta(days=d)).weekday() != 6
    ]
    result = build_fingerprint(rows)

    assert result.ok
    assert result.weekend_suppression > 0.5
    assert result.n_observations == len(rows)
    assert result.observation_span_days > 14


def test_weekend_suppression_withheld_for_a_short_history():
    """One Sunday against one Monday is not a weekly comparison."""
    rows = [
        _row(days=d, daynight=dn, hour_utc=8 if dn == "D" else 20)
        for d in range(10) for dn in ("D", "N")
    ]
    result = build_fingerprint(rows)
    assert result.weekend_suppression is None
    assert "too short" in result.reason
    # Two slots, so sharpness is still computable and the fingerprint is
    # still usable despite the weekly comparison being withheld.
    assert result.shift_sharpness is not None
    assert result.ok


# --- periodicity --------------------------------------------------------

def test_daily_series_fills_quiet_days_with_zero():
    """An absence is information; skipping empty days would make the
    series non-uniform again and reintroduce the sampling problem."""
    rows = [_row(days=0), _row(days=3)]
    days, values = daily_series(rows)
    assert len(days) == 4
    assert values == [1.0, 0.0, 0.0, 1.0]


def test_weekly_cycle_is_detected():
    """Active Monday to Saturday, quiet Sunday, for eight weeks."""
    rows = [
        _row(days=d) for d in range(56)
        if (BASE + timedelta(days=d)).weekday() != 6
    ]
    result = detect_periodicity(rows)
    assert result.dominant_period_days == pytest.approx(7.0, abs=0.75)
    assert result.is_weekly


def test_periodicity_declines_on_a_short_history():
    result = detect_periodicity([_row(days=d) for d in range(5)])
    assert result.dominant_period_days is None
    assert "days of history" in result.reason


def test_constant_activity_has_no_cycle():
    rows = [_row(days=d) for d in range(30)]
    result = detect_periodicity(rows)
    assert result.dominant_period_days is None


def test_sub_daily_limitation_is_documented_in_code():
    note = sub_daily_periodicity_is_not_recoverable()
    assert "Nyquist" in note and "geostationary" in note
