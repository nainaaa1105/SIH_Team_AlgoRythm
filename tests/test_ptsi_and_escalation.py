"""PTSI, escalation forecasting and freshness."""
from datetime import datetime, timedelta, timezone

import pytest

from temporal.forecast.escalation import (
    critical_threshold,
    forecast,
    predict_frp_at,
)
from temporal.forecast.freshness import FRESH, MODERATE, STALE, assess, median_cadence_hours
from temporal.forecast.kalman import initial_state, run_filter
from temporal.ptsi.baseline import (
    Baseline,
    build_baseline,
    deviation_multiple,
    deviation_sigma,
    is_behaving_normally,
)
from temporal.ptsi.index import (
    INTERMITTENT,
    PERSISTENT,
    TRANSIENT,
    classify_source,
    compute_ptsi,
    longevity_score,
    reliability_score,
    stability_score,
)

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _row(days, frp):
    return {"acq_datetime": BASE + timedelta(days=days), "frp": frp}


# --- baseline -----------------------------------------------------------

def test_baseline_needs_enough_observations():
    result = build_baseline([_row(d, 10.0) for d in range(3)])
    assert not result.ok
    assert "prior observations" in result.reason


def test_baseline_summarises_a_steady_source():
    baseline = build_baseline([_row(d, 10.0) for d in range(20)])
    assert baseline.ok
    assert baseline.mean_frp == pytest.approx(10.0)
    assert baseline.coefficient_of_variation == pytest.approx(0.05, abs=0.05)


def test_baseline_excludes_the_event_being_judged():
    """The rule that matters: a spike folded into its own baseline
    inflates the thing it is compared against and partly hides itself."""
    history = [_row(d, 10.0) for d in range(20)]
    spike = _row(20, 500.0)

    contaminated = build_baseline(history + [spike])
    correct = build_baseline(history + [spike], before=spike["acq_datetime"])

    assert correct.mean_frp == pytest.approx(10.0)
    assert contaminated.mean_frp > 20.0        # the spike dragged it up
    assert deviation_sigma(500.0, correct) > deviation_sigma(500.0, contaminated)


def test_baseline_respects_the_lookback_window():
    old = [_row(-200 + d, 100.0) for d in range(10)]     # far outside 90 days
    recent = [_row(d, 10.0) for d in range(10)]
    baseline = build_baseline(old + recent, before=BASE + timedelta(days=10), window_days=90)
    assert baseline.mean_frp == pytest.approx(10.0)


def test_baseline_std_has_a_floor():
    """A perfectly constant source would otherwise make every deviation
    infinite."""
    baseline = build_baseline([_row(d, 10.0) for d in range(20)])
    assert baseline.std_frp >= 0.5


# --- deviation ----------------------------------------------------------

def test_deviation_multiple_is_the_briefs_four_point_two_x():
    baseline = build_baseline([_row(d, 10.0) for d in range(20)])
    assert deviation_multiple(42.0, baseline) == pytest.approx(4.2, abs=0.01)


def test_deviation_sigma_accounts_for_how_variable_the_source_is():
    """4.2x is alarming for a steady flare and unremarkable for a wildfire
    that swings between 5 and 40 MW. Only the sigma captures that."""
    steady = build_baseline([_row(d, 10.0) for d in range(20)])
    erratic = build_baseline([_row(d, 10.0 + (d % 7) * 6.0) for d in range(20)])

    assert deviation_sigma(42.0, steady) > deviation_sigma(42.0, erratic)


def test_behaving_normally_is_none_without_a_baseline():
    """'We don't know what normal is' must not look like 'this is
    abnormal' on an operations dashboard."""
    assert is_behaving_normally(100.0, Baseline()) is None


def test_behaving_normally_flags_a_real_excursion():
    baseline = build_baseline([_row(d, 10.0) for d in range(20)])
    assert is_behaving_normally(11.0, baseline) is True
    assert is_behaving_normally(500.0, baseline) is False


# --- PTSI components ----------------------------------------------------

def test_longevity_saturates_at_a_month():
    assert longevity_score(0) == 0.0
    assert longevity_score(15) == pytest.approx(0.5)
    assert longevity_score(30) == 1.0
    assert longevity_score(365) == 1.0


def test_reliability_uses_the_bias_corrected_rate():
    assert reliability_score(1.0) == 1.0
    assert reliability_score(0.5) == 0.5
    assert reliability_score(None) == 0.0
    assert reliability_score(5.0) == 1.0       # clamped


def test_stability_is_high_for_a_steady_source():
    steady = build_baseline([_row(d, 10.0) for d in range(20)])
    erratic = build_baseline([_row(d, 1.0 if d % 2 else 60.0) for d in range(20)])
    assert stability_score(steady) > stability_score(erratic)


def test_classification_thresholds():
    assert classify_source(0.9) == PERSISTENT
    assert classify_source(0.45) == INTERMITTENT
    assert classify_source(0.1) == TRANSIENT


# --- PTSI end to end ----------------------------------------------------

def test_a_flare_scores_as_persistent():
    """Long-lived, seen on nearly every pass, steady output."""
    baseline = build_baseline([_row(d, 12.0) for d in range(60)])
    result = compute_ptsi(baseline, detection_rate=0.95, observation_span_days=60.0, current_frp=12.5)

    assert result.source_class == PERSISTENT
    assert result.behaving_normally
    assert "persistent" in result.summary.lower()


def test_an_accidental_fire_scores_as_transient():
    """Short-lived, erratic, seen a couple of times."""
    baseline = build_baseline([_row(d * 0.1, 5.0 + d * 40) for d in range(6)])
    result = compute_ptsi(baseline, detection_rate=0.1, observation_span_days=0.6, current_frp=250.0)
    assert result.source_class == TRANSIENT


def test_an_abnormal_flare_is_flagged_with_the_briefs_phrasing():
    baseline = build_baseline([_row(d, 10.0) for d in range(60)])
    result = compute_ptsi(baseline, detection_rate=0.9, observation_span_days=60.0, current_frp=42.0)

    assert result.source_class == PERSISTENT
    assert result.behaving_normally is False
    assert result.deviation_multiple == pytest.approx(4.2, abs=0.05)
    assert "4.2x its normal" in result.summary


def test_summary_says_so_when_there_is_no_baseline():
    result = compute_ptsi(Baseline(), detection_rate=None, observation_span_days=0.2, current_frp=5.0)
    assert result.behaving_normally is None
    assert "unknown" in result.summary.lower()


# --- critical threshold -------------------------------------------------

def test_threshold_is_relative_to_the_sources_own_baseline():
    """A steel furnace at 80 MW is normal; 20 MW at a site that never
    exceeds 5 MW is not."""
    quiet = critical_threshold(baseline_mean=2.0, baseline_std=0.5)
    busy = critical_threshold(baseline_mean=80.0, baseline_std=10.0)
    assert busy > quiet


def test_threshold_has_an_absolute_floor():
    """A near-zero baseline must not produce a near-zero threshold that
    alarms on the first real detection."""
    assert critical_threshold(baseline_mean=0.1, baseline_std=0.01) >= 10.0


def test_threshold_is_none_without_a_baseline():
    assert critical_threshold(None, None) is None


# --- escalation forecast ------------------------------------------------

def _rising_state(rate=5.0, frp=50.0):
    state = initial_state(frp)
    state.rate = rate
    state.covariance = [[25.0, 0.0], [0.0, 0.25]]
    return state


def test_a_rising_source_reaches_critical():
    result = forecast(_rising_state(rate=5.0, frp=50.0), threshold_frp=100.0)
    assert result.escalating
    assert result.time_to_critical_hours == pytest.approx(10.0, rel=0.4)
    assert result.probability_reaches_critical > 0.9


def test_a_falling_source_does_not_escalate():
    """Regression: with a widened covariance even a clearly decaying
    source has a thin positive tail, and any non-zero count used to set
    escalating=True. A dying fire was flagged as escalating on 0.8% of
    samples — false alerts on exactly the events an operator wants
    filtered out."""
    result = forecast(_rising_state(rate=-3.0, frp=50.0), threshold_frp=100.0)
    assert not result.escalating
    assert result.time_to_critical_hours is None
    assert result.probability_reaches_critical < 0.1


def test_a_marginal_probability_is_reported_but_not_called_an_escalation():
    """The probability is never hidden — only the boolean is gated."""
    from temporal.forecast.escalation import MIN_ESCALATION_PROBABILITY

    state = _rising_state(rate=-2.0, frp=50.0)
    result = forecast(state, threshold_frp=100.0)
    assert not result.escalating
    assert result.probability_reaches_critical < MIN_ESCALATION_PROBABILITY
    assert result.probability_reaches_critical >= 0.0


def test_already_critical_is_reported_separately():
    result = forecast(_rising_state(frp=150.0), threshold_frp=100.0)
    assert result.already_critical
    assert result.time_to_critical_hours == 0.0


def test_confidence_intervals_bracket_the_estimate():
    result = forecast(_rising_state(rate=5.0, frp=50.0), threshold_frp=100.0)
    assert result.p90_low <= result.p50_low <= result.time_to_critical_hours
    assert result.time_to_critical_hours <= result.p50_high <= result.p90_high


def test_more_uncertain_state_gives_wider_intervals():
    confident = _rising_state()
    confident.covariance = [[1.0, 0.0], [0.0, 0.01]]
    uncertain = _rising_state()
    uncertain.covariance = [[100.0, 0.0], [0.0, 4.0]]

    tight = forecast(confident, 100.0)
    loose = forecast(uncertain, 100.0)
    assert (loose.p90_high - loose.p90_low) > (tight.p90_high - tight.p90_low)


def test_an_uncertain_rate_straddling_zero_lowers_the_probability():
    """The case a Taylor-expansion CI gets badly wrong: 'it might not be
    growing at all' means 'it might never get there', and only the
    sampling approach can express that."""
    state = _rising_state(rate=0.5)
    state.covariance = [[25.0, 0.0], [0.0, 4.0]]      # rate std 2.0 >> rate
    result = forecast(state, 100.0)
    assert 0.0 < result.probability_reaches_critical < 0.8


def test_forecast_is_deterministic_for_a_fixed_seed():
    """Demos must be reproducible."""
    first = forecast(_rising_state(), 100.0, seed=7)
    second = forecast(_rising_state(), 100.0, seed=7)
    assert first.time_to_critical_hours == second.time_to_critical_hours


def test_no_threshold_means_no_forecast():
    result = forecast(_rising_state(), threshold_frp=None)
    assert not result.escalating
    assert "no baseline" in result.reason


def test_horizon_limits_the_projection():
    """A very slow rise should not report an ETA weeks out."""
    result = forecast(_rising_state(rate=0.01, frp=50.0), 100.0, max_horizon_hours=48.0)
    assert result.probability_reaches_critical < 0.5


def test_predict_frp_at_widens_with_horizon():
    state = run_filter([(i * 6.0, 50.0 + i) for i in range(10)])
    _, near_std = predict_frp_at(state, 6.0)
    _, far_std = predict_frp_at(state, 72.0)
    assert far_std > near_std


# --- freshness ----------------------------------------------------------

def test_cadence_is_the_median_gap():
    times = [BASE + timedelta(hours=12 * i) for i in range(10)]
    assert median_cadence_hours(times) == pytest.approx(12.0)


def test_cadence_resists_one_long_outage():
    """A single multi-day cloud gap would drag a mean far enough to make
    a stale source look freshly observed."""
    times = [BASE + timedelta(hours=12 * i) for i in range(10)]
    times.append(times[-1] + timedelta(days=10))
    assert median_cadence_hours(times) == pytest.approx(12.0, abs=1.0)


def test_cadence_needs_a_few_points():
    assert median_cadence_hours([BASE]) is None


def test_recent_observation_is_fresh():
    times = [BASE + timedelta(hours=12 * i) for i in range(10)]
    result = assess(times[-1], times, now=times[-1] + timedelta(hours=4))
    assert result.label == FRESH
    assert result.is_usable


def test_overdue_observation_is_moderate_then_stale():
    times = [BASE + timedelta(hours=12 * i) for i in range(10)]
    moderate = assess(times[-1], times, now=times[-1] + timedelta(hours=24))
    stale = assess(times[-1], times, now=times[-1] + timedelta(hours=50))
    assert moderate.label == MODERATE
    assert stale.label == STALE
    assert not stale.is_usable


def test_freshness_is_relative_to_each_sources_own_cadence():
    """12 hours of silence is stale for a source seen every pass and
    unremarkable for one seen twice a week."""
    chatty = [BASE + timedelta(hours=6 * i) for i in range(10)]
    sparse = [BASE + timedelta(days=3 * i) for i in range(10)]

    now = BASE + timedelta(days=40)
    chatty_result = assess(chatty[-1], chatty, now=chatty[-1] + timedelta(hours=20))
    sparse_result = assess(sparse[-1], sparse, now=sparse[-1] + timedelta(hours=20))

    assert chatty_result.label == STALE
    assert sparse_result.label == FRESH


def test_absolute_limit_applies_however_sparse_the_source():
    sparse = [BASE + timedelta(days=10 * i) for i in range(5)]
    result = assess(sparse[-1], sparse, now=sparse[-1] + timedelta(days=5))
    assert result.label == STALE


def test_no_observation_is_stale():
    assert assess(None).label == STALE


def test_naive_timestamp_does_not_raise():
    """M1 stores tz-aware UTC, but a naive datetime reaching here would
    otherwise raise on subtraction."""
    naive = datetime(2026, 6, 1, 12, 0)
    result = assess(naive, now=datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc))
    assert result.age_hours == pytest.approx(2.0)
