"""Backfilled/older detections must never reach the frontend at all.

By request: no user-facing toggle for this -- older data stays in the
database (still useful there for training/analysis), but the frontend
itself must never hold it in memory, so it can never show as a spot on
the map and can never leak through any overlay (WUI/crown/RDI) that
reads allFires directly, bypassing the ordinary date checkboxes.

Earlier attempt used a checkbox (filt.showOlderData) gating
passesSensorDateFilter -- correct in spirit but not what was asked for.
The real fix filters at load time in loadFires(), before anything is
ever assigned into allFires, so there's no separate flag or control to
keep in sync and no bypass path left to find.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_no_older_data_toggle_exists_anywhere():
    src = html()
    assert "cb-older" not in src
    assert "showOlderData" not in src


def test_load_fires_drops_dates_outside_the_three_most_recent():
    src = html()
    fn = src.split("async function loadFires() {")[1].split("\n    }")[0]
    assert "recentDates" in fn
    assert ".slice(0, 3)" in fn
    assert "allFires = fires.filter(f => !f.acqDate || recentDates.has(f.acqDate));" in fn


def test_pending_fires_with_no_acq_date_yet_are_not_dropped():
    """A cluster whose acqDate hasn't landed yet is not the same as an
    old one -- same fail-open reasoning the rest of the filter chain
    already uses for missing type/confidence, must not be silently
    dropped just because its date is unknown rather than old."""
    src = html()
    fn = src.split("async function loadFires() {")[1].split("\n    }")[0]
    assert "!f.acqDate ||" in fn


def test_passes_sensor_date_filter_no_longer_has_a_separate_older_branch():
    """With filtering done once at load time, every fire reaching
    applyFilt() already has a date within sensorDateOptions()'s own set
    -- a second exclusion path here would be dead code duplicating the
    same decision in two places that could drift."""
    src = html()
    fn = src.split("function passesSensorDateFilter(f) {")[1].split("\n    }")[0]
    assert "showOlderData" not in fn
    assert "offeredDates.has(f.acqDate)" in fn


def test_reset_filters_has_nothing_older_data_related_to_reset():
    src = html()
    reset_fn = src.split("btn-reset-f').addEventListener('click', () => {")[1].split("\n      });")[0]
    assert "showOlderData" not in reset_fn
    assert "cb-older" not in reset_fn
