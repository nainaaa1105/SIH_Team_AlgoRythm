"""Bug report: unticking every Sensor Data date checkbox still left fires
on the map.

Root cause: sensorDateOptions() only ever offers checkboxes for the 3
most-recent distinct acqDates present in the loaded data. Before the
90-day backfill, every fire's acqDate fell inside that 3-day window, so
this never showed up. Once older (backfilled) dates existed too,
passesSensorDateFilter's check --

    offeredDates.has(f.acqDate) && !filt.sensorDates.has(...)

-- was false for any fire outside the offered 3 dates (since
offeredDates.has() is false for them), so the whole bracketed condition
was false and the fire passed unconditionally, regardless of any
checkbox's state. showOlderData gives that older data its own real,
explicit on/off control instead of an implicit always-on.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_older_data_checkbox_exists_in_the_sensor_panel():
    src = html()
    sensor_panel = src.split('id="ac-sensor"')[1].split("</div>\n        </div>")[0]
    assert 'id="cb-older"' in sensor_panel
    assert 'id="cb-older" checked' in sensor_panel  # visible and on by default


def test_dates_outside_the_offered_three_are_gated_by_show_older_data():
    src = html()
    fn = src.split("function passesSensorDateFilter(f) {")[1].split("\n    }")[0]
    assert "offeredDates.has(f.acqDate)" in fn
    assert "return filt.showOlderData;" in fn


def test_show_older_data_defaults_to_true():
    """Off by default would silently hide the just-loaded 90-day
    backfill the moment anyone opened the dashboard -- must default on,
    same as every other class/date checkbox."""
    src = html()
    filt_block = src.split("let filt = {")[1].split("};")[0]
    assert "showOlderData: true" in filt_block


def test_older_data_checkbox_is_wired_to_the_filter():
    src = html()
    fn = src.split("function initSensorCBs() {")[1].split("\n    }")[0]
    assert "cb-older" in fn
    assert "filt.showOlderData = olderCb.checked" in fn
    assert "applyFilt()" in fn.split("cb-older")[1]


def test_reset_filters_restores_show_older_data_to_true_and_checks_the_box():
    src = html()
    reset_fn = src.split("btn-reset-f').addEventListener('click', () => {")[1].split("\n      });")[0]
    assert "showOlderData: true" in reset_fn
    assert "cb-older" in reset_fn
