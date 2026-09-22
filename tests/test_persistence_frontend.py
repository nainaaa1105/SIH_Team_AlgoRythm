"""Persistent-vs-transient thermal source indicator in the detail panel.

temporal.ptsi (see temporal/ptsi/index.py) already classifies every
cluster with enough history as persistent/intermittent/transient — real
data from longevity, overpass reliability and FRP stability, not a
guess. This wires that existing classification into the dashboard's
per-fire detail panel, between DETECTED and SENSOR as requested.

Same "no browser, just assert the hooks a live page needs" pattern as
test_wui.py / test_suppression_frontend.py.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_to_fire_carries_the_source_class_through():
    src = html()
    fn = src.split("function toFire(d) {")[1].split("\n    function ")[0]
    assert "sourceClass: d.source_class" in fn


def test_persistent_row_sits_between_detected_and_sensor():
    src = html()
    open_dp = src.split("function openDP(fire)")[1].split("async function loadEventDetail")[0]
    detected_idx = open_dp.index('<div class="dp-k">DETECTED</div>')
    persistent_idx = open_dp.index('<div class="dp-k">PERSISTENT</div>')
    sensor_idx = open_dp.index('<div class="dp-k">SENSOR</div>')
    assert detected_idx < persistent_idx < sensor_idx


def test_persistent_row_reads_not_available_without_a_baseline_not_a_guess():
    """A null source_class means M5 hasn't built enough history to judge
    persistence yet — must read as unknown, never default to a class."""
    src = html()
    open_dp = src.split("function openDP(fire)")[1].split("async function loadEventDetail")[0]
    assert "fire.sourceClass\n" in open_dp or "fire.sourceClass\n        ?" in open_dp
    assert "? '<span style=" in open_dp
    assert ": NA;" in open_dp


def test_persistent_row_labels_all_three_real_source_classes():
    src = html()
    open_dp = src.split("function openDP(fire)")[1].split("async function loadEventDetail")[0]
    assert "persistent: 'Yes — Persistent'" in open_dp
    assert "intermittent: 'Intermittent'" in open_dp
    assert "transient: 'No — Transient'" in open_dp


def test_persistence_never_fabricated_client_side():
    assert "Math.random" not in html()
