"""The "Active Fires" row added to the Fire Classification panel.

Same no-browser hook-assertion pattern as test_wui.py: checks the control
exists in the same style as the WUI Threat / Crown Fire rows next to it,
and that checking it actually narrows what's drawn (an isolate filter on
the server-computed fire.status, not just an overlay toggle like
cb-show-wui/cb-show-crown).
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_active_fires_row_exists_in_the_classification_panel():
    src = html()
    ac_types = src.split('id="ac-types"')[1].split("</div>\n        </div>")[0]
    assert 'id="cb-active-only"' in ac_types
    assert 'class="ft-lbl">Active Fires</span>' in ac_types
    assert 'class="ft-cnt" id="cnt-active">0</span>' in ac_types


def test_active_only_is_a_real_narrowing_filter_not_just_an_overlay_toggle():
    """Unlike cb-show-wui/cb-show-crown (which never hide a fire's own
    dot), checking Active Fires must actually remove non-active fires
    from visFires, since the request was "show only the fires which are
    active"."""
    src = html()
    apply_fn = src.split("function applyFilt() {")[1].split("\n    }")[0]
    assert "filt.activeOnly && f.status !== 'Active'" in apply_fn


def test_active_only_checkbox_is_wired_to_the_filter():
    src = html()
    fn = src.split("function initTypeCBs() {")[1].split("\n    }")[0]
    assert "cb-active-only" in fn
    assert "filt.activeOnly = activeOnlyCb.checked" in fn
    assert "applyFilt()" in fn.split("cb-active-only")[1].split("\n")[1]


def test_active_only_defaults_to_off():
    """Same default as wuiOnly/crownOnly/rdiOnly -- an isolate switch that
    starts off so it doesn't silently hide fires nobody asked to hide."""
    src = html()
    filt_block = src.split("let filt = {")[1].split("};")[0]
    assert "activeOnly: false" in filt_block


def test_active_fires_count_reflects_the_currently_visible_fires():
    """Same pattern as cnt-wui/cnt-crown in updateWuiPanel() -- counts
    off `scoped` (visFires), so the number matches the map, not the
    unfiltered dataset."""
    src = html()
    fn = src.split("function updateWuiPanel() {")[1].split("\n    }")[0]
    assert "cnt-active" in fn
    assert "scoped.filter(f => f.status === 'Active').length" in fn


def test_reset_filters_clears_active_only():
    src = html()
    reset_fn = src.split("btn-reset-f').addEventListener('click', () => {")[1].split("\n      });")[0]
    assert "activeOnly: false" in reset_fn
    assert "cb-active-only" in reset_fn
