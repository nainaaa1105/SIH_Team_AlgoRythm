"""Two zoom fixes and a LOCATION enhancement, reported together.

1. Zoom-in past real imagery: maxZoom was 19 everywhere, but Esri
   silently upsamples/blurs past where real detail exists instead of
   erroring — a 200 OK the earlier tileerror-based clamp never caught.
   Capped to 18 (the level whose scale-bar reading is "50 m" over
   India's latitude band, matching what was reported).
2. Globe stuck after returning from Leaflet: switchTo3D() released its
   `switching` guard synchronously while its own 1000ms camera flight
   was still animating from a low starting altitude (< 0.35 — the exact
   value that triggers switchTo2D), so onGlobeCameraChange saw that low
   mid-flight altitude and fought the transition.
3. LOCATION now shows the real attributed facility's name when one
   exists, or a live-looked-up nearby named OSM feature when the
   matched facility itself has no name tag (see test_osm_facilities.py
   for the backend side of that fallback).
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


# --- zoom cap ---------------------------------------------------------

def test_max_map_zoom_constant_is_18_not_19():
    src = html()
    assert "const MAX_MAP_ZOOM = 18;" in src


def test_no_hardcoded_maxzoom_19_left_anywhere():
    """Regression guard: every tile layer and the map constructor itself
    must go through the one shared constant, not a scattered literal
    that could drift back to 19 in one spot and not another."""
    assert "maxZoom: 19" not in html()
    assert "setMaxZoom(19)" not in html()


def test_all_tile_layers_and_the_map_constructor_use_the_shared_cap():
    src = html()
    assert src.count("maxZoom: MAX_MAP_ZOOM") >= 5
    assert "L.map('map-div', { zoomControl: true, maxZoom: MAX_MAP_ZOOM }" in src


def test_basemap_switch_resets_to_the_shared_cap_not_19():
    src = html()
    view_ctrls = src.split("function initViewCtrls()")[1].split("function initFilterCtrls")[0]
    assert "leafletMap.setMaxZoom(MAX_MAP_ZOOM);" in view_ctrls


# --- globe-stuck-after-return fix ------------------------------------------

def test_switch_to_3d_keeps_the_switching_guard_for_the_whole_flight():
    src = html()
    fn = src.split("function switchTo3D()")[1].split("function ")[0]
    assert "const FLIGHT_MS = 1000;" in fn
    assert "setTimeout(() => { switching = false; }, FLIGHT_MS);" in fn
    # Regression guard: must NOT release it synchronously right after
    # starting the animated pointOfView call.
    assert "switching = false;\n    }" not in fn


def test_switch_to_3d_still_sets_switching_true_immediately():
    """The synchronous guard at the start must remain — only the release
    moved to match the animation's real duration."""
    src = html()
    fn = src.split("function switchTo3D()")[1].split("function ")[0]
    assert "switching = true; is2D = false;" in fn.split("\n")[1]


# --- facility name in the LOCATION field -----------------------------

def test_location_row_has_a_stable_id_for_async_update():
    assert 'id="dp-location"' in html()


def test_location_prefers_the_real_facility_name():
    src = html()
    fn = src.split("async function loadEventDetail(fire)")[1].split("function buildSpreadHtml")[0]
    assert "attribution.name" in fn
    assert "attribution.nearby_named_feature" in fn


def test_location_labels_the_fallback_as_near_not_a_confirmed_match():
    """The live-looked-up nearby feature is a second, independent OSM
    element, not a confirmed match to the attributed facility record —
    must read 'Near X', not present it as the exact site."""
    src = html()
    fn = src.split("async function loadEventDetail(fire)")[1].split("function buildSpreadHtml")[0]
    assert "'Near ' + attribution.nearby_named_feature" in fn


def test_location_falls_back_to_district_state_when_neither_name_exists():
    src = html()
    fn = src.split("async function loadEventDetail(fire)")[1].split("function buildSpreadHtml")[0]
    assert "let headline = place;" in fn


def test_facility_name_never_uses_math_random():
    assert "Math.random" not in html()
