"""Gateway composition and dashboard delivery.

These run without a database. Anything needing Postgres is covered by
asserting the *degradation* behaviour instead, because that is the state
an operator actually hits first.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import MOUNTED_MODULES, app

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


@pytest.fixture(scope="module")
def client():
    # raise_server_exceptions=False makes TestClient behave like a real
    # HTTP client: it returns the response the exception handler
    # produced instead of re-raising the original exception, which is
    # what we need in order to assert on the 503 degradation path.
    return TestClient(app, raise_server_exceptions=False)


# --- composition --------------------------------------------------------

def test_every_member_router_is_mounted():
    """The whole point of the gateway: one app, all five members."""
    for label in ("M1 ingestion", "M2 classification", "M3 geospatial",
                  "M4 imagery", "M5 temporal", "M6 dashboard"):
        assert label in MOUNTED_MODULES, f"{label} did not mount"


def test_all_expected_routes_exist(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    expected = {
        "/hotspots", "/clusters", "/facilities",                      # M1
        "/classify/{cluster_id}",                                      # M2
        "/plume/{cluster_id}", "/threat/{cluster_id}",
        "/attribution/{cluster_id}",                                   # M3
        "/imagery/{cluster_id}",                                       # M4
        "/ptsi/{cluster_id}", "/forecast/{cluster_id}",
        "/rhythm/{cluster_id}", "/escalating",                         # M5
        "/dashboard/detections", "/dashboard/event/{cluster_id}",
        "/dashboard/alerts", "/dashboard/summary",                     # M6
        "/dashboard/event/{cluster_id}/sms-dispatch",
    }
    missing = expected - paths
    assert not missing, f"missing routes: {sorted(missing)}"


def test_health_reports_which_modules_mounted(client):
    """Introspecting app.routes returned an empty list on this FastAPI
    version, so modules are tracked explicitly. Guard that."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert len(body["modules"]) >= 6


def test_no_route_collides_across_members(client):
    paths = client.get("/openapi.json").json()["paths"]
    seen = set()
    for path, methods in paths.items():
        for method in methods:
            assert (method, path) not in seen
            seen.add((method, path))


# --- static dashboard ---------------------------------------------------

def test_dashboard_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_static_mount_does_not_shadow_the_api(client):
    """A StaticFiles mount at "/" swallows unmatched paths, so it must be
    registered after every router."""
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200


# --- degradation without a database -------------------------------------

def test_detections_are_windowed_to_3_days_not_the_full_history():
    """Explicit product requirement: the dashboard shows a rolling recent
    window, old detections stay in Postgres untouched but must not keep
    accumulating into what the map/Live Summary render. Verified against
    the query construction (no live DB in this suite — see the module
    docstring), the same way other PostGIS-dependent logic here is
    checked live rather than unit-tested against a real Postgres."""
    import inspect

    from gateway import routes_dashboard

    assert routes_dashboard.DASHBOARD_DETECTIONS_WINDOW_DAYS == 3
    src = inspect.getsource(routes_dashboard._cluster_rows)
    assert "func.max(Cluster.last_seen)" in src
    assert "timedelta(days=DASHBOARD_DETECTIONS_WINDOW_DAYS)" in src
    assert "Cluster.last_seen >= cutoff" in src


def test_detections_expose_the_ptsi_source_class():
    """Persistent-vs-transient (temporal.ptsi) is real per-cluster data,
    already computed for the detail panel's ptsi block — this makes it
    available on the flattened /dashboard/detections item too, since the
    front page needs it for every fire, not just the one clicked open."""
    import inspect

    from gateway import routes_dashboard

    src = inspect.getsource(routes_dashboard.list_detections)
    assert '"source_class": (ptsi or {}).get("source_class")' in src


def test_database_endpoints_return_503_not_500(client):
    """Before `alembic upgrade head` there is no database. A bare 500
    looks identical to a bug; 503 with a hint tells the operator what to
    do."""
    for path in ("/dashboard/detections", "/dashboard/summary", "/hotspots"):
        r = client.get(path)
        assert r.status_code in (200, 503), f"{path} → {r.status_code}"
        if r.status_code == 503:
            assert "hint" in r.json()


# --- the HTML itself ----------------------------------------------------
#
# These protected the previous Cesium dashboard by asserting its internal
# function names, its Cesium bootstrap tokens and its bundled `HOTSPOTS`
# sample array. That dashboard has been replaced by the Agni Pehchan UI
# (Globe.gl 3D globe + Leaflet 2D map) wired to the live gateway, so every
# one of those tokens is gone and the assertions were checking for a page
# that no longer ships.
#
# They are rewritten here against the dashboard that does ship, keeping
# what they were actually protecting: the map implementation is not
# silently swapped out, no control is decorative, the data layer stays
# separated from the UI, and — new, because it is now a hard product
# requirement — the page contains no fabricated detections at all.

@pytest.fixture(scope="module")
def html():
    return STATIC.read_text(encoding="utf-8")


def test_the_map_implementation_is_intact(html):
    """The 3D globe and the 2D map are the locked components. If any of
    these disappears, the map has been replaced rather than extended."""
    required = [
        "unpkg.com/globe.gl@2.46.2/dist/globe.gl.min.js",
        "unpkg.com/leaflet@1.9.4/dist/leaflet.js",
        "earth-blue-marble.jpg",
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        ".showAtmosphere(",
        "function initGlobe",
        "function initLeaflet",
        "function switchTo2D",
        "function switchTo3D",
    ]
    for token in required:
        assert token in html, f"protected map code missing: {token}"


def test_core_functions_are_all_still_present(html):
    for fn in ["function toFire", "function loadFires", "function applyFilt",
               "function updateStats", "function buildDonut", "function buildBD",
               "function openDP", "function closeDP",
               "function loadEventDetail", "function buildSpreadHtml",
               "function plotGlobePoints", "function plotLeafletMarkers",
               "function initSearch", "function initAccordions", "function initSidebar",
               "function initTypeCBs", "function initSensorCBs", "function initFilterCtrls",
               "function initDetailCtrls", "function initRightPanel",
               "function initLiveUpdates", "function initStateFilter",
               "function downloadReport", "function setBanner"]:
        assert fn in html, f"core function removed: {fn}"


def test_the_dashboard_fabricates_no_detections(html):
    """The hard requirement: every fire on this page is a real classified
    detection from the pipeline.

    The page this replaced generated its own dataset client-side —
    `genFires()` invented ~300 fires with Math.random and `classify()`
    assigned them a class from hardcoded geofences. Nothing of that kind
    may come back: a plausible-looking fabricated detection on an
    operator's screen is worse than an empty map.
    """
    for banned in ["function genFires", "const IND = [", "const FZ = [",
                   "const UZ = [", "const AZ = [", "const HOTSPOTS = ["]:
        assert banned not in html, f"synthetic data generator is back: {banned}"

    # Math.random must not be used to manufacture any displayed value.
    assert "Math.random" not in html, "Math.random has no place in a live feed"


def test_data_layer_is_separated_from_ui(html):
    """The UI must read the gateway, not a literal."""
    assert "const API = {" in html
    assert "function toFire" in html
    for endpoint in ["/dashboard/detections", "/dashboard/event/", "/dashboard/summary",
                     "/dashboard/states", "/health", "/ws/live-updates"]:
        assert endpoint in html, f"dashboard does not call {endpoint}"


def test_classifier_classes_map_one_to_one_to_marker_styles(html):
    """A class the model can emit but the UI cannot draw would be
    invisible on the map."""
    block = html.split("const CLASS_TO_TYPE = {")[1].split("};")[0]
    for predicted in ["wildfire", "industrial_fire", "gas_flare",
                      "agricultural_burning", "mining"]:
        assert predicted in block, f"no marker style for class {predicted}"


def test_every_control_has_a_handler(html):
    """No decorative buttons."""
    for control, handler in [
        ("btn-apply", "getElementById('btn-apply').addEventListener"),
        ("btn-reset-f", "getElementById('btn-reset-f').addEventListener"),
        ("btn-zoom", "getElementById('btn-zoom').addEventListener"),
        ("btn-report", "getElementById('btn-report').addEventListener"),
        ("dp-close", "getElementById('dp-close').addEventListener"),
    ]:
        assert handler in html, f"{control} has no event handler"

    # btn-start is a type="submit" button inside #lp-login-form — the
    # form's submit handler is its real event handler (also picks up
    # the Enter key, which a bare click listener would not).
    assert "loginForm.addEventListener('submit'" in html, "btn-start's form has no submit handler"

    # The five class checkboxes and the six sensor/day checkboxes are
    # wired in loops rather than individually.
    assert "cb.addEventListener('change'" in html


def test_every_fire_class_has_a_filter_row(html):
    """A class with no filter row cannot be turned off, and its count
    never renders."""
    block = html.split("const FT = {")[1].split("};")[0]
    keys = re.findall(r"(\w+):\s*\{ label:", block)
    assert set(keys) == {"WILD", "INDUSTRIAL", "GAS_FLARE", "AGRI", "MINING"}
    for key in keys:
        assert f'id="cb-{key}"' in html, f"no filter checkbox for {key}"
        assert f'id="cnt-{key}"' in html, f"no count element for {key}"


def test_sensor_day_checkboxes_actually_filter(html):
    """Regression: the six Sensor Data checkboxes collapsed to a two-value
    sensors set, so the three day rows under each instrument were
    decorative — unticking a day changed nothing while any box in that
    group stayed ticked."""
    assert "sensorDates" in html
    assert "function readSensorFilter" in html
    assert "function sensorDateOptions" in html
    # The per-detection check now lives in passesSensorDateFilter(), shared
    # by applyFilt() and the WUI/crown/RDI safety overlays (test_wui.py's
    # test_unchecking_every_sensor_date_hides_wui_too covers that sharing).
    date_fn = html.split("function passesSensorDateFilter(f)")[1].split("\n    function ")[0]
    assert "filt.sensorDates.has(f.sensor + '|' + f.acqDate)" in date_fn
    applyfilt = html.split("function applyFilt")[1].split("function updateStats")[0]
    assert "passesSensorDateFilter(f)" in applyfilt


def test_initial_view_is_drawn_through_the_filters(html):
    """Regression: the first paint plotted every loaded detection while
    the sidebar advertised a narrower filter, so the count jumped the
    first time any control was touched."""
    boot = html.split("window.addEventListener('load'")[1]
    assert "applyFilt();" in boot, "bootstrap must draw through applyFilt"


def test_missing_values_are_shown_as_missing(html):
    """A field the pipeline did not produce must read as absent, not as a
    plausible number."""
    assert "not available" in html
    assert "dp-na" in html
    assert "awaiting classification" in html


def test_detail_panel_has_the_sections_the_brief_specifies(html):
    detail = html.split("function openDP")[1].split("async function loadEventDetail")[0]
    for field in ["LOCATION", "LAT / LON", "DETECTED", "SENSOR",
                  "FRP (MW)", "BRIGHTNESS", "MODEL CONF.", "STATUS"]:
        assert field in detail, f"detail field missing: {field}"


def test_detail_panel_no_longer_shows_the_raw_sensor_detection_confidence(html):
    """Removed by request: DETECTION CONF. (the raw satellite-reported
    LOW/NOMINAL/HIGH quality flag) confused operators sitting right next
    to MODEL CONF. (the classifier's own confidence in the predicted
    class) — two independent numbers that don't have to agree. Only
    MODEL CONF. remains."""
    detail = html.split("function openDP")[1].split("async function loadEventDetail")[0]
    assert "DETECTION CONF." not in detail


def test_explanations_come_from_the_model_not_the_client(html):
    """The reasoning shown to an operator must be the model's SHAP output
    carried through from the gateway."""
    block = html.split("async function loadEventDetail")[1].split("function buildSpreadHtml")[0]
    assert "cls.reasons" in block
    assert "explanation_method" in block
    assert "model_version" in block


def test_assumed_wind_is_labelled_as_assumed(html):
    """M3 falls back to a default wind when Open-Meteo is unreachable and
    flags it. A plume drawn from assumed wind must say so."""
    assert "wind_is_assumed" in html
    assert "wind assumed" in html


def test_xai_spread_prediction_is_a_row_list_not_a_br_joined_paragraph(html):
    """Redesigned by request: the plume/escalation/temperature/PTSI
    readings used to be one <br>-joined wall of text — now each reading
    is its own row (same rhythm as Classification Reasoning's
    checkmarked list, just with a plain bullet instead of a checkmark),
    in plain black (var(--t2)) text, with no enclosing box."""
    block = html.split("function buildSpreadHtml(ev)")[1].split("function formatIn")[0]
    assert "parts.join('<br>')" not in block
    assert 'class="xai-row"' in block
    assert 'class="xai-bullet"' in block

    css = html.split(".xai-row {")[1].split("}")[0]
    assert "color: var(--t2)" in css
    assert "border" not in css
    assert "background" not in css
    assert "box-shadow" not in css


def test_responsive_breakpoints_exist(html):
    """The layout is driven by --sw/--rw/--hh, so a breakpoint that does
    not re-declare them is almost certainly not doing anything."""
    for bp in ["@media (max-width: 1180px)", "@media (max-width: 900px)",
               "@media (max-width: 640px)", "@media (max-width: 520px)"]:
        assert bp in html, f"missing breakpoint: {bp}"


def test_no_leftover_debug_statements(html):
    assert html.count("console.log") == 0
    assert "debugger" not in html
