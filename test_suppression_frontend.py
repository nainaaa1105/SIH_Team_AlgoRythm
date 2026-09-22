"""Fire Suppression tab: the dashboard-side wiring, checked the way
test_gateway.py checks the rest of the page. Two concerns:

1. The frontend's suppressionEstimate() is a JS mirror of
   geospatial/suppression.py's estimate_suppression() — same reasoning
   destPoint() mirrors geospatial/geometry.py::destination_point (see
   its comment) — so the bulk "every fire" list can be rendered client-
   side from data already fetched, instead of 900+ individual live API
   calls. This file pins the mirrored constants to the exact same
   numeric values as the Python source, so the two can never drift.
2. Every fire is listed (no size/class exclusion) and sorted by the
   real, already-computed RDI priority tier — not a new, separate
   priority scale invented for this tab.
"""
import re
from pathlib import Path

from geospatial.suppression import (
    AIR_SUPPORT_AREA_HA,
    DUMP_TRUCK_CAPACITY_M3,
    FOAM_APPLICATION_MINUTES,
    FOAM_RATE_L_PER_MIN_PER_M2,
    FRP_INTENSITY_MAX_MULTIPLIER,
    FRP_INTENSITY_REFERENCE_MW,
    GAS_FLARE_COOLING_RATE_L_PER_HA,
    SENSOR_PIXEL_AREA_HA,
    SMOTHER_DEPTH_M,
    UNCLASSIFIED_FALLBACK_RATE_L_PER_HA,
    WATER_RATE_L_PER_HA,
    WATER_TANKER_CAPACITY_L,
)

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def _js_number(src: str, name: str) -> float:
    match = re.search(re.escape(name) + r"\s*=\s*([\d.]+)", src)
    assert match, f"{name} not found in JS source"
    return float(match.group(1))


# --- the JS mirror's constants match the Python source exactly ------------

def test_js_water_rates_match_python():
    src = html()
    fn = src.split("const SUPP_WATER_RATE_L_PER_HA")[1].split(";")[0]
    assert str(WATER_RATE_L_PER_HA["agricultural_burning"]) in fn
    assert str(WATER_RATE_L_PER_HA["wildfire"]) in fn


def test_js_foam_rate_matches_python():
    src = html()
    assert _js_number(src, "SUPP_FOAM_RATE_L_PER_MIN_PER_M2") == FOAM_RATE_L_PER_MIN_PER_M2
    assert _js_number(src, "SUPP_FOAM_APPLICATION_MINUTES") == FOAM_APPLICATION_MINUTES


def test_js_smother_depth_matches_python():
    src = html()
    assert _js_number(src, "SUPP_SMOTHER_DEPTH_M") == SMOTHER_DEPTH_M


def test_js_fleet_capacities_match_python():
    src = html()
    assert _js_number(src, "SUPP_WATER_TANKER_CAPACITY_L") == WATER_TANKER_CAPACITY_L
    assert _js_number(src, "SUPP_DUMP_TRUCK_CAPACITY_M3") == DUMP_TRUCK_CAPACITY_M3


def test_js_air_support_threshold_matches_python():
    src = html()
    assert _js_number(src, "SUPP_AIR_SUPPORT_AREA_HA") == AIR_SUPPORT_AREA_HA


def test_js_sensor_pixel_floor_matches_python():
    """The real fix for the reported bug (582/903 real clusters were
    reporting zero material) — the JS mirror's floor must match the
    Python source's exactly, not a rounded-off approximation."""
    src = html()
    fn = src.split("const SUPP_SENSOR_PIXEL_AREA_HA")[1].split(";")[0]
    assert str(SENSOR_PIXEL_AREA_HA["VIIRS"]) in fn
    assert str(SENSOR_PIXEL_AREA_HA["MODIS"]) in fn


def test_js_intensity_scaling_matches_python():
    src = html()
    assert _js_number(src, "SUPP_FRP_INTENSITY_REFERENCE_MW") == FRP_INTENSITY_REFERENCE_MW
    assert _js_number(src, "SUPP_FRP_INTENSITY_MAX_MULTIPLIER") == FRP_INTENSITY_MAX_MULTIPLIER


def test_js_gas_flare_cooling_rate_matches_python():
    src = html()
    assert _js_number(src, "SUPP_GAS_FLARE_COOLING_RATE_L_PER_HA") == GAS_FLARE_COOLING_RATE_L_PER_HA


def test_js_unclassified_fallback_rate_matches_python():
    src = html()
    fn = src.split("const SUPP_UNCLASSIFIED_FALLBACK_RATE_L_PER_HA")[1].split(";")[0]
    assert "SUPP_WATER_RATE_L_PER_HA.agricultural_burning" in fn
    assert UNCLASSIFIED_FALLBACK_RATE_L_PER_HA == WATER_RATE_L_PER_HA["agricultural_burning"]


def test_js_priority_levels_match_python_low_medium_high():
    src = html()
    fn = src.split("const SUPP_PRIORITY_LEVELS")[1].split(";")[0]
    assert "FULL_SUPPRESSION_AIR_TANKERS: 'HIGH'" in fn
    assert "POINT_ZONE_PROTECTION: 'MEDIUM'" in fn
    assert "MONITOR_ECOLOGICAL_BENEFIT: 'LOW'" in fn


# --- never zero, never N/A --------------------------------------------------

def test_js_never_reports_zero_tanker_trips_for_a_positive_volume():
    """suppressionCeilAtLeastOne must floor at 1, not 0, for any real
    positive volume — the client-side mirror of _ceil_div_at_least_one."""
    src = html()
    fn = src.split("function suppressionCeilAtLeastOne(value, divisor)")[1].split("function suppressionEstimate")[0]
    assert "Math.max(1, Math.ceil(value / divisor))" in fn


def test_js_effective_area_applies_the_sensor_floor():
    src = html()
    fn = src.split("function suppressionEffectiveArea(areaHa, sensor)")[1].split("function suppressionIntensityMultiplier")[0]
    assert "Math.max(areaHa || 0.0, floor)" in fn


def test_gas_flare_gets_a_real_quantity_and_fleet_client_side_too():
    """No 'N/A' branch anywhere in the estimator for gas_flare — cooling
    volume and a well-control unit count instead."""
    src = html()
    fn = src.split("function suppressionEstimate(fire)")[1].split("function suppressionSummaryLine")[0]
    assert "wellControlUnits: 1" in fn
    assert "primaryVolumeL: null" not in fn


def test_detail_panel_quantity_never_falls_back_to_na():
    src = html()
    fn = src.split("const quantityEl = document.getElementById('supp-quantity');")[1].split("const fleetEl")[0]
    assert "'N/A'" not in fn


def test_detail_panel_fleet_never_falls_back_to_na():
    src = html()
    fn = src.split("const fleetEl = document.getElementById('supp-fleet');")[1].split("const airEl")[0]
    assert "'N/A'" not in fn
    assert "wellControlUnits" not in fn  # server-side snake_case field name is well_control_units
    assert "well_control_units" in fn


# --- priority row in the detail panel ---------------------------------------

def test_detail_panel_shows_the_low_medium_high_priority():
    src = html()
    assert 'id="supp-priority"' in src
    fn = src.split("const priorityEl = document.getElementById('supp-priority');")[1].split("const resourceEl")[0]
    assert "est.priority_level" in fn


# --- the tab lists every fire, unfiltered by size/class --------------------

def test_suppression_panel_reads_allfires_not_visfires():
    """'For all fires irrespective of how big or small' — the sidebar's
    class/date/confidence filters must not exclude anything from this
    list, same reasoning as the WUI/RDI panels."""
    src = html()
    fn = src.split("function updateSuppressionPanel()")[1].split("function ")[0]
    assert "allFires.filter(inStateScope)" in fn
    assert "visFires" not in fn


def test_suppression_panel_does_not_filter_on_area_or_class():
    """No .filter(f => f.areaHa > X) or class allow-list anywhere in the
    panel builder — every fire gets a row, including unclassified or
    zero-footprint ones."""
    src = html()
    fn = src.split("function updateSuppressionPanel()")[1].split("function ")[0]
    assert "f.areaHa" not in fn
    assert "f.predictedClass" not in fn


def test_suppression_panel_sorts_by_the_real_rdi_priority_tier():
    """'I also want their priority' — reuses coaType (the real,
    already-computed RDI tier), not a second invented priority scale."""
    src = html()
    assert "SUPP_PRIORITY_ORDER" in html()
    fn = src.split("const SUPP_PRIORITY_ORDER")[1].split(";")[0]
    assert "FULL_SUPPRESSION_AIR_TANKERS" in fn
    assert "POINT_ZONE_PROTECTION" in fn
    assert "MONITOR_ECOLOGICAL_BENEFIT" in fn


def test_suppression_card_exists_in_the_detail_panel():
    """The Live Summary panel's 'Fire Suppression' list (every fire, by
    priority) was removed by request — the per-fire Suppression Estimate
    card in the detail popup (openDP -> supp-card) is a different,
    still-present feature and is untouched."""
    src = html()
    assert 'id="rp-suppression-sec"' not in src
    assert 'id="rp-suppression-list"' not in src
    assert 'id="supp-card"' in src


def test_suppression_list_is_bounded_so_900_rows_do_not_break_layout():
    src = html()
    assert "rp-supp-list" in src
    css_rule = src.split(".rp-supp-list {")[1].split("}")[0]
    assert "overflow-y: auto" in css_rule


def test_suppression_card_moved_above_classification_reasoning_with_white_background():
    """By request: the Suppression Resource Estimate card now sits right
    after dp-body, above Classification Reasoning (previously it was the
    last card, after WUI — the WUI Ember-Jump Assessment card was later
    removed from the panel entirely, see test_wui.py). Its background is
    white, not the old light blue — that light-blue fill only ever meant
    anything when it shared the .wfdss-card class with the now-removed
    WFDSS Resource Decision Matrix card."""
    src = html()
    dp_body_idx = src.index('id="dp-body"')
    supp_card_idx = src.index('id="supp-card"')
    cls_reasons_idx = src.index('id="cls-reasons"')
    dp_acts_idx = src.index('class="dp-acts"')
    assert dp_body_idx < supp_card_idx < cls_reasons_idx < dp_acts_idx

    css_rule = src.split(".wfdss-card {")[1].split("}")[0]
    assert "background: #fff" in css_rule


def test_sms_button_lives_inside_the_suppression_card_styled_like_the_right_panel_badges():
    """By request: the SMS button moved inside the Suppression Resource
    Estimate card (no longer a separate dp-acts row) and takes the same
    blue (#0014a8) used by the header, footer, and the WUI Critical /
    Crown Fires badges in the right panel, with white text — a rule it
    now shares with Zoom/Download Report (see test_gateway.py)."""
    src = html()
    supp_card = src.split('id="supp-card"')[1].split('<div class="cls-box">')[0]
    assert 'id="btn-sms-dispatch"' in supp_card
    css_rule = src.split("#btn-sms-dispatch,")[1].split("}")[0]
    assert "#0014a8" in css_rule
    assert "color: #fff" in css_rule


def test_industrial_class_never_uses_the_water_rate_table_client_side():
    src = html()
    fn = src.split("function suppressionEstimate(fire)")[1].split("function suppressionSummaryLine")[0]
    assert "'industrial_fire'" in fn
    assert "'gas_flare'" in fn
    assert "'mining'" in fn


def test_suppression_card_states_it_is_an_estimate_not_a_measurement():
    """Replaced by request: the bulleted rate-citation notes (water/fuel
    caveats, NFPA rate, footprint-floor note, FRP scaling note) are gone
    entirely — just a single plain-language disclaimer line remains."""
    src = html()
    assert "This is an approximate estimate, not the actual answer." in src
    assert "Doctrine-based estimate, not a measurement" not in src
    assert 'id="supp-notes"' not in src


def test_detail_panel_fetches_the_live_endpoint_for_every_selected_fire():
    """Unlike WUI/RDI (gated on rdiScore != null), the suppression card
    must load for every fire, including ones with no RDI score yet."""
    src = html()
    open_fn = src.split("function openDP(fire)")[1].split("async function loadResourceDemandDetail")[0]
    assert "loadSuppressionDetail(fire);" in open_fn
    assert "if (fire.rdiScore != null) loadSuppressionDetail" not in open_fn


def test_suppression_never_uses_math_random():
    assert "Math.random" not in html()


def test_all_three_detail_panel_headings_are_blue():
    """By request: 'Suppression Resource Estimate', 'Classification
    Reasoning' and 'XAI Spread Prediction' must all share the same blue
    heading color — the base .cls-title rule (which the latter two use
    with no override) now matches the blue every section already used
    for its own override, rather than the earlier black."""
    src = html()
    rule = src.split(".cls-title {")[1].split("}")[0]
    assert "color: #1565c0" in rule


def test_cls_title_font_matches_the_fire_detail_header():
    """By request: the section headings must use the exact same font as
    'FIRE DETAIL' (.dp-hdr-lbl) at the top of the panel — same family
    (already shared), same size (19px) and letter-spacing (2px), not the
    smaller 15px/1.2px the section headings used to carry."""
    src = html()
    header_rule = src.split(".dp-hdr-lbl {")[1].split("}")[0]
    title_rule = src.split(".cls-title {")[1].split("}")[0]
    for prop in ("font-family: Calibri, 'Segoe UI', Candara, sans-serif",
                 "font-size: 19px", "letter-spacing: 2px"):
        assert prop in header_rule
        assert prop in title_rule
