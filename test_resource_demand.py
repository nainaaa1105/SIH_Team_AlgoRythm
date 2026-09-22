"""WFDSS-inspired Resource Demand Index (RDI).

Same split as test_wui.py: the pure scoring function is tested directly
(no DB, no network), and the dashboard-side wiring is checked the way
test_gateway.py checks the rest of the page — that the hooks a live
dashboard actually needs are present, and nothing fabricates a value.
"""
from pathlib import Path

from geospatial.decision_engine import (
    RDI_FULL_SUPPRESSION_THRESHOLD,
    RDI_POINT_ZONE_THRESHOLD,
    area_score,
    compute_rdi,
    footprint_area_ha,
    frp_score,
    proximity_score,
)
from geospatial.wui_analysis import EMBER_JUMP_THRESHOLD_M

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


# --- compute_rdi: the WFDSS Fire A vs Fire B example --------------------

def test_small_close_hot_fire_outranks_huge_remote_cool_fire():
    """The exact scenario this feature is modelled on: a small,
    WUI-adjacent, high-FRP fire must outrank a huge, remote, low-FRP
    one, even though the low-FRP fire has burned far more area."""
    fire_a_huge_remote_cool = compute_rdi(frp_max_mw=8.0, distance_m=25_000, area_ha=4000.0)
    fire_b_small_close_hot = compute_rdi(frp_max_mw=140.0, distance_m=1_800, area_ha=200.0)

    assert fire_b_small_close_hot["rdi_score"] > fire_a_huge_remote_cool["rdi_score"]
    assert fire_a_huge_remote_cool["coa_type"] == "MONITOR_ECOLOGICAL_BENEFIT"
    assert fire_b_small_close_hot["coa_type"] == "FULL_SUPPRESSION_AIR_TANKERS"


def test_rdi_score_is_bounded_0_to_100():
    result = compute_rdi(frp_max_mw=10_000.0, distance_m=0.0, area_ha=1_000_000.0)
    assert result["rdi_score"] <= 100.0


def test_rdi_components_sum_to_the_total():
    result = compute_rdi(frp_max_mw=60.0, distance_m=5_000, area_ha=150.0)
    total = result["frp_component"] + result["proximity_component"] + result["area_component"]
    assert result["rdi_score"] == round(total, 1)


# --- course-of-action thresholds -----------------------------------------

def test_coa_thresholds_match_their_boundaries():
    assert compute_rdi(0, None, 0)["rdi_score"] < RDI_POINT_ZONE_THRESHOLD
    assert compute_rdi(0, None, 0)["coa_type"] == "MONITOR_ECOLOGICAL_BENEFIT"

    # Maxed FRP + maxed proximity alone (75) falls just short of the
    # full-suppression boundary (80) — a real fire always has some
    # nonzero footprint, so this is the boundary that matters in
    # practice: does adding a modest area component cross it?
    maxed_frp_and_proximity = compute_rdi(frp_max_mw=150.0, distance_m=0.0, area_ha=0.0)
    assert maxed_frp_and_proximity["coa_type"] == "POINT_ZONE_PROTECTION"

    at_boundary = compute_rdi(frp_max_mw=150.0, distance_m=0.0, area_ha=100.0)
    assert at_boundary["rdi_score"] >= RDI_FULL_SUPPRESSION_THRESHOLD
    assert at_boundary["coa_type"] == "FULL_SUPPRESSION_AIR_TANKERS"


# --- frp_score --------------------------------------------------------

def test_frp_score_caps_at_the_weight_not_beyond():
    assert frp_score(150.0) == 35.0
    assert frp_score(10_000.0) == 35.0


def test_frp_score_is_zero_for_missing_or_nonpositive_frp():
    assert frp_score(None) == 0.0
    assert frp_score(0.0) == 0.0
    assert frp_score(-5.0) == 0.0


# --- proximity_score ----------------------------------------------------

def test_proximity_score_is_maximum_within_ember_jump_range():
    assert proximity_score(0.0) == 40.0
    assert proximity_score(EMBER_JUMP_THRESHOLD_M) == 40.0


def test_proximity_score_tapers_between_ember_range_and_10km():
    ember_km = EMBER_JUMP_THRESHOLD_M / 1000.0
    just_inside = proximity_score(EMBER_JUMP_THRESHOLD_M + 1)
    midpoint = proximity_score((ember_km + 10.0) / 2.0 * 1000.0)
    just_before_far = proximity_score(9_999.0)
    assert 40.0 > just_inside > midpoint > just_before_far > 5.0


def test_proximity_score_floors_at_far_beyond_10km():
    assert proximity_score(10_000.0) == 5.0
    assert proximity_score(1_000_000.0) == 5.0


def test_none_distance_is_a_confirmed_far_result_not_a_worst_case_guess():
    """None means the WUI check found no built-up land within its
    search radius at all — that is itself evidence of low proximity
    risk, so it must score the same as a confirmed-far fire rather than
    a fabricated worst-case penalty."""
    assert proximity_score(None) == proximity_score(50_000.0) == 5.0


# --- area_score / footprint_area_ha --------------------------------------

def test_area_score_caps_at_the_weight():
    assert area_score(500.0) == 25.0
    assert area_score(1_000_000.0) == 25.0


def test_area_score_is_zero_for_missing_or_nonpositive_area():
    assert area_score(None) == 0.0
    assert area_score(0.0) == 0.0


def test_footprint_area_ha_is_a_real_geometric_conversion_not_fabricated():
    """A 2 km-diameter footprint has a real circle area of pi*1^2 = pi
    km^2 = 100*pi ha — this must be an actual geometric computation,
    not an arbitrary made-up multiplier."""
    import math

    result = footprint_area_ha(2.0)
    assert abs(result - (math.pi * 1.0 * 1.0 * 100.0)) < 1e-6


def test_footprint_area_ha_is_zero_for_a_stationary_source():
    assert footprint_area_ha(0.0) == 0.0
    assert footprint_area_ha(None) == 0.0


# --- frontend wiring: no fabricated values, hooks present ----------------

def html():
    return STATIC.read_text(encoding="utf-8")


def test_rdi_fields_flow_from_the_gateway_into_the_fire_object():
    src = html()
    assert "rdiScore: d.rdi_score" in src
    assert "coaType: d.coa_type" in src


def test_rdi_never_uses_math_random():
    assert "Math.random" not in html()


def test_rdi_badge_and_control_panel_checkbox_are_both_removed():
    """The 'Full-Suppression Priority' Control Panel row and the Live
    Summary badge were both removed by request (the panel now shows
    only WUI and Crown Fire toggles) — the filter logic itself
    (filt.rdiOnly, the isolate override, the reset) is untouched and
    stays reachable programmatically even with no UI trigger left; see
    test_rdi_isolate_filter_is_a_full_override_like_wui_and_crown and
    test_rdi_reset_clears_the_new_filter_state."""
    src = html()
    assert 'id="cb-rdi-only"' not in src
    assert 'id="badge-rdi"' not in src
    assert 'id="badge-rdi-n"' not in src


def test_rdi_isolate_filter_is_a_full_override_like_wui_and_crown():
    """Same bug class the WUI isolate checkbox already had fixed for it:
    ticking 'Full-Suppression Priority' must not be AND'd with the
    (default-restrictive) confidence/type filters, or a real LOW-
    confidence full-suppression fire could tick the box and still see
    an empty map."""
    src = html()
    assert "filt.wuiOnly || filt.crownOnly || filt.rdiOnly" in src
    assert "filt.rdiOnly && f.coaType === 'FULL_SUPPRESSION_AIR_TANKERS'" in src


def test_rdi_reset_clears_the_new_filter_state():
    src = html()
    assert "rdiOnly: false" in src
    assert 'cb-rdi-only\'); if (rdiCb) rdiCb.checked = false' in src


def test_wfdss_card_and_dispatch_removed_from_the_detail_panel():
    """Removed by request: the 'WFDSS Resource Decision Matrix' card
    (RDI score, FRP/proximity/size, course of action, and its download
    button) is gone from the fire detail panel entirely. The underlying
    rdiScore/coaType data and compute_rdi() itself are untouched — they
    still drive the Suppression Resource Estimate card and the RDI
    isolate filter (see test_rdi_isolate_filter_is_a_full_override_like_wui_and_crown)."""
    src = html()
    assert 'id="wfdss-card"' not in src
    assert 'id="wfdss-rdi"' not in src
    assert 'id="wfdss-coa"' not in src
    assert 'id="wfdss-components"' not in src
    assert 'id="btn-rdi-dispatch"' not in src
    assert "downloadRdiDispatch" not in src
    assert "loadResourceDemandDetail" not in src


def test_rp_rdi_ranking_reads_visfires_an_absolute_filter():
    """By request: the Control Panel's type/date/confidence checkboxes
    are an absolute filter — an unticked class must disappear from the
    resource-ranking list too, not just its map dot, so this now reads
    visFires (which already respects filt.state via inStateScope)
    rather than a separate allFires+scope re-filter."""
    src = html()
    start = src.index("function updateWuiPanel")
    end = src.index("function updateWuiQuadChart")
    body = src[start:end]
    assert "const scoped = visFires;" in body
    assert "scoped.filter(f => f.coaType === 'FULL_SUPPRESSION_AIR_TANKERS')" in body
