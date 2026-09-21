"""WUI (wildland-urban interface) proximity alerts.

The physics/priority logic is tested directly, the same way
`test_corridor.py` tests `corridor.py` — no DB, no network, just the pure
functions. The dashboard-side wiring (banner, ring, projection line,
operator panel) is checked the way `test_gateway.py` checks the rest of
the page: that the hooks a live dashboard actually needs are present in
`static/index.html`, and that nothing about it fabricates a value.
"""
from pathlib import Path

import pytest

from geospatial.crown_fire import (
    CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR,
    CROWN_FIRE_FRP_THRESHOLD_MW,
    evaluate_crown_fire,
    frp_acceleration_mw_per_hour,
)
from geospatial.threat.corridor import SPREADING_CLASSES
from geospatial.wui_analysis import (
    CRITICAL_ETA_HOURS,
    EMBER_JUMP_THRESHOLD_M,
    SETTLEMENT_NAME_MAX_EXCESS_M,
    WATCH_MULTIPLIER,
    gate_critical_priority,
    priority_for,
    settlement_matches_builtup_pixel,
)

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


# --- priority_for ---------------------------------------------------------

def test_within_ember_jump_range_is_always_critical():
    """Proximity alone triggers CRITICAL, independent of how fast the
    fire is advancing — an ember can jump regardless of front speed."""
    assert priority_for(0.0, None) == "CRITICAL_AIR_TANKER_DISPATCH"
    assert priority_for(EMBER_JUMP_THRESHOLD_M, None) == "CRITICAL_AIR_TANKER_DISPATCH"
    assert priority_for(EMBER_JUMP_THRESHOLD_M - 1, 999) == "CRITICAL_AIR_TANKER_DISPATCH"


def test_beyond_ember_range_but_fast_eta_is_critical():
    """The other independent trigger: 'within a 12-hour projected burn
    window', regardless of raw distance."""
    far = EMBER_JUMP_THRESHOLD_M * 2
    assert priority_for(far, CRITICAL_ETA_HOURS) == "CRITICAL_AIR_TANKER_DISPATCH"
    assert priority_for(far, CRITICAL_ETA_HOURS - 0.01) == "CRITICAL_AIR_TANKER_DISPATCH"


def test_beyond_ember_range_and_slow_eta_is_not_critical():
    far = EMBER_JUMP_THRESHOLD_M * 2
    assert priority_for(far, CRITICAL_ETA_HOURS + 0.01) != "CRITICAL_AIR_TANKER_DISPATCH"


def test_watch_band_between_ember_range_and_watch_multiple():
    just_outside = EMBER_JUMP_THRESHOLD_M + 1
    still_watch = EMBER_JUMP_THRESHOLD_M * WATCH_MULTIPLIER
    beyond_watch = still_watch + 1
    assert priority_for(just_outside, None) == "WATCH"
    assert priority_for(still_watch, None) == "WATCH"
    assert priority_for(beyond_watch, None) == "MONITOR"


def test_no_distance_means_no_builtup_land_found_means_monitor():
    """`None` distance is what `_compute_threat` passes when the
    WorldCover search found nothing within range — a real result, not a
    failure, and it must never read as CRITICAL by accident."""
    assert priority_for(None, 3.0) == "MONITOR"
    assert priority_for(None, None) == "MONITOR"


def test_negative_eta_does_not_falsely_trigger_critical():
    """An ETA can only be non-negative in practice, but the boundary
    check is written as `0 <= eta_hours`, not just `eta_hours <=
    CRITICAL_ETA_HOURS` — guard the case explicitly so a future change
    to time_to_impact_hours's sign convention gets caught here."""
    far = EMBER_JUMP_THRESHOLD_M * 2
    assert priority_for(far, -1.0) != "CRITICAL_AIR_TANKER_DISPATCH"


# --- gate_critical_priority -------------------------------------------------
# Real bug found live: a count of the actual dashboard data showed 225 of
# 228 CRITICAL WUI fires were simultaneously labelled "Under Control" (not
# escalating), and none of the built-up-pixel proximity checks required a
# real named settlement — a fire a few metres from an unmapped compound or
# bare industrial pavement got the same CRITICAL/air-tanker-dispatch badge
# as one genuinely bearing down on a named village. A follow-up request:
# the named-place-only requirement then rejected real, visibly inhabited
# clusters of houses too, whenever OSM had no place node for them — a real
# WorldPop headcount at the same built-up point (is_populated) gives
# CRITICAL a second, independent way to confirm people actually live
# there, alongside (not instead of) the named-place check.

def test_critical_survives_gating_on_named_place_and_active():
    assert gate_critical_priority("CRITICAL_AIR_TANKER_DISPATCH", True, False, True) \
        == "CRITICAL_AIR_TANKER_DISPATCH"


def test_critical_survives_gating_on_population_and_active():
    """Real but unnamed cluster of houses — no OSM place node, but a
    real WorldPop headcount confirms people live there."""
    assert gate_critical_priority("CRITICAL_AIR_TANKER_DISPATCH", False, True, True) \
        == "CRITICAL_AIR_TANKER_DISPATCH"


def test_critical_downgrades_to_watch_without_a_named_place_or_population():
    """Real built-up land, but neither a real settlement name nor a real
    headcount — industrial pavement or a genuinely empty structure, not
    confirmed as a place anyone lives near."""
    assert gate_critical_priority("CRITICAL_AIR_TANKER_DISPATCH", False, False, True) == "WATCH"


def test_critical_downgrades_to_watch_when_not_active():
    """A stalled/Under-Control fire near a real named village — still
    genuinely close, just not currently escalating, so it doesn't get
    the same urgency as one that is. Named place and population both
    present, active alone is what's missing."""
    assert gate_critical_priority("CRITICAL_AIR_TANKER_DISPATCH", True, True, False) == "WATCH"


def test_critical_downgrades_to_watch_when_nothing_holds():
    assert gate_critical_priority("CRITICAL_AIR_TANKER_DISPATCH", False, False, False) == "WATCH"


def test_gate_never_touches_watch_or_monitor():
    """The gate only ever downgrades a CRITICAL read — WATCH/MONITOR
    don't claim a specific place is under imminent threat, so none of
    the named-place/population/active-fire conditions apply to them."""
    assert gate_critical_priority("WATCH", False, False, False) == "WATCH"
    assert gate_critical_priority("MONITOR", False, False, False) == "MONITOR"


def test_is_meaningfully_populated_requires_the_real_threshold():
    from geospatial.wui_analysis import POPULATED_THRESHOLD_PEOPLE, is_meaningfully_populated

    assert is_meaningfully_populated(POPULATED_THRESHOLD_PEOPLE) is True
    assert is_meaningfully_populated(POPULATED_THRESHOLD_PEOPLE - 0.01) is False


def test_is_meaningfully_populated_treats_unknown_as_not_populated():
    """None means WorldPop wasn't sampled/unavailable — an unknown
    headcount must not silently count as a confirmed one."""
    from geospatial.wui_analysis import is_meaningfully_populated

    assert is_meaningfully_populated(None) is False
    assert is_meaningfully_populated(0.0) is False


def test_compute_threat_is_actually_wired_to_the_gate():
    """Regression guard: the gate function existing and being correct in
    isolation is not enough — _compute_threat must call it with the real
    settlement-match, population, and is_active values, not compute
    wui_threat/priority straight off priority_for() the way it used to."""
    import inspect

    import geospatial.wui_analysis as wui_analysis_src

    src = inspect.getsource(wui_analysis_src._compute_threat)
    assert "raw_priority = priority_for(distance_m, eta_hours)" in src
    assert "has_named_place = bool(settlement and settlement.get(\"name\"))" in src
    assert "is_populated = is_meaningfully_populated(population_at_target)" in src
    assert (
        "priority = gate_critical_priority(\n"
        "        raw_priority, has_named_place, is_populated, bool(context.get(\"is_active\")),\n"
        "    )"
    ) in src


def test_population_is_only_sampled_when_it_could_change_the_verdict():
    """WorldPop sampling is a local raster read, not free — must only
    run when raw_priority is already CRITICAL and no named place has
    already settled the question, not unconditionally for every fire."""
    import inspect

    import geospatial.wui_analysis as wui_analysis_src

    src = inspect.getsource(wui_analysis_src._compute_threat)
    guard = src.split("is_populated = False")[1].split("priority = gate_critical_priority")[0]
    assert 'if raw_priority == "CRITICAL_AIR_TANKER_DISPATCH" and not has_named_place:' in src
    assert "sample_population_at_point(" in guard


def test_read_context_supplies_is_active():
    """_read_context must actually populate the is_active key
    _compute_threat's gate reads — not just have the gate function exist
    unused."""
    import inspect

    import geospatial.wui_analysis as wui_analysis_src

    src = inspect.getsource(wui_analysis_src._read_context)
    assert "from temporal.status import is_fire_active" in src
    assert '"is_active": is_active' in src


# --- settlement_matches_builtup_pixel --------------------------------------
# Regression coverage for a real bug found via a live cluster: fire 80
# (Odisha, next to Tata Steel Meramandali) measured 6.6 m from the
# nearest ESA WorldCover "built-up" pixel — almost certainly the plant's
# own paved yard, since WorldCover does not distinguish industrial
# pavement from a village. The dashboard was showing that measurement
# labelled "Motunga", a real OSM-tagged village — but Motunga is
# actually 5.1 km away, found only because the settlement-name search
# runs over a much larger 15 km radius than the built-up-pixel search.
# The two searches are independent and can find two different places;
# only trust the name when it's close enough to be the same one.

def test_the_real_motunga_mismatch_is_rejected():
    """The exact real-world numbers that exposed this bug."""
    settlement = {"name": "Motunga", "distance_m": 5147.5}
    assert settlement_matches_builtup_pixel(settlement, 6.6) is False


def test_a_settlement_at_the_same_distance_as_the_builtup_pixel_matches():
    settlement = {"name": "Dhenkanal", "distance_m": 42.0}
    assert settlement_matches_builtup_pixel(settlement, 42.0) is True


def test_a_settlement_within_one_villages_width_beyond_the_pixel_matches():
    """A settlement's OSM node marks roughly its centre, so its own
    distance from the fire is allowed to exceed the built-up-pixel
    distance by up to SETTLEMENT_NAME_MAX_EXCESS_M — the pixel is the
    edge of the built extent, the node can be further inside it."""
    settlement = {"name": "Realvillage", "distance_m": 200.0 + SETTLEMENT_NAME_MAX_EXCESS_M}
    assert settlement_matches_builtup_pixel(settlement, 200.0) is True


def test_a_settlement_just_beyond_the_excess_margin_does_not_match():
    settlement = {"name": "Toofar", "distance_m": 200.0 + SETTLEMENT_NAME_MAX_EXCESS_M + 0.1}
    assert settlement_matches_builtup_pixel(settlement, 200.0) is False


def test_no_settlement_found_never_matches():
    assert settlement_matches_builtup_pixel(None, 6.6) is False


def test_missing_distances_fail_closed_not_open():
    """A missing distance on either side must read as 'not the same
    place', not be treated as an automatic match — an unlabelled
    measurement is honest, a wrongly-labelled one is not."""
    assert settlement_matches_builtup_pixel({"name": "X", "distance_m": None}, 6.6) is False
    assert settlement_matches_builtup_pixel({"name": "X", "distance_m": 6.6}, None) is False


# --- gating reuses the industrial corridor's own class list --------------

def test_wui_gating_reuses_corridor_spreading_classes():
    """WUI evaluation must apply to exactly the classes the industrial
    threat corridor already treats as advancing — a wildfire and an
    industrial fire should never disagree about which classes 'spread'.
    """
    assert "wildfire" in SPREADING_CLASSES
    assert "agricultural_burning" in SPREADING_CLASSES
    assert "industrial_fire" not in SPREADING_CLASSES
    assert "gas_flare" not in SPREADING_CLASSES
    assert "mining" not in SPREADING_CLASSES


# --- crown-fire FRP threshold ---------------------------------------------

def _row(hour_offset, frp):
    import datetime

    base = datetime.datetime(2026, 1, 1, 0, 0)
    return {"acq_datetime": base + datetime.timedelta(hours=hour_offset), "frp": frp}


def test_absolute_frp_threshold_alone_triggers_crown_fire():
    result = evaluate_crown_fire(CROWN_FIRE_FRP_THRESHOLD_MW, [])
    assert result["is_crown_fire"] is True
    assert result["frp_acceleration_mw_per_hour"] is None


def test_below_threshold_and_no_history_is_not_crown_fire():
    result = evaluate_crown_fire(CROWN_FIRE_FRP_THRESHOLD_MW - 1, [])
    assert result["is_crown_fire"] is False


def test_rate_of_change_alone_triggers_crown_fire_below_absolute_threshold():
    rows = [_row(0, 10.0), _row(1, 10.0 + CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR)]
    frp_max = max(r["frp"] for r in rows)
    assert frp_max < CROWN_FIRE_FRP_THRESHOLD_MW
    result = evaluate_crown_fire(frp_max, rows)
    assert result["is_crown_fire"] is True
    assert result["frp_acceleration_mw_per_hour"] == pytest.approx(
        CROWN_FIRE_FRP_RATE_THRESHOLD_MW_PER_HOUR
    )


def test_a_cooling_fire_reports_no_acceleration_not_a_negative_one():
    """A fire that flared and is now dying down is not 'accelerating' in
    the sense this module cares about — the steepest *rise* is what
    matters, and a monotonically falling series has none."""
    rows = [_row(0, 80.0), _row(1, 20.0)]
    rate = frp_acceleration_mw_per_hour(rows)
    assert rate is None


def test_a_single_detection_has_no_rate_but_can_still_trip_the_threshold():
    """One lone very-hot pixel has nothing to compute a rate against, but
    must still be judged on the absolute threshold — the conservative
    direction to be wrong in."""
    result = evaluate_crown_fire(200.0, [_row(0, 200.0)])
    assert result["frp_acceleration_mw_per_hour"] is None
    assert result["is_crown_fire"] is True


def test_crown_fire_is_not_gated_by_spreading_class():
    """Unlike WUI, crown-fire detection applies to every class — an
    industrial fire radiating at crown-fire intensity is exactly as
    urgent as a wildfire doing the same. Nothing in evaluate_crown_fire
    takes a predicted_class argument at all, which is itself the
    guarantee; this test documents that as intentional."""
    import inspect

    params = inspect.signature(evaluate_crown_fire).parameters
    assert "predicted_class" not in params


# --- the dashboard hooks ---------------------------------------------------

@pytest.fixture(scope="module")
def html():
    return STATIC.read_text(encoding="utf-8")


def test_wui_fields_flow_from_the_gateway_into_the_fire_object(html):
    """The dashboard must read wui_threat etc. from the API response,
    not compute or guess them client-side."""
    block = html.split("function toFire(d) {")[1].split("function setBanner")[0]
    for field in ("d.wui_threat", "d.wui_distance_m", "d.wui_eta_hours",
                  "d.wui_bearing_deg", "d.wui_threatened_asset"):
        assert field in block, f"toFire() does not read {field} from the API"


def test_wui_overlay_never_uses_math_random(html):
    """The whole page already bans Math.random (test_gateway.py); this
    reasserts it specifically inside the WUI functions, since a pulsing
    ring or a random jitter would be the easiest place to accidentally
    reach for it."""
    for fn_name in ("updateWuiPanel", "updateWuiQuadChart",
                     "plotGlobePoints", "plotLeafletMarkers"):
        assert f"function {fn_name}" in html, f"{fn_name} missing"
    assert "Math.random" not in html


def test_wui_panel_hooks_exist(html):
    for marker in ('id="rp-wui-list"', 'id="wui-quad-chart"', "function updateWuiPanel"):
        assert marker in html, f"WUI hook missing: {marker}"


def test_the_header_critical_banner_was_removed_on_purpose(html):
    """The operator asked for the top-of-page "CRITICAL ESCALATION: AIR
    TANKER DISPATCH RECOMMENDED" strip to go — the sidebar badge/list/
    ring overlay still surface the same threat, this was specifically
    the header banner."""
    assert 'id="wui-banner"' not in html
    assert "function updateWuiBanner" not in html
    assert "CRITICAL ESCALATION" not in html


def test_wui_list_reads_visfires_an_absolute_filter(html):
    """By later request, the Control Panel's type/date/confidence
    checkboxes became an absolute filter — an operator who unticks a
    class must lose it from the ops list too, same as the map dot. The
    ops list reads visFires directly (which already respects filt.state
    via inStateScope — see test_state_scope_leak.py), not a separate
    allFires+scope re-filter."""
    panel_fn = html.split("function updateWuiPanel()")[1].split("function updateWuiQuadChart")[0]
    assert "const scoped = visFires;" in panel_fn
    assert "scoped.filter(f => f.wuiThreat)" in panel_fn


def test_ember_jump_ring_uses_the_real_threshold(html):
    """The drawn 2.4 km ring must be the literal ember-jump distance
    `geospatial/wui_analysis.py` evaluates against, not an arbitrary
    decorative radius that could silently drift out of sync with it.

    Drawn in `drawWuiOverlay`, the helper both `plotLeafletMarkers` (for
    a threat whose own marker survived the sidebar filter) and the
    always-visible fallback pass (for one that didn't) call — the ring
    is decorative-only in neither case."""
    overlay_fn = html.split("function drawWuiOverlay(")[1].split("function drawCrownOnlyMarker")[0]
    assert "radius: 2400" in overlay_fn
    assert "drawWuiOverlay(" in html.split("function plotLeafletMarkers()")[1].split("function applyFilt")[0]


def test_live_update_handles_wui_alert_push(html):
    live = html.split("function initLiveUpdates()")[1].split("function initStateFilter")[0]
    assert "WUI_THREAT_ALERT" in live


def test_wui_ember_jump_assessment_card_removed_from_the_detail_panel(html):
    """Removed by request: the boxed 'WUI Ember-Jump Assessment' card
    (target asset / distance / ETA / priority rows plus the 'Trigger Air
    Tanker Dispatch' download button) is gone from the fire detail panel
    entirely. The underlying wuiThreat/wuiDistanceM/etc. fields and the
    sidebar WUI Threats list are untouched — only this one card and its
    honest-record-download button went."""
    assert 'id="wui-card"' not in html
    assert 'id="btn-wui-dispatch"' not in html
    assert "function downloadWuiDispatch" not in html
    assert "WUI Ember-Jump Assessment" not in html
    assert "Trigger Air Tanker Dispatch" not in html


# --- crown fire dashboard hooks ---------------------------------------------

def test_crown_fire_field_flows_from_the_gateway(html):
    block = html.split("function toFire(d) {")[1].split("function setBanner")[0]
    assert "d.is_crown_fire" in block


# --- ETA reads as minutes under an hour, combined "1h 30min" over -----

def test_fmt_wui_eta_helper_switches_to_minutes_under_an_hour(html):
    fn = html.split("function fmtWuiEta(hours)")[1].split("\n    function ")[0]
    assert "hours < 1" in fn
    assert "Math.round(hours * 60) + ' min'" in fn


def test_fmt_wui_eta_helper_combines_hours_and_minutes_at_or_over_an_hour(html):
    """1.5h must read '1h 30min', not a bare '1.5h' — and an exact hour
    (or anything that rounds up to one, e.g. 1.999h) must not show a
    dangling '0min'."""
    fn = html.split("function fmtWuiEta(hours)")[1].split("\n    function ")[0]
    assert "const totalMin = Math.round(hours * 60);" in fn
    assert "const h = Math.floor(totalMin / 60);" in fn
    assert "const m = totalMin % 60;" in fn
    assert "m === 0 ? h + 'h' : h + 'h ' + m + 'min'" in fn


def test_live_summary_wui_list_shows_eta_and_distance_together(html):
    """Regression: the list used to pick ETA *or* distance, whichever
    happened to be non-null, hiding the other real measurement. They
    are two independent values (spread-rate physics vs. nearest
    built-up boundary) and both must show, joined, not one at a time."""
    map_fn = html.split("wuiActive.map(f => {")[1].split("}).join('')")[0]
    assert "if (f.wuiEtaHours != null) parts.push(fmtWuiEta(f.wuiEtaHours));" in map_fn
    assert "if (f.wuiDistanceM != null) parts.push((f.wuiDistanceM / 1000).toFixed(1) + 'km');" in map_fn
    assert "parts.join(' · ')" in map_fn


def test_every_wui_eta_display_site_uses_the_shared_formatter(html):
    """Regression guard: a site left on the old fire.wuiEtaHours.toFixed(1)
    pattern would keep showing '0.3 h' for a 20-minute threat instead of
    going through fmtWuiEta()."""
    assert "wuiEtaHours.toFixed(1)" not in html
    # +1 for the function's own declaration/definition. The detail panel
    # and dispatch report call sites went with the removed WUI Ember-Jump
    # Assessment card (see test_wui_ember_jump_assessment_card_removed_
    # from_the_detail_panel) — warning tooltip and summary list remain.
    assert html.count("fmtWuiEta(") == 3


def test_crown_fire_frp_marker_and_summary_badges_exist(html):
    """The isolate-filter checkboxes were removed from the Control
    Panel by request — the badges in the Live Summary panel are now the
    only UI for these filters, so they (and the FRP marker) must still
    be present and wired (see test_crown_fire_and_wui_only_filters_are_wired_and_reset)."""
    for marker in ('id="frp-crown-mark"', 'id="badge-wui"', 'id="badge-crown"'):
        assert marker in html, f"missing: {marker}"


def test_control_panel_no_longer_has_the_isolate_filter_checkboxes(html):
    for marker in ('id="cb-crown-only"', 'id="cb-wui-only"', 'id="cb-rdi-only"'):
        assert marker not in html, f"should have been removed: {marker}"


def test_wui_and_crown_badges_both_stay_clickable_at_zero_unlike_rdi(html):
    """WUI Critical and Crown Fires used to disagree on this — Crown Fires
    was fixed to stay clickable at zero first, leaving WUI Critical
    greyed out and unclickable at zero on its own, which visually read
    as a mismatched, half-broken pair of buttons whenever one was at
    zero and the other wasn't (real bug report from a live screenshot:
    WUI Critical showing washed-out/disabled next to a solid Crown
    Fires). Both now stay clickable — opening either just shows its own
    existing "No active ..." empty state (rp-wui-list / rp-crown-list)
    instead of disabling. RDI keeps the old disabled-at-zero behaviour;
    it has no dedicated sidebar panel of its own to open."""
    assert "badgeCrown.disabled" not in html
    assert "badgeWui.disabled" not in html
    assert "badgeRdi.disabled = !rdiActive.length" in html


def test_crown_fire_and_wui_only_filters_are_wired_and_reset(html):
    assert "filt.crownOnly" in html
    assert "filt.wuiOnly" in html
    # Reset Filters must clear both, not just the five class checkboxes —
    # otherwise an isolated "crown fire only" view survives a reset.
    assert "crownOnly: false" in html
    assert "wuiOnly: false" in html


def test_isolate_filters_bypass_the_other_filters(html):
    """Regression: ticking 'WUI Threat Zone' AND'd with the other filters
    (class/confidence/date/FRP), so a real WUI threat that happened to be
    LOW confidence — exactly the case the checkbox exists for — ticked
    the box and still saw an empty map. The isolate switches must be a
    full override, not an additional narrowing condition."""
    apply_fn = html.split("function applyFilt()")[1].split("function updateWuiPanel")[0]
    isolate_block = apply_fn.split("if (filt.wuiOnly || filt.crownOnly || filt.rdiOnly)")[1].split("return true;")[0]
    assert "(filt.wuiOnly && f.wuiThreat) || (filt.crownOnly && f.isCrownFire)" in isolate_block
    # And it must be checked before the ordinary filters have a chance to
    # reject the fire — i.e. it appears before the confidence check.
    assert apply_fn.index("filt.wuiOnly || filt.crownOnly") < apply_fn.index("filt.conf.has")


def test_unchecking_every_sensor_date_hides_wui_too(html):
    """Regression: sensor-date is a data-inclusion filter — unchecking
    every date must empty the WUI/crown/RDI views too, not leave every
    warning on screen with the rest of the dashboard claiming zero
    detections. It's checked first in applyFilt(), before the isolate-
    switch override, and updateWuiPanel/updateWuiQuadChart now inherit
    it for free by reading visFires (built by that same applyFilt) —
    updateSuppressionPanel is a separate, still-allFires-based function
    (its own list was removed from the UI; it applies the filter
    directly since it isn't fed by applyFilt's visFires)."""
    assert "function passesSensorDateFilter(f)" in html
    apply_fn = html.split("function applyFilt()")[1].split("function updateWuiPanel")[0]
    # Checked first, before the isolate-switch override.
    assert apply_fn.index("if (!passesSensorDateFilter(f)) return false;") \
        < apply_fn.index("filt.wuiOnly || filt.crownOnly")
    for fn_name in ("updateWuiPanel", "updateWuiQuadChart"):
        fn = html.split("function " + fn_name + "(")[1].split("\n    function ")[0]
        assert "visFires" in fn, f"{fn_name} does not derive from the filtered visFires"
    supp_fn = html.split("function updateSuppressionPanel(")[1].split("\n    function ")[0]
    assert "passesSensorDateFilter" in supp_fn


def test_crown_fire_marker_disappears_when_its_class_is_unticked(html):
    """By later request, reversing the earlier 'always visible' rule:
    a crown-fire cluster is drawn only when visFires includes it — the
    Control Panel's type/date/confidence checkboxes are now an absolute
    filter, with no safety exception for a critical detection."""
    marker_fn = html.split("function plotLeafletMarkers()")[1].split("function passesSensorDateFilter")[0]
    assert "drawCrownOnlyMarker" not in marker_fn
    assert "visFires.forEach(fire => {" in marker_fn
    assert "allFires" not in marker_fn
    assert "fire.isCrownFire && filt.showCrown" in marker_fn


def test_crown_fire_frp_row_is_highlighted_in_the_detail_panel(html):
    open_dp = html.split("function openDP(fire)")[1].split("async function loadEventDetail")[0]
    assert "isCrownFire" in open_dp
    assert "CROWN FIRE THRESHOLD EXCEEDED" in open_dp


def test_wui_and_crown_checkboxes_show_a_count_like_every_other_type_row(html):
    """Request: the WUI Threat / Crown Fire checkboxes should show a
    count, same as Wild/Industrial/Gas/Agricultural/Mining above them.
    Reuses the existing cnt-wui/cnt-crown ids updateWuiPanel() already
    populates from wuiActive/crownActive (state+date scoped, matching
    the 'always visible regardless of class/confidence filters'
    semantics every other WUI/crown count already uses) rather than
    introducing a second, differently-scoped counter."""
    src = html
    ac_types = src.split('id="ac-types"')[1].split("</div>\n        </div>")[0]
    assert 'class="ft-cnt" id="cnt-wui">0</span>' in ac_types
    assert 'class="ft-cnt" id="cnt-crown">0</span>' in ac_types

    panel_fn = src.split("function updateWuiPanel()")[1].split("function updateWuiQuadChart")[0]
    assert "cntWui.textContent = wuiActive.length" in panel_fn
    assert "cntCrown.textContent = crownActive.length" in panel_fn


def test_unnamed_builtup_wording_removed_everywhere(html):
    """Removed by request: 'unnamed built-up' cluttered the WUI ⚠️
    tooltip and every other place a threatened asset without a real OSM
    settlement name falls back to a district/state-derived label —
    geospatial/wui_analysis.py (the source of fire.wuiThreatenedAsset),
    the alert-feed fallback in routes_dashboard.py, and the dispatch
    report fallback in static/index.html all used the exact phrase.
    (The frontend's compatibility regex that strips a legacy cached
    value is the one legitimate remaining mention — see
    test_stale_cached_wui_asset_names_are_cleaned_up_client_side — so
    this only checks the quoted string literals a user would see.)"""
    import inspect

    import gateway.routes_dashboard as routes_dashboard_src
    import geospatial.wui_analysis as wui_analysis_src

    for module in (wui_analysis_src, routes_dashboard_src):
        assert "unnamed built-up" not in inspect.getsource(module)
    assert "'unnamed built-up area'" not in html
    assert '"unnamed built-up area"' not in html


def test_stale_cached_wui_asset_names_are_cleaned_up_client_side(html):
    """Fixing the source string doesn't retroactively fix clusters whose
    wui_threatened_asset was already computed and cached in the DB
    before this wording change — the frontend must strip a legacy
    "unnamed built-up " prefix on the way in too, so an old record
    displays the same shorter form without a backend recompute. And it
    must re-capitalize what's left ("area near X" -> "Area near X"),
    since the tooltip/list start the sentence with this word — matches
    what geospatial/wui_analysis.py now produces directly for new
    clusters ("Area near X")."""
    fn = html.split("wuiThreatenedAsset: d.wui_threatened_asset")[1].split("isCrownFire:")[0]
    assert "replace(/^unnamed built-up area/, 'Area')" in fn

    import inspect

    import geospatial.wui_analysis as wui_analysis_src
    assert '"Area near {0}"' in inspect.getsource(wui_analysis_src)
