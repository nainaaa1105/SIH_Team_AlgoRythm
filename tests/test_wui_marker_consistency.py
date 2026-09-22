"""A WUI-critical fire's marker went through several design iterations
in response to real user reports:

1. Originally, a WUI-critical fire whose own marker the sidebar's class/
   confidence/date filters would have hidden was drawn as a hardcoded
   solid-blue circleMarker instead of its real classification colour,
   with its own custom tooltip text instead of the format every other
   point uses — exactly what a real cluster (agricultural_burning, LOW
   confidence, WUI-critical) hit in practice.

2. The fix for #1 layered a ⚠️ badge (drawWuiWarningIcon) on top of the
   fire's real classification dot (drawClassifiedMarker) at the fire's
   own coordinates — correct colour, but two overlapping shapes at one
   point, which looked cluttered on zoom-in.

3. Next design: the fire keeps its ordinary classification dot, same as
   any other point (no replacement, no layering). The ⚠️ badge moves to
   the actual threatened settlement — the real point the server's
   bearing/distance measurement points at (the same destPoint() the
   projection line's endpoint already uses), not the fire's own
   location. No bearing/distance measurement means no known location
   for the badge, so it is skipped rather than drawn somewhere guessed.
   A WUI-critical or crown-fire cluster whose sidebar filters would
   otherwise have hidden it stayed visible anyway (criticalBypassFires),
   a deliberate safety override.

4. Current design, by explicit later request: that safety override is
   removed. The Control Panel's type/date/confidence checkboxes are now
   an absolute filter — unticking a class hides every fire of that
   class, including WUI-critical and crown-fire ones, no exceptions.
   plotLeafletMarkers only ever draws from visFires.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_classified_marker_is_a_single_shared_source_of_colour_and_tooltip():
    src = html()
    assert "function drawClassifiedMarker(fire)" in src
    fn = src.split("function drawClassifiedMarker(fire)")[1].split("function drawWuiWarningIcon")[0]
    assert "FT[fire.type]?.color" in fn
    assert "${FT[fire.type]?.icon} ${FT[fire.type]?.label} — ${fire.state}" in fn


def test_wui_overlay_draws_the_fires_own_ordinary_dot():
    """The fire itself must look like any other point — same colour,
    same shape — not a badge and not a hardcoded stand-in colour."""
    src = html()
    fn = src.split("function drawWuiOverlay(fire)")[1].split("function plotLeafletMarkers")[0]
    assert "const m = drawClassifiedMarker(fire);" in fn
    assert "radius: 7, fillColor:" not in fn  # the old stand-in fire dot


def test_wui_overlay_no_longer_takes_a_markeralreadydrawn_flag():
    src = html()
    assert "function drawWuiOverlay(fire, markerAlreadyDrawn)" not in src
    assert "drawWuiOverlay(fire, true)" not in src
    assert "drawWuiOverlay(fire, false)" not in src


def test_wui_and_crown_fallback_tooltips_match_ordinary_markers():
    """Regression guard: hovering a WUI-critical or crown-fire fire's
    own marker must show the exact same tooltip format as any other
    point, not a bespoke 'WUI threat — ...' / 'Crown fire — ...'
    string."""
    src = html()
    assert "'&#x1F6A8; WUI threat — '" not in src
    assert "'&#x1F525; Crown fire — FRP '" not in src


def test_ordinary_visfires_loop_routes_wui_fires_through_the_overlay():
    src = html()
    fn = src.split("function plotLeafletMarkers()")[1].split("function passesSensorDateFilter")[0]
    assert "if (fire.wuiThreat) { drawWuiOverlay(fire); return; }" in fn
    assert "drawClassifiedMarker(fire)" in fn  # still used for the non-WUI branch


def test_no_bypass_route_remains_for_a_filtered_out_critical_fire():
    """By explicit later request, reversing the earlier 'critical stays
    visible regardless of filters' design: the criticalBypassFires()
    safety override is gone entirely, and plotLeafletMarkers only ever
    iterates visFires — an unticked type/date/confidence checkbox hides
    a WUI-critical or crown-fire detection exactly like any other."""
    src = html()
    assert "function criticalBypassFires" not in src
    assert "function drawCrownOnlyMarker" not in src

    fn = src.split("function plotLeafletMarkers()")[1].split("function passesSensorDateFilter")[0]
    assert fn.count("Fires.forEach") == 1  # only the one visFires.forEach loop
    assert "allFires" not in fn


# --- the ⚠️ badge lives on the threatened settlement, not the fire --------

def test_warning_icon_is_placed_at_the_threatened_point_not_the_fire():
    src = html()
    fn = src.split("function drawWuiWarningIcon(fire, atPoint)")[1].split("function drawCrownFireIcon")[0]
    assert "&#x26A0;&#xFE0F;" in fn  # ⚠️
    assert "L.divIcon" in fn
    assert "L.marker(atPoint" in fn
    assert "[fire.lat, fire.lon]" not in fn


def test_warning_icon_call_site_passes_the_real_destpoint_target():
    """The badge's coordinate must be the same destPoint() result the
    projection line's endpoint already uses — the real server-measured
    bearing/distance — not a second, independent computation."""
    src = html()
    fn = src.split("function drawWuiOverlay(fire)")[1].split("function plotLeafletMarkers")[0]
    assert "const target = destPoint(fire.lat, fire.lon, fire.wuiBearingDeg, fire.wuiDistanceM);" in fn
    assert "drawWuiWarningIcon(fire, target);" in fn


def test_warning_icon_is_never_drawn_without_a_known_target_location():
    """No bearing/distance means no known settlement location — the
    badge must not be drawn somewhere guessed. Also gated by the
    Control Panel's WUI Threat show/hide checkbox (filt.showWui)."""
    src = html()
    fn = src.split("function drawWuiOverlay(fire)")[1].split("function plotLeafletMarkers")[0]
    guarded = fn.split("if (filt.showWui && fire.wuiBearingDeg != null && fire.wuiDistanceM != null) {")[1]
    assert "drawWuiWarningIcon(fire, target);" in guarded
    unguarded_tail = fn.split("drawWuiWarningIcon(fire, target);")[1] if "drawWuiWarningIcon(fire, target);" in fn else ""
    assert "drawWuiWarningIcon(" not in unguarded_tail


def test_warning_icon_tooltip_describes_the_threatened_asset():
    src = html()
    fn = src.split("function drawWuiWarningIcon(fire, atPoint)")[1].split("function drawCrownFireIcon")[0]
    assert "fire.wuiThreatenedAsset" in fn


def test_ember_jump_ring_still_real_and_still_2400m():
    """The dashed circle itself is unchanged by this fix — it's the
    real 2.4 km ember-jump radius, not decoration."""
    src = html()
    fn = src.split("function drawWuiOverlay(fire)")[1].split("function plotLeafletMarkers")[0]
    assert "radius: 2400" in fn


def test_wui_marker_fix_never_uses_math_random():
    assert "Math.random" not in html()


# --- Control Panel show/hide checkboxes for the WUI/crown badges ---------

def test_control_panel_has_wui_and_crown_show_hide_checkboxes():
    """Request: alongside the fire-type checkboxes, add a checkbox for
    the WUI ⚠️ badge and one for the crown-fire 🔥 badge, each on by
    default (checked) like the existing type checkboxes."""
    src = html()
    ac_types = src.split('id="ac-types"')[1].split("</div>\n        </div>")[0]
    assert 'id="cb-show-wui" checked' in ac_types
    assert '&#x26A0;&#xFE0F;' in ac_types
    assert 'id="cb-show-crown" checked' in ac_types
    assert '&#x1F525;' in ac_types


def test_show_wui_and_show_crown_checkboxes_are_wired():
    src = html()
    fn = src.split("function initTypeCBs()")[1].split("\n    }")[0]
    assert "filt.showWui = showWuiCb.checked" in fn
    assert "filt.showCrown = showCrownCb.checked" in fn
    assert "applyFilt()" in fn


def test_filt_defaults_and_reset_both_start_wui_and_crown_visible():
    src = html()
    assert "showWui: true" in src
    assert "showCrown: true" in src
    reset_fn = src.split("btn-reset-f').addEventListener('click'")[1].split("initDetailCtrls")[0]
    assert "showWuiCb.checked = true" in reset_fn
    assert "showCrownCb.checked = true" in reset_fn


# --- crown fire gets a real 🔥 badge, same as WUI's ⚠️ ---------------------

def test_crown_fire_gets_its_own_fire_emoji_badge_like_wui():
    """Request: crown fires must work the same way as WUI — a real icon
    shown on the map (drawCrownFireIcon), not just the pre-existing CSS
    glow, which stays as an extra visual cue rather than being removed."""
    src = html()
    fn = src.split("function drawCrownFireIcon(fire)")[1].split("function markCrownFire")[0]
    assert "&#x1F525;" in fn  # 🔥
    assert "L.divIcon" in fn
    assert "[fire.lat, fire.lon]" in fn  # at the fire itself, no separate target point


def test_mark_crown_fire_is_the_single_shared_application_point():
    """The glow + badge combo must be applied identically wherever a
    crown fire is drawn — the ordinary loop and the WUI-overlay branch
    (the third, filters-bypass call site was removed along with
    criticalBypassFires) — via one shared helper, not hand-rolled
    copies that could drift."""
    src = html()
    assert src.count("markCrownFire(m, fire)") == 2  # ordinary loop, drawWuiOverlay
    fn = src.split("function markCrownFire(marker, fire)")[1].split("\n    }")[0]
    assert "wui-glowing-spot" in fn
    assert "crown-glow" in fn
    assert "drawCrownFireIcon(fire)" in fn


def test_crown_fire_badge_never_appears_when_show_crown_is_off():
    """filt.showCrown must gate the badge/glow at every remaining call
    site — the ordinary loop and drawWuiOverlay (the bypass path this
    test used to also check was removed along with criticalBypassFires,
    see test_no_bypass_route_remains_for_a_filtered_out_critical_fire)."""
    src = html()
    ordinary_fn = src.split("function plotLeafletMarkers()")[1].split("function passesSensorDateFilter")[0]
    assert "fire.isCrownFire && filt.showCrown" in ordinary_fn

    overlay_fn = src.split("function drawWuiOverlay(fire)")[1].split("function plotLeafletMarkers")[0]
    assert "fire.isCrownFire && filt.showCrown" in overlay_fn
