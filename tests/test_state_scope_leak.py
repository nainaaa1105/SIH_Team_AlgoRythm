"""A state-role login's account boundary must hold everywhere filt.state
is supposed to apply — the map, the WUI/crown/RDI lists and badges, the
globe rings, and the quad chart.

Historically the WUI/crown/RDI views deliberately bypassed the ordinary
class/confidence/date filters so a critical threat could never be
filtered into invisibility — see test_wui.py's older revisions. By
explicit request that bypass was removed: the Control Panel's type/date/
confidence checkboxes are now an absolute filter, and everything derives
from the single visFires array instead of a second allFires+scope
re-filter. This file guards that filt.state — a state account's hard
access boundary, not an ordinary narrowing filter — still holds under
that simpler design, including in isolate mode (the WUI Critical / Crown
Fires badges).
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_in_state_scope_helper_exists_and_is_permissive_for_admins():
    src = html()
    fn = src.split("function inStateScope(f)")[1].split("\n")[0]
    assert "!filt.state" in fn
    assert "f.state === filt.state" in fn


def test_globe_rings_read_visfires_an_absolute_filter():
    """The pulsing globe rings around WUI/crown-fire clusters now read
    visFires (which already respects filt.state via inStateScope, and
    every ordinary type/date/confidence checkbox), same as pointsData
    above it — not a separate allFires+scope re-filter that could
    disagree with what the dots show. Still gated by the Control Panel's
    WUI Threat / Crown Fire show/hide checkboxes (filt.showWui/
    filt.showCrown) for the ring itself."""
    src = html()
    fn = src.split("function plotGlobePoints()")[1].split("function destPoint")[0]
    assert ".pointsData(visFires)" in fn
    assert ".ringsData(visFires.filter(d => (d.wuiThreat && filt.showWui) || (d.isCrownFire && filt.showCrown)))" in fn


def test_leaflet_markers_read_only_visfires_no_bypass():
    """By request: unchecking every Fire Classification checkbox must
    hide every fire from the map, including WUI-critical and crown-fire
    ones — no exceptions. The old criticalBypassFires() safety override
    (which kept a WUI/crown-critical fire visible even past an unticked
    type/confidence filter) is removed entirely; plotLeafletMarkers only
    ever draws from visFires now."""
    src = html()
    assert "function criticalBypassFires" not in src
    assert "drawCrownOnlyMarker" not in src

    fn = src.split("function plotLeafletMarkers()")[1].split("function passesSensorDateFilter")[0]
    assert "visFires.forEach(fire => {" in fn
    assert "allFires.filter(inStateScope)" not in fn


def test_wui_crown_rdi_lists_and_badges_read_visfires():
    src = html()
    fn = src.split("function updateWuiPanel()")[1].split("function updateWuiQuadChart")[0]
    assert "const scoped = visFires;" in fn
    assert "scoped.filter(f => f.wuiThreat)" in fn
    assert "scoped.filter(f => f.isCrownFire)" in fn
    assert "scoped.filter(f => f.coaType" in fn


def test_isolate_mode_itself_respects_state_scope():
    """Real bug found live: a Maharashtra state login (36 real detections,
    22 WUI-critical) clicking the WUI Critical badge saw 228 — the full
    India WUI-critical count. applyFilt's isolate branch
    (filt.wuiOnly/crownOnly/rdiOnly) returned straight from
    passesSensorDateFilter without ever checking filt.state, unlike the
    ordinary narrowing branch below it which does — so a state account's
    own access boundary only held when isolate mode was off. The isolate
    branch must reject anything outside inStateScope() before checking
    wuiThreat/isCrownFire/coaType."""
    src = html()
    apply_fn = src.split("function applyFilt() {")[1].split("\n    function ")[0]
    isolate_block = apply_fn.split("if (filt.wuiOnly || filt.crownOnly || filt.rdiOnly) {")[1].split("return true;")[0]
    assert "if (!inStateScope(f)) return false;" in isolate_block
    # Must be checked before the wuiThreat/isCrownFire/coaType return, not after.
    assert isolate_block.index("inStateScope(f)") < isolate_block.index("filt.wuiOnly && f.wuiThreat")


def test_quad_chart_reads_visfires():
    src = html()
    combined = src.split("function updateWuiQuadChart()")[1][:2000]
    assert "const points = visFires" in combined
    assert "f.wuiDistanceM" in combined


def test_admin_login_is_unaffected_since_filt_state_is_empty():
    """inStateScope must be a no-op (everything passes) whenever
    filt.state is unset — an administrator, or a state account before
    login sets it, must not lose any fires to this guard."""
    src = html()
    fn = src.split("function inStateScope(f)")[1].split("\n")[0]
    assert fn.strip().startswith("{ return !filt.state ||")


def test_state_scope_never_uses_math_random():
    assert "Math.random" not in html()


# --- Live Summary stats must agree with what the map draws ---------------

def test_classification_breakdown_reads_visfires_no_effFires():
    """By request, checkboxes became an absolute filter: unticking a
    class must zero it out of the Classification Breakdown and Live
    Summary total too, not just hide the dot while the count secretly
    still includes it. The old effFires (visFires + a safety bypass) is
    gone — every stat reads visFires directly, at every scope, since My
    State and All India both derive visFires from the same
    filt.state-aware pipeline."""
    src = html()
    assert "effFires" not in src

    stats_fn = src.split("function updateStats() {")[1].split("\n    function ")[0]
    assert "visFires.length" in stats_fn
    assert "visFires.forEach(f => { if (f.type" in stats_fn
