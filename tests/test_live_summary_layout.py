"""Live Summary (right panel) layout, per an explicit request:

1. Total-fires + Classification Breakdown at the top.
2. Exactly two toggle boxes below that — WUI and Crown Fire (the third,
   Full-Suppression/RDI, is gone — see test_resource_demand.py).
3. Clicking a toggle shows ONLY that box's own content below it;
   clicking neither (the default) shows Data Sources there instead.
4. No emojis anywhere in the panel; same existing style/colour scheme,
   no new theme.

Same "no browser, just assert the hooks a live page needs" pattern as
test_wui.py / test_gateway.py.
"""
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def _rp_block():
    src = html()
    return src.split('<div id="rp" class="dashboard-ui">')[1].split('<div id="bb"')[0]


def test_total_and_breakdown_come_before_the_toggle_boxes():
    rp = _rp_block()
    total_idx = rp.index('id="st-total"')
    breakdown_idx = rp.index("Classification Breakdown")
    badges_idx = rp.index('class="big-badges"')
    assert total_idx < badges_idx
    assert breakdown_idx < badges_idx


def test_exactly_two_toggle_boxes_wui_and_crown():
    rp = _rp_block()
    assert 'id="badge-wui"' in rp
    assert 'id="badge-crown"' in rp
    assert 'id="badge-rdi"' not in rp
    badges_block = rp.split('class="big-badges"')[1].split("</div>")[0]
    assert badges_block.count('class="big-badge') == 2


def test_wui_and_crown_sections_start_hidden_data_sources_starts_shown():
    rp = _rp_block()
    wui_sec = rp.split('id="rp-wui-sec"')[1].split(">")[0]
    crown_sec = rp.split('id="rp-crown-sec"')[1].split(">")[0]
    ds_sec = rp.split('id="rp-datasources-sec"')[1].split(">")[0]
    assert "display:none" in wui_sec
    assert "display:none" in crown_sec
    assert "display:none" not in ds_sec


def test_clicking_a_badge_toggles_its_section_and_hides_data_sources():
    src = html()
    fn = src.split("function updateRpPanelVisibility()")[1].split("\n    function ")[0]
    assert "rpActivePanel === 'wui'" in fn
    assert "rpActivePanel === 'crown'" in fn
    assert "dsSec.style.display = rpActivePanel ? 'none' : 'block'" in fn


def test_clicking_the_active_badge_again_turns_it_off():
    """A single-select toggle, not a one-way switch: rpActivePanel flips
    back to null (showing Data Sources again) on a second click of the
    same badge, and both filt.wuiOnly/crownOnly follow it exactly."""
    src = html()
    fn = src.split("function setRpActivePanel(panel)")[1].split("\n      }")[0]
    assert "rpActivePanel = rpActivePanel === panel ? null : panel;" in fn
    assert "filt.wuiOnly = rpActivePanel === 'wui';" in fn
    assert "filt.crownOnly = rpActivePanel === 'crown';" in fn


def test_top_states_section_removed_along_with_its_dead_code():
    """Regression guard for the crash this would otherwise cause:
    buildTS() used to do `document.getElementById('ts-list').innerHTML`
    with no null guard — removing the HTML without also removing the
    function/call would throw on every updateStats() call."""
    src = html()
    assert 'id="ts-list"' not in src
    assert "function buildTS" not in src
    assert "buildTS()" not in src


_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)


def test_no_emojis_anywhere_in_the_right_panel():
    rp = _rp_block()
    found = _EMOJI_PATTERN.findall(rp)
    assert not found, f"emoji(s) left in the right panel: {found}"


def test_no_emoji_escapes_in_the_badge_and_section_labels():
    rp = _rp_block()
    for escape in ("&#x1F6A8;", "&#x1F525;", "&#x1F6E9;", "&#xFE0F;"):
        assert escape not in rp, f"leftover emoji escape in right panel: {escape}"


def test_right_panel_still_uses_the_same_style_classes_not_a_new_theme():
    """Reorganised, not redesigned: the panel keeps rp-sec/rp-title/
    big-badge and reuses rp-wui-item/rp-wui-empty for the new crown
    list, rather than introducing parallel one-off classes."""
    rp = _rp_block()
    assert 'class="rp-sec' in rp
    assert 'class="big-badge' in rp
    assert 'id="rp-crown-list"' in rp
    src = html()
    fn = src.split("function updateWuiPanel()")[1].split("function updateWuiQuadChart")[0]
    assert "'<div class=\"rp-wui-item\" data-cid=\"' + f.clusterId" in fn


def test_crown_list_never_fabricates_a_name_or_value():
    src = html()
    fn = src.split("function updateWuiPanel()")[1].split("function updateWuiQuadChart")[0]
    crown_block = fn.split("getElementById('rp-crown-list')")[1].split("crownList.querySelectorAll")[0]
    assert "f.name || ('Cluster ' + f.clusterId)" in crown_block
    assert "f.frp != null ? f.frp.toFixed(1) + ' MW' : '--'" in crown_block
