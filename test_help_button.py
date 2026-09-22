"""The "i" help button: a floating, always-available button in the
bottom-right of the dashboard (#app) that opens a modal explaining how
to navigate the site. No browser — same pattern as test_gateway.py /
test_wui.py: assert the hooks a live page needs are present in
static/index.html and that the styling actually reuses the app's own
design tokens rather than inventing new ones.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_help_button_exists_inside_the_inner_app_not_the_login_screen():
    src = html()
    app_body = src.split('<div id="app">')[1].split("\n  <script")[0]
    assert 'id="help-btn"' in app_body
    assert 'id="help-backdrop"' in app_body
    assert 'id="help-modal"' in app_body

    login_body = src.split('<div id="login-screen">')[1].split('<div id="app">')[0]
    assert 'id="help-btn"' not in login_body


def test_help_button_is_pinned_to_the_map_not_the_whole_viewport():
    """Real bug: a viewport-fixed button at 'right: 16px' sat directly
    on top of the Live Summary panel's own content (e.g. the Data
    Sources 'Engine' row) whenever #rp was open, since #rp reserves
    real layout width rather than overlaying the map. It must be
    absolutely positioned inside #gw (position:relative) instead, so
    it can never collide with #rp regardless of that panel's state."""
    src = html()
    gw_body = src.split('<div id="gw">')[1].split('<button id="rp-tog"')[0]
    assert 'id="help-btn"' in gw_body
    assert 'id="help-backdrop"' not in gw_body  # the modal stays viewport-wide

    rule = src.split("#help-btn {")[1].split("}")[0]
    assert "position: absolute" in rule
    assert "right: 16px" in rule
    assert "bottom: 16px" in rule


def test_help_button_hides_behind_the_fire_detail_panel_not_on_top_of_it():
    """Real bug found live: the button used to sit above #dp (z-index
    1000 > #dp's 999), so opening a fire's detail floated it directly
    on top of the Suppression Resource Estimate card's own text (Fleet
    Needed / Air Support). #dp is opaque and covers this exact corner
    of #gw when open, so the button's z-index must be below #dp's — it
    is then naturally hidden behind the panel while it's open and
    reappears on its own once it closes, with no JS toggle needed."""
    src = html()
    rule = src.split("#help-btn {")[1].split("}")[0]
    z_index = int(rule.split("z-index:")[1].split(";")[0].strip())
    dp_rule = src.split("\n    #dp {")[1].split("}")[0]
    dp_z_index = int(dp_rule.split("z-index:")[1].split(";")[0].strip())
    assert z_index < dp_z_index


def test_help_button_reuses_the_app_blue_not_a_new_colour():
    """Same #0014a8 as the header, footer, and the WUI Critical / Crown
    Fires badges — not an arbitrary new accent colour for a one-off
    button. Colours swapped by request: white fill, blue "i" and
    border, rather than a solid blue fill."""
    src = html()
    rule = src.split("#help-btn {")[1].split("}")[0]
    assert "background: #fff" in rule
    assert "color: #0014a8" in rule
    assert "border: 2px solid #0014a8" in rule


def test_help_button_has_no_colored_glow_shadow():
    """The dark-glow AI-slop pattern this project has already been
    caught on once (the SMS button's box-shadow) — a neutral elevation
    shadow only, no colour tint."""
    src = html()
    rule = src.split("#help-btn {")[1].split("}")[0]
    assert "box-shadow" in rule
    shadow_line = rule.split("box-shadow:")[1].split(";")[0]
    assert "rgba(15, 23, 42" in shadow_line  # neutral dark
    assert "#0014a8" not in shadow_line and "0, 20, 168" not in shadow_line


def test_help_button_uses_the_info_circle_icon_not_a_text_letter():
    """Request: swap the plain "i" text glyph for a real info-circle
    icon (matching the supplied reference), recoloured to the button's
    own scheme — fill="currentColor" so it always matches whatever
    color the button itself is set to, rather than a second hardcoded
    colour that could drift from it."""
    src = html()
    btn = src.split('id="help-btn"')[1].split("</button>")[0]
    assert "<svg" in btn
    assert 'fill="currentColor"' in btn
    assert "<circle" in btn  # the dot
    assert ">i<" not in btn


def test_help_modal_reuses_the_fire_detail_header_and_section_title_classes():
    """Request: font, size, colour must match the rest of the prototype
    — achieved by literally reusing .dp-hdr/.dp-hdr-lbl (Fire Detail's
    own header) and .cls-title (every section heading in the app), not
    a parallel set of near-identical classes."""
    src = html()
    modal = src.split('id="help-modal"')[1].split("</div>\n    </div>\n  </div>")[0]
    assert 'class="dp-hdr"' in modal
    assert 'class="dp-hdr-lbl"' in modal
    assert modal.count('class="cls-title"') >= 5  # one per help section


def test_help_body_paragraphs_use_the_app_font_and_black_text_explicitly():
    """Request: body text must match the right panel's font/colour.
    Hardcoded literal values with !important (not var(--font)/var(--t2)
    indirection) so a rendering environment where the custom-property
    cascade doesn't apply as expected still gets the right font/colour
    on a fresh load — same font family/colour the rest of the app uses,
    just spelled out rather than routed through a variable."""
    src = html()
    rule = src.split(".help-sec p {")[1].split("}")[0]
    assert "font-family: 'Space Grotesk', sans-serif !important" in rule
    assert "color: #000000 !important" in rule


def test_help_modal_content_names_the_real_panels_not_fabricated_ones():
    """Every section referenced must be a real, existing feature —
    checked against markers that only exist if those features do."""
    src = html()
    modal = src.split('id="help-modal"')[1].split("</div>\n    </div>\n  </div>")[0]
    for real_feature in (
        "Fire Classification", "Sensor Data", "Filters", "Map View",
        "My State", "All India", "WUI Threat", "Crown Fire",
        "Classification Breakdown", "SHAP", "suppression",
    ):
        assert real_feature in modal, f"help text references a feature that doesn't appear: {real_feature}"


def test_help_button_is_wired_open_close_backdrop_and_escape():
    src = html()
    fn = src.split("function initHelp()")[1].split("\n    }")[0]
    assert "btn.addEventListener('click', open)" in fn
    assert "help-close').addEventListener('click', close)" in fn
    assert "backdrop.addEventListener('click'" in fn
    assert "e.target === backdrop" in fn  # only the dim area closes it, not the card
    assert "e.key === 'Escape'" in fn


def test_help_button_is_included_in_the_dashboard_ui_reveal():
    """Like every other chrome element, it must stay invisible until
    enterDashboard() reveals .dashboard-ui elements — not appear over
    the login screen."""
    src = html()
    assert 'id="help-btn" class="dashboard-ui"' in src


def test_help_init_is_actually_called_on_bootstrap():
    src = html()
    boot = src.split("window.addEventListener('load'")[1].split("\n    });")[0]
    assert "initHelp();" in boot


def test_help_button_never_fabricates_data():
    assert "Math.random" not in html()
