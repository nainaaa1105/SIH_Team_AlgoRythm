"""National vs state exploration for a state-role login.

Spec: an administrator lands on the all-India view with national stats.
A state account lands scoped to its own state — stats default to
state-level — but must be able to explore nationally on the map with
everything (markers, WUI/crown/RDI overlays) visible and working, not
permanently locked to its own state. #scope-toggle is that switch; it
rides entirely on the pre-existing filt.state / inStateScope() pipeline
rather than a second, parallel filtering mechanism.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_scope_toggle_exists_with_both_buttons():
    src = html()
    assert 'id="scope-toggle"' in src
    assert 'id="scope-state-btn"' in src
    assert 'id="scope-national-btn"' in src


def test_scope_toggle_hidden_by_default_and_only_shown_for_state_role():
    src = html()
    assert 'id="scope-toggle" style="display:none"' in src
    fn = src.split("function updateAuthUi()")[1].split("function setNationalScope")[0]
    assert "authRole === 'state' && authState" in fn
    assert "toggle.style.display = " in fn


def test_national_scope_clears_filt_state_not_a_second_mechanism():
    """Switching to All India must go through the exact same filt.state
    / applyFilt() pipeline everything else already respects — not a
    separate client-side unlock that could drift out of sync with
    inStateScope()."""
    src = html()
    fn = src.split("function setNationalScope(national)")[1].split("function initScopeToggle")[0]
    assert "filt.state = national ? '' : (authState || '')" in fn
    assert "applyFilt()" in fn


def test_national_button_activates_all_india_and_state_button_activates_own_state():
    src = html()
    fn = src.split("function setNationalScope(national)")[1].split("function initScopeToggle")[0]
    assert "stateBtn.classList.toggle('on', !national)" in fn
    assert "natBtn.classList.toggle('on', national)" in fn


def test_toggle_buttons_are_wired_to_setNationalScope():
    src = html()
    fn = src.split("function initScopeToggle()")[1].split("\n    }")[0]
    assert "setNationalScope(false)" in fn
    assert "setNationalScope(true)" in fn


def test_state_login_lands_scoped_via_the_same_toggle_function():
    """Regression guard: the landing behaviour and the toggle must not
    be two independent implementations that could disagree — enterDashboard
    delegates straight to setNationalScope(false)."""
    src = html()
    fn = src.split("function enterDashboard()")[1].split("function ")[0]
    assert "authRole === 'state' && authState" in fn
    assert "setNationalScope(false)" in fn


def test_admin_login_never_calls_setNationalScope_and_stays_unscoped():
    src = html()
    fn = src.split("function enterDashboard()")[1].split("function ")[0]
    # The call is inside the state-only guard, not unconditional.
    guard_and_call = fn.split("if (authRole === 'state' && authState) {")[1].split("}")[0]
    assert "setNationalScope(false)" in guard_and_call


def test_manual_state_filter_dropdown_keeps_the_toggle_honest():
    """A state account could also change #flt-state directly (existing
    sidebar control) instead of the header toggle — both must agree on
    what's actually applied afterward."""
    src = html()
    apply_fn = src.split("btn-apply').addEventListener('click'")[1].split("btn-reset-f")[0]
    assert "filt.state === authState" in apply_fn
    reset_fn = src.split("btn-reset-f').addEventListener('click'")[1].split("initDetailCtrls")[0]
    assert "natBtn.classList.add('on')" in reset_fn


def test_scope_toggle_never_fabricates_data_client_side():
    assert "Math.random" not in html()


def test_live_summary_label_names_the_current_scope_not_hardcoded_india():
    """The 'Fire detections across ___' label must track filt.state, the
    same source of truth as the map/donut/overlays — not stay hardcoded
    to 'India' when a state account is scoped to its own state, and not
    stay stuck on a state name after switching to All India."""
    src = html()
    assert 'id="st-lbl">Fire detections across India</div>' in src
    fn = src.split("function updateStats()")[1].split("\n    function ")[0]
    assert "stLbl.textContent = 'Fire detections across ' + (filt.state || 'India')" in fn
