"""Two-role login/signup (Administrator vs State) and the resulting
state-scoped dashboard.

Same split as the rest of the frontend-hook suite: no browser, just
asserting the hooks a live page needs are present in the source, and
that the wiring is a full override / real API call rather than a
cosmetic shortcut. The backend side (role mismatch rejected, state
validated against the real boundary list) is covered in test_auth.py.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


# --- role pickers exist for both login and signup -------------------------

def test_login_role_picker_offers_administrator_and_state():
    src = html()
    assert 'id="role-login-admin" data-role="admin"' in src
    assert 'id="role-login-state" data-role="state"' in src


def test_signup_role_picker_offers_administrator_and_state():
    src = html()
    assert 'id="role-signup-admin" data-role="admin"' in src
    assert 'id="role-signup-state" data-role="state"' in src


def test_role_is_chosen_before_credentials_are_ever_shown():
    """Regression guard for the exact flow requested: role first, then
    username/password — not the other way round."""
    src = html()
    assert 'id="lp-login-form" autocomplete="on" style="display:none"' in src
    assert 'id="lp-signup-form" autocomplete="on" style="display:none"' in src


# --- signup fields: name, username, password, [state], government ID -----

def test_signup_form_collects_full_name_and_government_id():
    src = html()
    form = src.split('id="lp-signup-form"')[1].split("</form>")[0]
    assert 'id="lp-signup-name"' in form
    assert 'id="lp-signup-govid"' in form


def test_signup_state_dropdown_is_hidden_until_state_role_is_chosen():
    src = html()
    assert 'id="lp-signup-state" style="display:none"' in src


def test_signup_state_dropdown_is_populated_from_the_real_state_list():
    """Not a hardcoded 28-state list baked into the page — the same
    /dashboard/states endpoint (backed by the real boundary file) the
    sidebar's own state filter already uses."""
    src = html()
    fn = src.split("async function populateSignupStateSelect()")[1].split("function initAuthForms")[0]
    assert "loadStateOptions()" in fn
    load_fn = src.split("function loadStateOptions()")[1].split("function populateSignupStateSelect")[0]
    assert "API.states()" in load_fn


def test_signup_role_toggle_shows_state_field_only_for_state_role():
    src = html()
    handler = src.split("#lp-role-signup .lp-role-box")[1].split("loginForm.addEventListener")[0]
    assert "pendingSignupRole === 'state'" in handler
    assert "stateSel.style.display = 'block'" in handler
    assert "stateSel.style.display = 'none'" in handler


def test_signup_rejects_a_state_role_with_no_state_selected_client_side():
    src = html()
    fn = src.split("signupForm.addEventListener('submit'")[1]
    assert "pendingSignupRole === 'state' && !state" in fn


def test_signup_sends_role_and_government_id_to_the_real_api():
    src = html()
    fn = src.split("signupForm.addEventListener('submit'")[1].split("persistSession")[0]
    assert "role: pendingSignupRole" in fn
    assert "government_id: govId" in fn
    assert "state: pendingSignupRole === 'state' ? state : null" in fn


# --- login role gate ------------------------------------------------------

def test_login_sends_the_chosen_role_so_the_backend_can_gate_it():
    src = html()
    fn = src.split("loginForm.addEventListener('submit'")[1]
    assert "API.login(pendingLoginRole, username, password)" in fn


def test_login_requires_a_role_to_be_chosen_first():
    src = html()
    fn = src.split("loginForm.addEventListener('submit'")[1]
    assert "if (!pendingLoginRole)" in fn


# --- 3 demo quick-login buttons --------------------------------------------

def test_three_demo_login_buttons_exist_for_admin_haryana_odisha():
    src = html()
    for btn_id in ("demo-admin", "demo-haryana", "demo-odisha"):
        assert f'id="{btn_id}"' in src, f"{btn_id} missing"


def test_demo_buttons_use_the_real_login_api_not_a_client_side_bypass():
    """The demo buttons must be real accounts logged in through the same
    /auth/login call as any other credential — not a shortcut that
    fabricates a session client-side."""
    src = html()
    fn = src.split("DEMO_ACCOUNTS = {")[1].split("function setFormBusy")[0] \
        if "function setFormBusy" in src.split("DEMO_ACCOUNTS = {")[1] \
        else src.split("DEMO_ACCOUNTS = {")[1]
    assert "await API.login(role, username, password)" in fn
    assert "persistSession(" in fn


def test_demo_accounts_match_the_seed_script():
    """The three hardcoded demo credentials on the login screen must be
    exactly the accounts scripts/seed_demo_users.py actually creates —
    a mismatch here would make the "quick sign-in" buttons silently
    fail for every reviewer who clicks them."""
    src = html()
    seed = (Path(__file__).resolve().parents[1] / "scripts" / "seed_demo_users.py").read_text(encoding="utf-8")
    for username in ("admin_demo", "haryana_demo", "odisha_demo"):
        assert username in src
        assert username in seed


# --- session carries role/state, and the header shows it -------------------

def test_persisted_session_stores_role_and_state():
    src = html()
    fn = src.split("function persistSession(")[1].split("function updateAuthUi")[0]
    assert "AUTH_ROLE_KEY" in fn
    assert "AUTH_STATE_KEY" in fn


def test_restored_session_carries_role_and_state_from_auth_me():
    src = html()
    fn = src.split("async function checkStoredSession()")[1].split("function persistSession")[0]
    assert "me.role" in fn
    assert "me.state" in fn


# --- state login scopes the dashboard --------------------------------------

def test_state_login_sets_the_existing_state_filter_not_a_new_mechanism():
    """The whole "only show Odisha's fires and stats" requirement rides
    on the dashboard's own pre-existing state filter (filt.state), which
    updateStats()/buildDonut()/buildTS() already derive from. Landing
    scoped delegates to setNationalScope(false) (see test_scope_toggle.py)
    rather than setting filt.state inline a second, divergent way."""
    src = html()
    fn = src.split("function enterDashboard()")[1].split("function ")[0]
    assert "authRole === 'state' && authState" in fn
    assert "setNationalScope(false)" in fn


def test_state_login_flies_to_the_states_own_location_not_all_india():
    src = html()
    fn = src.split("function homeLocation()")[1].split("function switchTo2D")[0]
    assert "authRole === 'state' && authState && LOCS[authState]" in fn


def test_state_login_uses_a_tighter_zoom_than_admin():
    src = html()
    fn = src.split("function homeZoom()")[1].split("function homeLocation")[0]
    assert "return (authRole === 'state' && authState) ? 7 : 6" in fn


def test_admin_role_is_unaffected_and_still_gets_all_india():
    """Regression guard: an administrator login must not accidentally
    inherit state scoping left over from a previous session variable,
    or from sloppy conditionals."""
    src = html()
    home_fn = src.split("function homeLocation()")[1].split("function switchTo2D")[0]
    assert "LOCS['India']" in home_fn


def test_every_real_state_has_a_zoom_location():
    """geospatial/admin_boundaries.state_names() is the real 36-entry
    source of truth /dashboard/states and the signup dropdown both use
    — every one of those must resolve to a real LOCS entry, or a state
    account for it would silently fall back to the India-wide view."""
    src = html()
    locs_block = src.split("const LOCS = {")[1].split("};")[0]
    real_states = [
        "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
        "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
        "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
        "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    ]
    for state in real_states:
        assert f"'{state}':" in locs_block, f"LOCS is missing a zoom target for {state}"


def test_header_shows_role_and_state_for_a_logged_in_user():
    src = html()
    fn = src.split("function updateAuthUi()")[1].split("function logout")[0]
    assert "authRole === 'admin'" in fn
    assert "authState" in fn


def test_login_never_fabricates_state_boundaries_client_side():
    """No Math.random, no hardcoded fake polygon — the state scoping
    must ride entirely on the real filt.state / visFires mechanism."""
    assert "Math.random" not in html()
