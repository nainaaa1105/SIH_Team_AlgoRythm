"""Frontend hooks for two things fixed together in one pass:

1. The login/signup screen actually authenticating against the real
   /auth/login and /auth/signup endpoints (app/auth/routes.py), instead
   of "Sign In" dismissing the overlay on any click with no check at
   all.
2. Two globe/map zoom glitches: zooming in past where Esri's imagery
   actually has coverage showed blank "no data" tiles instead of
   stopping; zooming out to the 3D globe left it unable to be manually
   zoomed back in, because #globe-div was permanently pointer-events:
   none with no override for normal dashboard use.

Same pattern as test_wui.py / test_gateway.py: no browser, just
asserting the hooks a live page actually needs are present in the
source, and that nothing here fabricates a login success.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


# --- login / signup wiring -------------------------------------------------

def test_login_form_posts_to_the_real_auth_endpoint():
    src = html()
    assert "API.login(pendingLoginRole, username, password)" in src
    assert "this.post('/auth/login'" in src


def test_signup_form_posts_to_the_real_auth_endpoint():
    src = html()
    assert "API.signup({" in src
    assert "this.post('/auth/signup'" in src


def test_signin_no_longer_dismisses_the_login_screen_unconditionally():
    """Regression: the old handler was a bare click listener on
    #btn-start that flew into the dashboard no matter what was typed
    (or left blank). It must now be gated on a successful API call."""
    src = html()
    assert "document.getElementById('btn-start').addEventListener('click', () => {" not in src
    # The success path only runs inside the login form's submit handler,
    # after the API call resolves — not unconditionally.
    login_fn = src.split("loginForm.addEventListener('submit'")[1].split("signupForm.addEventListener")[0]
    assert "await API.login" in login_fn
    assert "enterDashboard()" in login_fn
    assert login_fn.index("await API.login") < login_fn.index("enterDashboard()")


def test_login_failure_shows_the_real_server_message_and_does_not_enter_dashboard():
    src = html()
    login_fn = src.split("loginForm.addEventListener('submit'")[1].split("signupForm.addEventListener")[0]
    catch_block = login_fn.split("catch (err)")[1].split("finally")[0]
    assert "loginErr.textContent = err.message" in catch_block
    assert "enterDashboard()" not in catch_block


def test_signup_failure_shows_the_real_server_message_and_does_not_enter_dashboard():
    src = html()
    signup_fn = src.split("signupForm.addEventListener('submit'")[1]
    catch_block = signup_fn.split("catch (err)")[1].split("finally")[0]
    assert "signupErr.textContent = err.message" in catch_block
    assert "enterDashboard()" not in catch_block


def test_signup_requires_matching_passwords_before_hitting_the_api():
    src = html()
    signup_fn = src.split("signupForm.addEventListener('submit'")[1].split("setFormBusy(signupForm, true)")[0]
    assert "password !== password2" in signup_fn


def test_password_field_uses_password_input_type_not_plaintext():
    src = html()
    assert 'id="lp-login-pass" placeholder="Password" autocomplete="current-password" required>' in src
    assert src.count('type="password"') >= 3  # login pass, signup pass, signup confirm


def test_stored_session_is_verified_against_the_server_not_trusted_blindly():
    """A token sitting in sessionStorage must be checked against
    /auth/me before it's treated as a valid session — an expired or
    tampered token must not silently grant entry."""
    src = html()
    fn = src.split("async function checkStoredSession()")[1].split("function persistSession")[0]
    assert "await API.me(token)" in fn
    assert "sessionStorage.removeItem(AUTH_TOKEN_KEY)" in fn


def test_logout_clears_the_session_rather_than_just_hiding_ui():
    src = html()
    fn = src.split("function logout()")[1].split("\n    }")[0]
    assert "sessionStorage.removeItem(AUTH_TOKEN_KEY)" in fn


def test_session_uses_sessionstorage_not_localstorage_so_closing_the_tab_logs_out():
    """Regression guard for the explicit requirement: closing the tab
    must always require signing back in. localStorage would survive a
    closed tab/browser restart; sessionStorage is cleared with it."""
    src = html()
    assert "localStorage.setItem" not in src
    assert "localStorage.getItem" not in src
    assert "localStorage.removeItem" not in src
    assert "sessionStorage.setItem(AUTH_TOKEN_KEY" in src


def test_login_never_uses_math_random_to_fake_success():
    assert "Math.random" not in html()


# --- zoom bug #1: stop at the imagery provider's real coverage limit ------

def test_tile_error_clamps_zoom_instead_of_showing_blank_tiles():
    src = html()
    assert "function clampZoomOnTileError" in src
    fn = src.split("function clampZoomOnTileError")[1].split("function initLeaflet")[0]
    assert "'tileerror'" in fn
    assert "leafletMap.setMaxZoom(clamped)" in fn
    assert "leafletMap.setZoom(clamped)" in fn


def test_both_satellite_and_street_basemaps_are_clamp_protected():
    """Regression risk: the basemap switcher creates fresh tile layers
    on every style change — the clamp has to be re-attached to each new
    layer, not just the ones created at init."""
    src = html()
    view_ctrls = src.split("function initViewCtrls()")[1].split("function initFilterCtrls")[0]
    assert view_ctrls.count("clampZoomOnTileError(") == 2


# --- zoom bug #2: globe must stay interactive after returning to 3D -------

def test_globe_div_is_interactive_by_default_not_permanently_blocked():
    """Regression: #globe-div was permanently pointer-events:none with
    no override anywhere, so a manual mouse zoom on the globe never
    reached the renderer's canvas — the only reason it ever worked was
    the one scripted (non-mouse-driven) fly-in on Sign In. Returning to
    the globe via switchTo3D() had no such script, so it looked stuck."""
    src = html()
    base_rule = src.split("#globe-div {")[1].split("}")[0]
    assert "pointer-events: auto" in base_rule


def test_pointer_events_are_only_disabled_during_the_scripted_intro_phases():
    src = html()
    assert "#globe-div.intro-pos,\n    #globe-div.front-pos,\n    #globe-div.login-pos {\n      pointer-events: none;" in src


# --- globe manual zoom-out capped 500km past wherever it already is -------

def test_globe_zoom_out_ceiling_constant_is_500km():
    src = html()
    assert "const GLOBE_MANUAL_ZOOM_OUT_KM = 500;" in src


def test_fly_globe_helper_suppresses_the_clamp_during_scripted_flights():
    src = html()
    fn = src.split("function flyGlobe(pov, ms)")[1].split("\n    function ")[0]
    assert "scriptedGlobeFlightActive = true;" in fn
    assert "scriptedGlobeFlightActive = false;" in fn
    assert "globeZoomOutCeiling = landedAlt + GLOBE_MANUAL_ZOOM_OUT_KM / EARTH_RADIUS_KM;" in fn


def test_every_scripted_globe_flight_goes_through_fly_globe_not_pointofview_directly():
    """Regression guard: a call site that still uses globeInst.pointOfView()
    directly would never update globeZoomOutCeiling, so the ceiling would
    stay wherever the last flyGlobe() call left it — either trapping the
    user way too close after a flight to a farther target, or leaving the
    500km cap keyed off a stale, no-longer-current position."""
    src = html()
    assert src.count("globeInst.pointOfView(") == 4  # the setter inside flyGlobe, its own
    # read-only lookups (settle(), onGlobeCameraChange), and the clamp's own corrective call


def test_on_globe_camera_change_clamps_past_the_ceiling():
    src = html()
    fn = src.split("function onGlobeCameraChange()")[1].split("\n    function ")[0]
    assert "scriptedGlobeFlightActive" in fn
    assert "clampingGlobeZoom" in fn
    assert "alt > globeZoomOutCeiling" in fn


def test_zoom_out_clamp_does_not_swallow_the_3d_to_2d_switch_during_a_flight():
    """Regression guard for the exact bug reported: gating the whole
    function on scriptedGlobeFlightActive (instead of just the clamp
    branch) silently broke every scripted flight's alt<0.35 -> switchTo2D
    trigger, leaving the dashboard stuck on a close, un-switched 3D globe
    with no header/sidebar ever shown."""
    src = html()
    fn = src.split("function onGlobeCameraChange()")[1].split("\n    function ")[0]
    assert "if (switching) return;" in fn
    assert "if (switching || scriptedGlobeFlightActive" not in fn
    assert "if (alt < 0.35 && !is2D) switchTo2D(pov.lat, pov.lng);" in fn


# --- same cap, translated to the 2D Leaflet map's discrete zoom levels ----

def test_leaflet_zoom_out_floor_constant_is_500km():
    src = html()
    assert "const LEAFLET_ZOOM_OUT_SCALE_BAR_KM = 500;" in src


def test_update_leaflet_zoom_out_floor_targets_the_scale_bars_own_500km_bucket():
    """Regression guard for the exact bug reported: an earlier viewport-
    width-based guess landed the floor at a 200km scale-bar reading
    instead of the requested 500km. This must solve for the zoom whose
    L.control.scale readout is the 500km bucket directly, using the
    current latitude (Web Mercator distorts ground distance away from
    the equator) rather than a hardcoded distance-per-zoom-level table."""
    src = html()
    fn = src.split("function updateLeafletZoomOutFloor()")[1].split("\n    function ")[0]
    assert "leafletMap.getCenter()" in fn
    assert "Math.cos(center.lat" in fn
    assert "LEAFLET_ZOOM_OUT_SCALE_BAR_KM * 1000) / 100" in fn
    assert "leafletMap.setMinZoom(" in fn


def test_every_scripted_leaflet_move_goes_through_fly_leaflet():
    """Regression guard, mirroring the globe's: a setView() call that
    bypasses flyLeaflet() would never refresh the minZoom floor, so it
    would stay keyed off wherever the map was before that move instead
    of the new position — either too tight or, if the user had zoomed
    out further in the meantime, not applied at all."""
    src = html()
    assert src.count("leafletMap.setView(") == 1  # the one call inside flyLeaflet itself


def test_switch_to_2d_and_search_and_detail_zoom_all_use_fly_leaflet():
    src = html()
    assert "flyLeaflet(lat || homeLocation().lat, lng || homeLocation().lon, homeZoom()" in src
    assert "flyLeaflet(l.lat, l.lon, 6)" in src
    assert "flyLeaflet(lat, lng, 10)" in src


# --- map attribution: shrunk, never fully removed --------------------------

def test_leaflet_attribution_is_shrunk_not_disabled():
    """OpenStreetMap's and Esri's free tile terms both require visible
    attribution — the operator asked for it removed, but it can only be
    made subtler without risking the tile providers blocking this app."""
    src = html()
    assert "attributionControl: false" not in src
    assert ".leaflet-control-attribution {" in src
    rule = src.split(".leaflet-control-attribution {")[1].split("}")[0]
    assert "font-size:" in rule
