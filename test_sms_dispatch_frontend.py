"""Frontend hooks for the emergency SMS dispatch button (TEST MODE).

No browser, just asserting the hooks a live page needs are present in
static/index.html — same pattern as test_wui.py / test_gateway.py.
Design constraint from the request: the frontend must never contain or
hardcode the test phone number.
"""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static" / "index.html"


def html():
    return STATIC.read_text(encoding="utf-8")


def test_sms_button_exists_in_the_detail_panel():
    """Moved by request into the Suppression Resource Estimate card
    (id="supp-card") rather than the standalone dp-acts button row."""
    src = html()
    assert 'id="btn-sms-dispatch"' in src
    supp_card = src.split('id="supp-card"')[1].split('<div class="cls-box">')[0]
    assert 'id="btn-sms-dispatch"' in supp_card


def test_sms_button_wired_to_the_backend_not_a_client_side_composer():
    src = html()
    fn = src.split("function initDetailCtrls() {")[1].split("\n    }")[0]
    assert "btn-sms-dispatch" in fn
    assert "sendEmergencySms" in fn


def test_send_emergency_sms_calls_the_real_backend_endpoint():
    src = html()
    fn = src.split("async function sendEmergencySms()")[1].split("\n    async function ")[0]
    assert "API.dispatchSms(fire.clusterId)" in fn
    # No client-side message composer — the server builds it from real
    # incident data (see test_sms_dispatch.py::test_sms_body_...).
    assert "FIREOPS EMERGENCY ALERT" not in fn


def test_api_dispatch_sms_hits_the_dashboard_sms_route():
    src = html()
    assert "dispatchSms(clusterId)" in src
    assert "'/dashboard/event/' + clusterId + '/sms-dispatch'" in src


def test_frontend_never_hardcodes_the_test_phone_number():
    src = html()
    assert "9773207681" not in src


def test_sms_button_disabled_for_the_whole_request_not_just_on_click():
    """Prevents a second click firing a second request while the first
    is still in flight — the backend's own 2-minute de-dupe is a second,
    independent layer, not the only protection."""
    src = html()
    fn = src.split("async function sendEmergencySms()")[1].split("\n    async function ")[0]
    assert "btn.disabled = true;" in fn
    assert "btn.disabled = false;" in fn
    assert "finally" in fn


def test_sms_failure_shows_a_real_error_not_a_claimed_success():
    src = html()
    fn = src.split("async function sendEmergencySms()")[1].split("\n    async function ")[0]
    assert "res.error" in fn
    assert "'err'" in fn


def test_cluster_deep_link_opens_the_right_fire_after_login():
    """The "Map Location" link in the SMS (?cluster=<id>) must actually
    open that fire, not just land on the home view."""
    src = html()
    fn = src.split("function enterDashboard()")[1].split("\n    function ")[0]
    assert "URLSearchParams(location.search).get('cluster')" in fn
    assert "openDP(fire)" in fn


def test_sms_banner_auto_dismisses_instead_of_staying_forever():
    """Real bug report: the success/duplicate/failure banner from an SMS
    dispatch stayed on screen indefinitely — setBanner never had an
    auto-dismiss path at all before this fix. All four outcomes (success,
    duplicate, backend-reported failure, request error) must pass a
    timeout so the banner clears itself a few seconds later, the same
    way the pre-existing "clusters still awaiting classification"
    banner already does."""
    src = html()
    fn = src.split("async function sendEmergencySms()")[1].split("\n    async function ")[0]
    assert fn.count(", 6000)") == 4


def test_set_banner_supports_a_real_auto_dismiss_not_a_second_mechanism():
    """The auto-dismiss timer lives inside setBanner itself (tracked so
    a new banner arriving early cancels the previous one's pending
    dismiss, rather than that stale timer wiping out the new banner) —
    not a parallel setTimeout(() => setBanner('', null), ...) pattern
    duplicated at every call site."""
    src = html()
    fn = src.split("function setBanner(message, kind, autoDismissMs)")[1].split("\n    }")[0]
    assert "clearTimeout(bannerDismissTimer)" in fn
    assert "setTimeout(() => setBanner('', null), autoDismissMs)" in fn
    # The older hand-rolled version of this pattern must be gone, not
    # left duplicated alongside the new built-in one.
    assert "setTimeout(() => setBanner('', null), 9000)" not in src
