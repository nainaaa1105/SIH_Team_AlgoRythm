"""temporal.status.is_active_from_signals — the one "is this fire
currently Active" definition shared by gateway.routes_dashboard's
Active/Under Control/Extinguished/Verified status label and
geospatial.wui_analysis's WUI-critical gate (see
tests/test_wui.py::test_critical_downgrades_to_watch_when_not_active).
Pure logic only, no DB — is_fire_active's own two-row read is exercised
indirectly wherever a cluster fixture already exists.
"""
from temporal.status import is_active_from_signals


def test_escalating_alone_is_active():
    assert is_active_from_signals(True, None) is True


def test_ptsi_abnormal_alone_is_active():
    assert is_active_from_signals(False, {"is_behaving_normally": False}) is True


def test_neither_signal_is_not_active():
    assert is_active_from_signals(False, None) is False
    assert is_active_from_signals(False, {"is_behaving_normally": True}) is False


def test_missing_ptsi_never_crashes_and_reads_as_not_abnormal():
    """No PTSI baseline yet (is_behaving_normally is None) must not be
    treated as 'abnormal' — that would be inventing a signal, not
    reading one that exists."""
    assert is_active_from_signals(False, {"is_behaving_normally": None}) is False
