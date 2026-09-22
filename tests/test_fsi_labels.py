from datetime import datetime, timezone

import pytest

from classifier.labels.fsi_labels import load_fsi_alerts, matches_fsi_alert

pytest.importorskip("pandas")

CLUSTER_TIME = datetime(2026, 3, 10, 6, 30, tzinfo=timezone.utc)


def _alert(lon=79.0, lat=21.0, date="2026-03-10"):
    import pandas as pd

    return {"lon": lon, "lat": lat, "date": pd.to_datetime(date)}


def test_matches_a_nearby_alert_on_the_same_day():
    assert matches_fsi_alert(79.0, 21.0, CLUSTER_TIME, [_alert()])


def test_does_not_match_a_distant_alert():
    assert not matches_fsi_alert(79.0, 21.0, CLUSTER_TIME, [_alert(lon=85.0, lat=25.0)])


def test_does_not_match_an_alert_from_a_different_week():
    assert not matches_fsi_alert(79.0, 21.0, CLUSTER_TIME, [_alert(date="2026-01-01")])


def test_tolerates_timezone_aware_cluster_vs_naive_alert():
    """M1's acq_datetime is tz-aware UTC; FSI exports carry no timezone.
    Subtracting one from the other raises TypeError if not handled."""
    assert matches_fsi_alert(79.0, 21.0, CLUSTER_TIME, [_alert()])


def test_matches_on_location_alone_when_the_alert_date_is_unparseable():
    """pd.to_datetime yields NaT for a blank date; NaT comparisons are
    always False, which would silently drop a valid location match."""
    import pandas as pd

    alert = {"lon": 79.0, "lat": 21.0, "date": pd.to_datetime(None)}
    assert matches_fsi_alert(79.0, 21.0, CLUSTER_TIME, [alert])


def test_matches_when_the_cluster_has_no_timestamp():
    assert matches_fsi_alert(79.0, 21.0, None, [_alert()])


def test_within_the_distance_tolerance_but_not_exact():
    """FIRMS pixels are 375m-1km and revisits re-detect the same fire, so
    an exact coordinate match would essentially never fire."""
    assert matches_fsi_alert(79.005, 21.005, CLUSTER_TIME, [_alert()], max_distance_km=1.0)


def test_empty_alert_list_never_matches():
    assert not matches_fsi_alert(79.0, 21.0, CLUSTER_TIME, [])


def test_loader_accepts_alternative_column_spellings(tmp_path):
    import pandas as pd

    path = tmp_path / "fsi.csv"
    pd.DataFrame(
        [{"LATITUDE": 21.0, "LONGITUDE": 79.0, "ACQ_DATE": "2026-03-10"}]
    ).to_csv(path, index=False)

    alerts = load_fsi_alerts(str(path))
    assert len(alerts) == 1
    assert alerts[0]["lat"] == 21.0
    assert alerts[0]["lon"] == 79.0


def test_loader_rejects_a_file_with_no_coordinates(tmp_path):
    import pandas as pd

    path = tmp_path / "bad.csv"
    pd.DataFrame([{"something_else": 1}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="No usable rows"):
        load_fsi_alerts(str(path))
