"""FSI (Forest Survey of India) forest-fire alert loader.

FSI publishes near-real-time forest-fire alerts derived from MODIS and
SNPP-VIIRS thermal anomalies, downloadable from their Forest Fire Alert
System dashboard as CSV. There is no public REST API (this is the same
constraint recorded in the project research notes), so this is a manual
bulk export loaded from disk, not a live feed — which is also why FSI
sits with M2 as a *label* source rather than with M1 as an ingestion
source.

Caveat worth keeping in mind when reading the resulting metrics: FSI's
alerts are themselves derived from the same VIIRS/MODIS thermal
anomalies our pipeline ingests, and FSI explicitly filters out known
mining and industrial areas to suppress false forest-fire alerts. So an
FSI match is strong evidence of "forest fire", but the absence of a
match is weak evidence of "not a forest fire".
"""
import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# FSI exports vary by download; these are the column names seen on the
# public dashboard export. Extend rather than rename if a new export
# differs, so older files keep loading.
_LAT_COLUMNS = ("latitude", "LATITUDE", "Latitude", "lat")
_LON_COLUMNS = ("longitude", "LONGITUDE", "Longitude", "lon", "long")
_DATE_COLUMNS = ("acq_date", "ACQ_DATE", "date", "DATE", "alert_date")


def _first_present(row: dict, candidates: Tuple[str, ...]):
    for candidate in candidates:
        if candidate in row and row[candidate] not in (None, ""):
            return row[candidate]
    return None


def load_fsi_alerts(csv_path: str) -> List[Dict]:
    """Parse an FSI alert CSV export into {lon, lat, date} records."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    alerts: List[Dict] = []
    for row in df.to_dict(orient="records"):
        lat = _first_present(row, _LAT_COLUMNS)
        lon = _first_present(row, _LON_COLUMNS)
        date = _first_present(row, _DATE_COLUMNS)
        if lat is None or lon is None:
            continue
        alerts.append({
            "lon": float(lon),
            "lat": float(lat),
            "date": pd.to_datetime(date) if date is not None else None,
        })

    if not alerts:
        raise ValueError(
            f"No usable rows in {csv_path}. Expected latitude/longitude columns "
            f"(one of {_LAT_COLUMNS} / {_LON_COLUMNS})."
        )
    logger.info("Loaded %d FSI forest-fire alerts from %s", len(alerts), csv_path)
    return alerts


def matches_fsi_alert(
    lon: float,
    lat: float,
    cluster_datetime,
    alerts: List[Dict],
    max_distance_km: float = 1.0,
    max_days: int = 1,
) -> bool:
    """Is there an FSI alert near this cluster, around the same time?

    The tolerances matter: FSI's own alerts come from ~375m-1km pixels
    and the same real fire is re-detected across satellite revisits, so
    an exact coordinate match would almost never fire.
    """
    from classifier.features.thermal import haversine_km

    import pandas as pd

    for alert in alerts:
        if haversine_km(lon, lat, alert["lon"], alert["lat"]) > max_distance_km:
            continue
        alert_date = alert.get("date")
        # pd.to_datetime turns an unparseable/blank date into NaT, which is
        # not None — comparing against it silently yields False forever, so
        # treat it the same as a missing date: match on location alone.
        if alert_date is None or pd.isna(alert_date) or cluster_datetime is None:
            return True
        # Compare naive-to-naive: FSI exports carry no timezone, while M1's
        # acq_datetime is tz-aware UTC. Stripping tzinfo avoids a
        # "can't subtract offset-naive and offset-aware" TypeError.
        cluster_naive = cluster_datetime.replace(tzinfo=None)
        alert_naive = alert_date.to_pydatetime() if hasattr(alert_date, "to_pydatetime") else alert_date
        alert_naive = alert_naive.replace(tzinfo=None)
        if abs((cluster_naive - alert_naive)) <= timedelta(days=max_days):
            return True
    return False
