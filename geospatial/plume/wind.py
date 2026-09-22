"""Open-Meteo wind client.

M1 has `open_meteo_url` in its config but never calls it per-cluster —
wind is contextual data that only matters once you're modelling where
smoke goes, which is M3's job. So the call lives here.

Design note from the project research: wind is supporting evidence for
plume behaviour and spread, NOT a classification signal. Nothing in this
module feeds M2's feature matrix; it only shapes M3's geometry.

Open-Meteo needs no API key. The parsing is split from the fetching so
the awkward part (picking the hour nearest the detection, and the
meteorological direction convention) is unit-testable offline.
"""
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_WIND_SPEED_MS = 3.0     # light breeze; a neutral-ish fallback
DEFAULT_WIND_DIRECTION = 0.0    # from the north
_KMH_TO_MS = 1.0 / 3.6


@dataclass
class WindObservation:
    speed_ms: float
    direction_deg: float          # meteorological: direction wind comes FROM
    cloud_cover_fraction: Optional[float]
    is_daytime: bool
    source: str                   # "open-meteo" or "fallback"
    observed_at: Optional[str] = None

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"


def fallback_wind(at: Optional[datetime] = None) -> WindObservation:
    """Used when Open-Meteo is unreachable.

    Returns a clearly-labelled default rather than raising: a plume drawn
    from assumed wind with `source='fallback'` on the record is more
    useful to a responder than no plume at all, provided the UI can see
    that it was assumed. `is_fallback` exists so it can.
    """
    moment = at or datetime.now(timezone.utc)
    return WindObservation(
        speed_ms=DEFAULT_WIND_SPEED_MS,
        direction_deg=DEFAULT_WIND_DIRECTION,
        cloud_cover_fraction=None,
        is_daytime=6 <= moment.hour <= 18,
        source="fallback",
        observed_at=moment.isoformat(),
    )


def _nearest_hour_index(times: list, target: datetime) -> int:
    """Index of the hourly sample closest to the detection time.

    Open-Meteo returns naive local-to-the-requested-timezone strings; we
    request UTC, so they are parsed as UTC. Comparing a naive parse
    against a tz-aware target raises TypeError, hence the explicit
    normalisation on both sides.
    """
    target_naive = target.replace(tzinfo=None)
    best_index, best_delta = 0, None
    for i, value in enumerate(times):
        try:
            parsed = datetime.fromisoformat(str(value)).replace(tzinfo=None)
        except ValueError:
            continue
        delta = abs((parsed - target_naive).total_seconds())
        if best_delta is None or delta < best_delta:
            best_index, best_delta = i, delta
    return best_index


def parse_open_meteo(payload: Dict[str, Any], at: datetime) -> WindObservation:
    """Turn an Open-Meteo hourly response into a WindObservation."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    speeds = hourly.get("wind_speed_10m") or hourly.get("windspeed_10m") or []
    directions = hourly.get("wind_direction_10m") or hourly.get("winddirection_10m") or []
    clouds = hourly.get("cloud_cover") or hourly.get("cloudcover") or []
    daylight = hourly.get("is_day") or []

    if not times or not speeds or not directions:
        raise ValueError("Open-Meteo response missing hourly wind fields")

    index = _nearest_hour_index(times, at)

    def _at(series, default=None):
        return series[index] if index < len(series) else default

    speed_kmh = _at(speeds)
    if speed_kmh is None:
        raise ValueError("Open-Meteo response has no wind speed at the requested hour")

    cloud_pct = _at(clouds)
    is_day_flag = _at(daylight)
    if is_day_flag is None:
        parsed_hour = datetime.fromisoformat(str(times[index])).hour
        is_day = 6 <= parsed_hour <= 18
    else:
        is_day = bool(is_day_flag)

    return WindObservation(
        # Open-Meteo reports wind speed in km/h by default; the whole
        # dispersion model is in m/s.
        speed_ms=float(speed_kmh) * _KMH_TO_MS,
        direction_deg=float(_at(directions, DEFAULT_WIND_DIRECTION)),
        cloud_cover_fraction=None if cloud_pct is None else max(0.0, min(1.0, float(cloud_pct) / 100.0)),
        is_daytime=is_day,
        source="open-meteo",
        observed_at=str(times[index]),
    )


def choose_endpoint(at: datetime, forecast_url: str, archive_url: str, cutoff_days: int) -> str:
    """Forecast endpoint for recent times, archive endpoint for old ones.

    Open-Meteo's forecast API only covers roughly the last few days.
    Asking it for a date from the 90-day retrospective replay returns no
    usable hours, so every historical plume would silently fall back to
    assumed wind — the backfill would look like it worked while producing
    meteorologically meaningless geometry.
    """
    age_days = (datetime.now(timezone.utc) - at).total_seconds() / 86400.0
    return archive_url if age_days > cutoff_days else forecast_url


def fetch_wind(lon: float, lat: float, at: Optional[datetime] = None, timeout: int = 20) -> WindObservation:
    """Fetch wind for a location/time, falling back rather than raising."""
    import requests

    from app.config import get_settings

    from geospatial.config import get_m3_settings

    moment = at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    settings = get_settings()
    m3_settings = get_m3_settings()
    url = choose_endpoint(
        moment,
        settings.open_meteo_url,
        m3_settings.open_meteo_archive_url,
        m3_settings.archive_cutoff_days,
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m,cloud_cover,is_day",
        "timezone": "UTC",
        "start_date": moment.strftime("%Y-%m-%d"),
        "end_date": moment.strftime("%Y-%m-%d"),
    }

    # Cached on disk, keyed to the hour actually used for the lookup —
    # one Open-Meteo call per cluster per cycle otherwise, and plume
    # modelling runs for every classified cluster. A *fallback* result is
    # never cached: it is an assumption, not an observation, and caching
    # it would keep serving assumed wind after the network recovered.
    from app import geo_cache

    cache_key = geo_cache.make_key("wind", lon, lat, moment.strftime("%Y-%m-%dT%H"))
    cached = geo_cache.get(cache_key)
    if cached is not geo_cache.MISS:
        return WindObservation(**cached)

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        observation = parse_open_meteo(response.json(), moment)
    except Exception:
        logger.warning(
            "Open-Meteo lookup failed for (%s, %s) at %s via %s — using assumed wind, "
            "flagged as fallback", lon, lat, moment, url, exc_info=True,
        )
        return fallback_wind(moment)

    if not observation.is_fallback:
        geo_cache.put(cache_key, asdict(observation))
    return observation
