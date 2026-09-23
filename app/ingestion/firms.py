"""NASA FIRMS ingestion — the primary hotspot feed.

API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
Endpoint shape:
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/{day_range}[/{date}]
where bbox is "west,south,east,north" and SOURCE is one of
VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, MODIS_NRT, ...

Requires a MAP_KEY registered at https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""
import io
import logging
from typing import Dict, List, Optional

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.ingestion.normalize import firms_row_to_canonical

logger = logging.getLogger(__name__)

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox}/{day_range}"
# Archive queries add a trailing /{date}, e.g. .../5/2026-06-01 — FIRMS
# rejects day_range outside [1..5] with a 400 ("Invalid day range. Expects
# [1..5]."), confirmed directly against the live API — the docs/tutorials
# that say 10 are wrong (or describe a different, non-archive endpoint).
# 90-day backfills must loop in <=5-day windows (see scripts/backfill_90day.py).
FIRMS_MAX_ARCHIVE_DAY_RANGE = 5


class FirmsClientError(RuntimeError):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, FirmsClientError)),
)
def _fetch_csv(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    # FIRMS returns a plain-text error message (not CSV) for bad keys/params.
    if text.startswith("Invalid") or "error" in text[:200].lower():
        raise FirmsClientError(f"FIRMS API returned an error payload: {text[:200]}")
    return text


def fetch_source(
    source: str, settings: Settings, day_range: Optional[int] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """Fetch one FIRMS source (e.g. VIIRS_SNPP_NRT) for the configured bbox.

    `end_date` (YYYY-MM-DD) + `day_range` together select a historical
    window ending on that date — used by the 90-day backfill. Omit both
    for the default NRT behaviour (most recent `firms_day_range` days).
    """
    if not settings.firms_map_key:
        raise FirmsClientError("FIRMS_MAP_KEY is not set — register one at "
                                "https://firms.modaps.eosdis.nasa.gov/api/map_key/")

    bbox = ",".join(str(v) for v in settings.firms_bbox_tuple)
    url = FIRMS_AREA_URL.format(
        map_key=settings.firms_map_key,
        source=source,
        bbox=bbox,
        day_range=day_range or settings.firms_day_range,
    )
    if end_date:
        url = f"{url}/{end_date}"
    csv_text = _fetch_csv(url)
    if not csv_text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(csv_text))
    return df


def fetch_all_sources(settings: Settings | None = None) -> List[Dict]:
    """Fetch every configured FIRMS source and return canonical records."""
    settings = settings or get_settings()
    canonical_records: List[Dict] = []

    for source in settings.firms_source_list:
        try:
            df = fetch_source(source, settings)
        except Exception:
            logger.exception("FIRMS fetch failed for source=%s — skipping this cycle", source)
            continue

        if df.empty:
            logger.info("FIRMS source=%s returned no detections for this window", source)
            continue

        for row in df.to_dict(orient="records"):
            try:
                canonical_records.append(firms_row_to_canonical(row, source))
            except (KeyError, ValueError):
                logger.warning("Skipping malformed FIRMS row from %s: %s", source, row)

    logger.info("FIRMS fetch complete: %d canonical records across %d sources",
                len(canonical_records), len(settings.firms_source_list))
    return canonical_records


def fetch_historical_window(end_date: str, day_range: int, settings: Optional[Settings] = None) -> List[Dict]:
    """Fetch every configured FIRMS source for a single historical window
    (<=10 days, FIRMS' archive-query cap) ending on `end_date`.
    """
    settings = settings or get_settings()
    if day_range > FIRMS_MAX_ARCHIVE_DAY_RANGE:
        raise ValueError(f"day_range must be <= {FIRMS_MAX_ARCHIVE_DAY_RANGE} for FIRMS archive queries")

    canonical_records: List[Dict] = []
    for source in settings.firms_source_list:
        try:
            df = fetch_source(source, settings, day_range=day_range, end_date=end_date)
        except Exception:
            logger.exception("FIRMS historical fetch failed for source=%s end_date=%s", source, end_date)
            continue
        if df.empty:
            continue
        for row in df.to_dict(orient="records"):
            try:
                canonical_records.append(firms_row_to_canonical(row, source))
            except (KeyError, ValueError):
                logger.warning("Skipping malformed FIRMS row from %s: %s", source, row)
    return canonical_records
