"""On-disk cache for the expensive per-location lookups.

Three of the enrichment inputs cost a network round-trip per cluster:

    cloud fraction   CDSE catalogue search   ~1 s
    land cover       ESA WorldCover COG      ~2-5 s
    wind             Open-Meteo              ~0.5 s

A live cycle touches thousands of clusters, so paying those serially
every run puts the pipeline in the tens of minutes. None of the three
changes meaningfully between runs at the precision the model consumes
them, so they are cached on disk keyed by rounded coordinates.

SQLite rather than a dict or a pickle because the cache has to survive
process restarts (the point is that the *second* run of the day is fast)
and be safe to read from the API process while the enrichment worker
writes to it. WAL mode plus SQLite's own locking handles that; no extra
service to run, which matters because this project already has Redis
unavailable in the local setup.

Nothing here ever fabricates a value: a miss returns the sentinel MISS
and the caller does the real lookup. The cache only ever replays a real
measurement that was actually fetched earlier.
"""
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Overridable so a test run, or a second checkout, never shares one
# process's cache with another's.
DEFAULT_CACHE_PATH = Path(os.environ.get("FIRESIGHT_GEO_CACHE", "data/cache/geo_cache.sqlite"))


def is_disabled() -> bool:
    """Whether caching is turned off for this process.

    Set FIRESIGHT_GEO_CACHE_DISABLED=1 to make every lookup go to the
    real source. Tests need this: a cached value survives between tests,
    so a test that patches the underlying fetcher would otherwise be
    served an answer recorded by an earlier test and never call its own
    mock. The conftest sets it for the whole suite.
    """
    return os.environ.get("FIRESIGHT_GEO_CACHE_DISABLED", "").strip().lower() in {"1", "true", "yes"}

# Coordinate rounding per namespace, in decimal places. 2dp ~ 1.1 km,
# which is finer than the sampling buffers these values are derived from
# (cloud fraction uses a 2 km box, land cover a 0.5-20 km buffer), so
# rounding cannot move an answer outside the window it was measured over.
_COORD_PRECISION = 2

# How long a cached value stays usable, per namespace, in seconds.
# Land cover is an annual product, so a long TTL is honest. Cloud
# fraction and wind are observations tied to a moment and are keyed by
# date as well, so their TTL only bounds cache growth.
_TTL_SECONDS = {
    "landcover": 30 * 24 * 3600,
    "cloud_fraction": 7 * 24 * 3600,
    "wind": 6 * 3600,
}
_DEFAULT_TTL = 24 * 3600


class _Miss:
    """Distinct from None, because None is a legitimate cached answer.

    'This location has no WorldCover tile' is a real, expensive-to-derive
    result worth remembering; collapsing it into the miss sentinel would
    re-fetch it on every single run.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<MISS>"


MISS = _Miss()

_local = threading.local()


def _connect(path: Path) -> sqlite3.Connection:
    """One connection per thread — SQLite connections are not shareable
    across threads, and the prefetch paths are thread-pooled.
    """
    existing = getattr(_local, "conn", None)
    if existing is not None and getattr(_local, "path", None) == str(path):
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")       # concurrent reader + writer
    conn.execute("PRAGMA synchronous=NORMAL")     # durability is not critical for a cache
    conn.execute(
        "CREATE TABLE IF NOT EXISTS geo_cache ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  stored_at REAL NOT NULL"
        ")"
    )
    conn.commit()
    _local.conn = conn
    _local.path = str(path)
    return conn


def make_key(namespace: str, lon: float, lat: float, *extra: Any) -> str:
    """Cache key from a namespace, a rounded coordinate and any extras
    (e.g. the acquisition date, or the sampling radius).
    """
    parts = [
        namespace,
        f"{round(float(lon), _COORD_PRECISION):.2f}",
        f"{round(float(lat), _COORD_PRECISION):.2f}",
    ]
    parts.extend("" if e is None else str(e) for e in extra)
    return "|".join(parts)


def get(key: str, cache_path: Optional[Path] = None) -> Any:
    """Cached value, or MISS if absent/expired/unreadable."""
    if is_disabled():
        return MISS
    path = cache_path or DEFAULT_CACHE_PATH
    try:
        row = _connect(path).execute(
            "SELECT value, stored_at FROM geo_cache WHERE key = ?", (key,)
        ).fetchone()
    except sqlite3.Error:
        logger.debug("Geo cache read failed for %s", key, exc_info=True)
        return MISS

    if row is None:
        return MISS

    value_json, stored_at = row
    ttl = _TTL_SECONDS.get(key.split("|", 1)[0], _DEFAULT_TTL)
    if time.time() - stored_at > ttl:
        return MISS

    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return MISS


def put(key: str, value: Any, cache_path: Optional[Path] = None) -> None:
    """Store a value. Cache failures are never fatal — a cache that
    cannot be written must degrade to 'slow', not to 'broken'.
    """
    if is_disabled():
        return
    path = cache_path or DEFAULT_CACHE_PATH
    try:
        conn = _connect(path)
        conn.execute(
            "INSERT OR REPLACE INTO geo_cache (key, value, stored_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        conn.commit()
    except (sqlite3.Error, TypeError, ValueError):
        logger.debug("Geo cache write failed for %s", key, exc_info=True)


def get_or_compute(key: str, compute: Callable[[], Any], cache_path: Optional[Path] = None) -> Any:
    """Return the cached value, else call `compute` and cache its result.

    An exception from `compute` propagates and is deliberately NOT
    cached: caching a failure would turn one transient network error
    into a value that looks settled for the whole TTL.
    """
    cached = get(key, cache_path)
    if cached is not MISS:
        return cached
    value = compute()
    put(key, value, cache_path)
    return value


def stats(cache_path: Optional[Path] = None) -> dict:
    """Row counts per namespace — for the ops/status endpoint."""
    path = cache_path or DEFAULT_CACHE_PATH
    try:
        rows = _connect(path).execute(
            "SELECT substr(key, 1, instr(key, '|') - 1) AS ns, count(*) "
            "FROM geo_cache GROUP BY ns"
        ).fetchall()
    except sqlite3.Error:
        return {}
    return {ns: count for ns, count in rows if ns}
