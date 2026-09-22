"""Deduplication of canonical hotspot records.

Two duplicate scenarios are handled:
1. Exact re-poll duplicates: the same sensor reports the same pixel again
   because a scheduled poll window overlaps the previous one. Matched by
   (source, lon, lat, acq_datetime) — this mirrors the DB's
   `uq_hotspot_identity` unique constraint, so this pass is a fast
   in-memory pre-filter before the DB's ON CONFLICT DO NOTHING catches
   anything that slips through.
2. Near-duplicate re-detections: the same physical pixel reported with
   floating-point jitter in lat/lon, or the same sensor's overlapping
   swaths on consecutive passes. Matched by rounding coordinates to a
   tolerance and truncating time to a coarse bucket.

Cross-sensor "duplicates" (e.g. FIRMS VIIRS and Sentinel-3 FRP both
seeing the same physical fire) are intentionally NOT collapsed here —
that's clustering's job (cluster.py), and provenance per sensor must
survive into the `hotspots` table for M2's evidence-weighting engine.
"""
from typing import Dict, List, Tuple

_COORD_DECIMALS = 4  # ~11m at the equator, tight enough to only catch true re-reports


def _exact_key(record: Dict) -> Tuple:
    return (
        record["source"],
        round(record["lon"], 6),
        round(record["lat"], 6),
        record["acq_datetime"],
    )


def _near_dup_key(record: Dict) -> Tuple:
    return (
        record["source"],
        round(record["lon"], _COORD_DECIMALS),
        round(record["lat"], _COORD_DECIMALS),
        record["acq_datetime"].strftime("%Y-%m-%dT%H:%M"),
    )


def dedup_exact(records: List[Dict]) -> List[Dict]:
    """Drop records that are byte-for-byte identical detections."""
    seen = set()
    result = []
    for record in records:
        key = _exact_key(record)
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def dedup_near(records: List[Dict]) -> List[Dict]:
    """Drop near-identical re-detections (same sensor, same minute, coords
    within ~11m) that survive the exact-match pass due to float jitter.
    Keeps the first occurrence (records are expected to already be
    time-ordered by the caller).
    """
    seen = set()
    result = []
    for record in records:
        key = _near_dup_key(record)
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def exact_key_for_db_row(source: str, lon: float, lat: float, acq_datetime) -> Tuple:
    """Public counterpart of `_exact_key`, for callers (e.g. the pipeline)
    building comparison keys from rows already read back out of the DB,
    where they don't have a canonical record dict on hand.
    """
    return (source, round(lon, 6), round(lat, 6), acq_datetime)


def dedup_against_existing(records: List[Dict], existing_keys: set) -> List[Dict]:
    """Filter out records already persisted in a previous ingestion cycle.

    `existing_keys` must be a set of tuples shaped like `_exact_key(...)`
    / `exact_key_for_db_row(...)` — (source, round(lon,6), round(lat,6),
    acq_datetime) — pulled from the DB for the relevant time window
    before calling this.
    """
    return [r for r in records if _exact_key(r) not in existing_keys]


def full_dedup(records: List[Dict], existing_keys: set | None = None) -> List[Dict]:
    """Convenience pipeline: exact -> near -> against-existing-DB-rows."""
    deduped = dedup_exact(records)
    deduped = dedup_near(deduped)
    if existing_keys is not None:
        deduped = dedup_against_existing(deduped, existing_keys)
    return deduped
