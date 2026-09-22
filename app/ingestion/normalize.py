"""Canonical detection schema + per-source normalizers.

Every ingestion connector (FIRMS, INSAT-3DS, Sentinel-3 FRP, Himawari)
converts its raw records into this one canonical dict shape before
anything else in the pipeline (dedup, clustering, DB insert) touches it.
Keeping the mapping isolated here means dedup/clustering never need to
know sensor-specific field names.

Canonical fields (all required except where noted):
    source: str            e.g. 'VIIRS_SNPP', 'MODIS', 'INSAT3DS', 'SENTINEL3_FRP', 'HIMAWARI'
    lon: float
    lat: float
    acq_datetime: datetime (UTC, tz-aware)
    frp: float | None      fire radiative power in MW
    brightness: float | None   brightness temperature in Kelvin
    confidence: float | None   normalized to 0.0-1.0 (raw scales differ per sensor)
    daynight: 'D' | 'N' | None
    scan: float | None     along-scan pixel size (km) -- FIRMS only
    track: float | None    along-track pixel size (km) -- FIRMS only
    raw_payload: dict      original record, kept for audit/debugging
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

CANONICAL_FIELDS = (
    "source", "lon", "lat", "acq_datetime", "frp", "brightness",
    "confidence", "daynight", "scan", "track", "raw_payload",
)


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_viirs_confidence(raw: Any) -> Optional[float]:
    """VIIRS confidence is categorical: 'l' (low), 'n' (nominal), 'h' (high)."""
    mapping = {"l": 0.3, "low": 0.3, "n": 0.7, "nominal": 0.7, "h": 0.95, "high": 0.95}
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return mapping.get(key)


def normalize_modis_confidence(raw: Any) -> Optional[float]:
    """MODIS confidence is numeric 0-100."""
    val = _to_float(raw)
    if val is None:
        return None
    return max(0.0, min(1.0, val / 100.0))


def firms_row_to_canonical(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Map one row of a FIRMS CSV (VIIRS or MODIS) to the canonical schema.

    FIRMS CSV columns (both VIIRS and MODIS variants share the same core
    set): latitude, longitude, acq_date, acq_time, confidence, frp,
    daynight, scan, track, and either bright_ti4 (VIIRS) or brightness (MODIS).
    """
    acq_date = row["acq_date"]  # 'YYYY-MM-DD'
    acq_time = str(row["acq_time"]).zfill(4)  # 'HHMM'
    acq_datetime = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M").replace(
        tzinfo=timezone.utc
    )

    is_viirs = "VIIRS" in source
    brightness = _to_float(row.get("bright_ti4")) if is_viirs else _to_float(row.get("brightness"))
    confidence = (
        normalize_viirs_confidence(row.get("confidence"))
        if is_viirs
        else normalize_modis_confidence(row.get("confidence"))
    )

    return {
        "source": source,
        "lon": _to_float(row["longitude"]),
        "lat": _to_float(row["latitude"]),
        "acq_datetime": acq_datetime,
        "frp": _to_float(row.get("frp")),
        "brightness": brightness,
        "confidence": confidence,
        "daynight": row.get("daynight"),
        "scan": _to_float(row.get("scan")),
        "track": _to_float(row.get("track")),
        "raw_payload": dict(row),
    }


def generic_pixel_to_canonical(
    *,
    source: str,
    lon: float,
    lat: float,
    acq_datetime: datetime,
    frp: Optional[float] = None,
    brightness: Optional[float] = None,
    confidence: Optional[float] = None,
    daynight: Optional[str] = None,
    raw_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical-schema builder for non-FIRMS sources (INSAT-3DS, Sentinel-3
    FRP, Himawari) whose native record shape varies by product and is best
    parsed close to the source rather than forced through the FIRMS mapper.
    """
    if acq_datetime.tzinfo is None:
        acq_datetime = acq_datetime.replace(tzinfo=timezone.utc)
    return {
        "source": source,
        "lon": float(lon),
        "lat": float(lat),
        "acq_datetime": acq_datetime,
        "frp": _to_float(frp),
        "brightness": _to_float(brightness),
        "confidence": _to_float(confidence),
        "daynight": daynight,
        "scan": None,
        "track": None,
        "raw_payload": raw_payload or {},
    }
