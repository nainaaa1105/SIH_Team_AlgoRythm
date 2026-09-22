"""Thermal / temporal / spatial feature computation.

These are pure functions over plain dicts (one per `hotspots` row of a
cluster) so they can be unit-tested without a database. `assemble.py`
does the DB reads and hands the rows here.

Why these particular features, per the project's research notes:
  - temporal behaviour separates a flare (same spot for weeks, steady
    FRP) from a wildfire (sudden onset, spreads, vanishes) from an
    accidental industrial fire (sharp FRP spike then gone);
  - spatial behaviour is the other half of that — a wildfire expands
    across vegetation while a flare stays a fixed point, so footprint
    extent and its growth rate are first-class discriminators, not
    afterthoughts;
  - source_diversity (how many independent sensors saw it) is a
    corroboration signal that feeds the evidence-weighting engine.
"""
import math
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence

_EARTH_RADIUS_KM = 6371.0


def _values(rows: Sequence[Dict], key: str) -> List[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def _safe_max(values: Sequence[float]) -> Optional[float]:
    return float(max(values)) if values else None


def _safe_std(values: Sequence[float]) -> Optional[float]:
    """Population stdev; 0.0 for a single observation (no spread, not unknown)."""
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def spatial_extent_km(rows: Sequence[Dict]) -> float:
    """Max pairwise distance between detections in the cluster — the
    footprint diameter. A persistent flare stays near 0; a spreading
    wildfire grows.

    O(n^2), which is fine: a cluster is a handful to a few hundred
    detections, not millions.
    """
    points = [(r["lon"], r["lat"]) for r in rows if r.get("lon") is not None and r.get("lat") is not None]
    if len(points) < 2:
        return 0.0
    return max(
        haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )


def spatial_growth_rate(rows: Sequence[Dict]) -> float:
    """Change in footprint extent per day, comparing the first half of the
    cluster's life to the whole of it. Positive => spreading (wildfire-like);
    ~0 => stationary (flare/furnace-like).
    """
    dated = [r for r in rows if r.get("acq_datetime") is not None]
    if len(dated) < 4:
        return 0.0

    ordered = sorted(dated, key=lambda r: r["acq_datetime"])
    span_days = (ordered[-1]["acq_datetime"] - ordered[0]["acq_datetime"]).total_seconds() / 86400.0
    if span_days <= 0:
        return 0.0

    midpoint = len(ordered) // 2
    early_extent = spatial_extent_km(ordered[:midpoint])
    full_extent = spatial_extent_km(ordered)
    return (full_extent - early_extent) / span_days


def persistence_days(rows: Sequence[Dict]) -> float:
    dated = _values(rows, "acq_datetime")
    if not dated:
        return 0.0
    span = (max(dated) - min(dated)).total_seconds() / 86400.0
    return float(span)


def frp_zscore(rows: Sequence[Dict], baseline_mean: Optional[float], baseline_std: Optional[float]) -> Optional[float]:
    """How anomalous this cluster's peak FRP is versus its own history.

    The baseline comes from the cluster's longer FRP time series (M1
    stores 90 days of detections in TimescaleDB). This is the
    "currently 4.2x its normal FRP" signal from the project brief — the
    normal-vs-abnormal discriminator for an already-known industrial
    source.

    Returns None when there's no usable baseline, and 0.0 when the
    baseline has no variance but the value matches it (a perfectly steady
    flare is not anomalous).
    """
    current = _safe_max(_values(rows, "frp"))
    if current is None or baseline_mean is None:
        return None

    if not baseline_std:
        # A zero-variance baseline means a perfectly steady source. Any
        # departure from it is anomalous but the z-score is undefined
        # (division by zero), so clamp to a signed sentinel rather than
        # returning inf, which XGBoost cannot split on sensibly.
        if math.isclose(current, baseline_mean):
            return 0.0
        return 3.0 if current > baseline_mean else -3.0

    return float((current - baseline_mean) / baseline_std)


def night_fraction(rows: Sequence[Dict]) -> Optional[float]:
    flags = [r["daynight"] for r in rows if r.get("daynight") in ("D", "N")]
    if not flags:
        return None
    return sum(1 for f in flags if f == "N") / len(flags)


def source_diversity(rows: Sequence[Dict]) -> int:
    """Distinct *platforms*, not distinct product feeds.

    VIIRS_SNPP_NRT and VIIRS_NOAA20_NRT are two satellites carrying the
    same instrument, so they do count as independent looks; but the NRT
    suffix and instrument variants of one platform must not inflate the
    count, hence the normalisation below.
    """
    platforms = set()
    for row in rows:
        source = (row.get("source") or "").upper()
        if not source:
            continue
        platforms.add(source.replace("_NRT", "").replace("_RT", ""))
    return len(platforms)


def compute_thermal_features(
    rows: Sequence[Dict],
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """All M2-owned features for one cluster, from its detection rows.

    Each row is a dict with keys: lon, lat, acq_datetime (datetime),
    frp, brightness, confidence, daynight, source.
    """
    frps = _values(rows, "frp")
    brightnesses = _values(rows, "brightness")
    confidences = _values(rows, "confidence")

    span = persistence_days(rows)
    n = len(rows)

    return {
        "frp_mean": _safe_mean(frps),
        "frp_max": _safe_max(frps),
        "frp_std": _safe_std(frps),
        "frp_zscore": frp_zscore(rows, baseline_mean, baseline_std),
        "brightness_mean": _safe_mean(brightnesses),
        "brightness_max": _safe_max(brightnesses),
        "confidence_mean": _safe_mean(confidences),
        "persistence_days": span,
        "n_detections": float(n),
        # A cluster seen many times in one day is a different animal from
        # one seen once a day for a month; guard the <1-day case so a
        # short burst doesn't divide by a near-zero span and explode.
        "detections_per_day": float(n) / max(span, 1.0),
        "night_fraction": night_fraction(rows),
        "source_diversity": float(source_diversity(rows)),
        "spatial_extent_km": spatial_extent_km(rows),
        "spatial_growth_rate": spatial_growth_rate(rows),
    }
