"""Reverse-geocode a detection to its Indian state and district.

`gateway/routes_dashboard.py` shipped `"state": None, "district": None`
with a note saying they were "resolved client-side from the geo tables".
No such client-side resolution existed, so every detection reached the
dashboard with no administrative context at all — which is what forced
the old dashboard to label its hierarchy "representative sample values".

This resolves them for real, by point-in-polygon against published
district boundaries (data/boundaries/india_districts.geojson: 760
districts, 36 states/UTs, current post-2019 names and splits, so
Telangana and Ladakh are present rather than folded into their
predecessors).

An STRtree index over the district polygons makes a lookup ~10 us, so
this runs inline on the detections endpoint rather than needing to be
precomputed into a table. The index is built once per process and held
in module state.

Returns (None, None) outside India's land boundaries rather than
snapping to the nearest district: a detection just offshore or across
the border is genuinely not in an Indian district, and inventing one
would put a fabricated location in front of an operator.
"""
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BOUNDARY_PATH = Path("data/boundaries/india_districts.geojson")

_lock = threading.Lock()
_index = None          # shapely STRtree over district polygons
_geometries: List = []  # polygon per tree entry, index-aligned
_properties: List[Dict] = []  # {"state", "district"} per tree entry, index-aligned
# id(geometry) -> position, for the Shapely 1.8 query() return shape.
_geometry_positions: Dict[int, int] = {}
_load_failed = False


def _load(boundary_path: Optional[Path] = None) -> bool:
    """Build the spatial index once. Returns whether it is usable.

    A missing or unreadable boundary file is logged once and then
    degrades to "no administrative context" — the dashboard shows the
    coordinates it always had, rather than the whole endpoint failing.
    """
    global _index, _geometries, _properties, _geometry_positions, _load_failed

    if _index is not None:
        return True
    if _load_failed:
        return False

    with _lock:
        if _index is not None:
            return True
        if _load_failed:
            return False

        path = boundary_path or DEFAULT_BOUNDARY_PATH
        try:
            from shapely.geometry import shape
            from shapely.strtree import STRtree
        except ImportError:
            logger.warning("shapely not installed — state/district resolution unavailable")
            _load_failed = True
            return False

        if not path.is_file():
            logger.warning(
                "District boundaries not found at %s — state/district will be null. "
                "Fetch them with scripts/bulk_load_boundaries.py", path,
            )
            _load_failed = True
            return False

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            geometries, properties = [], []
            for feature in payload.get("features", []):
                geometry = feature.get("geometry")
                if not geometry:
                    continue
                props = feature.get("properties", {})
                geometries.append(shape(geometry))
                properties.append({
                    "state": props.get("st_nm") or props.get("ST_NM"),
                    "district": props.get("district") or props.get("DISTRICT"),
                })

            if not geometries:
                raise ValueError("boundary file contains no usable features")

            _geometries = geometries
            _properties = properties
            _geometry_positions = {id(g): i for i, g in enumerate(geometries)}
            _index = STRtree(geometries)
        except Exception:
            logger.warning("Failed to load district boundaries from %s", path, exc_info=True)
            _load_failed = True
            return False

    logger.info("Loaded %d district boundaries from %s", len(_geometries), path)
    return True


def resolve(lon: float, lat: float) -> Tuple[Optional[str], Optional[str]]:
    """(state, district) containing the point, or (None, None) outside India."""
    if not _load():
        return None, None

    try:
        from shapely.geometry import Point
    except ImportError:
        return None, None

    point = Point(lon, lat)

    # STRtree narrows to bounding-box candidates; the containment test
    # against the real polygon is still needed because a district's bbox
    # overlaps its neighbours'.
    #
    # Every containing polygon is checked rather than returning the first
    # hit: alongside its 726 district polygons the source file carries 34
    # whole-state outlines with no `district` property, and those overlap
    # the districts inside them. Taking the first match returned a state
    # with a null district for any point that happened to hit the outline
    # first, so a district-bearing match wins and the outline is only the
    # fallback.
    fallback: Optional[Tuple[Optional[str], Optional[str]]] = None

    for candidate in _index.query(point):
        # Shapely >=2 yields positional indices (as numpy integers);
        # Shapely 1.8 yielded the geometries themselves. Handle both so a
        # version bump does not silently stop resolving every point.
        if isinstance(candidate, (int, np.integer)):
            index = int(candidate)
            geometry = _geometries[index]
        else:
            geometry = candidate
            index = _geometry_positions.get(id(candidate))
            if index is None:
                continue

        if not geometry.contains(point):
            continue

        props = _properties[index]
        if props.get("district"):
            return props.get("state"), props.get("district")
        if fallback is None:
            fallback = (props.get("state"), None)

    return fallback if fallback is not None else (None, None)


def resolve_many(points: List[Tuple[float, float]]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Resolve a batch — one index build, one pass, for the detections endpoint."""
    return [resolve(lon, lat) for lon, lat in points]


def available() -> bool:
    """Whether resolution is working — surfaced on /health so a missing
    boundary file is visible rather than silently blanking every state.
    """
    return _load()


def state_names() -> List[str]:
    """Every state/UT present in the boundary file, sorted.

    The dashboard's state filter is populated from this rather than a
    hardcoded list, so the dropdown can never drift from what the
    resolver can actually return.
    """
    if not _load():
        return []
    return sorted({p["state"] for p in _properties if p.get("state")})
