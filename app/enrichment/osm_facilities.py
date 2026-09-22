"""OSM industrial-facility ingestion (Overpass API) + facility proximity
scoring.

Overpass docs: https://wiki.openstreetmap.org/wiki/Overpass_API
Tags queried: landuse=industrial, power=plant, man_made=works,
pipeline=*, industrial=oil / refinery / mine, per the architecture's
"Custom Industrial Infrastructure Database" (Phase 2, item 9).

Proximity scoring formula (used to seed `facilities.prior_weight` context
for M2's classifier and M3's attribution): prior_weight = base_weight *
exp(-distance_m / 500), matching the decay function referenced in
SIH26162 planning notes for facility-attribution probability.
"""
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import requests
from shapely.geometry import Point, Polygon

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_FACILITY_TAGS = [
    '["landuse"="industrial"]',
    '["power"="plant"]',
    '["man_made"="works"]',
    '["industrial"="oil"]',
    '["industrial"="refinery"]',
    '["man_made"="petroleum_well"]',
    '["landuse"="quarry"]',
]

_FACILITY_TYPE_BY_TAG = {
    "power=plant": "power",
    "landuse=industrial": "industrial",
    "man_made=works": "industrial",
    "industrial=oil": "refinery",
    "industrial=refinery": "refinery",
    "man_made=petroleum_well": "oil_well",
    "landuse=quarry": "mine",
}


def build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    """bbox = (west, south, east, north) -> Overpass (south,west,north,east) order."""
    west, south, east, north = bbox
    bbox_str = f"{south},{west},{north},{east}"
    clauses = "\n".join(f"  node{tag}({bbox_str});\n  way{tag}({bbox_str});" for tag in _FACILITY_TAGS)
    return f"""
[out:json][timeout:120];
(
{clauses}
);
out geom;
""".strip()


# Overpass rejects a single whole-India query for these seven tag sets
# outright ("406 Not Acceptable" — the query is too heavy for the public
# instance), so the bbox is walked in tiles and the results concatenated.
# 2 degrees keeps each tile comfortably inside the public rate/complexity
# limits while keeping the tile count (~150 over India) low enough that
# the polite inter-request delay doesn't dominate the run.
OVERPASS_TILE_DEGREES = 2.0
# The public instance 429s anonymous clients that hammer it; identifying
# the caller and pausing between tiles is what its usage policy asks for.
OVERPASS_USER_AGENT = "FireSight-SIH162/1.0 (thermal anomaly classification; academic)"
OVERPASS_TILE_DELAY_SECONDS = 1.0


def tile_bbox(
    bbox: Tuple[float, float, float, float], step_degrees: float = OVERPASS_TILE_DEGREES
) -> List[Tuple[float, float, float, float]]:
    """Split (west, south, east, north) into <=step_degrees square tiles."""
    west, south, east, north = bbox
    tiles: List[Tuple[float, float, float, float]] = []
    lat = south
    while lat < north:
        lat_hi = min(lat + step_degrees, north)
        lon = west
        while lon < east:
            lon_hi = min(lon + step_degrees, east)
            tiles.append((lon, lat, lon_hi, lat_hi))
            lon = lon_hi
        lat = lat_hi
    return tiles


# The public instances all enforce per-client slot limits and answer 429
# (no free slot) or 504 (query exceeded its slot's time) under load.
# Rotating across independent mirrors and backing off is what turns a
# 240-tile walk from "mostly 429s" into a complete load. Ordered by
# observed throughput for this query shape.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OVERPASS_MAX_ATTEMPTS = 6
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


def _fetch_tile(
    url: str,
    bbox: Tuple[float, float, float, float],
    preferred: Optional[str] = None,
) -> List[Dict]:
    """One tile's raw Overpass elements; [] once every attempt is spent.

    `url` is the configured primary; the other mirrors are tried in turn
    on a retryable failure. `preferred` pins the first attempt to one
    specific mirror, which is how the parallel loader keeps each of its
    workers on a different instance instead of all of them queueing for
    a slot on the same one. A single tile ultimately failing must not
    lose the other tiles' work, so it is logged and skipped rather than
    raised — the caller reports how many tiles came back empty.
    """
    query = build_overpass_query(bbox)
    endpoints = [url] + [m for m in OVERPASS_MIRRORS if m != url]
    if preferred:
        endpoints = [preferred] + [e for e in endpoints if e != preferred]

    for attempt in range(OVERPASS_MAX_ATTEMPTS):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            resp = requests.post(
                endpoint, data={"data": query}, timeout=240,
                headers={"User-Agent": OVERPASS_USER_AGENT},
            )
            if resp.status_code in _RETRYABLE_STATUS:
                # Exponential backoff, capped — a 429 means "no slot free
                # yet", so waiting is the correct response, not failing.
                delay = min(2 ** attempt, 60)
                logger.info(
                    "Overpass %s on %s for tile %s — retrying in %ss (attempt %d/%d)",
                    resp.status_code, endpoint, bbox, delay, attempt + 1, OVERPASS_MAX_ATTEMPTS,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except (requests.RequestException, ValueError):
            delay = min(2 ** attempt, 60)
            logger.info(
                "Overpass request error on %s for tile %s — retrying in %ss (attempt %d/%d)",
                endpoint, bbox, delay, attempt + 1, OVERPASS_MAX_ATTEMPTS, exc_info=True,
            )
            time.sleep(delay)

    logger.warning("Overpass tile %s exhausted all %d attempts — skipping", bbox, OVERPASS_MAX_ATTEMPTS)
    return []


def fetch_osm_facilities(settings: Optional[Settings] = None) -> List[Dict]:
    """Query Overpass for industrial facilities across the configured
    India bbox and return normalized facility records ready for the
    `facilities` table.

    Walks the bbox in tiles (see OVERPASS_TILE_DEGREES) and de-duplicates
    on OSM (type, id), because a way straddling a tile edge is returned
    by both tiles that cover it.
    """
    settings = settings or get_settings()
    tiles = tile_bbox(settings.osm_bbox_tuple)
    logger.info("Overpass: fetching %d tiles across %s", len(tiles), settings.osm_bbox_tuple)

    elements: List[Dict] = []
    seen_ids = set()
    for i, tile in enumerate(tiles, 1):
        for el in _fetch_tile(settings.overpass_url, tile):
            identity = (el.get("type"), el.get("id"))
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            elements.append(el)
        if i % 20 == 0:
            logger.info("Overpass: %d/%d tiles, %d unique elements", i, len(tiles), len(elements))
        time.sleep(OVERPASS_TILE_DELAY_SECONDS)

    facilities = normalize_elements(elements)
    logger.info("Fetched %d OSM facilities", len(facilities))
    return facilities


def normalize_elements(elements: List[Dict]) -> List[Dict]:
    """Map raw Overpass elements to `facilities`-table records.

    Split out from `fetch_osm_facilities` so the tiled bulk loader can
    normalise and commit one tile at a time instead of holding every
    tile's elements in memory until the whole walk finishes.
    """
    facilities = []
    for el in elements:
        tags = el.get("tags", {})
        facility_type = "industrial"
        for key_val, ftype in _FACILITY_TYPE_BY_TAG.items():
            key, _, val = key_val.partition("=")
            if tags.get(key) == val:
                facility_type = ftype
                break

        if el["type"] == "node":
            lon, lat = el.get("lon"), el.get("lat")
            if lon is None or lat is None:
                continue
            geom = Point(lon, lat)
        else:
            # way -> Overpass "out geom" gives the full vertex list, not
            # just a centroid, so a real Polygon can be stored instead of
            # collapsing the facility to a point. This is what lets
            # ST_Contains actually work in the attribution query instead
            # of always evaluating false.
            geometry = el.get("geometry")
            if not geometry:
                continue
            coords = [(pt["lon"], pt["lat"]) for pt in geometry if pt]
            if len(coords) < 2:
                continue

            closed = len(coords) >= 4 and coords[0] == coords[-1]
            if closed:
                try:
                    geom = Polygon(coords)
                except ValueError:
                    continue
                if not geom.is_valid or geom.is_empty:
                    geom = geom.buffer(0)
                if not geom.is_valid or geom.is_empty:
                    continue
            else:
                # Open way (e.g. a pipeline) has no interior — fall back
                # to its midpoint the same way a node would be stored.
                mid_lon, mid_lat = coords[len(coords) // 2]
                geom = Point(mid_lon, mid_lat)

            centroid = geom.centroid
            lon, lat = centroid.x, centroid.y

        facilities.append(
            {
                "name": tags.get("name"),
                "facility_type": facility_type,
                "source": "OSM",
                "lon": lon,
                "lat": lat,
                "geom": geom,
                "prior_weight": 1.0,
                "metadata": tags,
            }
        )
    return facilities


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def facility_proximity_score(distance_m: float, base_weight: float = 1.0, decay_m: float = 500.0) -> float:
    """prior_weight = base_weight * exp(-distance_m / decay_m).

    At distance=0, score=base_weight; at distance=decay_m, score has
    decayed to ~37% of base_weight; effectively negligible beyond ~2km.
    """
    return base_weight * math.exp(-distance_m / decay_m)


def find_nearest_facility(
    cluster_lon: float,
    cluster_lat: float,
    facilities: List[Dict],
    max_distance_m: float = 5000.0,
) -> Optional[Dict]:
    """Pure-Python nearest-facility search (used for unit testing and
    small batches); production enrichment should prefer the PostGIS
    ST_DWithin + ST_Distance query for speed at scale, see
    `nearest_facility_sql` below.
    """
    best = None
    best_distance = None
    for facility in facilities:
        distance = haversine_m(cluster_lon, cluster_lat, facility["lon"], facility["lat"])
        if distance > max_distance_m:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = facility, distance

    if best is None:
        return None
    return {
        **best,
        "distance_m": best_distance,
        "prior_weight": facility_proximity_score(best_distance, best.get("prior_weight", 1.0)),
    }


NEAREST_FACILITY_SQL = """
SELECT f.id, f.name, f.facility_type,
       ST_Distance(f.geom::geography, c.centroid::geography) AS distance_m
FROM facilities f, clusters c
WHERE c.id = :cluster_id
  AND ST_DWithin(f.geom::geography, c.centroid::geography, :max_distance_m)
ORDER BY distance_m ASC
LIMIT 1;
"""

# A landuse=industrial polygon bulk-imported from Maxar-derived land-use
# data (the common case in this DB) routinely carries no `name` tag at
# all — the actual named company is a separate OSM element the bulk
# ingestion's tag list above never queried for. Rather than leave the
# dashboard's LOCATION field silent for every such fire, or re-walk all
# of India with a broader tag set, this does one live, narrow Overpass
# query for the nearest genuinely NAMED industrial-ish feature — same
# "real data, or nothing" contract as
# geospatial.wui_analysis._nearest_settlement_name, which this mirrors.
NAMED_FACILITY_MAX_EXCESS_M = 400.0


def named_feature_is_same_site(
    candidate: Optional[Dict], matched_distance_m: Optional[float],
) -> bool:
    """Is `candidate` (a nearest_named_industrial_feature() result)
    close enough to the already-matched (but unnamed) facility's own
    distance to plausibly be the same site, or a second, unrelated
    named feature the wider search radius happened to pick up? Same
    reasoning as geospatial.wui_analysis.settlement_matches_builtup_pixel.
    """
    if candidate is None or matched_distance_m is None:
        return False
    candidate_distance = candidate.get("distance_m")
    if candidate_distance is None:
        return False
    return candidate_distance <= matched_distance_m + NAMED_FACILITY_MAX_EXCESS_M


def nearest_named_industrial_feature(
    lon: float, lat: float, radius_m: float = 1500.0,
) -> Optional[Dict]:
    """The nearest OSM element with both a `name` tag and an
    industrial/power tag within `radius_m` — or None if nothing real is
    that close. Never invents a name; a facility with no named OSM
    neighbour simply gets none."""
    from app import geo_cache

    cache_key = geo_cache.make_key("named_facility", lon, lat, int(radius_m))
    cached = geo_cache.get(cache_key)
    if cached is not geo_cache.MISS:
        return cached

    tag_clauses = "\n  ".join(
        f'{kind}["name"]{tag}(around:{{radius:.0f}},{{lat}},{{lon}});'
        for tag in _FACILITY_TAGS
        for kind in ("node", "way")
    )
    query = (
        "[out:json][timeout:15];\n(\n  " + tag_clauses + "\n);\nout center 10;"
    ).format(radius=radius_m, lat=lat, lon=lon)

    try:
        resp = requests.post(
            OVERPASS_MIRRORS[0], data={"data": query}, timeout=15,
            headers={"User-Agent": OVERPASS_USER_AGENT},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        logger.info(
            "Named-facility lookup failed for (%s, %s)", lon, lat, exc_info=True,
        )
        return None

    candidates = []
    for el in elements:
        name = (el.get("tags") or {}).get("name")
        if not name:
            continue
        center = el.get("center") or el  # nodes carry lat/lon directly, ways use "center"
        el_lon, el_lat = center.get("lon"), center.get("lat")
        if el_lon is None or el_lat is None:
            continue
        candidates.append({
            "name": name,
            "distance_m": haversine_m(lon, lat, el_lon, el_lat),
            "lon": el_lon, "lat": el_lat,
        })

    result = min(candidates, key=lambda c: c["distance_m"]) if candidates else None
    geo_cache.put(cache_key, result)
    return result
