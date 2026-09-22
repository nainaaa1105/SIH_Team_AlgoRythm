"""WUI (Wildland-Urban Interface) proximity evaluation.

The problem statement, in one sentence: a spreading fire that gets close
enough to human development starts an ember-cast risk that has nothing
to do with the main fire front actually reaching the buildings — burning
embers can start new fires up to a couple of kilometres ahead of it. This
module answers, for one already-classified cluster, "how close is it to
the nearest built-up land, and how urgent is that."

Deliberately built as a thin layer over machinery that already exists
and is already tested rather than a parallel implementation:

  * the fire-front trajectory and conservative spread rate come from
    `geospatial.threat.trajectory` — the same physics
    `geospatial.tasks._build_and_store_corridor` uses for the industrial
    threat corridor, so a fire's WUI ETA and its industrial-facility ETA
    are never inconsistent with each other;
  * "how far to the nearest settlement" is answered two ways and
    combined honestly: `landcover_worldcover_cog.nearest_builtup_pixel_cog`
    gives the authoritative distance/bearing from a real 10 m 2021
    satellite land-cover product (this is the actual wildland-urban
    boundary, not a proxy for it), and `_nearest_settlement_name` below
    gives a human-readable label from real OpenStreetMap place data when
    one exists nearby. Neither ever invents a name or a number — a
    cluster with no built-up land in range gets `wui_threat: False` and
    a stated reason, not a plausible-looking guess;
  * the corridor polygon reuses `geospatial.threat.corridor.corridor_polygon`,
    the same wedge geometry the industrial threat corridor draws, aimed
    at the built-up pixel's bearing instead of the general spread bearing.

Gating: only fires the classifier put in `corridor.SPREADING_CLASSES`
(wildfire, agricultural_burning) are evaluated at all. A gas flare, an
industrial fire or a mine does not advance across the landscape, so a
WUI corridor for one would be actively misleading — same reasoning
`corridor.should_build_corridor` already applies, reused here rather
than re-derived.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Standard ember-cast / spotting distance cited in wildland-urban
# interface fire-behaviour literature (CAL FIRE / NFPA guidance uses the
# same figure): a fire within this range of built-up land can ignite it
# by ember-cast alone, independent of whether the main front ever
# arrives. This is a policy/physical threshold, not an environment
# tunable — see the note in geospatial/config.py.
EMBER_JUMP_THRESHOLD_M = 2400.0

# Anything within this multiple of the ember-jump distance is worth a
# human glance even though it is not (yet) in ember range.
WATCH_MULTIPLIER = 3.0

# "within a 12-hour projected burn window" — the second, independent
# trigger alongside raw proximity. A fire far from any settlement but
# closing fast is just as urgent as a slow fire that is already close.
CRITICAL_ETA_HOURS = 12.0

# `nearest_builtup_pixel_cog` (the real, authoritative ember-jump
# distance) and `_nearest_settlement_name` (a human-readable label) run
# as two INDEPENDENT searches, over very different radii — the built-up
# search stops at a few km, the settlement name search runs out to
# wui_settlement_search_radius_m (15 km by default). ESA WorldCover's
# "built-up" class covers industrial pavement as readily as a village,
# so a fire can measure a few metres from built-up land that is not
# anywhere near a named settlement at all — an industrial facility's
# own yard, for instance. Attaching whatever named village happens to
# be nearest within 15 km to that measurement would label it with a
# real name for the wrong place: a genuine village 5 km away is not the
# "threatened asset" 6 m from the fire. A settlement's OSM node marks
# roughly its centre, so its own distance from the fire is allowed to
# exceed the built-up-pixel distance by up to this much (a generous
# single village's extent) before it is treated as a different,
# unrelated place.
SETTLEMENT_NAME_MAX_EXCESS_M = 1500.0

# The named-place requirement (see gate_critical_priority) rejects a
# real but genuinely unnamed cluster of houses just as readily as it
# rejects bare industrial pavement — OSM's place-node coverage is real
# but incomplete, and a WorldCover "built-up" pixel with no named place
# within SETTLEMENT_NAME_MAX_EXCESS_M can still be a visibly inhabited
# hamlet, just one nobody has mapped a name for. WorldPop's unconstrained
# ~100 m-pixel product (already used for the classifier's own
# population_density feature and for plume exposure) gives a real,
# independent headcount at the same built-up point regardless of
# whether OSM has a name for it. Roughly two average rural Indian
# households (~4.4 people each per census figures) — "multiple houses,
# multiple people", not a single isolated structure, which is what a
# name-only requirement risks missing.
POPULATED_THRESHOLD_PEOPLE = 8.0


def priority_for(distance_m: Optional[float], eta_hours: Optional[float]) -> str:
    """CRITICAL if the fire is already within ember-jump range OR is
    projected to arrive within `CRITICAL_ETA_HOURS`; WATCH if it is
    merely close; MONITOR otherwise.

    Two independent triggers because ember spotting is a proximity
    phenomenon (it does not care how fast the front is moving) while the
    12-hour window is a rate phenomenon — a slow fire 500 m from a
    village and a fast fire 8 km away are both genuinely urgent, for
    different reasons, and either alone should raise the alarm rather
    than needing both at once.
    """
    if distance_m is None:
        return "MONITOR"
    if distance_m <= EMBER_JUMP_THRESHOLD_M:
        return "CRITICAL_AIR_TANKER_DISPATCH"
    if eta_hours is not None and 0.0 <= eta_hours <= CRITICAL_ETA_HOURS:
        return "CRITICAL_AIR_TANKER_DISPATCH"
    if distance_m <= EMBER_JUMP_THRESHOLD_M * WATCH_MULTIPLIER:
        return "WATCH"
    return "MONITOR"


def is_meaningfully_populated(population_count: Optional[float]) -> bool:
    """A real headcount at the built-up target point, not just "some
    built-up land was found" — see POPULATED_THRESHOLD_PEOPLE. `None`
    (WorldPop unavailable, not sampled) reads as not populated: an
    unknown headcount must not silently count as a confirmed one."""
    return population_count is not None and population_count >= POPULATED_THRESHOLD_PEOPLE


def gate_critical_priority(
    raw_priority: str, has_named_place: bool, is_populated: bool, is_active: bool,
) -> str:
    """Downgrades a raw CRITICAL_AIR_TANKER_DISPATCH read to WATCH unless
    both confirming conditions hold. Three real problems this closes:

    1. ESA WorldCover's "built-up" class covers industrial pavement and
       an unmapped compound as readily as a real settlement — proximity
       to bare built-up land alone was firing CRITICAL for target
       distances of a few metres from land nobody actually lives near.
       `has_named_place` requires a real OSM-named settlement
       (settlement_matches_builtup_pixel already vetted it against this
       same built-up measurement) before the badge can claim there is a
       specific place at risk.

    2. That named-place requirement alone rejected real, visibly
       inhabited clusters of houses just as readily as it rejected bare
       pavement, whenever OSM had no place node for them — India's rural
       settlement coverage is dense but incomplete. `is_populated`
       (is_meaningfully_populated, a real WorldPop headcount at the same
       built-up point) gives CRITICAL a second, independent way to
       confirm people actually live there, so an unnamed-but-inhabited
       hamlet is not treated the same as an empty industrial yard.

    3. A live count on this dashboard found 225 of 228 CRITICAL WUI
       fires were simultaneously labelled "Under Control" — the module
       never read the fire's own escalation/PTSI state at all, so a
       stalled fire that happened to be close to a village got the same
       air-tanker-dispatch urgency as one actively bearing down on it.
       `is_active` (temporal.status.is_fire_active) requires the same
       Active read the dashboard's own status label uses.

    Never downgrades WATCH or MONITOR — those don't claim a specific
    place is under imminent threat, so none of the three conditions
    applies to them; the fire is still worth a human glance either way,
    just not the full CRITICAL treatment.
    """
    if raw_priority != "CRITICAL_AIR_TANKER_DISPATCH":
        return raw_priority
    if (has_named_place or is_populated) and is_active:
        return raw_priority
    return "WATCH"


def settlement_matches_builtup_pixel(
    settlement: Optional[Dict[str, Any]], builtup_distance_m: Optional[float],
) -> bool:
    """Is `settlement` (a `_nearest_settlement_name` result) actually the
    same built-up land `nearest_builtup_pixel_cog` measured `builtup_distance_m`
    to, or a real but unrelated village the much-larger settlement-name
    search radius happened to pick up? See SETTLEMENT_NAME_MAX_EXCESS_M.

    False whenever either distance is missing, rather than assuming a
    match — an unlabelled measurement is honest; a mislabelled one is not.
    """
    if settlement is None or builtup_distance_m is None:
        return False
    settlement_distance_m = settlement.get("distance_m")
    if settlement_distance_m is None:
        return False
    return settlement_distance_m <= builtup_distance_m + SETTLEMENT_NAME_MAX_EXCESS_M


def _nearest_settlement_name(lon: float, lat: float, radius_m: float) -> Optional[Dict[str, Any]]:
    """The nearest named place (city/town/village/hamlet) within
    `radius_m`, via a single Overpass `around` query. Real OSM data —
    this never generates or guesses a name.

    One fast attempt against the primary mirror with a short timeout,
    not the tiled multi-mirror retry machinery `osm_facilities.py` uses
    for its bulk load: this runs inline in the per-cluster evaluation
    path, so it must fail fast and let the caller fall back to the
    district/state name from `admin_boundaries` rather than stall
    waiting on a rate-limited public server for one cluster's label.
    """
    from app import geo_cache

    cache_key = geo_cache.make_key("settlement", lon, lat, int(radius_m))
    cached = geo_cache.get(cache_key)
    if cached is not geo_cache.MISS:
        return cached

    query = (
        '[out:json][timeout:15];\n'
        'node["place"~"^(city|town|village|hamlet)$"](around:{radius:.0f},{lat},{lon});\n'
        'out body 5;'
    ).format(radius=radius_m, lat=lat, lon=lon)

    try:
        import requests

        from app.enrichment.osm_facilities import OVERPASS_MIRRORS, OVERPASS_USER_AGENT

        resp = requests.post(
            OVERPASS_MIRRORS[0], data={"data": query}, timeout=15,
            headers={"User-Agent": OVERPASS_USER_AGENT},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        logger.info(
            "Nearest-settlement lookup failed for (%s, %s) — falling back to district/state",
            lon, lat, exc_info=True,
        )
        return None

    if not elements:
        geo_cache.put(cache_key, None)
        return None

    from geospatial.geometry import haversine_m

    best = min(
        elements,
        key=lambda el: haversine_m(lon, lat, el.get("lon", lon), el.get("lat", lat)),
    )
    result = {
        "name": (best.get("tags") or {}).get("name"),
        "distance_m": haversine_m(lon, lat, best.get("lon", lon), best.get("lat", lat)),
        "lon": best.get("lon"),
        "lat": best.get("lat"),
    }
    geo_cache.put(cache_key, result)
    return result


def _read_context(session, cluster_id: int) -> Optional[Dict[str, Any]]:
    """Everything the evaluation needs, read from the DB in one pass:
    the cluster's location, its classification, its detection history
    (for the trajectory) and M2's footprint-growth feature if it has
    landed. Returns None if the cluster does not exist.
    """
    from geoalchemy2.shape import to_shape

    from app.db.models import Cluster
    from geospatial.features_io import load_hotspot_rows

    cluster = session.get(Cluster, cluster_id)
    if cluster is None:
        return None
    point = to_shape(cluster.centroid)

    predicted_class = None
    try:
        from classifier.db.models import Classification

        row = session.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
        predicted_class = row.predicted_class if row else None
    except ImportError:
        pass

    footprint_growth = None
    try:
        from classifier.db.models import ClusterFeatures

        features = session.get(ClusterFeatures, cluster_id)
        footprint_growth = getattr(features, "spatial_growth_rate", None) if features else None
    except ImportError:
        pass

    from temporal.status import is_fire_active

    is_active = is_fire_active(session, cluster_id)

    return {
        "cluster_id": cluster_id,
        "lon": point.x,
        "lat": point.y,
        "predicted_class": predicted_class,
        "hotspot_rows": load_hotspot_rows(session, cluster_id),
        "footprint_growth_km_day": footprint_growth,
        "is_active": is_active,
    }


def _compute_threat(context: Dict[str, Any], settings=None) -> Dict[str, Any]:
    """The pure-ish physics/network half: no DB session touched here, so
    this is safe to call after the caller's read transaction has closed
    (or, for the convenience `evaluate_wui_threat` entry point below,
    while it is still open — see that function's docstring).
    """
    from geospatial.config import get_m3_settings
    from geospatial.threat.corridor import (
        SPREADING_CLASSES,
        corridor_polygon,
        half_angle_for_confidence,
        time_to_impact_hours,
    )
    from geospatial.threat.trajectory import compute_trajectory, conservative_spread_rate

    settings = settings or get_m3_settings()
    lon, lat = context["lon"], context["lat"]
    predicted_class = context["predicted_class"]

    result: Dict[str, Any] = {
        "cluster_id": context["cluster_id"],
        "wui_threat": False,
        "priority": "NOT_APPLICABLE",
        "distance_m": None,
        "eta_hours": None,
        "bearing_deg": None,
        "threatened_asset": None,
        "settlement_name": None,
        "spread_rate_km_day": None,
        "corridor_geojson": None,
        "reason": None,
    }

    if not predicted_class or predicted_class.lower() not in SPREADING_CLASSES:
        result["reason"] = "class={0} does not advance across terrain".format(predicted_class)
        return result

    trajectory = compute_trajectory(context["hotspot_rows"])
    spread_rate = conservative_spread_rate(trajectory, context.get("footprint_growth_km_day"))
    result["spread_rate_km_day"] = round(spread_rate, 3)

    nearest = nearest_builtup_pixel_cog(lon, lat, settings.wui_builtup_search_radius_m)
    if nearest is None:
        result["reason"] = "no built-up land within the search radius"
        return result

    distance_m = nearest["distance_m"]
    bearing = nearest["bearing_deg"]
    eta_hours = time_to_impact_hours(distance_m / 1000.0, spread_rate)

    settlement = None
    if distance_m <= settings.wui_settlement_search_radius_m:
        settlement = _nearest_settlement_name(lon, lat, settings.wui_settlement_search_radius_m)
    if not settlement_matches_builtup_pixel(settlement, distance_m):
        settlement = None
    has_named_place = bool(settlement and settlement.get("name"))

    raw_priority = priority_for(distance_m, eta_hours)
    # Only worth sampling WorldPop (a local raster read, cheap but not
    # free) when it could actually change the verdict — a fire that's
    # already confirmed near a named place, or isn't CRITICAL by
    # distance/eta in the first place, doesn't need it.
    is_populated = False
    if raw_priority == "CRITICAL_AIR_TANKER_DISPATCH" and not has_named_place:
        from app.enrichment.population import sample_population_at_point

        population_at_target = sample_population_at_point(
            settings.population_raster, nearest["lon"], nearest["lat"],
        )
        is_populated = is_meaningfully_populated(population_at_target)
    priority = gate_critical_priority(
        raw_priority, has_named_place, is_populated, bool(context.get("is_active")),
    )
    threat = priority == "CRITICAL_AIR_TANKER_DISPATCH"

    if settlement and settlement.get("name"):
        asset_name = settlement["name"]
    else:
        asset_name = None
        try:
            from geospatial.admin_boundaries import resolve as resolve_admin

            admin_state, admin_district = resolve_admin(lon, lat)
            place = ", ".join(p for p in (admin_district, admin_state) if p) or None
        except Exception:  # noqa: BLE001
            place = None
        if place:
            asset_name = "Area near {0}".format(place)

    corridor_geojson = None
    length_km = max(distance_m / 1000.0 * 1.2, 0.3)
    half_angle = half_angle_for_confidence(trajectory.confidence)
    try:
        ring = corridor_polygon(lon, lat, bearing, length_km, half_angle)
        corridor_geojson = {"type": "Polygon", "coordinates": [[list(p) for p in ring]]}
    except ValueError:
        corridor_geojson = None

    result.update({
        "wui_threat": threat,
        "priority": priority,
        "distance_m": round(distance_m, 1),
        "eta_hours": round(eta_hours, 2) if eta_hours is not None else None,
        "bearing_deg": round(bearing, 1),
        "threatened_asset": asset_name,
        "settlement_name": settlement.get("name") if settlement else None,
        "corridor_geojson": corridor_geojson,
    })
    return result


def nearest_builtup_pixel_cog(lon: float, lat: float, search_radius_m: float):
    """Re-exported for convenience so callers of this module do not also
    need to import from `geospatial.attribution.landcover_worldcover_cog`
    directly. The real implementation lives there, next to
    `zonal_landcover_cog`, which it shares its windowed-read machinery
    with.
    """
    from geospatial.attribution.landcover_worldcover_cog import (
        nearest_builtup_pixel_cog as _impl,
    )

    return _impl(lon, lat, search_radius_m)


def evaluate_wui_threat(cluster_id: int, db_session) -> Dict[str, Any]:
    """Full WUI evaluation for one cluster: reads what it needs from
    `db_session`, then does the (cache-miss) network lookups and the
    physics, and returns one shaped dict — no writes.

    This is the single entry point matching the shape a route handler or
    a test wants: one function, the session already in hand, one dict
    back. It is used directly by `GET /geospatial/wui/{cluster_id}`
    (called on the request's own `db: Session = Depends(get_db)`, the
    same pattern `get_plume`/`get_threat` already use) and by tests.

    The higher-volume pipeline path (`geospatial.tasks.wui_threat`)
    splits the read / network / write phases the way every other M3 task
    in this project does — no DB transaction held open across a network
    call — and calls the same `_read_context` / `_compute_threat` this
    function calls, so there is exactly one implementation of the
    physics either way, not two that could drift apart.
    """
    context = _read_context(db_session, cluster_id)
    if context is None:
        return {
            "cluster_id": cluster_id, "wui_threat": False,
            "priority": "NOT_APPLICABLE", "reason": "cluster not found",
        }
    return _compute_threat(context)
