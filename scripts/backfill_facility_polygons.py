"""Backfill: upgrade already-ingested point-only OSM facilities to real
polygon geometry.

`bulk_load_osm.py --resume` skips any tile that already has facility rows,
so it can never upgrade a tile it loaded before `out geom` replaced
`out center tags` in app/enrichment/osm_facilities.py. A full non-resumed
re-crawl is hours of heavy public-Overpass rate limiting to re-fetch
~240 tiles for zero benefit outside the ones the live DB actually uses.

This instead re-queries Overpass around each already-ingested facility's
stored point and replaces it with the real way polygon Overpass returns,
when one exists within a tight radius of that point. By default it only
touches facilities with a live FacilityAttribution row -- i.e. the ones
that actually affect a real classification today -- which is minutes of
work, not hours.

Usage: python -m scripts.backfill_facility_polygons [--all]
"""
import argparse
import logging
import time

from shapely.geometry import Polygon
from sqlalchemy import text

from app.db.session import session_scope
from app.enrichment.osm_facilities import (
    OVERPASS_MIRRORS,
    OVERPASS_TILE_DELAY_SECONDS,
    OVERPASS_USER_AGENT,
    _FACILITY_TAGS,
    haversine_m,
)
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATCH_RADIUS_M = 250.0


def nearest_way_polygon(lon: float, lat: float, radius_m: float = MATCH_RADIUS_M):
    """Real polygon geometry for the industrial way nearest (lon, lat), or
    None if nothing matches within radius_m. Picks the way whose own
    centroid is closest, not just the first hit, since a facility can sit
    near more than one tagged way."""
    clauses = "\n  ".join(f"way{tag}(around:{radius_m:.0f},{lat},{lon});" for tag in _FACILITY_TAGS)
    query = f"[out:json][timeout:30];\n(\n  {clauses}\n);\nout geom;"
    resp = requests.post(
        OVERPASS_MIRRORS[0], data={"data": query}, timeout=30,
        headers={"User-Agent": OVERPASS_USER_AGENT},
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    best = None
    best_distance = None
    for el in elements:
        geometry = el.get("geometry")
        if not geometry:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geometry if pt]
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        try:
            poly = Polygon(coords)
        except ValueError:
            continue
        if not poly.is_valid or poly.is_empty:
            poly = poly.buffer(0)
        if not poly.is_valid or poly.is_empty:
            continue
        centroid = poly.centroid
        distance = haversine_m(lon, lat, centroid.x, centroid.y)
        if best_distance is None or distance < best_distance:
            best, best_distance = poly, distance
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true",
        help="backfill every point-geometry facility, not just ones with a "
             "live attribution (default: attributed-only, bounded to minutes)",
    )
    args = parser.parse_args()

    with session_scope() as session:
        if args.all:
            rows = session.execute(
                text(
                    "SELECT id, ST_X(geom) AS lon, ST_Y(geom) AS lat FROM facilities "
                    "WHERE GeometryType(geom) = 'POINT'"
                )
            ).fetchall()
        else:
            rows = session.execute(
                text(
                    "SELECT DISTINCT f.id, ST_X(f.geom) AS lon, ST_Y(f.geom) AS lat "
                    "FROM facilities f "
                    "JOIN facility_attributions fa ON fa.facility_id = f.id "
                    "WHERE GeometryType(f.geom) = 'POINT'"
                )
            ).fetchall()

    logger.info("Backfill candidates: %d point-geometry facilities", len(rows))

    upgraded = 0
    for row in rows:
        try:
            poly = nearest_way_polygon(row.lon, row.lat)
        except Exception:
            logger.warning("Overpass lookup failed for facility %d", row.id, exc_info=True)
            time.sleep(OVERPASS_TILE_DELAY_SECONDS)
            continue

        if poly is None:
            logger.info("Facility %d: no polygon within %.0fm — left as point", row.id, MATCH_RADIUS_M)
        else:
            with session_scope() as session:
                session.execute(
                    text("UPDATE facilities SET geom = ST_GeomFromText(:wkt, 4326) WHERE id = :id"),
                    {"wkt": poly.wkt, "id": row.id},
                )
            upgraded += 1
            logger.info("Facility %d: upgraded to a %d-vertex polygon", row.id, len(poly.exterior.coords))

        time.sleep(OVERPASS_TILE_DELAY_SECONDS)

    logger.info("Backfill complete: %d/%d facilities upgraded to real polygons", upgraded, len(rows))


if __name__ == "__main__":
    main()
