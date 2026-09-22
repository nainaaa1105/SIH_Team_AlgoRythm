"""One-time (Day 1) bulk load: fetch OSM industrial facilities across
India and insert into the `facilities` table.

Usage: python -m scripts.bulk_load_osm [--resume]

The public Overpass instances cannot answer a single whole-India query
for these tag sets, so `osm_facilities.tile_bbox` walks the bbox in 2
degree tiles. This script commits each tile as it lands rather than
accumulating every tile and writing once at the end: a full walk is 240
requests with rate-limit backoff and can run for the better part of an
hour, and losing all of it to one failure at tile 230 is not acceptable.

Committing per tile also makes `--resume` cheap — already-loaded tiles
are skipped by checking whether the `facilities` table already has rows
inside that tile's bounds.
"""
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy import text

from app.db.models import Facility
from app.db.session import session_scope
from app.enrichment.osm_facilities import (
    OVERPASS_MIRRORS,
    _fetch_tile,
    normalize_elements,
    tile_bbox,
)
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _tile_already_loaded(session, tile) -> bool:
    """Has any facility already been stored inside this tile?

    Cheap proxy for "this tile was done on a previous run". A tile that
    genuinely contains no industrial features re-runs on every resume,
    which costs one request and is the safe direction to be wrong in.
    """
    west, south, east, north = tile
    count = session.execute(
        text(
            "SELECT count(*) FROM facilities "
            "WHERE ST_Within(geom, ST_MakeEnvelope(:w, :s, :e, :n, 4326))"
        ),
        {"w": west, "s": south, "e": east, "n": north},
    ).scalar()
    return bool(count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true",
        help="skip tiles that already have facilities stored (re-run after an interruption)",
    )
    parser.add_argument(
        "--priority-from-clusters", action="store_true",
        help="fetch tiles containing detected clusters first, so facility "
             "attribution becomes available where it can actually change a "
             "classification before the whole country finishes loading",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="parallel tile fetches; keep at or below the number of Overpass "
             "mirrors so each worker gets its own instance (default: 3)",
    )
    args = parser.parse_args()

    settings = get_settings()
    tiles = tile_bbox(settings.osm_bbox_tuple)
    logger.info("Overpass: %d tiles to walk across %s", len(tiles), settings.osm_bbox_tuple)

    if args.resume:
        with session_scope() as session:
            pending = [t for t in tiles if not _tile_already_loaded(session, t)]
        logger.info("Resume: %d/%d tiles still to fetch", len(pending), len(tiles))
    else:
        pending = list(tiles)

    if args.priority_from_clusters:
        # A full India walk is ~240 Overpass requests against rate-limited
        # public instances and takes hours. Facility distance only affects
        # a cluster that actually exists, so walking in raw west-to-east
        # order spends the first hour on empty ocean tiles while the
        # industrial belt — where facility attribution changes the verdict
        # most — waits until the end. Ordering by how many clusters each
        # tile contains front-loads the tiles that can change a
        # classification, and tiles with none are still fetched, just last.
        with session_scope() as session:
            counts = []
            for tile in pending:
                west, south, east, north = tile
                n = session.execute(
                    text(
                        "SELECT count(*) FROM clusters "
                        "WHERE ST_Within(centroid::geometry, "
                        "ST_MakeEnvelope(:w, :s, :e, :n, 4326))"
                    ),
                    {"w": west, "s": south, "e": east, "n": north},
                ).scalar()
                counts.append((n or 0, tile))
        counts.sort(key=lambda pair: pair[0], reverse=True)
        pending = [tile for _, tile in counts]
        populated = sum(1 for n, _ in counts if n)
        logger.info(
            "Prioritised: %d of %d pending tiles contain clusters; fetching those first",
            populated, len(pending),
        )

    state = {"inserted": 0, "done": 0, "empty": 0}
    seen_ids = set()
    lock = threading.Lock()

    def _handle(indexed_tile):
        """Fetch one tile and commit it. Runs on a worker thread."""
        index, tile = indexed_tile
        # Pin each worker to its own mirror: the public instances limit
        # concurrent slots *per instance*, so N workers all hitting the
        # same one just queue behind each other and 429.
        mirror = OVERPASS_MIRRORS[index % len(OVERPASS_MIRRORS)]
        elements = _fetch_tile(settings.overpass_url, tile, preferred=mirror)

        # A way straddling a tile edge comes back from both tiles, and
        # workers can see the same way concurrently, so the dedup set is
        # guarded.
        with lock:
            fresh = []
            for el in elements:
                identity = (el.get("type"), el.get("id"))
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                fresh.append(el)

        facilities = normalize_elements(fresh)
        if facilities:
            with session_scope() as session:
                for f in facilities:
                    session.add(
                        Facility(
                            name=f.get("name"),
                            facility_type=f.get("facility_type"),
                            source="OSM",
                            geom=from_shape(f.get("geom") or Point(f["lon"], f["lat"]), srid=4326),
                            prior_weight=f.get("prior_weight", 1.0),
                            facility_metadata=f.get("metadata"),
                        )
                    )

        with lock:
            state["inserted"] += len(facilities)
            state["done"] += 1
            if not elements:
                state["empty"] += 1
            logger.info(
                "[%d/%d] %s -> %d new facilities (running total %d)",
                state["done"], len(pending), tile, len(facilities), state["inserted"],
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_handle, enumerate(pending)))

    logger.info(
        "OSM bulk load complete: %d facilities inserted, %d/%d tiles returned nothing",
        state["inserted"], state["empty"], len(pending),
    )


if __name__ == "__main__":
    main()
