"""One-time bulk load: fetch Indian district boundaries for the
state/district resolver.

    python -m scripts.bulk_load_boundaries

`gateway/routes_dashboard` reports which state and district every
detection falls in, and `geospatial.admin_boundaries` answers that by
point-in-polygon against this file. Without it the dashboard still works
but every detection reads "outside Indian boundaries", the Top States
panel is empty, and the state filter has nothing to filter on.

The source is a published GeoJSON of the 2011 census districts with
post-2019 state boundaries, so Telangana and Ladakh are present as their
own states rather than folded into Andhra Pradesh and Jammu & Kashmir.
It carries 726 district polygons plus 34 whole-state outlines; the
resolver prefers a district-bearing match and treats the outlines as a
fallback (see `admin_boundaries.resolve`).
"""
import argparse
import json
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
DEFAULT_DEST = Path("data/boundaries/india_districts.geojson")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="GeoJSON URL to fetch")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="where to write it")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file already exists"
    )
    args = parser.parse_args()

    if args.dest.is_file() and not args.force:
        logger.info("%s already present (%d bytes) — pass --force to refetch",
                    args.dest, args.dest.stat().st_size)
        return

    logger.info("Fetching district boundaries from %s", args.source)
    response = requests.get(args.source, timeout=180)
    response.raise_for_status()
    payload = response.json()

    features = payload.get("features") or []
    if not features:
        raise SystemExit("Downloaded file contains no features — refusing to write it")

    states = {
        (f.get("properties") or {}).get("st_nm")
        for f in features
        if (f.get("properties") or {}).get("st_nm")
    }
    districts = {
        (f.get("properties") or {}).get("district")
        for f in features
        if (f.get("properties") or {}).get("district")
    }

    # A file that parses but carries none of the properties the resolver
    # reads would silently blank every state on the dashboard, so fail
    # here instead, where the cause is obvious.
    if not states:
        raise SystemExit(
            "Downloaded file has no 'st_nm' property on any feature — the resolver "
            "reads that field, so this source is not usable as-is"
        )

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_text(json.dumps(payload), encoding="utf-8")

    logger.info(
        "Wrote %s (%d features, %d states/UTs, %d districts, %.1f MB)",
        args.dest, len(features), len(states), len(districts),
        args.dest.stat().st_size / (1024 * 1024),
    )

    for expected in ("Telangana", "Ladakh"):
        if expected not in states:
            logger.warning(
                "%s is absent — this source predates a state reorganisation and "
                "detections there will be attributed to its predecessor", expected,
            )


if __name__ == "__main__":
    main()
