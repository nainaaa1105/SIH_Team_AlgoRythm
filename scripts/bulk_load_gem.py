"""One-time (Day 1) bulk load: Global Energy Monitor plant trackers
(coal/oil-gas/steel) + GGFR flare sites, into the `facilities` table.

GEM trackers are distributed as XLSX downloads (registration required at
globalenergymonitor.org) — point GEM_XLSX_PATHS at the local files after
downloading them manually; there's no public GEM REST API.

Usage: python -m scripts.bulk_load_gem --gem path/to/Coal_Plant_Tracker.xlsx --ggfr path/to/ggfr_flares.csv
"""
import argparse
import logging
from typing import List

import pandas as pd
from geoalchemy2.shape import from_shape
from shapely.geometry import Point

from app.db.models import Facility
from app.db.session import session_scope
from app.enrichment.flares import load_ggfr_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GEM trackers vary slightly in column naming release-to-release; these
# are the common names as of the 2024/2025 Global Coal/Steel/Oil&Gas
# Plant Tracker releases. Adjust if a newer release renames columns.
_GEM_LAT_COLS = ("Latitude", "Plant latitude")
_GEM_LON_COLS = ("Longitude", "Plant longitude")
_GEM_NAME_COLS = ("Plant name", "Plant / Project name")
_GEM_TYPE_COLS = ("Type", "Plant type")


def _first_present(row: dict, candidates) -> object:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return None


def load_gem_xlsx(path: str, facility_type_default: str) -> List[dict]:
    df = pd.read_excel(path)
    facilities = []
    for row in df.to_dict(orient="records"):
        lat = _first_present(row, _GEM_LAT_COLS)
        lon = _first_present(row, _GEM_LON_COLS)
        if lat is None or lon is None:
            continue
        facilities.append(
            {
                "name": _first_present(row, _GEM_NAME_COLS),
                "facility_type": _first_present(row, _GEM_TYPE_COLS) or facility_type_default,
                "source": "GEM",
                "lon": float(lon),
                "lat": float(lat),
                "prior_weight": 1.0,
                "metadata": {k: v for k, v in row.items() if pd.notna(v)},
            }
        )
    return facilities


def insert_facilities(facilities: List[dict]) -> None:
    with session_scope() as session:
        for f in facilities:
            session.add(
                Facility(
                    name=str(f.get("name")) if f.get("name") is not None else None,
                    facility_type=str(f.get("facility_type")) if f.get("facility_type") else None,
                    source=f["source"],
                    geom=from_shape(Point(f["lon"], f["lat"]), srid=4326),
                    prior_weight=f.get("prior_weight", 1.0),
                    facility_metadata=f.get("metadata"),
                )
            )
    logger.info("Inserted %d facilities", len(facilities))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gem", action="append", default=[], help="Path to a GEM tracker XLSX (repeatable)")
    parser.add_argument("--ggfr", help="Path to the GGFR flare-site CSV")
    args = parser.parse_args()

    all_facilities: List[dict] = []
    for path in args.gem:
        all_facilities.extend(load_gem_xlsx(path, facility_type_default="industrial"))
    if args.ggfr:
        all_facilities.extend(load_ggfr_csv(args.ggfr))

    if not all_facilities:
        logger.warning("No --gem or --ggfr paths given, nothing to load")
        return

    insert_facilities(all_facilities)


if __name__ == "__main__":
    main()
