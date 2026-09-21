"""GGFR (Global Gas Flaring Reduction, World Bank) flare-site loader.

GGFR publishes an annual flare-site catalogue (CSV/shapefile, site
lon/lat + estimated flared gas volume) — bulk-downloaded once and loaded
as `facilities` rows with facility_type='flare'. This gives M2's
classifier a strong prior for the "gas flare" class independent of OSM
tagging (OSM rarely tags individual flare stacks).
"""
import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"Longitude", "Latitude"}


def load_ggfr_csv(csv_path: str) -> List[Dict]:
    """Parse a GGFR flare-site CSV export into facility records."""
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"GGFR CSV at {csv_path} is missing expected columns: {missing}")

    facilities = []
    for row in df.to_dict(orient="records"):
        volume = row.get("Flared_Volume_MCM") or row.get("Flare Volume (MCM)")
        facilities.append(
            {
                "name": row.get("Country") and f"Flare site ({row['Country']})",
                "facility_type": "flare",
                "source": "GGFR",
                "lon": float(row["Longitude"]),
                "lat": float(row["Latitude"]),
                "prior_weight": 1.0,
                "metadata": {"flared_volume_mcm": volume, **row},
            }
        )
    logger.info("Loaded %d GGFR flare sites", len(facilities))
    return facilities
