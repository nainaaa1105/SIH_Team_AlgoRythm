"""Picks which geostationary supplementary source to poll this cycle.

INSAT-3DS (via MOSDAC) is the intended long-term source. Until that
account is approved, EUMETSAT's Meteosat-9 IODC active-fire product
stands in as a real (not sample) temporary backup — see
`eumetsat_iodc.py` for what is and isn't verified about it.

The switch is just "which credentials are present," so filling in
`MOSDAC_USERNAME`/`MOSDAC_PASSWORD` in .env switches back to MOSDAC on
the very next poll with no code change and no redeploy. If both are
configured, MOSDAC wins because it's the source the architecture spec
actually names.
"""
import logging
from typing import Dict, List, Optional

from app.config import Settings, get_settings
from app.ingestion import eumetsat_iodc, insat3ds

logger = logging.getLogger(__name__)


def active_source_name(settings: Optional[Settings] = None) -> str:
    """Which source `fetch_all()` will use this cycle, for logging/health
    checks — an operator should be able to see this without reading code."""
    settings = settings or get_settings()
    if settings.mosdac_username and settings.mosdac_password:
        return "MOSDAC_INSAT3DS"
    if settings.eumetsat_consumer_key and settings.eumetsat_consumer_secret:
        return "EUMETSAT_IODC_BACKUP"
    return "none_configured"


def fetch_all(settings: Optional[Settings] = None) -> List[Dict]:
    settings = settings or get_settings()
    source = active_source_name(settings)

    if source == "MOSDAC_INSAT3DS":
        return insat3ds.fetch_all(settings)
    if source == "EUMETSAT_IODC_BACKUP":
        logger.info(
            "MOSDAC not yet configured — using EUMETSAT IODC as the temporary "
            "geostationary supplementary source this cycle"
        )
        return eumetsat_iodc.fetch_all(settings)

    logger.info(
        "Neither MOSDAC nor EUMETSAT credentials configured — no geostationary "
        "supplementary source this cycle (FIRMS/Sentinel-3 continue unaffected)"
    )
    return []
