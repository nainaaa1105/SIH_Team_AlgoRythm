"""INSAT-3DS ingestion via MOSDAC (ISRO's Meteorological & Oceanographic
Satellite Data Archival Centre).

IMPORTANT — known limitation (read before wiring credentials):
MOSDAC (https://mosdac.gov.in) does not expose a public, documented REST
API like FIRMS does. Access is session/cookie-based behind a login form,
and bulk/NRT product delivery is normally arranged per-user (order-based
download or an FTP drop) after manual registration approval. The URL
pattern and product filename below are therefore best-effort scaffolding
based on MOSDAC's typical open-data catalogue layout, not a verified
endpoint. Before relying on this in production:
    1. Register at https://mosdac.gov.in and request access to the
       INSAT-3DS "Fire" / hotspot product (or the older INSAT-3D/3DR
       equivalent product if 3DS is not yet listed for your account).
    2. Confirm the actual product delivery path (HTTPS catalogue vs
       order-and-FTP) and the NetCDF variable names for lat/lon/fire-flag,
       then adjust `_FIRE_VARS` and `fetch_latest()` below accordingly.

Until that's confirmed, this module fetches whatever the configured URL
returns and parses it with the variable names in `_FIRE_VARS`, raising a
clear error if the file doesn't have that shape — so a schema mismatch
fails loudly instead of silently dropping data.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

from app.config import Settings, get_settings
from app.ingestion.normalize import generic_pixel_to_canonical

logger = logging.getLogger(__name__)

# Best-effort variable names for ISRO fire products (matches the
# convention used in NRSC/FSI's INSAT-3D-based forest fire alert product).
# CONFIRM against the real file once MOSDAC access is granted.
_FIRE_VARS = {
    "lat": "latitude",
    "lon": "longitude",
    "fire_flag": "fire_confidence",  # non-zero / above threshold => detection
    "frp": "FRP",
}


class Insat3dsClientError(RuntimeError):
    pass


def _authenticated_session(settings: Settings) -> requests.Session:
    session = requests.Session()
    if settings.mosdac_username and settings.mosdac_password:
        # MOSDAC's real login flow is form/cookie based, not HTTP basic auth.
        # Placeholder for the session bootstrap — swap in the real login POST
        # once the endpoint is confirmed from a registered account.
        session.auth = (settings.mosdac_username, settings.mosdac_password)
    return session


def download_latest_product(settings: Settings | None = None, dest_dir: str = "data/insat3ds") -> Path | None:
    """Download the most recent INSAT-3DS fire product NetCDF file.

    Returns the local path, or None if credentials are missing (so the
    scheduler can skip this source gracefully instead of crashing the
    whole ingestion cycle).
    """
    settings = settings or get_settings()
    if not (settings.mosdac_username and settings.mosdac_password):
        logger.warning("MOSDAC credentials not configured — skipping INSAT-3DS ingestion this cycle")
        return None

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    session = _authenticated_session(settings)

    # Best-effort catalogue path — see module docstring.
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    url = f"{settings.mosdac_base_url.rstrip('/')}/data/insat3ds/fire/{today}/latest_fire.nc"

    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("INSAT-3DS download failed (url=%s) — verify MOSDAC access/path", url)
        return None

    dest = Path(dest_dir) / f"insat3ds_fire_{today}.nc"
    dest.write_bytes(resp.content)
    return dest


def parse_fire_product(path: Path) -> List[Dict]:
    """Parse a downloaded INSAT-3DS fire-product NetCDF file into canonical
    detection records. Only pixels flagged as fire (fire_flag > 0) become
    detections.
    """
    try:
        import netCDF4  # noqa: WPS433 - optional heavy dep, imported lazily
    except ImportError as exc:
        raise Insat3dsClientError(
            "netCDF4 is required to parse INSAT-3DS products (pip install netCDF4)"
        ) from exc

    canonical_records: List[Dict] = []
    with netCDF4.Dataset(path) as ds:
        missing = [v for v in _FIRE_VARS.values() if v not in ds.variables]
        if missing:
            raise Insat3dsClientError(
                f"INSAT-3DS product at {path} is missing expected variables {missing} — "
                "confirm the real variable names against a live MOSDAC product and update "
                "app/ingestion/insat3ds.py::_FIRE_VARS"
            )

        lats = ds.variables[_FIRE_VARS["lat"]][:]
        lons = ds.variables[_FIRE_VARS["lon"]][:]
        fire_flag = ds.variables[_FIRE_VARS["fire_flag"]][:]
        frp = ds.variables[_FIRE_VARS["frp"]][:] if _FIRE_VARS["frp"] in ds.variables else None

        # Product timestamp: prefer a global attribute, fall back to "now".
        acq_dt = datetime.now(timezone.utc)
        if hasattr(ds, "observation_time"):
            try:
                acq_dt = datetime.strptime(ds.observation_time, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        flat_lat, flat_lon, flat_flag = lats.flatten(), lons.flatten(), fire_flag.flatten()
        flat_frp = frp.flatten() if frp is not None else None

        for idx in range(flat_flag.size):
            if flat_flag[idx] <= 0:
                continue
            canonical_records.append(
                generic_pixel_to_canonical(
                    source="INSAT3DS",
                    lon=float(flat_lon[idx]),
                    lat=float(flat_lat[idx]),
                    acq_datetime=acq_dt,
                    frp=float(flat_frp[idx]) if flat_frp is not None else None,
                    confidence=None,
                    raw_payload={"fire_flag": float(flat_flag[idx])},
                )
            )

    return canonical_records


def fetch_all(settings: Settings | None = None) -> List[Dict]:
    """Full INSAT-3DS ingestion cycle: download + parse. Returns an empty
    list (never raises) if the source is unavailable this cycle, so the
    scheduler can keep polling FIRMS/Sentinel-3 uninterrupted.
    """
    settings = settings or get_settings()
    try:
        path = download_latest_product(settings)
        if path is None:
            return []
        return parse_fire_product(path)
    except Insat3dsClientError:
        logger.exception("INSAT-3DS parsing failed this cycle")
        return []
