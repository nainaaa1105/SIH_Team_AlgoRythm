"""EUMETSAT Meteosat-9 IODC (Indian Ocean Data Coverage) active-fire
product — a TEMPORARY, real-data backup for INSAT-3DS while the MOSDAC
account (`app/ingestion/insat3ds.py`) is pending approval.

Why this and not something else: Meteosat-9 sits at 45.5°E as the IODC
prime satellite and looks straight down on India, unlike Himawari-8/9
(140.7°E — India is at the far, distorted edge of its disk, and this
project's `himawari.py` module has no fire-detection algorithm behind it
anyway, only raw L1B scene listing). EUMETSAT's SEVIRI-based "Active Fire
Monitoring" product is described by EUMETSAT as "operational, free and
unrestricted" and covers the IODC region every 15 minutes.

Registration reality, stated plainly: this needs a free EUMETSAT account
and an API consumer key/secret from https://api.eumetsat.int (Profile ->
API key), which is instant and self-service — not the multi-day manual
MOSDAC approval. It is a NEW registration, not something already in
place; `eumetsat_consumer_key`/`_secret` in app/config.py are blank until
you create one.

Honesty about what's verified and what isn't, same standard as
insat3ds.py:
  - The OAuth2 client-credentials token exchange
    (`POST /token` with HTTP Basic auth) and the Data Store search/download
    REST endpoints are EUMETSAT's documented, stable public API surface
    (the same one the official `eumdac` client wraps) — this part is a
    real, working integration once a key is entered.
  - The exact internal structure of the IODC fire CAP (Common Alerting
    Protocol) XML file — how many <info>/<area>/<circle> blocks per
    file, whether one circle is one fire pixel or an aggregated hazard
    region, and what severity/certainty values the product actually
    emits — is NOT verified against a real downloaded file, because
    without an API key there is nothing to download. CAP 1.2 is an open
    OASIS standard (not a guessed vendor format like MOSDAC's), so the
    parser below is written against that public spec; confirm it against
    one real file the first time this runs with real credentials, and
    adjust `parse_cap_fire_product` if the product nests differently.
    Like insat3ds.py, a schema mismatch raises a clear error rather than
    silently returning nothing.

Auto switch-back: see geostationary_supplementary.py — the moment
MOSDAC credentials are filled in, this module stops being called at all,
with no code change needed.
"""
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from app.config import Settings, get_settings
from app.ingestion.normalize import generic_pixel_to_canonical

logger = logging.getLogger(__name__)

CAP_NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}

# CAP defines an ordered severity scale; used only to derive a rough
# confidence proxy since the product's own confidence field is unverified
# (see module docstring). CONFIRM against a real file.
_SEVERITY_CONFIDENCE = {
    "extreme": 0.9, "severe": 0.75, "moderate": 0.6, "minor": 0.4, "unknown": 0.5,
}


class EumetsatClientError(RuntimeError):
    pass


_token_cache: Dict[str, object] = {"token": None, "expires_at": 0.0}


def _get_access_token(settings: Settings) -> Optional[str]:
    """OAuth2 client-credentials exchange against api.eumetsat.int.

    Cached in-process until ~60s before expiry, matching the pattern
    every EUMETSAT client (including the official `eumdac`) uses to
    avoid re-authenticating on every request.
    """
    if not (settings.eumetsat_consumer_key and settings.eumetsat_consumer_secret):
        return None

    now = time.monotonic()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]  # type: ignore[return-value]

    try:
        resp = requests.post(
            settings.eumetsat_token_url,
            data={"grant_type": "client_credentials"},
            auth=(settings.eumetsat_consumer_key, settings.eumetsat_consumer_secret),
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException:
        logger.exception("EUMETSAT token request failed — check consumer key/secret")
        return None

    token = payload.get("access_token")
    expires_in = float(payload.get("expires_in", 600))
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + max(expires_in - 60.0, 30.0)
    return token


def search_latest_product(settings: Settings, token: str) -> Optional[Dict]:
    """Most recent product in the IODC fire collection.

    Data Store search is a documented, stable REST endpoint (unlike
    MOSDAC's undocumented catalogue path) — sorted descending by start
    time, one result.
    """
    try:
        resp = requests.get(
            settings.eumetsat_search_url,
            params={
                "format": "json",
                "pi": settings.eumetsat_iodc_fire_collection,
                "si": 0,
                "c": 1,
                "sort": "start,time,0",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException:
        logger.exception("EUMETSAT product search failed")
        return None

    features = payload.get("features") or []
    if not features:
        logger.info("EUMETSAT IODC fire collection returned no products this cycle")
        return None
    return features[0]


def download_latest_product(
    settings: Settings | None = None, dest_dir: str = "data/eumetsat_iodc"
) -> Path | None:
    """Download the most recent IODC fire-monitoring CAP file.

    Returns None (never raises) whenever the source isn't usable this
    cycle — no key configured, token exchange failed, nothing published
    yet — so the caller can fall back or skip gracefully.
    """
    settings = settings or get_settings()
    token = _get_access_token(settings)
    if token is None:
        return None

    product = search_latest_product(settings, token)
    if product is None:
        return None

    product_id = product.get("properties", {}).get("identifier") or product.get("id")
    if not product_id:
        logger.warning("EUMETSAT search result had no product identifier: %s", product)
        return None

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(dest_dir) / f"{product_id.replace('/', '_')}.cap"

    try:
        resp = requests.get(
            f"{settings.eumetsat_download_url}/{product_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("EUMETSAT product download failed (id=%s)", product_id)
        return None

    dest.write_bytes(resp.content)
    return dest


def _confidence_for(info_el: ET.Element) -> float:
    severity = (info_el.findtext("cap:severity", default="unknown", namespaces=CAP_NS) or "unknown").lower()
    return _SEVERITY_CONFIDENCE.get(severity, 0.5)


def _timestamp_for(info_el: ET.Element) -> datetime:
    raw = (
        info_el.findtext("cap:effective", namespaces=CAP_NS)
        or info_el.findtext("cap:onset", namespaces=CAP_NS)
    )
    if raw:
        try:
            return datetime.fromisoformat(raw).astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def parse_cap_fire_product(path: Path) -> List[Dict]:
    """Parse a CAP 1.2 XML fire-monitoring file into canonical detections.

    Each <circle>lat,lon radius</circle> inside an <area> block is taken
    as one fire detection at that centroid. CAP is an open standard, so
    the tag names here are correct by specification; what is NOT
    guaranteed is that EUMETSAT emits exactly one circle per fire pixel
    rather than, say, one circle covering an aggregated hazard region —
    confirm against a real downloaded file (see module docstring).

    Raises (does not silently drop data) if the file has no <alert>/<info>
    structure at all, since that means the whole schema assumption above
    is wrong, not just a naming detail.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    infos = root.findall(".//cap:info", CAP_NS)
    if not infos:
        raise EumetsatClientError(
            f"EUMETSAT product at {path} has no CAP <info> blocks — this is not the "
            "CAP structure this parser expects. Inspect the file and update "
            "app/ingestion/eumetsat_iodc.py::parse_cap_fire_product."
        )

    records: List[Dict] = []
    for info in infos:
        acq_dt = _timestamp_for(info)
        confidence = _confidence_for(info)
        for area in info.findall("cap:area", CAP_NS):
            for circle in area.findall("cap:circle", CAP_NS):
                if not circle.text:
                    continue
                try:
                    latlon_part, _radius_km = circle.text.strip().split(" ")
                    lat_str, lon_str = latlon_part.split(",")
                    lat, lon = float(lat_str), float(lon_str)
                except (ValueError, AttributeError):
                    logger.warning("Unparseable CAP <circle> value: %r", circle.text)
                    continue

                records.append(
                    generic_pixel_to_canonical(
                        source="EUMETSAT_IODC",
                        lon=lon,
                        lat=lat,
                        acq_datetime=acq_dt,
                        frp=None,  # CAP carries a hazard description, not radiometric FRP
                        confidence=confidence,
                        raw_payload={"severity": info.findtext("cap:severity", namespaces=CAP_NS)},
                    )
                )

    return records


def fetch_all(settings: Settings | None = None) -> List[Dict]:
    """Full EUMETSAT IODC ingestion cycle: token -> search -> download ->
    parse. Returns [] (never raises) if the source is unavailable or
    unconfigured this cycle, matching every other connector's contract.
    """
    settings = settings or get_settings()
    try:
        path = download_latest_product(settings)
        if path is None:
            return []
        return parse_cap_fire_product(path)
    except EumetsatClientError:
        logger.exception("EUMETSAT IODC parsing failed this cycle")
        return []
