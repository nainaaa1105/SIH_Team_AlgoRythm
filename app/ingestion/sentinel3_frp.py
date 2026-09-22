"""Sentinel-3 SLSTR Fire Radiative Power (FRP) ingestion.

Product: SL_2_FRP___ (SLSTR Level-2 Fire Radiative Power, land), delivered
via the Copernicus Data Space Ecosystem (CDSE) OData/STAC catalogue.
Docs: https://documentation.dataspace.copernicus.eu/APIs/OData.html

This is a genuine independent fire-detection product (not just optical
imagery) — its FRP retrieval is analogous to MODIS/VIIRS but is derived
from SLSTR's own mid-infrared channels, so it's used here as a
cross-validation source for FIRMS/INSAT-3DS detections, not a duplicate.

Auth: OAuth2 client-credentials grant (register a CDSE application to get
CDSE_CLIENT_ID / CDSE_CLIENT_SECRET).
"""
import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.ingestion.normalize import generic_pixel_to_canonical

logger = logging.getLogger(__name__)

# The FRP-PIXEL sub-product file inside the SL_2_FRP___ product package.
_FRP_FILE_CANDIDATES = ("FRP_in.nc",)


class Sentinel3ClientError(RuntimeError):
    pass


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_access_token(settings: Settings) -> str:
    if not (settings.cdse_client_id and settings.cdse_client_secret):
        raise Sentinel3ClientError("CDSE_CLIENT_ID/CDSE_CLIENT_SECRET not configured")

    resp = requests.post(
        settings.cdse_token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": settings.cdse_client_id,
            "client_secret": settings.cdse_client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _search_products(settings: Settings, token: str, hours_back: int = 12) -> List[dict]:
    """Query the CDSE OData catalogue for recent SLSTR FRP products over India."""
    west, south, east, north = settings.firms_bbox_tuple  # reuse the same India bbox
    aoi_wkt = (
        f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    )
    since = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    filter_expr = (
        "Collection/Name eq 'SENTINEL-3' and "
        "contains(Name,'SL_2_FRP___') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') and "
        f"ContentDate/Start gt {since}"
    )
    url = f"{settings.cdse_catalog_url}/Products?$filter={filter_expr}&$top=20"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("value", [])


# CDSE answers a product download with a 30x to `download.dataspace...`,
# a different host from the `catalogue.dataspace...` one the OData query
# goes to. `requests` deliberately strips the Authorization header when a
# redirect crosses hosts, so following redirects automatically delivered
# an unauthenticated request to the download host and CDSE answered 401 —
# every Sentinel-3 FRP fetch failed at the download step even though the
# catalogue search above had just succeeded with the same token.
_MAX_DOWNLOAD_REDIRECTS = 5


def _download_product(settings: Settings, token: str, product_id: str, dest_dir: str) -> Path:
    url = f"{settings.cdse_catalog_url}/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    session = requests.Session()
    resp = session.get(url, headers=headers, timeout=120, stream=True, allow_redirects=False)

    # Follow the chain by hand, re-attaching the bearer token at each hop.
    redirects = 0
    while resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("Location")
        if not location:
            break
        redirects += 1
        if redirects > _MAX_DOWNLOAD_REDIRECTS:
            resp.close()
            raise requests.TooManyRedirects(
                f"CDSE download exceeded {_MAX_DOWNLOAD_REDIRECTS} redirects for {product_id}"
            )
        resp.close()
        resp = session.get(
            location, headers=headers, timeout=120, stream=True, allow_redirects=False
        )

    resp.raise_for_status()
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(dest_dir) / f"{product_id}.zip"
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    return dest


def _extract_frp_netcdf(zip_path: Path, dest_dir: str) -> Path | None:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if any(name.endswith(candidate) for candidate in _FRP_FILE_CANDIDATES):
                zf.extract(name, dest_dir)
                return Path(dest_dir) / name
    return None


def parse_frp_netcdf(path: Path) -> List[Dict]:
    try:
        import netCDF4  # noqa: WPS433
    except ImportError as exc:
        raise Sentinel3ClientError(
            "netCDF4 is required to parse Sentinel-3 FRP products (pip install netCDF4)"
        ) from exc

    canonical_records: List[Dict] = []
    with netCDF4.Dataset(path) as ds:
        required = ("latitude", "longitude", "FRP_MWIR")
        missing = [v for v in required if v not in ds.variables]
        if missing:
            raise Sentinel3ClientError(
                f"Sentinel-3 FRP product at {path} missing expected variables {missing}"
            )

        lats = ds.variables["latitude"][:].flatten()
        lons = ds.variables["longitude"][:].flatten()
        frp = ds.variables["FRP_MWIR"][:].flatten()  # mid-wave-infrared FRP, MW

        acq_dt = datetime.now(timezone.utc)
        if hasattr(ds, "start_time"):
            try:
                acq_dt = datetime.strptime(ds.start_time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        for idx in range(frp.size):
            value = frp[idx]
            if value is None or (hasattr(value, "mask") and value.mask):
                continue
            if float(value) <= 0:
                continue
            canonical_records.append(
                generic_pixel_to_canonical(
                    source="SENTINEL3_FRP",
                    lon=float(lons[idx]),
                    lat=float(lats[idx]),
                    acq_datetime=acq_dt,
                    frp=float(value),
                    confidence=0.8,  # SLSTR FRP L2 land product ships pre-filtered/QC'd detections
                )
            )
    return canonical_records


def fetch_all(settings: Settings | None = None, workdir: str = "data/sentinel3") -> List[Dict]:
    """Full Sentinel-3 SLSTR FRP ingestion cycle. Never raises — returns
    an empty list if auth/search/parse fails, so this optional
    cross-validation source can't take down the main pipeline.
    """
    settings = settings or get_settings()
    try:
        token = _get_access_token(settings)
        products = _search_products(settings, token)
        records: List[Dict] = []
        for product in products:
            zip_path = _download_product(settings, token, product["Id"], workdir)
            nc_path = _extract_frp_netcdf(zip_path, workdir)
            if nc_path is None:
                logger.warning("No FRP_in.nc found inside %s", zip_path)
                continue
            records.extend(parse_frp_netcdf(nc_path))
        logger.info("Sentinel-3 FRP fetch complete: %d records from %d products", len(records), len(products))
        return records
    except Sentinel3ClientError:
        logger.exception("Sentinel-3 FRP ingestion failed this cycle")
        return []
    except requests.HTTPError as exc:
        # CDSE distinguishes "your token is not valid" from "your token is
        # valid but was not issued for this service", and the difference
        # decides whether an operator should re-check the secret or go and
        # register a different kind of client. A bare stack trace hides
        # that, so the actionable case is named explicitly.
        response = getattr(exc, "response", None)
        body = (response.text[:400] if response is not None else "")
        if response is not None and response.status_code == 401 and "audience" in body.lower():
            logger.warning(
                "Sentinel-3 FRP unavailable: the CDSE credentials in use are a "
                "Sentinel Hub client (client_id starts 'sh-'). That token is "
                "accepted by the OData catalogue — which is why the cloud-cover "
                "gate works — but NOT by the product download service, which "
                "rejects it with DAT-ZIP-609 'Token audience not allowed'. "
                "Downloading full SLSTR products needs a CDSE OAuth client "
                "registered at dataspace.copernicus.eu, not a Sentinel Hub one. "
                "This is a supplementary cross-validation source; FIRMS remains "
                "the primary feed and is unaffected."
            )
        else:
            logger.exception("Sentinel-3 FRP network error this cycle")
        return []
    except requests.RequestException:
        logger.exception("Sentinel-3 FRP network error this cycle")
        return []
