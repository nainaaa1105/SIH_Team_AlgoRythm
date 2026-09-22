"""Central configuration, loaded from environment variables / .env.

Every ingestion/enrichment module reads its credentials and tunables from
here rather than calling os.environ directly, so the whole pipeline's
external-service surface is visible in one place.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg2://sih_user:sih_pass@localhost:5432/sih_thermal"
    redis_url: str = "redis://localhost:6379/0"

    # Auth — signs the session token issued by /auth/login and /auth/signup.
    # Set a real secret via the AUTH_SECRET_KEY env var before any real
    # deployment; the default is fine for local dev only, since anyone
    # who read this file could forge a session with it.
    auth_secret_key: str = "dev-insecure-change-me"
    auth_token_ttl_hours: int = 24 * 7

    # FIRMS
    firms_map_key: str = ""
    firms_sources: str = "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT,VIIRS_NOAA21_NRT,MODIS_NRT"
    firms_bbox: str = "68.0,6.5,97.5,37.5"  # India bounding box (minLon,minLat,maxLon,maxLat)
    firms_day_range: int = 1

    # Google Earth Engine
    gee_service_account: str = ""
    gee_private_key_file: str = "secrets/gee-service-account.json"

    # Copernicus Data Space Ecosystem
    cdse_client_id: str = ""
    cdse_client_secret: str = ""
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    cdse_catalog_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    # MOSDAC / INSAT-3DS
    mosdac_username: str = ""
    mosdac_password: str = ""
    mosdac_base_url: str = "https://mosdac.gov.in"

    # EUMETSAT Data Store — temporary geostationary-fire backup while the
    # MOSDAC account is pending (see app/ingestion/eumetsat_iodc.py).
    # Free, instant, self-service registration at https://api.eumetsat.int
    # (Profile -> API key), unlike MOSDAC's multi-day manual approval.
    # Leave blank to skip this source; it auto-activates the moment both
    # are set, and steps aside again the moment mosdac_username/password
    # are filled in (see geostationary_supplementary.py's ordering).
    eumetsat_consumer_key: str = ""
    eumetsat_consumer_secret: str = ""
    eumetsat_token_url: str = "https://api.eumetsat.int/token"
    eumetsat_search_url: str = "https://api.eumetsat.int/data/search-products/v1/search"
    eumetsat_download_url: str = "https://api.eumetsat.int/data/download/products"
    # "Active Fire Monitoring (CAP) - MSG - Indian Ocean" in the EUMETSAT
    # Data Store catalogue (https://data.eumetsat.int, search "IODC fire").
    # CONFIRM this collection ID against your own Data Store account before
    # relying on it — EUMETSAT's product/collection IDs are catalogue
    # entries, not guaranteed stable across catalogue revisions.
    eumetsat_iodc_fire_collection: str = "EO:EUM:DAT:MSG:FIRC-IODC"

    # Himawari
    himawari_s3_bucket: str = "noaa-himawari8"
    himawari_region_bbox: str = "88.0,20.0,97.5,29.5"  # NE India / border zone only

    # OSM
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    osm_india_bbox: str = "68.0,6.5,97.5,37.5"

    # Open-Meteo
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"

    # Pipeline tuning
    dbscan_eps_degrees: float = 0.005
    dbscan_min_samples: int = 1
    cloud_fraction_threshold: float = 0.30
    poll_interval_minutes: int = 30

    # Emergency SMS dispatch (HttpSMS) — TEST MODE ONLY for now.
    # sms_mode="test" always sends to test_sms_recipient regardless of
    # the incident; nearest-fire-station routing is a later phase (see
    # app/notifications/sms_recipient.py::get_sms_recipient — that's the
    # one place to change when that phase starts, nothing else in the
    # dispatch flow needs to know where the recipient came from).
    sms_mode: str = "test"
    test_sms_recipient: str = ""
    httpsms_api_key: str = ""
    httpsms_from_number: str = ""
    httpsms_base_url: str = "https://api.httpsms.com/v1"
    # Used to build the "view this fire on the dashboard" link in the SMS
    # body — no deep-link/permalink mechanism existed before this feature.
    frontend_base_url: str = "http://localhost:8000"

    @property
    def firms_source_list(self) -> List[str]:
        return [s.strip() for s in self.firms_sources.split(",") if s.strip()]

    @property
    def firms_bbox_tuple(self) -> tuple:
        return tuple(float(x) for x in self.firms_bbox.split(","))

    @property
    def himawari_bbox_tuple(self) -> tuple:
        return tuple(float(x) for x in self.himawari_region_bbox.split(","))

    @property
    def osm_bbox_tuple(self) -> tuple:
        return tuple(float(x) for x in self.osm_india_bbox.split(","))


@lru_cache
def get_settings() -> Settings:
    return Settings()
