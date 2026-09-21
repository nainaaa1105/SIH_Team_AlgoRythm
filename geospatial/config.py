"""M3-specific settings.

Database/Redis/GEE/Open-Meteo settings are NOT redefined here — they come
from M1's `app.config.get_settings()` so all three services agree on
which Postgres, which broker and which Earth Engine account they use.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class M3Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Attribution
    facility_search_radius_m: float = 5000.0
    attribution_confidence_margin: float = 0.25

    # Land cover
    worldcover_collection: str = "ESA/WorldCover/v200"

    # Plume
    plume_polygon_steps: int = 24
    # M1's `open_meteo_url` points at the forecast endpoint, which only
    # covers roughly the last few days. The 90-day retrospective replay
    # needs the archive endpoint or every historical plume silently falls
    # back to assumed wind.
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    archive_cutoff_days: int = 5
    population_raster: str = "data/population/worldpop_india.tif"
    population_raster_is_density: bool = False

    # Threat corridor
    corridor_projection_hours: float = 24.0
    corridor_arc_steps: int = 12

    # Evacuation routing (OSMnx). Off by default: building a road graph
    # is slow and network-bound, which is the wrong thing to do inside a
    # per-cluster task that runs every 30 minutes.
    routing_enabled: bool = False
    routing_search_radius_m: float = 10000.0

    # WUI (wildland-urban interface) proximity. The ember-jump distance
    # itself (2400 m) and the watch/critical thresholds derived from it
    # are module constants in geospatial/wui_analysis.py, matching how
    # threat/corridor.py keeps its own half-angle bounds as constants
    # rather than settings — they are physical/policy thresholds, not
    # environment tunables. The two search radii here ARE environment
    # tunables (how far to look), same category as facility_search_radius_m.
    wui_builtup_search_radius_m: float = 10000.0
    wui_settlement_search_radius_m: float = 15000.0


@lru_cache
def get_m3_settings() -> M3Settings:
    return M3Settings()
