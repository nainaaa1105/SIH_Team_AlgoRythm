"""M4-specific settings.

Database, Redis, GEE and Open-Meteo settings come from M1's
`app.config.get_settings()` — all four services must agree on which
Postgres, which broker and which Earth Engine account they use.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class M4Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Sentinel-2 patch fetch ---
    s2_collection: str = "COPERNICUS/S2_SR_HARMONIZED"  # GEE collection id (fallback tier)
    s2_cdse_collection: str = "sentinel-2-l2a"  # CDSE/Sentinel Hub collection id (primary tier)
    # Patch half-width when the cluster footprint is unknown. 2 km covers
    # a plant and its immediate surroundings at 10 m resolution (~400 px).
    s2_default_halfwidth_m: float = 2000.0
    s2_max_halfwidth_m: float = 10000.0
    s2_search_days: int = 7          # +/- window around the detection
    s2_max_cloud_percentage: float = 40.0
    s2_patch_dir: str = "data/patches"
    s2_thumbnail_dir: str = "data/thumbnails"

    # --- Dozier ---
    # Sentinel-2 reflectance in GEE is scaled by 10000.
    reflectance_scale: float = 10000.0

    # --- Image classifier ---
    model_dir: str = "data/image_models"
    model_version: str = "v1"
    image_size_px: int = 224          # EfficientNet-B0's native input size
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 3e-4
    # Below this the CNN's verdict is too weak to be worth fusing; M2
    # then runs tabular-only rather than being dragged by a guess.
    min_image_confidence: float = 0.35


@lru_cache
def get_m4_settings() -> M4Settings:
    return M4Settings()
