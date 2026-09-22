"""M2-specific settings.

Database/Redis URLs are NOT redefined here — those come from M1's
`app.config.get_settings()`, so both services always agree on which
Postgres and which broker they're talking to.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class M2Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model registry
    model_dir: str = "data/models"
    model_version: str = "v1"

    # Late fusion (architecture doc: 0.70 x XGBoost + 0.30 x EfficientNet)
    fusion_weight_xgb: float = 0.70

    # Spatial split guard against the leakage failure mode documented in
    # "Spatial Leakage in Classifying NASA FIRMS Thermal Anomalies".
    split_tile_degrees: float = 0.5   # ~55 km tiles; no tile spans train and test
    test_fraction: float = 0.2
    random_seed: int = 42

    # Optuna
    optuna_trials: int = 30

    # Label-rule parameters
    flare_match_distance_m: float = 500.0
    industrial_match_distance_m: float = 500.0
    persistent_source_min_days: int = 14


@lru_cache
def get_m2_settings() -> M2Settings:
    return M2Settings()
