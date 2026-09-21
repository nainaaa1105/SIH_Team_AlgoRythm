"""M5-specific settings.

Database and Redis come from M1's `app.config.get_settings()` so all five
services agree on which Postgres and which broker they use.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class M5Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Kalman ---
    # Relative (scales with FRP^2). Empirically tuned against the
    # walk-forward calibration harness; see the tables on
    # DEFAULT_PROCESS_NOISE and CALIBRATED_STD_MULTIPLIER in
    # forecast/kalman.py for the measurements behind both values.
    process_noise: float = 1e-7
    relative_measurement_error: float = 0.30
    # Off by default: it double-counts now that q scales with FRP^2, and
    # measurably destroys the rate estimate.
    adaptive_process_noise: bool = False

    # --- Escalation ---
    critical_sigma_multiple: float = 3.0
    escalation_samples: int = 4000
    max_horizon_hours: float = 240.0
    escalation_seed: int = 42          # deterministic Monte Carlo for reproducible demos

    # --- Rhythm ---
    # Sunday only: the weekly off in Indian industry is Sunday, with
    # Saturday commonly a working day. Treating Sat+Sun as "weekend"
    # dilutes a real signal with a working day.
    weekend_days: str = "6"
    min_detections_for_rhythm: int = 8

    # --- PTSI ---
    baseline_window_days: float = 90.0
    persistent_threshold: float = 0.60

    @property
    def weekend_day_indices(self) -> tuple:
        return tuple(int(d) for d in self.weekend_days.split(",") if d.strip())


@lru_cache
def get_m5_settings() -> M5Settings:
    return M5Settings()
