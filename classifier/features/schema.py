"""The 28-feature contract.

This module is the single source of truth for (a) which features the
model consumes, (b) what order they appear in the vector, and (c) which
team member owns each one. Training and inference both import
FEATURE_COLUMNS from here, and the model registry persists a copy
alongside every saved model so a column-order drift between train and
predict fails loudly instead of silently scrambling inputs.

Feature groups exist so the evidence-weighting engine can reason about
whole blocks of evidence going missing (e.g. every imagery feature is
unavailable when the cloud gate says optical data isn't usable) rather
than about individual NaNs.
"""
from enum import Enum
from typing import Dict, List, Tuple


class FeatureGroup(str, Enum):
    CONTEXT = "context"    # M1: land cover, facility proximity, population
    THERMAL = "thermal"    # M2: FRP / brightness statistics
    TEMPORAL = "temporal"  # M2: persistence, cadence, day/night mix
    SPATIAL = "spatial"    # M2: footprint size and growth
    IMAGERY = "imagery"    # M4: Dozier temperature, spectral indices
    RHYTHM = "rhythm"      # M5: shift-schedule fingerprint, Kalman projection


# (column_name, group). Order here IS the model's feature vector order.
FEATURE_SPEC: Tuple[Tuple[str, FeatureGroup], ...] = (
    # --- CONTEXT (7) — produced by M1's tasks.enrich_cluster ---
    ("pct_cropland", FeatureGroup.CONTEXT),
    ("pct_forest", FeatureGroup.CONTEXT),
    ("pct_urban", FeatureGroup.CONTEXT),
    ("facility_distance_m", FeatureGroup.CONTEXT),
    ("facility_prior_weight", FeatureGroup.CONTEXT),
    ("population_density", FeatureGroup.CONTEXT),
    ("near_facility", FeatureGroup.CONTEXT),
    # --- THERMAL (7) — M2, from the `hotspots` rows of the cluster ---
    ("frp_mean", FeatureGroup.THERMAL),
    ("frp_max", FeatureGroup.THERMAL),
    ("frp_std", FeatureGroup.THERMAL),
    ("frp_zscore", FeatureGroup.THERMAL),
    ("brightness_mean", FeatureGroup.THERMAL),
    ("brightness_max", FeatureGroup.THERMAL),
    ("confidence_mean", FeatureGroup.THERMAL),
    # --- TEMPORAL (5) — M2 ---
    ("persistence_days", FeatureGroup.TEMPORAL),
    ("n_detections", FeatureGroup.TEMPORAL),
    ("detections_per_day", FeatureGroup.TEMPORAL),
    ("night_fraction", FeatureGroup.TEMPORAL),
    ("source_diversity", FeatureGroup.TEMPORAL),
    # --- SPATIAL (2) — M2 ---
    ("spatial_extent_km", FeatureGroup.SPATIAL),
    ("spatial_growth_rate", FeatureGroup.SPATIAL),
    # --- IMAGERY (4) — M4, only when clusters.optical_available is true ---
    ("dozier_temp", FeatureGroup.IMAGERY),
    ("ndvi", FeatureGroup.IMAGERY),
    ("ndbi", FeatureGroup.IMAGERY),
    ("smoke_red_blue_ratio", FeatureGroup.IMAGERY),
    # --- RHYTHM (3) — M5 ---
    ("shift_sharpness", FeatureGroup.RHYTHM),
    ("weekend_suppression", FeatureGroup.RHYTHM),
    ("kalman_time_to_critical", FeatureGroup.RHYTHM),
)

FEATURE_COLUMNS: List[str] = [name for name, _ in FEATURE_SPEC]
FEATURE_GROUP_BY_COLUMN: Dict[str, FeatureGroup] = {name: group for name, group in FEATURE_SPEC}

N_FEATURES = len(FEATURE_COLUMNS)

# The five source classes the PS asks us to segregate (Phase 1 deliverable 1).
CLASSES: Tuple[str, ...] = (
    "industrial_fire",
    "gas_flare",
    "wildfire",
    "agricultural_burning",
    "mining",
)

# Deliberately NOT a model feature, though it is stored on cluster_features:
# M3's threat corridor is computed *downstream of* classification (a
# wildfire-to-facility corridor is only drawn once something is already
# classed as a spreading fire), so feeding it back in would be target
# leakage. It lives in the table for the dashboard and alerting only.
NON_FEATURE_COLUMNS: Tuple[str, ...] = ("threat_corridor_present",)


def columns_in_group(group: FeatureGroup) -> List[str]:
    return [name for name, g in FEATURE_SPEC if g is group]


def group_of(column: str) -> FeatureGroup:
    return FEATURE_GROUP_BY_COLUMN[column]
