"""SHAP explanations for a single prediction.

Feeds the dashboard's XAI panel: "why was this tagged an industrial fire
rather than a flare?" — Phase 3 deliverable 11 asks for exactly this.

shap is an optional import. If it isn't installed, explanation degrades
to XGBoost's own gain-based feature importances rather than taking the
whole prediction path down with it — an unexplained classification is
still a useful classification.
"""
import logging
import weakref
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from classifier.features.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Keyed weakly on the model object itself, NOT on id(model): CPython
# recycles ids after garbage collection, so an id-keyed cache can hand
# back an explainer built for a previous, unrelated model — wrong
# explanations with no error. A WeakKeyDictionary drops its entry when
# the model is collected, so that can't happen.
_explainer_cache: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


def _get_explainer(model):
    """Cache one TreeExplainer per model — building it is the expensive
    part, and inference runs once per cluster per cycle.
    """
    cached = _explainer_cache.get(model)
    if cached is None:
        import shap

        cached = shap.TreeExplainer(model)
        _explainer_cache[model] = cached
    return cached


def _fallback_importances(model, feature_columns: List[str]) -> Dict[str, float]:
    try:
        importances = model.feature_importances_
    except (AttributeError, ValueError):
        return {}
    return {col: float(val) for col, val in zip(feature_columns, importances)}


def explain_prediction(
    model,
    feature_vector: np.ndarray,
    predicted_class_index: int,
    feature_columns: Optional[List[str]] = None,
    top_k: int = 5,
) -> Tuple[Dict[str, float], str]:
    """Per-feature SHAP contributions toward the predicted class.

    Returns (contributions, method) where method is "shap" or
    "feature_importance_fallback" so the UI can label the explanation
    honestly rather than implying SHAP when it wasn't available.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    row = np.asarray(feature_vector, dtype=float).reshape(1, -1)

    try:
        explainer = _get_explainer(model)
        shap_values = explainer.shap_values(row)
    except Exception:
        logger.warning("SHAP unavailable, falling back to gain importances", exc_info=True)
        importances = _fallback_importances(model, feature_columns)
        top = dict(sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])
        return top, "feature_importance_fallback"

    values = np.asarray(shap_values)

    # shap returns different shapes across versions/objectives for
    # multiclass models: a list of per-class arrays, a 3-D array indexed
    # (sample, feature, class), or a plain 2-D array for binary. Normalise
    # all of them to a 1-D per-feature vector for the predicted class.
    if isinstance(shap_values, list):
        per_class = np.asarray(shap_values[predicted_class_index])
        contributions = per_class[0] if per_class.ndim == 2 else per_class
    elif values.ndim == 3:
        contributions = values[0, :, predicted_class_index]
    else:
        contributions = values[0]

    contributions = np.asarray(contributions, dtype=float).ravel()
    if contributions.shape[0] != len(feature_columns):
        logger.warning(
            "SHAP returned %d contributions for %d features — falling back",
            contributions.shape[0], len(feature_columns),
        )
        importances = _fallback_importances(model, feature_columns)
        top = dict(sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])
        return top, "feature_importance_fallback"

    paired = {col: float(val) for col, val in zip(feature_columns, contributions)}
    top = dict(sorted(paired.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])
    return top, "shap"


def to_human_reasons(contributions: Dict[str, float], predicted_class: str) -> List[str]:
    """Turn SHAP numbers into the plain sentences the event card shows."""
    phrasing = {
        "facility_distance_m": "distance to the nearest industrial facility",
        "facility_prior_weight": "proximity weighting to a known facility",
        "near_facility": "sits inside a known industrial footprint",
        "pct_forest": "surrounding forest cover",
        "pct_cropland": "surrounding cropland cover",
        "pct_urban": "surrounding built-up cover",
        "population_density": "nearby population density",
        "frp_mean": "average fire radiative power",
        "frp_max": "peak fire radiative power",
        "frp_std": "variability in radiative power",
        "frp_zscore": "how far radiative power deviates from this source's own norm",
        "brightness_mean": "average brightness temperature",
        "brightness_max": "peak brightness temperature",
        "confidence_mean": "detection confidence",
        "persistence_days": "how many days this source has persisted",
        "n_detections": "number of satellite detections",
        "detections_per_day": "detection frequency",
        "night_fraction": "share of night-time detections",
        "source_diversity": "number of independent satellites that saw it",
        "spatial_extent_km": "physical footprint size",
        "spatial_growth_rate": "how fast the footprint is spreading",
        "dozier_temp": "sub-pixel combustion temperature",
        "ndvi": "vegetation index",
        "ndbi": "built-up index",
        "smoke_red_blue_ratio": "smoke plume colour signature",
        "shift_sharpness": "how sharply activity follows a shift schedule",
        "weekend_suppression": "drop in activity at weekends",
        "kalman_time_to_critical": "projected time to a critical threshold",
    }
    reasons = []
    for feature, value in contributions.items():
        label = phrasing.get(feature, feature.replace("_", " "))
        direction = "supports" if value > 0 else "argues against"
        reasons.append(f"{label.capitalize()} {direction} '{predicted_class}' (SHAP {value:+.3f})")
    return reasons
