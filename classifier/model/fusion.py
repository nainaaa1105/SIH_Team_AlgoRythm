"""Late fusion of the tabular classifier with M4's image classifier.

Architecture doc: 0.70 x XGBoost + 0.30 x EfficientNet-B0, applied only
when Sentinel-2 imagery was actually available for the cluster (M1's
cloud gate decides that and records it on `clusters.optical_available`).

When imagery is unavailable the tabular model carries the full weight —
we do NOT fall back to a zero-filled image vector, which would drag every
class probability toward whatever the image model predicts for a blank
patch.
"""
from typing import Dict, Optional, Tuple

from classifier.features.schema import CLASSES


def _normalise(probabilities: Dict[str, float]) -> Dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        # Degenerate input — fall back to a uniform distribution rather
        # than dividing by zero or returning all-zeros (which would make
        # argmax meaningless).
        return {cls: 1.0 / len(CLASSES) for cls in CLASSES}
    return {cls: value / total for cls, value in probabilities.items()}


def align_to_classes(probabilities: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Coerce a probability dict onto the canonical class list.

    Guards the realistic integration failure where M4's model emits a
    different class ordering, a subset of classes, or an extra label:
    unknown labels are dropped, missing ones filled with 0.0, and the
    result renormalised.
    """
    if probabilities is None:
        return None
    aligned = {cls: float(probabilities.get(cls, 0.0)) for cls in CLASSES}
    return _normalise(aligned)


def fuse(
    xgb_probabilities: Dict[str, float],
    image_probabilities: Optional[Dict[str, float]] = None,
    weight_xgb: float = 0.70,
) -> Tuple[Dict[str, float], float]:
    """Blend the two models' outputs.

    Returns (fused_probabilities, weight_actually_given_to_xgb) — the
    second value is persisted on the classification row so the dashboard
    can show whether a verdict had image corroboration or not.
    """
    if not 0.0 <= weight_xgb <= 1.0:
        raise ValueError("weight_xgb must be between 0 and 1")

    tabular = align_to_classes(xgb_probabilities)
    imagery = align_to_classes(image_probabilities)

    if imagery is None:
        return tabular, 1.0

    fused = {
        cls: weight_xgb * tabular[cls] + (1.0 - weight_xgb) * imagery[cls]
        for cls in CLASSES
    }
    return _normalise(fused), weight_xgb


def top_class(probabilities: Dict[str, float]) -> Tuple[str, float]:
    """Argmax with a deterministic tie-break (canonical class order)."""
    best = max(CLASSES, key=lambda cls: (probabilities.get(cls, 0.0), -CLASSES.index(cls)))
    return best, probabilities.get(best, 0.0)


def confidence_score(probabilities: Dict[str, float], confidence_multiplier: float = 1.0) -> float:
    """Headline 0-100 confidence for the UI.

    Built from the winning probability, discounted by the evidence
    multiplier from the evidence-weighting engine, so a verdict reached
    without optical corroboration visibly scores lower than the same
    verdict with it.
    """
    _, probability = top_class(probabilities)
    return round(max(0.0, min(1.0, probability * confidence_multiplier)) * 100.0, 2)
