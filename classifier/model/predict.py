"""Single-cluster inference: features -> evidence weighting -> XGBoost ->
fusion with M4 -> SHAP -> a classification result ready to persist.

`classify_features` is deliberately pure (no DB, no Celery): it takes a
feature dict and returns a dataclass. `tasks.py` does the DB reads and
writes around it, which keeps the decision logic unit-testable.
"""
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from classifier.config import get_m2_settings
from classifier.features.evidence_weighting import apply_evidence_weighting
from classifier.features.schema import CLASSES, FEATURE_COLUMNS
from classifier.model.fusion import align_to_classes, confidence_score, fuse, top_class
from classifier.model.registry import load_metadata, load_model, validate_feature_order
from classifier.model.shap_explain import explain_prediction, to_human_reasons

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    predicted_class: str
    class_probabilities: Dict[str, float]
    xgb_probabilities: Dict[str, float]
    image_probabilities: Optional[Dict[str, float]]
    fusion_weight_xgb: float
    confidence_score: float
    shap_values: Dict[str, float]
    explanation_method: str
    reasons: List[str]
    evidence: Dict[str, Any]
    model_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_feature_row(features: Dict[str, Optional[float]]) -> np.ndarray:
    """Order a feature dict into the model's expected vector.

    Missing keys become NaN rather than raising: a cluster whose M4/M5
    features haven't landed yet must still be classifiable, which is the
    whole point of the evidence-weighting design.
    """
    row = np.full((1, len(FEATURE_COLUMNS)), np.nan, dtype=float)
    for j, column in enumerate(FEATURE_COLUMNS):
        value = features.get(column)
        if value is None:
            continue
        row[0, j] = float(value)
    return row


def classify_features(
    features: Dict[str, Optional[float]],
    model=None,
    metadata=None,
    image_probabilities: Optional[Dict[str, float]] = None,
    optical_available: Optional[bool] = None,
    cloud_fraction: Optional[float] = None,
    model_dir: Optional[str] = None,
    version: str = "latest",
) -> ClassificationResult:
    """Classify one cluster from its assembled feature dict."""
    settings = get_m2_settings()
    model_dir = model_dir or settings.model_dir

    if model is None:
        model = load_model(model_dir, version)
    if metadata is None:
        metadata = load_metadata(model_dir, version)

    # Refuses to run on a scrambled/mismatched feature contract.
    validate_feature_order(metadata, FEATURE_COLUMNS)

    weighted_features, assessment = apply_evidence_weighting(
        features, optical_available=optical_available, cloud_fraction=cloud_fraction
    )
    row = build_feature_row(weighted_features)

    raw_probabilities = model.predict_proba(row)[0]
    xgb_probabilities = {
        cls: float(raw_probabilities[i]) for i, cls in enumerate(metadata.classes)
    }
    xgb_probabilities = align_to_classes(xgb_probabilities)

    fused, weight_used = fuse(
        xgb_probabilities, image_probabilities, weight_xgb=settings.fusion_weight_xgb
    )
    predicted_class, _ = top_class(fused)
    score = confidence_score(fused, assessment.confidence_multiplier)

    class_index = CLASSES.index(predicted_class)
    contributions, method = explain_prediction(model, row, class_index, FEATURE_COLUMNS)

    return ClassificationResult(
        predicted_class=predicted_class,
        class_probabilities={k: round(v, 6) for k, v in fused.items()},
        xgb_probabilities={k: round(v, 6) for k, v in xgb_probabilities.items()},
        image_probabilities=(
            {k: round(v, 6) for k, v in align_to_classes(image_probabilities).items()}
            if image_probabilities is not None
            else None
        ),
        fusion_weight_xgb=weight_used,
        confidence_score=score,
        shap_values={k: round(v, 6) for k, v in contributions.items()},
        explanation_method=method,
        reasons=to_human_reasons(contributions, predicted_class),
        evidence=assessment.as_json(),
        model_version=metadata.version,
    )
