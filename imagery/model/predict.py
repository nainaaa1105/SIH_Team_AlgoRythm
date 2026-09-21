"""Image-model inference for one patch.

Kept pure of DB and Celery so the decision logic is unit-testable; the
task layer does the reads and writes around it.

The `min_image_confidence` gate matters: M2 fuses at 0.70/0.30, so a
weak, near-uniform image verdict still shifts the final answer by up to
30%. When the CNN is basically guessing it is better to hand M2 nothing
and let it run tabular-only — which its evidence-weighting engine already
handles and explains — than to dilute a confident tabular verdict with
noise.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from imagery.config import get_m4_settings
from imagery.model.registry import (
    ImageModelMetadata,
    ImageRegistryError,
    load_metadata,
    load_state_dict,
    validate_band_order,
)

logger = logging.getLogger(__name__)

_model_cache: Dict[tuple, object] = {}


@dataclass
class ImagePredictionResult:
    class_probabilities: Dict[str, float]
    predicted_class: str
    confidence: float
    model_version: str
    usable_for_fusion: bool
    reason: str = ""


def top_class(probabilities: Dict[str, float]) -> tuple:
    """Argmax with a deterministic tie-break on the class name."""
    if not probabilities:
        return None, 0.0
    best = max(sorted(probabilities), key=lambda cls: probabilities[cls])
    return best, probabilities[best]


def load_cached_model(metadata: ImageModelMetadata, model_dir: str, version: str):
    """Load once per worker rather than per cluster.

    Keyed on the artefact's modification time as well as its path, so a
    retrain that overwrites `latest` is picked up without a restart —
    same approach as M2's registry.
    """
    from pathlib import Path

    from imagery.model.efficientnet import build_model

    artefact = Path(model_dir) / version / "model.pt"
    mtime = artefact.stat().st_mtime_ns if artefact.exists() else 0
    key = (str(artefact), mtime)

    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    model = build_model(
        n_classes=len(metadata.classes),
        in_channels=len(metadata.band_order),
        pretrained=False,          # weights come from the checkpoint
    )
    model.load_state_dict(load_state_dict(model_dir, version))
    model.eval()

    _model_cache.clear()           # only ever one model per path; bound the cache
    _model_cache[key] = model
    return model


def classify_patch(
    bands: Dict[str, np.ndarray],
    model=None,
    metadata: Optional[ImageModelMetadata] = None,
    model_dir: Optional[str] = None,
    version: str = "latest",
) -> Optional[ImagePredictionResult]:
    """Classify one Sentinel-2 patch.

    Returns None when no model is trained yet — a normal state early in
    the project, and M2 carries on tabular-only.
    """
    settings = get_m4_settings()
    model_dir = model_dir or settings.model_dir

    try:
        if metadata is None:
            metadata = load_metadata(model_dir, version)
        if model is None:
            model = load_cached_model(metadata, model_dir, version)
    except ImageRegistryError as exc:
        logger.info("Image model unavailable: %s", exc)
        return None
    except ImportError as exc:
        logger.info("Image model unavailable: %s", exc)
        return None

    present = [b for b in metadata.band_order if b in bands]
    # Fail loudly on a scrambled/mismatched band contract rather than
    # feeding the network a silently wrong tensor.
    validate_band_order(metadata, present)

    stack = np.stack([np.asarray(bands[b], dtype=float) for b in metadata.band_order], axis=0)

    from imagery.model.efficientnet import predict_probabilities

    probabilities = predict_probabilities(
        model, stack, metadata.classes, metadata.image_size_px
    )
    predicted, confidence = top_class(probabilities)

    usable = confidence >= settings.min_image_confidence
    return ImagePredictionResult(
        class_probabilities={k: round(v, 6) for k, v in probabilities.items()},
        predicted_class=predicted,
        confidence=confidence,
        model_version=metadata.version,
        usable_for_fusion=usable,
        reason=(
            "ok" if usable else
            f"image confidence {confidence:.2f} below the {settings.min_image_confidence:.2f} "
            "fusion threshold — withheld so a weak guess cannot pull the tabular verdict"
        ),
    )
