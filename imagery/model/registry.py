"""Versioned artefacts for the image model.

Mirrors M2's registry discipline deliberately, because the same silent
bug class applies: if the band order at inference differs from the band
order at training, the model consumes a scrambled tensor and returns
confident nonsense with no error anywhere. M2 pins `feature_columns`;
M4 pins `band_order` and `classes`, and validates both before predicting.
"""
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ImageModelMetadata:
    version: str
    trained_at: str
    band_order: List[str]
    classes: List[str]
    image_size_px: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    n_train_samples: int = 0
    n_test_samples: int = 0
    architecture: str = "efficientnet_b0"
    split_strategy: str = "spatial_tile"
    notes: str = ""


class ImageRegistryError(RuntimeError):
    pass


class BandOrderMismatch(ImageRegistryError):
    """Raised when inference bands don't match the trained contract."""


def _version_dir(model_dir: str, version: str) -> Path:
    return Path(model_dir) / version


def save_model(state_dict, metadata: ImageModelMetadata, model_dir: str = "data/image_models") -> Path:
    target = _version_dir(model_dir, metadata.version)
    target.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise ImageRegistryError("torch is required to save an image model") from exc

    torch.save(state_dict, target / "model.pt")
    (target / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    # "latest" pointer, copied rather than symlinked because symlinks
    # need elevated privileges on Windows by default.
    latest = Path(model_dir) / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(target, latest)
    return target


def load_metadata(model_dir: str = "data/image_models", version: str = "latest") -> ImageModelMetadata:
    path = _version_dir(model_dir, version) / "metadata.json"
    if not path.exists():
        raise ImageRegistryError(
            f"No image-model metadata at {path} — train one first "
            f"(python -m scripts.train_image_model)"
        )
    return ImageModelMetadata(**json.loads(path.read_text(encoding="utf-8")))


def load_state_dict(model_dir: str = "data/image_models", version: str = "latest"):
    try:
        import torch
    except ImportError as exc:
        raise ImageRegistryError("torch is required to load an image model") from exc

    path = _version_dir(model_dir, version) / "model.pt"
    if not path.exists():
        raise ImageRegistryError(
            f"No image-model artefact at {path} — train one first "
            f"(python -m scripts.train_image_model)"
        )
    return torch.load(path, map_location="cpu")


def validate_band_order(metadata: ImageModelMetadata, band_order: List[str]) -> None:
    """Refuse to predict on a differently-ordered or differently-shaped stack."""
    if list(metadata.band_order) != list(band_order):
        expected, got = list(metadata.band_order), list(band_order)
        if set(expected) == set(got):
            raise BandOrderMismatch(
                f"Model {metadata.version} was trained on the same bands in a DIFFERENT "
                f"order. Expected {expected}, got {got}"
            )
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise BandOrderMismatch(
            f"Band contract mismatch for model {metadata.version}: "
            f"missing={missing} unexpected={extra}"
        )


def new_metadata(
    version: str,
    band_order: List[str],
    classes: List[str],
    image_size_px: int,
    metrics: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    n_train_samples: int = 0,
    n_test_samples: int = 0,
    notes: str = "",
) -> ImageModelMetadata:
    return ImageModelMetadata(
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        band_order=list(band_order),
        classes=list(classes),
        image_size_px=image_size_px,
        metrics=metrics or {},
        hyperparameters=hyperparameters or {},
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        notes=notes,
    )
