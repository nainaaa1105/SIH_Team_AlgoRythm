"""Model registry — versioned model artefacts plus the metadata needed to
use them safely.

The metadata is not decoration: `feature_columns` is persisted with every
model and re-validated at prediction time. Feature-order drift between
training and inference is the classic silent ML bug (the model happily
consumes a scrambled vector and returns confident nonsense), and it is
exactly the kind of thing that surfaces during a live demo. Here it
raises instead.
"""
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    version: str
    trained_at: str
    feature_columns: List[str]
    classes: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    n_train_samples: int = 0
    n_test_samples: int = 0
    split_strategy: str = "spatial_tile"
    notes: str = ""


class ModelRegistryError(RuntimeError):
    pass


class FeatureOrderMismatch(ModelRegistryError):
    """Raised when inference features don't match the trained contract."""


def _version_dir(model_dir: str, version: str) -> Path:
    return Path(model_dir) / version


def save_model(model, metadata: ModelMetadata, model_dir: str = "data/models") -> Path:
    """Persist an XGBoost booster + its metadata under data/models/<version>/."""
    target = _version_dir(model_dir, metadata.version)
    target.mkdir(parents=True, exist_ok=True)

    model_path = target / "model.json"
    model.save_model(str(model_path))

    (target / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    # A stable "latest" pointer so inference doesn't have to guess the
    # newest version string. Copied rather than symlinked because symlink
    # creation needs elevated privileges on Windows by default.
    latest = Path(model_dir) / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(target, latest)

    return target


def load_metadata(model_dir: str = "data/models", version: str = "latest") -> ModelMetadata:
    path = _version_dir(model_dir, version) / "metadata.json"
    if not path.exists():
        raise ModelRegistryError(
            f"No model metadata at {path} — train a model first "
            f"(python -m scripts.train_model)"
        )
    return ModelMetadata(**json.loads(path.read_text(encoding="utf-8")))


# Cache loaded models so the Celery worker doesn't re-read the artefact
# from disk for every single cluster it classifies. Keyed on the file's
# modification time as well as its path, so a retrain that overwrites
# `latest` is picked up automatically rather than serving a stale model
# until the worker restarts.
_model_cache: Dict[tuple, Any] = {}


def load_model(model_dir: str = "data/models", version: str = "latest"):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ModelRegistryError("xgboost is required to load a model") from exc

    model_path = _version_dir(model_dir, version) / "model.json"
    if not model_path.exists():
        raise ModelRegistryError(
            f"No model artefact at {model_path} — train a model first "
            f"(python -m scripts.train_model)"
        )

    cache_key = (str(model_path), model_path.stat().st_mtime_ns)
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached

    booster = xgb.XGBClassifier()
    booster.load_model(str(model_path))

    _model_cache.clear()  # only ever serve one model per path; bound the cache
    _model_cache[cache_key] = booster
    return booster


def validate_feature_order(metadata: ModelMetadata, feature_columns: List[str]) -> None:
    """Refuse to predict on a differently-shaped or differently-ordered vector."""
    if list(metadata.feature_columns) != list(feature_columns):
        expected, got = metadata.feature_columns, list(feature_columns)
        if set(expected) == set(got):
            raise FeatureOrderMismatch(
                f"Model {metadata.version} was trained on the same features in a DIFFERENT "
                f"order. Expected {expected[:3]}... got {got[:3]}..."
            )
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise FeatureOrderMismatch(
            f"Feature contract mismatch for model {metadata.version}: "
            f"missing={missing} unexpected={extra}"
        )


def new_metadata(
    version: str,
    feature_columns: List[str],
    classes: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    n_train_samples: int = 0,
    n_test_samples: int = 0,
    notes: str = "",
) -> ModelMetadata:
    return ModelMetadata(
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        feature_columns=list(feature_columns),
        classes=list(classes),
        metrics=metrics or {},
        hyperparameters=hyperparameters or {},
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        notes=notes,
    )
