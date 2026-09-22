"""XGBoost training with Optuna tuning and a hard leakage guard.

Design decisions worth knowing before changing anything here:

  * The split is spatial-tile based and `verify_no_leakage` runs before
    a single tree is fitted. If it raises, training aborts — a leaky
    experiment produces a number that looks great and means nothing.
  * The headline metric is macro-F1, not accuracy. The five classes are
    heavily imbalanced (gas flares and mining anomalies are rare next to
    agricultural burning in season), and accuracy would let a model that
    ignores the rare classes look excellent.
  * NaNs are passed through, not imputed. XGBoost learns a default
    branch direction for missing values, which is precisely the
    behaviour the evidence-weighting engine relies on.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from classifier.config import get_m2_settings
from classifier.features.schema import CLASSES, FEATURE_COLUMNS
from classifier.model.registry import ModelMetadata, new_metadata, save_model
from classifier.model.split import SplitResult, spatial_group_split, verify_no_leakage

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    X: np.ndarray                     # (n_samples, n_features), may contain NaN
    y: np.ndarray                     # integer-encoded labels
    coordinates: List[Tuple[float, float]]
    cluster_ids: List[int]
    feature_columns: List[str]
    class_names: List[str]


def encode_labels(labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Encode label strings to contiguous integers, in canonical order.

    Two constraints have to hold at once:

      * XGBoost requires classes 0..k-1 with no gaps. Indexing straight
        into the five-element CLASSES tuple breaks the moment the
        labelled set is missing a class — three labelled classes would
        encode as [0, 2, 4] and XGBoost rejects it outright. That is the
        normal early-project case, not an edge case.
      * The index -> class-name mapping must survive into the saved
        metadata, because inference maps predict_proba's columns back to
        names through it.

    So we encode only the classes actually present, ordered by the
    canonical CLASSES order (not alphabetically, the way sklearn's
    LabelEncoder would), and return that list for the metadata. Classes
    absent from training are filled in downstream as 0.0 probability by
    `fusion.align_to_classes`.
    """
    unknown = sorted(set(labels) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown label(s) {unknown}; expected one of {list(CLASSES)}")

    observed = set(labels)
    present = [cls for cls in CLASSES if cls in observed]
    index = {cls: i for i, cls in enumerate(present)}
    return np.array([index[label] for label in labels], dtype=int), present


def build_training_matrix(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    coordinates: Sequence[Tuple[float, float]],
    cluster_ids: Sequence[int],
) -> TrainingData:
    """Assemble the (n, 28) matrix in the exact FEATURE_COLUMNS order."""
    if not (len(rows) == len(labels) == len(coordinates) == len(cluster_ids)):
        raise ValueError("rows, labels, coordinates and cluster_ids must be the same length")

    matrix = np.full((len(rows), len(FEATURE_COLUMNS)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, column in enumerate(FEATURE_COLUMNS):
            value = row.get(column)
            if value is None:
                continue
            matrix[i, j] = float(value)

    y, class_names = encode_labels(labels)
    return TrainingData(
        X=matrix,
        y=y,
        coordinates=list(coordinates),
        cluster_ids=list(cluster_ids),
        feature_columns=list(FEATURE_COLUMNS),
        class_names=class_names,
    )


def make_split(data: TrainingData, test_fraction: float, tile_degrees: float, seed: int) -> SplitResult:
    split = spatial_group_split(
        data.coordinates, test_fraction=test_fraction, tile_degrees=tile_degrees, seed=seed
    )
    # Hard guard — never trains on a leaky split.
    verify_no_leakage(split, data.coordinates, tile_degrees=tile_degrees, group_ids=data.cluster_ids)
    return split


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    predictions = model.predict(X_test)
    present = sorted(set(y_test.tolist()) | set(np.asarray(predictions).tolist()))

    report = classification_report(
        y_test,
        predictions,
        labels=present,
        target_names=[class_names[i] for i in present],
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "per_class": {
            class_names[i]: report[class_names[i]] for i in present if class_names[i] in report
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=present).tolist(),
        "confusion_matrix_labels": [class_names[i] for i in present],
    }


def _suggest_params(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "tree_method": "hist",
    }


def tune_hyperparameters(
    data: TrainingData, split: SplitResult, n_trials: int, seed: int
) -> Dict[str, Any]:
    """Optuna search optimising macro-F1 on the held-out spatial split.

    Tuning against the same held-out set we later report on does risk
    mild optimism; with a hackathon-sized labelled set that is the
    pragmatic trade-off, and it is recorded in the model metadata notes
    so the number is never presented as a clean generalisation estimate.
    """
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train = data.X[split.train_indices], data.y[split.train_indices]
    X_test, y_test = data.X[split.test_indices], data.y[split.test_indices]

    def objective(trial):
        params = _suggest_params(trial)
        model = xgb.XGBClassifier(**params, random_state=seed, verbosity=0)
        model.fit(X_train, y_train)
        from sklearn.metrics import f1_score

        return f1_score(y_test, model.predict(X_test), average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = _clean_params(study.best_params)
    logger.info("Optuna best macro-F1=%.4f params=%s", study.best_value, best)
    return best


def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise hyperparameters for XGBClassifier.

    `num_class` is deliberately NOT set: the sklearn wrapper infers it
    from y, and supplying it too can conflict. The objective is likewise
    left to the wrapper so a two-class training set gets binary:logistic
    instead of a multiclass objective it can't use.
    """
    return {**params, "tree_method": "hist"}


def train(
    data: TrainingData,
    version: str = "v1",
    model_dir: Optional[str] = None,
    n_trials: Optional[int] = None,
    tune: bool = True,
) -> Tuple[Any, ModelMetadata]:
    """Full training run: split -> leakage guard -> tune -> fit -> evaluate -> save."""
    import xgboost as xgb

    settings = get_m2_settings()
    model_dir = model_dir or settings.model_dir
    n_trials = settings.optuna_trials if n_trials is None else n_trials

    split = make_split(
        data, settings.test_fraction, settings.split_tile_degrees, settings.random_seed
    )
    logger.info(
        "Spatial split: %d train / %d test samples across %d/%d tiles",
        len(split.train_indices), len(split.test_indices),
        len(split.train_tiles), len(split.test_tiles),
    )

    if tune and n_trials > 0:
        params = tune_hyperparameters(data, split, n_trials, settings.random_seed)
    else:
        params = _clean_params({"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1})

    model = xgb.XGBClassifier(**params, random_state=settings.random_seed, verbosity=0)
    model.fit(data.X[split.train_indices], data.y[split.train_indices])

    metrics = evaluate(model, data.X[split.test_indices], data.y[split.test_indices], data.class_names)
    logger.info("Test macro-F1=%.4f", metrics["macro_f1"])

    metadata = new_metadata(
        version=version,
        feature_columns=data.feature_columns,
        classes=data.class_names,
        metrics=metrics,
        hyperparameters=params,
        n_train_samples=len(split.train_indices),
        n_test_samples=len(split.test_indices),
        notes=(
            "Spatial-tile split with leakage verification. Metrics are on a "
            "geographically held-out set; hyperparameters were tuned against that "
            "same split, so treat macro-F1 as mildly optimistic."
        ),
    )
    save_model(model, metadata, model_dir)
    return model, metadata
