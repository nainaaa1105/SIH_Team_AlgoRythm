"""End-to-end: build a synthetic labelled set, train a real XGBoost
model through the real split/registry path, then classify with it.

This is the test that would catch a feature-order drift, a broken
registry round-trip, or a NaN-handling regression — the failure modes
that only appear when the pieces are wired together.
"""
import random

import numpy as np
import pytest

from classifier.features.schema import CLASSES, FEATURE_COLUMNS
from classifier.model.registry import (
    FeatureOrderMismatch,
    ModelRegistryError,
    load_metadata,
    load_model,
    new_metadata,
    validate_feature_order,
)
from classifier.model.train import build_training_matrix, encode_labels, train

pytest.importorskip("xgboost")


def _synthetic_dataset(n_per_class=30, seed=0):
    """Separable-by-construction samples spread over distinct regions.

    Each class gets its own geographic band so the spatial split has
    tiles to work with, and its own feature signature so a correctly
    wired model can actually learn something.
    """
    rng = random.Random(seed)
    signatures = {
        "industrial_fire": {"facility_distance_m": 100, "persistence_days": 40, "pct_urban": 0.7},
        "gas_flare": {"facility_distance_m": 50, "persistence_days": 80, "frp_std": 0.5},
        "wildfire": {"pct_forest": 0.9, "spatial_growth_rate": 0.8, "facility_distance_m": 9000},
        "agricultural_burning": {"pct_cropland": 0.9, "persistence_days": 1, "facility_distance_m": 7000},
        "mining": {"pct_urban": 0.2, "persistence_days": 60, "facility_distance_m": 400},
    }

    rows, labels, coordinates, cluster_ids = [], [], [], []
    cluster_id = 1
    for class_index, cls in enumerate(CLASSES):
        # Each class occupies its own longitude band, several tiles wide.
        base_lon = 70.0 + class_index * 4.0
        base_lat = 20.0 + class_index * 2.0
        for i in range(n_per_class):
            row = {column: rng.uniform(0.0, 1.0) for column in FEATURE_COLUMNS}
            for key, value in signatures[cls].items():
                row[key] = value * rng.uniform(0.9, 1.1)
            rows.append(row)
            labels.append(cls)
            coordinates.append((base_lon + (i % 6) * 0.6, base_lat + (i % 4) * 0.6))
            cluster_ids.append(cluster_id)
            cluster_id += 1
    return rows, labels, coordinates, cluster_ids


def test_encode_labels_uses_canonical_order_not_alphabetical():
    """sklearn's LabelEncoder would sort alphabetically; we keep the
    canonical CLASSES order so the mapping is stable everywhere."""
    _, class_names = encode_labels(["wildfire", "industrial_fire"])
    assert class_names == ["industrial_fire", "wildfire"]  # canonical, not alphabetical


def test_encode_labels_produces_contiguous_indices_for_a_class_subset():
    """Regression: indexing into the five-element CLASSES tuple yielded
    [0, 2, 4] for a three-class training set, which XGBoost rejects with
    'Invalid classes inferred from unique values of y'. That is the
    normal early-project case, not an edge case."""
    labels = ["industrial_fire", "wildfire", "mining"]
    encoded, class_names = encode_labels(labels)

    assert sorted(set(encoded.tolist())) == [0, 1, 2]
    assert class_names == ["industrial_fire", "wildfire", "mining"]


def test_encode_labels_round_trips_index_to_name():
    labels = ["mining", "wildfire", "mining", "gas_flare"]
    encoded, class_names = encode_labels(labels)
    assert [class_names[i] for i in encoded] == labels


def test_training_succeeds_when_only_some_classes_are_labelled(tmp_path):
    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=20)
    keep = {"industrial_fire", "wildfire", "mining"}
    filtered = [
        (r, l, c, i)
        for r, l, c, i in zip(rows, labels, coordinates, cluster_ids)
        if l in keep
    ]
    rows, labels, coordinates, cluster_ids = map(list, zip(*filtered))

    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    _, metadata = train(data, version="subset", model_dir=str(tmp_path), tune=False)

    assert metadata.classes == ["industrial_fire", "wildfire", "mining"]


def test_absent_classes_still_appear_with_zero_probability_at_inference(tmp_path):
    """A model trained on three classes must still return the full
    five-class distribution, so the API contract never changes shape."""
    from classifier.model.predict import classify_features

    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=20)
    keep = {"industrial_fire", "wildfire", "mining"}
    filtered = [
        (r, l, c, i)
        for r, l, c, i in zip(rows, labels, coordinates, cluster_ids)
        if l in keep
    ]
    rows, labels, coordinates, cluster_ids = map(list, zip(*filtered))

    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    model, metadata = train(data, version="subset2", model_dir=str(tmp_path), tune=False)

    result = classify_features(dict(rows[0]), model=model, metadata=metadata)
    assert set(result.class_probabilities) == set(CLASSES)
    assert result.class_probabilities["gas_flare"] == 0.0
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_encode_labels_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="Unknown label"):
        encode_labels(["wildfire", "volcano"])


def test_build_training_matrix_respects_feature_column_order():
    rows = [{column: float(i) for i, column in enumerate(FEATURE_COLUMNS)}]
    data = build_training_matrix(rows, ["mining"], [(77.0, 28.0)], [1])

    assert data.X.shape == (1, 28)
    assert data.feature_columns == FEATURE_COLUMNS
    np.testing.assert_array_equal(data.X[0], np.arange(len(FEATURE_COLUMNS), dtype=float))


def test_build_training_matrix_puts_nan_where_a_feature_is_missing():
    rows = [{"pct_forest": 0.5}]
    data = build_training_matrix(rows, ["wildfire"], [(77.0, 28.0)], [1])

    forest_index = FEATURE_COLUMNS.index("pct_forest")
    assert data.X[0, forest_index] == 0.5
    assert np.isnan(data.X[0, FEATURE_COLUMNS.index("ndvi")])


def test_build_training_matrix_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError):
        build_training_matrix([{}], ["mining", "wildfire"], [(1.0, 1.0)], [1])


def test_full_training_run_produces_a_usable_model(tmp_path):
    rows, labels, coordinates, cluster_ids = _synthetic_dataset()
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)

    model, metadata = train(
        data, version="test_v1", model_dir=str(tmp_path), tune=False
    )

    assert metadata.feature_columns == FEATURE_COLUMNS
    assert metadata.classes == list(CLASSES)
    assert metadata.n_train_samples > 0
    assert metadata.n_test_samples > 0
    # Classes are separable by construction, so a correctly wired
    # pipeline should comfortably beat random (0.2 for five classes).
    assert metadata.metrics["macro_f1"] > 0.5
    assert "confusion_matrix" in metadata.metrics


def test_optuna_tuning_path_runs(tmp_path):
    """Every other training test uses tune=False, so without this the
    whole Optuna branch is untested — a NameError in there would ship
    silently and only surface on the real training run."""
    pytest.importorskip("optuna")

    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=12)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)

    _, metadata = train(
        data, version="tuned", model_dir=str(tmp_path), n_trials=2, tune=True
    )

    assert metadata.hyperparameters, "tuned hyperparameters should be recorded"
    assert "n_estimators" in metadata.hyperparameters
    assert "num_class" not in metadata.hyperparameters  # inferred, never forced
    assert metadata.metrics["macro_f1"] >= 0.0


def test_saved_model_round_trips_through_the_registry(tmp_path):
    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=20)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    model, metadata = train(data, version="rt", model_dir=str(tmp_path), tune=False)

    reloaded = load_model(str(tmp_path), "rt")
    reloaded_metadata = load_metadata(str(tmp_path), "rt")

    assert reloaded_metadata.feature_columns == metadata.feature_columns
    sample = data.X[:1]
    np.testing.assert_allclose(
        model.predict_proba(sample), reloaded.predict_proba(sample), rtol=1e-5
    )


def test_latest_pointer_is_written(tmp_path):
    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=20)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    train(data, version="v9", model_dir=str(tmp_path), tune=False)

    latest = load_metadata(str(tmp_path), "latest")
    assert latest.version == "v9"


def test_registry_rejects_a_reordered_feature_contract():
    metadata = new_metadata("v1", FEATURE_COLUMNS, list(CLASSES))
    scrambled = list(reversed(FEATURE_COLUMNS))

    with pytest.raises(FeatureOrderMismatch, match="DIFFERENT"):
        validate_feature_order(metadata, scrambled)


def test_registry_reports_missing_and_extra_features():
    metadata = new_metadata("v1", FEATURE_COLUMNS, list(CLASSES))
    wrong = [c for c in FEATURE_COLUMNS if c != "ndvi"] + ["bogus_feature"]

    with pytest.raises(FeatureOrderMismatch) as exc:
        validate_feature_order(metadata, wrong)
    assert "ndvi" in str(exc.value)
    assert "bogus_feature" in str(exc.value)


def test_registry_accepts_the_matching_contract():
    metadata = new_metadata("v1", FEATURE_COLUMNS, list(CLASSES))
    validate_feature_order(metadata, FEATURE_COLUMNS)  # must not raise


def test_loading_a_missing_model_gives_an_actionable_error(tmp_path):
    with pytest.raises(ModelRegistryError, match="train a model first"):
        load_model(str(tmp_path), "nonexistent")


def test_classify_features_end_to_end(tmp_path):
    """The full inference path: features -> evidence weighting -> model
    -> fusion -> SHAP -> result."""
    from classifier.model.predict import classify_features

    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=25)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    model, metadata = train(data, version="infer", model_dir=str(tmp_path), tune=False)

    features = dict(rows[0])
    result = classify_features(
        features, model=model, metadata=metadata, optical_available=True
    )

    assert result.predicted_class in CLASSES
    assert 0.0 <= result.confidence_score <= 100.0
    assert result.fusion_weight_xgb == 1.0  # no image probabilities supplied
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert result.shap_values, "expected some explanation contributions"
    assert result.reasons
    assert result.model_version == "infer"


def test_classify_features_with_cloud_blocked_imagery_is_discounted(tmp_path):
    from classifier.model.predict import classify_features

    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=25)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    model, metadata = train(data, version="cloudy", model_dir=str(tmp_path), tune=False)

    features = dict(rows[0])
    for column in ("dozier_temp", "ndvi", "ndbi", "smoke_red_blue_ratio"):
        features[column] = None

    result = classify_features(
        features, model=model, metadata=metadata,
        optical_available=False, cloud_fraction=0.9,
    )

    assert "imagery" in result.evidence["missing_groups"]
    assert result.evidence["confidence_multiplier"] < 1.0
    assert "cloud" in result.evidence["notes"]["imagery"].lower()


def test_classify_features_accepts_a_full_database_row(tmp_path):
    """Regression: the real caller (tasks.m2_classify) passes every
    cluster_features column, not a clean feature dict — including
    facility_type (str), updated_at (datetime) and cluster_id (int)."""
    from datetime import datetime

    from classifier.model.predict import classify_features

    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=20)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    model, metadata = train(data, version="dbrow", model_dir=str(tmp_path), tune=False)

    db_row = dict(rows[0])
    db_row.update({
        "cluster_id": 123,
        "facility_type": "refinery",
        "threat_corridor_present": True,
        "updated_at": datetime(2026, 9, 12),
    })

    result = classify_features(db_row, model=model, metadata=metadata)
    assert result.predicted_class in CLASSES
    assert 0.0 <= result.confidence_score <= 100.0


def test_classify_features_applies_fusion_when_image_probabilities_present(tmp_path):
    from classifier.model.predict import classify_features

    rows, labels, coordinates, cluster_ids = _synthetic_dataset(n_per_class=25)
    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    model, metadata = train(data, version="fused", model_dir=str(tmp_path), tune=False)

    result = classify_features(
        dict(rows[0]),
        model=model, metadata=metadata,
        image_probabilities={"wildfire": 1.0},
        optical_available=True,
    )

    assert result.fusion_weight_xgb == pytest.approx(0.70)
    assert result.image_probabilities is not None
    assert result.image_probabilities["wildfire"] == pytest.approx(1.0)
