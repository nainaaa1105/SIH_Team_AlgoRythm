import math

import pytest

from classifier.features.evidence_weighting import (
    GROUP_WEIGHTS,
    MIN_CONFIDENCE_MULTIPLIER,
    apply_evidence_weighting,
    assess_evidence,
    group_is_present,
)
from classifier.features.schema import FeatureGroup, columns_in_group


def _full_features():
    from classifier.features.schema import FEATURE_COLUMNS

    return {column: 1.0 for column in FEATURE_COLUMNS}


def test_group_weights_sum_to_one():
    assert math.isclose(sum(GROUP_WEIGHTS.values()), 1.0)


def test_every_group_has_a_weight():
    for group in FeatureGroup:
        assert group in GROUP_WEIGHTS


def test_all_groups_present_gives_full_confidence():
    assessment = assess_evidence(_full_features())
    assert assessment.missing_groups == []
    assert assessment.available_weight == pytest.approx(1.0)
    assert assessment.confidence_multiplier == pytest.approx(1.0)


def test_missing_imagery_discounts_confidence_and_explains_why():
    features = _full_features()
    for column in columns_in_group(FeatureGroup.IMAGERY):
        features[column] = None

    assessment = assess_evidence(features, optical_available=False, cloud_fraction=0.82)

    assert FeatureGroup.IMAGERY in assessment.missing_groups
    assert assessment.confidence_multiplier < 1.0
    note = assessment.notes["imagery"]
    assert "cloud" in note.lower() and "82%" in note


def test_missing_imagery_without_cloud_reason_gets_a_different_note():
    features = _full_features()
    for column in columns_in_group(FeatureGroup.IMAGERY):
        features[column] = None

    assessment = assess_evidence(features, optical_available=True)
    assert "not yet computed" in assessment.notes["imagery"]


def test_nan_counts_as_missing_just_like_none():
    features = _full_features()
    for column in columns_in_group(FeatureGroup.RHYTHM):
        features[column] = float("nan")

    assessment = assess_evidence(features)
    assert FeatureGroup.RHYTHM in assessment.missing_groups


def test_partial_group_still_counts_as_present():
    """M4 may deliver Dozier temperature but no NDVI at night; partial
    evidence beats none."""
    features = _full_features()
    imagery = columns_in_group(FeatureGroup.IMAGERY)
    for column in imagery[1:]:
        features[column] = None

    assert group_is_present(features, FeatureGroup.IMAGERY)


def test_confidence_never_falls_below_the_floor():
    """Even with nothing but a bare detection, the thermal anomaly is
    real — confidence is discounted, not annihilated."""
    features = {column: None for column in _full_features()}
    assessment = assess_evidence(features)
    assert assessment.confidence_multiplier == pytest.approx(MIN_CONFIDENCE_MULTIPLIER)
    assert assessment.available_weight == 0.0


def test_apply_evidence_weighting_converts_none_to_nan_not_zero():
    """Imputing 0.0 would tell the model 'NDVI is exactly zero here',
    which is a lie; NaN means 'unknown' to XGBoost."""
    features = _full_features()
    features["ndvi"] = None

    weighted, _ = apply_evidence_weighting(features)
    assert math.isnan(weighted["ndvi"])
    assert weighted["ndvi"] != 0.0


def test_apply_evidence_weighting_preserves_real_zero_values():
    features = _full_features()
    features["pct_forest"] = 0.0

    weighted, _ = apply_evidence_weighting(features)
    assert weighted["pct_forest"] == 0.0
    assert not math.isnan(weighted["pct_forest"])


def test_whole_database_rows_do_not_crash_the_coercion():
    """Regression: tasks.py hands in every cluster_features column, which
    includes a string (facility_type), a datetime (updated_at), an int
    primary key and a bool. float('refinery') used to raise ValueError
    here, which would have crashed every classification in production."""
    from datetime import datetime

    features = _full_features()
    features.update({
        "cluster_id": 42,
        "facility_type": "refinery",
        "updated_at": datetime(2026, 9, 12),
        "threat_corridor_present": True,
    })

    weighted, assessment = apply_evidence_weighting(features)
    assert assessment.missing_groups == []


def test_returned_vector_contains_exactly_the_declared_features():
    """No extra DB columns leak into the model input, and none are lost."""
    from datetime import datetime

    from classifier.features.schema import FEATURE_COLUMNS

    features = _full_features()
    features.update({"cluster_id": 7, "facility_type": "power", "updated_at": datetime.now()})

    weighted, _ = apply_evidence_weighting(features)
    assert set(weighted) == set(FEATURE_COLUMNS)
    assert "cluster_id" not in weighted
    assert "facility_type" not in weighted


def test_unconvertible_value_in_a_feature_column_becomes_nan_not_an_exception():
    features = _full_features()
    features["frp_mean"] = "not-a-number"

    weighted, _ = apply_evidence_weighting(features)
    assert math.isnan(weighted["frp_mean"])


def test_boolean_feature_values_are_coerced_to_one_and_zero():
    features = _full_features()
    features["near_facility"] = True

    weighted, _ = apply_evidence_weighting(features)
    assert weighted["near_facility"] == 1.0


def test_missing_keys_entirely_are_treated_as_unknown():
    """A cluster whose M4/M5 tasks never ran has no such keys at all."""
    weighted, assessment = apply_evidence_weighting({"pct_forest": 0.5})
    assert math.isnan(weighted["ndvi"])
    assert FeatureGroup.IMAGERY in assessment.missing_groups


def test_assessment_serialises_for_the_database():
    assessment = assess_evidence(_full_features())
    payload = assessment.as_json()
    assert set(payload) == {
        "available_groups", "missing_groups", "available_weight",
        "confidence_multiplier", "notes",
    }
    assert all(isinstance(g, str) for g in payload["available_groups"])
