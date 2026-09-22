from classifier.features.schema import (
    CLASSES,
    FEATURE_COLUMNS,
    FEATURE_SPEC,
    N_FEATURES,
    NON_FEATURE_COLUMNS,
    FeatureGroup,
    columns_in_group,
    group_of,
)


def test_matrix_is_exactly_28_features():
    """The team task division specifies a 28-feature matrix; if this
    changes, every trained model's metadata must change with it."""
    assert N_FEATURES == 28
    assert len(FEATURE_COLUMNS) == 28


def test_feature_columns_are_unique():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_every_feature_belongs_to_exactly_one_group():
    assigned = [col for group in FeatureGroup for col in columns_in_group(group)]
    assert sorted(assigned) == sorted(FEATURE_COLUMNS)


def test_group_lookup_round_trips():
    for column, group in FEATURE_SPEC:
        assert group_of(column) is group
        assert column in columns_in_group(group)


def test_five_source_classes_match_the_problem_statement():
    assert set(CLASSES) == {
        "industrial_fire",
        "gas_flare",
        "wildfire",
        "agricultural_burning",
        "mining",
    }


def test_threat_corridor_is_excluded_from_the_model_matrix():
    """M3's threat corridor is computed downstream of classification, so
    including it as an input would be target leakage."""
    assert "threat_corridor_present" in NON_FEATURE_COLUMNS
    assert "threat_corridor_present" not in FEATURE_COLUMNS


def test_all_six_groups_are_represented():
    for group in FeatureGroup:
        assert columns_in_group(group), f"{group} has no features"
