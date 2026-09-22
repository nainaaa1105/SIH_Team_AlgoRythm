"""Contract tests against Member 1's package.

These are the tests that catch integration drift — the failures that
don't show up in either member's own unit tests but break the running
pipeline. Everything here imports M1's real code, not a stub.
"""
import pytest

pytest.importorskip("app.db.models", reason="app package must be importable")

from classifier.db.models import Classification, ClusterFeatures, TrainingLabel
from classifier.features.schema import FEATURE_COLUMNS, NON_FEATURE_COLUMNS


def test_m2_tables_register_on_m1s_shared_metadata():
    """One database, one MetaData. If M2 declared its own Base, Alembic
    autogenerate would try to drop M2's tables every time M1 ran it."""
    from app.db.models import Base

    tables = set(Base.metadata.tables)
    assert {"cluster_features", "classifications", "training_labels"} <= tables
    assert {"clusters", "hotspots", "fingerprints", "facilities"} <= tables


def test_cluster_features_foreign_keys_resolve_to_m1s_clusters_table():
    fks = {
        list(fk.columns)[0].name: fk.elements[0].target_fullname
        for fk in ClusterFeatures.__table__.foreign_key_constraints
    }
    assert fks["cluster_id"] == "clusters.id"


def test_every_model_feature_has_a_backing_database_column():
    """Drift here means the assembler silently writes nothing for a
    feature and the model trains on an all-NaN column."""
    db_columns = {c.name for c in ClusterFeatures.__table__.columns}
    missing = [c for c in FEATURE_COLUMNS if c not in db_columns]
    assert not missing, f"features with no cluster_features column: {missing}"


def test_non_feature_columns_exist_in_the_table_but_not_the_matrix():
    db_columns = {c.name for c in ClusterFeatures.__table__.columns}
    for column in NON_FEATURE_COLUMNS:
        assert column in db_columns
        assert column not in FEATURE_COLUMNS


def test_m2_only_claims_ownership_of_columns_that_exist():
    from classifier.features.assemble import M2_OWNED_COLUMNS

    db_columns = {c.name for c in ClusterFeatures.__table__.columns}
    unknown = [c for c in M2_OWNED_COLUMNS if c not in db_columns]
    assert not unknown, f"M2 claims non-existent columns: {unknown}"


def test_m2_does_not_claim_columns_owned_by_other_members():
    """The assembler must never overwrite M3/M4/M5's concurrent writes."""
    from classifier.features.assemble import M2_OWNED_COLUMNS

    other_members_columns = {
        "dozier_temp", "ndvi", "ndbi", "smoke_red_blue_ratio",     # M4
        "shift_sharpness", "weekend_suppression", "kalman_time_to_critical",  # M5
        "threat_corridor_present",                                  # M3
    }
    trespassing = other_members_columns & set(M2_OWNED_COLUMNS)
    assert not trespassing, f"M2 would clobber: {trespassing}"


def test_task_name_matches_what_m1_actually_dispatches():
    """The single most important integration contract: M1's
    dispatch_downstream_jobs sends this exact string, and if M2 registers
    a different one the message sits on the queue forever with no error
    anywhere."""
    import inspect

    import app.orchestration.tasks as m1_tasks

    source = inspect.getsource(m1_tasks.dispatch_downstream_jobs)
    assert "tasks.m2_build_feature_vector" in source

    import classifier.tasks as m2_tasks

    assert m2_tasks.build_feature_vector.name == "tasks.m2_build_feature_vector"


def test_m2_classify_task_is_registered_under_its_chained_name():
    import classifier.tasks as m2_tasks

    assert m2_tasks.classify_cluster.name == "tasks.m2_classify"


def test_both_members_tasks_share_one_celery_app():
    """Two Celery apps pointed at different brokers is a silent
    integration failure; assert they are literally the same object."""
    from app.orchestration.queue import celery_app as m1_app

    import classifier.tasks as m2_tasks

    assert m2_tasks.celery_app is m1_app


def test_classification_router_mounts_onto_m1s_fastapi_app():
    from fastapi import FastAPI

    from classifier.api.routes_classify import router

    app_under_test = FastAPI()
    app_under_test.include_router(router)
    paths = set(app_under_test.openapi()["paths"])

    assert "/classify/{cluster_id}" in paths
    assert "/classify" in paths


def test_classification_table_has_the_columns_the_api_reads():
    api_fields = {
        "predicted_class", "class_probabilities", "xgb_probabilities",
        "image_probabilities", "fusion_weight_xgb", "confidence_score",
        "shap_values", "explanation_method", "reasons",
        "evidence_weight_notes", "model_version",
    }
    db_columns = {c.name for c in Classification.__table__.columns}
    assert api_fields <= db_columns


def test_training_label_table_supports_the_ambiguity_workflow():
    columns = {c.name for c in TrainingLabel.__table__.columns}
    assert {"is_ambiguous", "verified_by_analyst", "label_source", "reason"} <= columns
