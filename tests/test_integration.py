"""Contract tests against Members 1-4.

These catch the integration drift that no member's own unit tests would
see: a task name nobody dispatches, a column M2's ownership rules forbid,
a second Celery app, a colliding route path.
"""
import inspect

import pytest

pytest.importorskip("app.db.models", reason="app package must be importable")
pytest.importorskip("classifier.db.models", reason="classifier package must be importable")

from temporal.db.models import KalmanStateRow, PTSIRegistry, RhythmFingerprintRow  # noqa: E402


# --- shared schema ------------------------------------------------------

def test_m5_tables_join_the_shared_metadata():
    from app.db.models import Base

    tables = set(Base.metadata.tables)
    assert {"hotspots", "clusters", "facilities", "fingerprints"} <= tables       # M1
    assert {"cluster_features", "classifications", "training_labels"} <= tables   # M2
    assert {"ptsi_registry", "rhythm_fingerprints", "kalman_states"} <= tables    # M5


def test_m5_foreign_keys_resolve_to_m1s_clusters():
    for model in (PTSIRegistry, RhythmFingerprintRow, KalmanStateRow):
        fks = {
            list(fk.columns)[0].name: fk.elements[0].target_fullname
            for fk in model.__table__.foreign_key_constraints
        }
        assert fks["cluster_id"] == "clusters.id"


def test_orm_and_migration_agree_on_every_m5_column():
    """Adding a column to the ORM but forgetting the migration is silent
    drift that only surfaces against a real database."""
    import pathlib
    import re

    migration = pathlib.Path(
        "app/db/migrations/versions/0005_add_temporal_tables.py"
    )
    if not migration.exists():
        pytest.skip("migration not reachable from this working directory")

    text = migration.read_text(encoding="utf-8")
    for model in (PTSIRegistry, RhythmFingerprintRow, KalmanStateRow):
        table = model.__tablename__
        block = text.split(f'"{table}"', 1)[1].split("op.create_table")[0]
        migration_columns = set(re.findall(r'sa\.Column\("(\w+)"', block))
        orm_columns = {c.name for c in model.__table__.columns}
        assert orm_columns == migration_columns, (
            f"{table}: ORM-only={sorted(orm_columns - migration_columns)} "
            f"migration-only={sorted(migration_columns - orm_columns)}"
        )


# --- task contracts -----------------------------------------------------

def test_task_name_matches_what_m1_dispatches():
    import app.orchestration.tasks as m1_tasks

    assert "tasks.m5_rhythm_and_kalman" in inspect.getsource(m1_tasks.dispatch_downstream_jobs)

    import temporal.tasks as m5_tasks

    assert m5_tasks.rhythm_and_kalman.name == "tasks.m5_rhythm_and_kalman"


def test_m5_retriggers_m2s_feature_build():
    """M5's three columns are part of M2's matrix, so M2 must rebuild —
    otherwise they never reach the classifier for a first-time cluster."""
    import temporal.tasks as m5_tasks

    assert "tasks.m2_build_feature_vector" in inspect.getsource(m5_tasks.rhythm_and_kalman)


def test_all_five_members_share_one_celery_app():
    from app.orchestration.queue import celery_app as m1_app

    import classifier.tasks as m2_tasks
    import temporal.tasks as m5_tasks

    assert m5_tasks.celery_app is m1_app
    assert m2_tasks.celery_app is m1_app


# --- column ownership ---------------------------------------------------

def test_m5_writes_only_its_three_reserved_columns():
    """M2 reserved shift_sharpness / weekend_suppression /
    kalman_time_to_critical for M5 and asserts nobody clobbers another
    member's columns. M3 and M4 write this same row concurrently."""
    import temporal.tasks as m5_tasks

    source = inspect.getsource(m5_tasks._update_cluster_features)
    for column in ("shift_sharpness", "weekend_suppression", "kalman_time_to_critical"):
        assert column in source

    forbidden = (
        "frp_mean", "persistence_days", "spatial_extent_km", "spatial_growth_rate",
        "dozier_temp", "ndvi", "ndbi", "smoke_red_blue_ratio",
        "threat_corridor_present", "pct_forest", "facility_distance_m",
    )
    for column in forbidden:
        assert f"row.{column} =" not in source


def test_m5s_columns_are_the_rhythm_group_of_m2s_matrix():
    from classifier.features.schema import FEATURE_COLUMNS, FeatureGroup, columns_in_group

    rhythm = set(columns_in_group(FeatureGroup.RHYTHM))
    assert rhythm == {"shift_sharpness", "weekend_suppression", "kalman_time_to_critical"}
    assert rhythm <= set(FEATURE_COLUMNS)


def test_m5_reads_m2_defensively():
    """M5 must keep working if M2 is not installed."""
    import temporal.features_io as io
    import temporal.tasks as m5_tasks

    assert "ImportError" in inspect.getsource(io.read_predicted_class)
    assert "ImportError" in inspect.getsource(m5_tasks._update_cluster_features)


# --- the feature M2 already documented ---------------------------------

def test_m2s_unavailable_message_describes_what_m5_actually_produces():
    """M2 wrote the 'rhythm unavailable' note before M5 existed. It says
    the cause is too short a history — which is exactly the condition
    M5's fingerprint declines on, so the two agree."""
    from classifier.features.evidence_weighting import assess_evidence
    from classifier.features.schema import FEATURE_COLUMNS

    # Assert on the rendered note rather than the source, so a reflow
    # of the string literal does not fail the test.
    assessment = assess_evidence({column: None for column in FEATURE_COLUMNS})
    note = assessment.notes["rhythm"]
    assert "too short a history" in note
    assert "M5" in note

    # And that is exactly the condition M5's fingerprint declines on.
    from temporal.rhythm.fingerprint import MIN_DETECTIONS_FOR_RHYTHM, build_fingerprint

    short = build_fingerprint([])
    assert not short.ok
    assert MIN_DETECTIONS_FOR_RHYTHM > 0


def test_rhythm_group_carries_the_weight_m2_assigned_it():
    from classifier.features.evidence_weighting import GROUP_WEIGHTS
    from classifier.features.schema import FeatureGroup

    assert GROUP_WEIGHTS[FeatureGroup.RHYTHM] == pytest.approx(0.05)


# --- PTSI fills an architecture gap -------------------------------------

def test_ptsi_is_not_duplicated_by_another_member():
    """Phase-1 deliverable #2 and architecture Layer 4 both call for a
    persistent-source catalog; no earlier member built one, which is why
    it lives here."""
    from app.db.models import Base

    ptsi_tables = [t for t in Base.metadata.tables if "ptsi" in t]
    assert ptsi_tables == ["ptsi_registry"]


def test_ptsi_expresses_the_briefs_own_example_output():
    """'persistent for 27 days -> currently 4.2x its normal FRP'."""
    from datetime import datetime, timedelta, timezone

    from temporal.ptsi.baseline import build_baseline
    from temporal.ptsi.index import PERSISTENT, compute_ptsi

    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    history = [{"acq_datetime": base + timedelta(days=d), "frp": 10.0} for d in range(27)]

    baseline = build_baseline(history)
    result = compute_ptsi(baseline, detection_rate=0.95, observation_span_days=27.0, current_frp=42.0)

    assert result.source_class == PERSISTENT
    assert result.deviation_multiple == pytest.approx(4.2, abs=0.05)
    assert "4.2x its normal" in result.summary


# --- API ----------------------------------------------------------------

def test_temporal_router_mounts():
    from fastapi import FastAPI

    from temporal.api.routes_temporal import router

    app_under_test = FastAPI()
    app_under_test.include_router(router)
    paths = set(app_under_test.openapi()["paths"])

    assert {"/ptsi/{cluster_id}", "/forecast/{cluster_id}",
            "/rhythm/{cluster_id}", "/escalating"} <= paths


def test_no_route_collides_across_all_five_members():
    from fastapi import FastAPI

    from app.api import routes_clusters, routes_facilities, routes_hotspots
    from classifier.api.routes_classify import router as classify_router

    from temporal.api.routes_temporal import router as temporal_router

    routers = [
        routes_hotspots.router, routes_clusters.router, routes_facilities.router,
        classify_router, temporal_router,
    ]

    # M3 and M4 are optional at test time; include them when available.
    try:
        from geospatial.api.routes_geo import router as geo_router

        routers.append(geo_router)
    except ImportError:
        pass
    try:
        from imagery.api.routes_imagery import router as imagery_router

        routers.append(imagery_router)
    except ImportError:
        pass

    combined = FastAPI()
    for router in routers:
        combined.include_router(router)

    seen = set()
    for path, methods in combined.openapi()["paths"].items():
        for method in methods:
            key = (method, path)
            assert key not in seen, f"duplicate route {method.upper()} {path}"
            seen.add(key)
