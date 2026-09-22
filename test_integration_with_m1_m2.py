"""Contract tests against Members 1 and 2.

These catch the integration drift that neither member's own unit tests
would see: a task name that nobody dispatches, a column M2's ownership
rules forbid M3 from touching, a second Celery app pointed elsewhere.
"""
import inspect

import pytest

pytest.importorskip("app.db.models", reason="app package must be importable")
pytest.importorskip("classifier.db.models", reason="classifier package must be importable")

from geospatial.db.models import (  # noqa: E402
    EvacuationRoute,
    FacilityAttribution,
    Plume,
    ThreatCorridor,
)


# --- shared schema ------------------------------------------------------

def test_all_three_members_share_one_metadata():
    """One database, one MetaData, one Alembic history. A second Base
    would make autogenerate try to drop the other members' tables."""
    from app.db.models import Base

    tables = set(Base.metadata.tables)
    assert {"hotspots", "clusters", "facilities", "fingerprints"} <= tables      # M1
    assert {"cluster_features", "classifications", "training_labels"} <= tables  # M2
    assert {"facility_attributions", "plumes", "threat_corridors",
            "evacuation_routes"} <= tables                                        # M3


def test_m3_foreign_keys_resolve_to_m1_tables():
    attribution_fks = {
        list(fk.columns)[0].name: fk.elements[0].target_fullname
        for fk in FacilityAttribution.__table__.foreign_key_constraints
    }
    assert attribution_fks["cluster_id"] == "clusters.id"
    assert attribution_fks["facility_id"] == "facilities.id"

    for model in (Plume, ThreatCorridor, EvacuationRoute):
        fks = {
            list(fk.columns)[0].name: fk.elements[0].target_fullname
            for fk in model.__table__.foreign_key_constraints
        }
        assert fks["cluster_id"] == "clusters.id"


def test_geometry_columns_use_the_project_srid():
    """Mixing SRIDs makes ST_Intersects silently return nothing."""
    assert Plume.__table__.c.geom.type.srid == 4326
    assert ThreatCorridor.__table__.c.geom.type.srid == 4326
    assert EvacuationRoute.__table__.c.geom.type.srid == 4326


# --- task contracts -----------------------------------------------------

def test_phase_a_task_name_matches_what_m1_dispatches():
    """M1 sends this exact string. A mismatch means the message sits on
    the queue forever with no error anywhere."""
    import app.orchestration.tasks as m1_tasks

    source = inspect.getsource(m1_tasks.dispatch_downstream_jobs)
    assert "tasks.m3_spatial_attribution" in source

    import geospatial.tasks as m3_tasks

    assert m3_tasks.spatial_attribution.name == "tasks.m3_spatial_attribution"


def test_phase_b_task_name_matches_what_m2_dispatches():
    """Plume/corridor modelling runs only after classification, so M2
    triggers it — this is the contract added to M2's classify task."""
    import classifier.tasks as m2_tasks

    source = inspect.getsource(m2_tasks.classify_cluster)
    assert "tasks.m3_threat_and_plume" in source

    import geospatial.tasks as m3_tasks

    assert m3_tasks.threat_and_plume.name == "tasks.m3_threat_and_plume"


def test_phase_a_retriggers_m2_feature_build():
    """M3 improves the attribution M2 snapshots, so M2 must rebuild or it
    trains on the coarser values M1 wrote."""
    import geospatial.tasks as m3_tasks

    source = inspect.getsource(m3_tasks.spatial_attribution)
    assert "tasks.m2_build_feature_vector" in source


def test_all_three_members_share_one_celery_app():
    from app.orchestration.queue import celery_app as m1_app

    import classifier.tasks as m2_tasks
    import geospatial.tasks as m3_tasks

    assert m3_tasks.celery_app is m1_app
    assert m2_tasks.celery_app is m1_app


# --- column ownership ---------------------------------------------------

def test_m3_writes_only_its_own_cluster_features_column():
    """M2's tests assert nobody clobbers another member's columns. M3's
    single scalar there is threat_corridor_present."""
    import geospatial.tasks as m3_tasks

    source = inspect.getsource(m3_tasks._set_threat_flag)
    assert "threat_corridor_present" in source

    forbidden = [
        "spatial_extent_km", "spatial_growth_rate", "frp_mean", "dozier_temp",
        "ndvi", "shift_sharpness", "kalman_time_to_critical",
    ]
    for column in forbidden:
        assert f"row.{column} =" not in source
        assert f"setattr(row, \"{column}\"" not in source


def test_threat_corridor_present_is_not_a_model_feature():
    """It is computed downstream of classification, so feeding it back
    would be target leakage — M2 excluded it deliberately, and M3 must
    not quietly reintroduce it."""
    from classifier.features.schema import FEATURE_COLUMNS, NON_FEATURE_COLUMNS

    assert "threat_corridor_present" in NON_FEATURE_COLUMNS
    assert "threat_corridor_present" not in FEATURE_COLUMNS


def test_m3_reads_m2_features_defensively():
    """M3 must keep working if M2 isn't installed or hasn't run yet."""
    import geospatial.tasks as m3_tasks

    for function in (m3_tasks._read_growth_rate, m3_tasks._read_spatial_extent):
        source = inspect.getsource(function)
        assert "ImportError" in source, f"{function.__name__} must tolerate M2 being absent"


# --- API ----------------------------------------------------------------

def test_geo_router_mounts_onto_a_fastapi_app():
    from fastapi import FastAPI

    from geospatial.api.routes_geo import router

    app_under_test = FastAPI()
    app_under_test.include_router(router)
    paths = set(app_under_test.openapi()["paths"])

    assert "/plume/{cluster_id}" in paths
    assert "/threat/{cluster_id}" in paths
    assert "/attribution/{cluster_id}" in paths


def test_geo_routes_do_not_collide_with_m1_or_m2_routes():
    """All three routers mount on one FastAPI app; a duplicated path
    would shadow whichever registered first."""
    from fastapi import FastAPI

    from classifier.api.routes_classify import router as classify_router
    from geospatial.api.routes_geo import router as geo_router

    combined = FastAPI()
    combined.include_router(classify_router)
    combined.include_router(geo_router)

    from app.api import routes_clusters, routes_facilities, routes_hotspots

    for router in (routes_hotspots.router, routes_clusters.router, routes_facilities.router):
        combined.include_router(router)

    seen = set()
    for path, methods in combined.openapi()["paths"].items():
        for method in methods:
            key = (method, path)
            assert key not in seen, f"duplicate route {method.upper()} {path}"
            seen.add(key)


def test_plume_facility_types_cover_what_m1_actually_writes():
    """M1's OSM loader maps tags to these facility_type values; any it
    emits that M3 doesn't recognise falls back to a generic profile."""
    from app.enrichment.osm_facilities import _FACILITY_TYPE_BY_TAG

    from geospatial.plume.chemicals import chemical_profile

    for facility_type in set(_FACILITY_TYPE_BY_TAG.values()):
        profile = chemical_profile(facility_type)
        assert profile["basis"] == f"facility_type={facility_type}", (
            f"M1 emits facility_type={facility_type!r} but M3 has no profile for it"
        )


def test_corridor_classes_are_real_classifier_classes():
    """A typo in SPREADING_CLASSES would silently disable corridors."""
    from classifier.features.schema import CLASSES

    from geospatial.threat.corridor import SPREADING_CLASSES

    for cls in SPREADING_CLASSES:
        assert cls in CLASSES, f"{cls!r} is not one of the classifier's classes"
