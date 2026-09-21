"""Contract tests against Members 1, 2 and 3.

These catch integration drift that no member's own unit tests would see:
a task name nobody dispatches, a column M2's ownership rules forbid, a
second Celery app, a route path that shadows another member's.
"""
import inspect

import pytest

pytest.importorskip("app.db.models", reason="app package must be importable")
pytest.importorskip("classifier.db.models", reason="classifier package must be importable")
pytest.importorskip("geospatial.db.models", reason="geospatial package must be importable")

from imagery.db.models import ImagePrediction, Sentinel2Patch, ThermalRetrieval  # noqa: E402


# --- shared schema ------------------------------------------------------

def test_all_four_members_share_one_metadata():
    from app.db.models import Base

    tables = set(Base.metadata.tables)
    assert {"hotspots", "clusters", "facilities", "fingerprints"} <= tables        # M1
    assert {"cluster_features", "classifications", "training_labels"} <= tables    # M2
    assert {"facility_attributions", "plumes", "threat_corridors"} <= tables       # M3
    assert {"sentinel2_patches", "image_predictions", "thermal_retrievals"} <= tables  # M4


def test_m4_foreign_keys_resolve():
    for model in (Sentinel2Patch, ImagePrediction, ThermalRetrieval):
        fks = {
            list(fk.columns)[0].name: fk.elements[0].target_fullname
            for fk in model.__table__.foreign_key_constraints
        }
        assert fks["cluster_id"] == "clusters.id"

    prediction_fks = {
        list(fk.columns)[0].name: fk.elements[0].target_fullname
        for fk in ImagePrediction.__table__.foreign_key_constraints
    }
    assert prediction_fks["patch_id"] == "sentinel2_patches.id"


def test_patch_bbox_uses_the_project_srid():
    """Mixing SRIDs makes spatial predicates silently return nothing."""
    assert Sentinel2Patch.__table__.c.bbox.type.srid == 4326


# --- task contracts -----------------------------------------------------

def test_task_name_matches_what_m1_dispatches():
    import app.orchestration.tasks as m1_tasks

    source = inspect.getsource(m1_tasks.dispatch_downstream_jobs)
    assert "tasks.m4_imagery_features" in source

    import imagery.tasks as m4_tasks

    assert m4_tasks.imagery_features.name == "tasks.m4_imagery_features"


def test_task_signature_tolerates_m1s_current_three_argument_dispatch():
    """M1 currently sends [cluster_id, lon, lat]. The optical_available
    flag must therefore default, or every dispatch fails on arity."""
    import imagery.tasks as m4_tasks

    signature = inspect.signature(m4_tasks.imagery_features)
    parameter = signature.parameters["optical_available"]
    assert parameter.default is True


def test_m1_passes_the_optical_flag_instead_of_gating_the_dispatch():
    """M1 used to only dispatch M4 when the cloud gate was open, which
    silently disabled the Dozier retrieval — a purely thermal product —
    for every cloudy cluster. It now dispatches unconditionally and lets
    M4 decide which half of its work is possible."""
    import app.orchestration.tasks as m1_tasks

    source = inspect.getsource(m1_tasks.dispatch_downstream_jobs)
    dispatch_line = [ln for ln in source.splitlines() if "m4_imagery_features" in ln]
    assert dispatch_line, "M1 must still dispatch M4"

    m4_call_index = source.index("tasks.m4_imagery_features")
    preceding = source[:m4_call_index]
    # The dispatch must not sit inside the `if optical_available:` block.
    assert not preceding.rstrip().endswith("if optical_available:")
    assert "optical_available" in source[m4_call_index:m4_call_index + 220]


def test_m2_does_not_remodel_m3_when_the_verdict_is_unchanged():
    """A cluster is classified up to three times per cycle (M1's
    dispatch, M3's Phase-A re-trigger, M4's fusion callback) and M3's
    Phase B calls Open-Meteo each time. Re-modelling an unchanged verdict
    is pure waste."""
    import classifier.tasks as m2_tasks

    source = inspect.getsource(m2_tasks.classify_cluster)
    assert "previous_class" in source
    assert "if previous_class != result.predicted_class:" in source


def test_thermal_half_is_not_gated_on_optical_availability():
    """Dozier uses VIIRS I4/I5 straight from FIRMS — gating it on cloud
    cover would throw away sub-pixel temperature for exactly the clusters
    with the least other evidence."""
    import imagery.tasks as m4_tasks

    source = inspect.getsource(m4_tasks.imagery_features)
    dozier_call = source.index("_run_dozier")
    optical_gate = source.index("if optical_available")
    assert dozier_call < optical_gate, "Dozier must run before/outside the optical gate"


def test_optical_failure_cannot_discard_the_thermal_result():
    """The Dozier retrieval is computed before the optical half but
    written after it, so an unguarded optical exception would throw away
    a perfectly good temperature."""
    import imagery.tasks as m4_tasks

    source = inspect.getsource(m4_tasks.imagery_features)
    optical_call = source.index("_run_optical(")
    preceding = source[:optical_call]
    assert "try:" in preceding[-400:], "the optical half must be wrapped"
    assert "except Exception" in source[optical_call:]

    # And the write must still happen afterwards.
    assert source.index("_store_thermal") > optical_call


def test_orm_and_migration_agree_on_every_m4_column():
    """Adding a column to the ORM but forgetting the migration is a
    silent drift that only shows up against a real database."""
    import pathlib
    import re

    migration = pathlib.Path(
        "app/db/migrations/versions/0004_add_imagery_tables.py"
    )
    if not migration.exists():
        pytest.skip("migration not reachable from this working directory")

    text = migration.read_text(encoding="utf-8")
    for model in (Sentinel2Patch, ImagePrediction, ThermalRetrieval):
        table = model.__tablename__
        block = text.split(f'"{table}"', 1)[1].split("op.create_table")[0]
        migration_columns = set(re.findall(r'sa\.Column\("(\w+)"', block))
        orm_columns = {c.name for c in model.__table__.columns}
        assert orm_columns == migration_columns, (
            f"{table}: ORM-only={sorted(orm_columns - migration_columns)} "
            f"migration-only={sorted(migration_columns - orm_columns)}"
        )


def test_fusion_is_dispatched_to_m2_with_image_probabilities():
    """M2's classify accepts image_probabilities and fuses 0.70/0.30."""
    import imagery.tasks as m4_tasks

    source = inspect.getsource(m4_tasks.imagery_features)
    assert "tasks.m2_classify" in source
    assert "image_probabilities" in source


def test_m2_classify_accepts_the_kwarg_m4_sends():
    import classifier.tasks as m2_tasks

    signature = inspect.signature(m2_tasks.classify_cluster)
    assert "image_probabilities" in signature.parameters


def test_m2_fusion_handles_m4s_class_probability_dict():
    """M4 may emit a subset of classes; M2's align_to_classes fills the
    rest with zero and renormalises rather than failing."""
    from classifier.features.schema import CLASSES
    from classifier.model.fusion import fuse

    m4_output = {"wildfire": 0.8, "mining": 0.2}
    tabular = {cls: 1.0 / len(CLASSES) for cls in CLASSES}

    fused, weight = fuse(tabular, m4_output, weight_xgb=0.70)
    assert weight == pytest.approx(0.70)
    assert sum(fused.values()) == pytest.approx(1.0)
    assert set(fused) == set(CLASSES)


def test_all_four_members_share_one_celery_app():
    from app.orchestration.queue import celery_app as m1_app

    import classifier.tasks as m2_tasks
    import geospatial.tasks as m3_tasks
    import imagery.tasks as m4_tasks

    assert m4_tasks.celery_app is m1_app
    assert m3_tasks.celery_app is m1_app
    assert m2_tasks.celery_app is m1_app


# --- column ownership ---------------------------------------------------

def test_m4_writes_only_its_four_reserved_feature_columns():
    """M2 reserved dozier_temp/ndvi/ndbi/smoke_red_blue_ratio for M4 and
    asserts nobody clobbers another member's columns."""
    import imagery.tasks as m4_tasks

    source = inspect.getsource(m4_tasks._update_cluster_features)
    assert "dozier_temp" in source
    for column in ("ndvi", "ndbi", "smoke_red_blue_ratio"):
        assert column in source

    forbidden = (
        "frp_mean", "persistence_days", "spatial_extent_km", "spatial_growth_rate",
        "shift_sharpness", "kalman_time_to_critical", "threat_corridor_present",
        "pct_forest", "facility_distance_m",
    )
    for column in forbidden:
        assert f'"{column}"' not in source
        assert f"row.{column} =" not in source


def test_m4s_feature_columns_exist_on_m2s_table():
    from classifier.db.models import ClusterFeatures
    from classifier.features.schema import FEATURE_COLUMNS

    db_columns = {c.name for c in ClusterFeatures.__table__.columns}
    for column in ("dozier_temp", "ndvi", "ndbi", "smoke_red_blue_ratio"):
        assert column in db_columns
        assert column in FEATURE_COLUMNS


def test_m4_reads_m2_and_m3_defensively():
    """M4 must keep working if M2 or M3 is not installed."""
    import imagery.features_io as io

    for function in (io.read_spatial_extent_km, io.read_modelled_plume_bearing):
        assert "ImportError" in inspect.getsource(function)


# --- cross-member data flow --------------------------------------------

def test_smoke_cross_check_reads_m3s_downwind_bearing():
    """M3 stores the travel direction (already converted out of the
    meteorological from-convention), which is directly comparable to the
    bearing M4 measures from pixels."""
    from geospatial.db.models import Plume

    assert "downwind_bearing_deg" in {c.name for c in Plume.__table__.columns}

    import imagery.features_io as io

    assert "downwind_bearing_deg" in inspect.getsource(io.read_modelled_plume_bearing)


def test_dual_band_recovery_matches_m1s_storage_layout():
    """M1 keeps the whole FIRMS row in raw_payload, so the second
    brightness band is available without altering its hot table."""
    from imagery.features_io import thermal_band_from_payload

    assert thermal_band_from_payload({"bright_ti5": "295.3"}) == pytest.approx(295.3)
    assert thermal_band_from_payload({"bright_t31": 290.1}) == pytest.approx(290.1)
    assert thermal_band_from_payload({"irrelevant": 1}) is None
    assert thermal_band_from_payload(None) is None


def test_dual_band_filter_excludes_single_band_sensors():
    """INSAT-3DS and Sentinel-3 FRP carry no second brightness band."""
    from imagery.features_io import dual_band_rows_only

    rows = [
        {"source": "VIIRS_SNPP_NRT", "brightness_mir": 340.0, "brightness_tir": 300.0},
        {"source": "MODIS_NRT", "brightness_mir": 335.0, "brightness_tir": 298.0},
        {"source": "INSAT3DS", "brightness_mir": 330.0, "brightness_tir": None},
        {"source": "SENTINEL3_FRP", "brightness_mir": None, "brightness_tir": None},
    ]
    usable = dual_band_rows_only(rows)
    assert {r["source"] for r in usable} == {"VIIRS_SNPP_NRT", "MODIS_NRT"}


def test_image_model_reuses_m2s_leakage_safe_splitter():
    """Writing a second splitter risks a random split, which is the
    spatial-leakage failure the research notes document."""
    import imagery.model.train as m4_train

    source = inspect.getsource(m4_train.train)
    assert "spatial_group_split" in source
    assert "verify_no_leakage" in source


def test_image_model_trains_against_m2s_class_list():
    from classifier.features.schema import CLASSES

    import imagery.model.train as m4_train

    assert "CLASSES" in inspect.getsource(m4_train.train)
    assert len(CLASSES) == 5


# --- API ----------------------------------------------------------------

def test_imagery_router_mounts():
    from fastapi import FastAPI

    from imagery.api.routes_imagery import router

    app_under_test = FastAPI()
    app_under_test.include_router(router)
    paths = set(app_under_test.openapi()["paths"])

    assert "/imagery/{cluster_id}" in paths
    assert "/imagery/{cluster_id}/patches" in paths


def test_no_route_collides_across_all_four_members():
    from fastapi import FastAPI

    from app.api import routes_clusters, routes_facilities, routes_hotspots
    from classifier.api.routes_classify import router as classify_router
    from geospatial.api.routes_geo import router as geo_router

    from imagery.api.routes_imagery import router as imagery_router

    combined = FastAPI()
    for router in (
        routes_hotspots.router, routes_clusters.router, routes_facilities.router,
        classify_router, geo_router, imagery_router,
    ):
        combined.include_router(router)

    seen = set()
    for path, methods in combined.openapi()["paths"].items():
        for method in methods:
            key = (method, path)
            assert key not in seen, f"duplicate route {method.upper()} {path}"
            seen.add(key)
