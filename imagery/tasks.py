"""M4's Celery task, registered on M1's Celery app.

`tasks.m4_imagery_features` is the name M1 already dispatches
(`app/orchestration/tasks.py`). One important detail about that dispatch:
M1 currently only fires it when the cloud gate says optical data is
usable.

That is right for the Sentinel-2 half of M4's work and wrong for the
thermal half. The Dozier retrieval uses VIIRS I4/I5 brightness
temperatures straight from FIRMS — it needs no optical imagery at all, so
gating it on cloud cover throws away sub-pixel fire temperature for
precisely the clusters that already have the least evidence. In monsoon
season that is most of India.

So this task takes `optical_available` as a parameter and branches
internally:

    thermal half   -> always runs
    optical half   -> runs only when the gate allows

The parameter defaults to True so the task still works against the
current M1, which passes only three arguments. The one-line change
proposed to M1 is to move its dispatch out of the `if optical_available:`
block and pass the flag through:

    celery_app.send_task("tasks.m4_imagery_features",
                         args=[cluster_id, lon, lat, optical_available])

Run the worker with all four packages importable:
    celery -A app.orchestration.queue.celery_app worker \
           -I classifier.tasks,geospatial.tasks,imagery.tasks
"""
import logging
import math
from typing import Any, Dict, List, Optional

from app.orchestration.queue import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.m4_imagery_features")
def imagery_features(
    cluster_id: int, lon: float, lat: float, optical_available: bool = True
) -> Dict[str, Any]:
    """Dozier retrieval (always) plus Sentinel-2 features (cloud permitting)."""
    from app.db.session import session_scope

    from imagery.features_io import (
        load_dual_band_rows,
        read_modelled_plume_bearing,
        read_spatial_extent_km,
    )

    # --- read phase (short transaction, no network) ---
    with session_scope() as session:
        rows = load_dual_band_rows(session, cluster_id)
        spatial_extent_km = read_spatial_extent_km(session, cluster_id)
        modelled_bearing = read_modelled_plume_bearing(session, cluster_id)
        last_seen = max((r["acq_datetime"] for r in rows if r.get("acq_datetime")), default=None)

    if not rows:
        logger.warning("Cluster %s has no detections — nothing to analyse", cluster_id)
        return {"cluster_id": cluster_id, "status": "no_detections"}

    # --- thermal half: no imagery needed, so never gated on cloud ---
    thermal = _run_dozier(rows)

    # --- optical half: network I/O, deliberately outside any transaction ---
    optical: Dict[str, Any] = {"status": "skipped", "reason": "cloud gate closed"}
    patch = None
    image_result = None
    smoke = None
    agreement = None

    if optical_available:
        # Best-effort, and deliberately wrapped: the thermal retrieval
        # above is already computed but not yet written, and an exception
        # anywhere in the optical half (a band contract mismatch, a disk
        # error writing the patch, a malformed scene) would otherwise
        # discard it. Dozier is the more reliable of the two products and
        # must not be lost to an imagery failure.
        try:
            patch, optical, image_result, smoke, agreement = _run_optical(
                cluster_id, lon, lat, last_seen, spatial_extent_km, modelled_bearing
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Optical analysis failed for cluster %s", cluster_id)
            patch = image_result = smoke = agreement = None
            optical = {"status": "failed", "reason": str(exc)}

    # --- write phase (short transaction) ---
    with session_scope() as session:
        _store_thermal(session, cluster_id, thermal, len(rows))
        patch_id = _store_patch(session, cluster_id, patch) if patch else None
        _store_image_prediction(session, cluster_id, patch_id, image_result, smoke, agreement)
        _update_cluster_features(session, cluster_id, thermal, optical)

    # Hand the image verdict to M2 for fusion. Only when the model is
    # confident enough to be worth 30% of the answer.
    fused = False
    if image_result is not None and image_result.usable_for_fusion:
        celery_app.send_task(
            "tasks.m2_classify",
            args=[cluster_id],
            kwargs={"image_probabilities": image_result.class_probabilities},
        )
        fused = True

    logger.info(
        "Imagery for cluster %s: dozier=%s optical=%s fused=%s",
        cluster_id,
        f"{thermal.fire_temperature_k:.0f}K" if thermal.ok else f"n/a ({thermal.reason[:40]})",
        optical.get("status"), fused,
    )
    return {
        "cluster_id": cluster_id,
        "status": "analysed",
        "dozier_temperature_k": thermal.fire_temperature_k,
        "optical": optical.get("status"),
        "sent_for_fusion": fused,
    }


# --------------------------------------------------------------------------
# thermal
# --------------------------------------------------------------------------

def _run_dozier(rows: List[Dict[str, Any]]):
    """Retrieve sub-pixel fire temperature from the cluster's hottest detection."""
    from imagery.features_io import dual_band_rows_only
    from imagery.thermal.background import estimate_background, is_daytime_from_rows
    from imagery.thermal.dozier import DozierResult, pixel_area_m2, solve_with_uncertainty

    usable = dual_band_rows_only(rows)
    if not usable:
        return DozierResult(
            fire_temperature_k=None, fire_fraction=None, fire_area_m2=None,
            background_temperature_k=float("nan"), converged=False,
            reason="no dual-band VIIRS/MODIS detections in this cluster",
        )

    background = estimate_background(
        [r["brightness_tir"] for r in usable], is_daytime_from_rows(usable)
    )

    # The hottest mid-infrared detection carries the strongest fire
    # signal and gives the best-conditioned retrieval; marginal edge
    # pixels are what the background estimate is built from instead.
    hottest = max(usable, key=lambda r: r["brightness_mir"])

    result = solve_with_uncertainty(
        t4_observed_k=hottest["brightness_mir"],
        t5_observed_k=hottest["brightness_tir"],
        background_t4_k=background.temperature_k,
        sensor=hottest.get("source", "VIIRS"),
        pixel_area_m2_value=pixel_area_m2(hottest.get("scan"), hottest.get("track")),
    )
    result.background_temperature_k = background.temperature_k
    result.background_source = background.source
    return result


# --------------------------------------------------------------------------
# optical
# --------------------------------------------------------------------------

def _run_optical(cluster_id, lon, lat, last_seen, spatial_extent_km, modelled_bearing):
    """Fetch a patch, derive indices, classify it, and check the smoke direction."""
    from imagery.model.predict import classify_patch
    from imagery.optical.indices import smoke_mask, summarise_patch
    from imagery.optical.sentinel2 import fetch_patch, save_patch, save_thumbnail
    from imagery.optical.smoke import cross_validate_with_model, smoke_bearing

    patch = fetch_patch(lon, lat, last_seen, spatial_extent_km)
    if not patch.available:
        return None, {"status": "unavailable", "reason": patch.reason}, None, None, None

    patch.file_path = save_patch(patch, cluster_id)
    patch.thumbnail_path = save_thumbnail(patch, cluster_id)

    features = summarise_patch(patch.bands)

    smoke = None
    agreement = None
    required = ("B2", "B3", "B4", "B8", "B11")
    if all(band in patch.bands for band in required):
        mask = smoke_mask(
            patch.bands["B2"], patch.bands["B3"], patch.bands["B4"],
            patch.bands["B8"], patch.bands["B11"],
        )
        smoke = smoke_bearing(mask.mask)
        agreement = cross_validate_with_model(smoke.bearing_deg, modelled_bearing)

    image_result = classify_patch(patch.bands)

    return (
        patch,
        {"status": "analysed", "features": features, "cloud_percentage": patch.cloud_percentage},
        image_result,
        smoke,
        agreement,
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def _store_thermal(session, cluster_id: int, thermal, n_rows: int) -> None:
    from imagery.db.models import ThermalRetrieval

    payload = {
        "fire_temperature_k": thermal.fire_temperature_k,
        "fire_fraction": thermal.fire_fraction,
        "fire_area_m2": thermal.fire_area_m2,
        # NaN background means the retrieval never ran (no dual-band rows);
        # store NULL rather than a NaN Postgres float.
        "background_temperature_k": (
            None if math.isnan(thermal.background_temperature_k)
            else thermal.background_temperature_k
        ),
        "background_source": getattr(thermal, "background_source", None),
        "sensor": thermal.sensor,
        "converged": thermal.converged,
        "reason": thermal.reason,
        "n_detections_used": n_rows,
        "temperature_low_k": thermal.temperature_low_k,
        "temperature_high_k": thermal.temperature_high_k,
        "well_constrained": thermal.well_constrained,
    }

    existing = session.get(ThermalRetrieval, cluster_id)
    if existing is None:
        session.add(ThermalRetrieval(cluster_id=cluster_id, **payload))
    else:
        for key, value in payload.items():
            setattr(existing, key, value)


def _store_patch(session, cluster_id: int, patch) -> Optional[int]:
    from geoalchemy2.shape import from_shape
    from shapely.geometry import box

    from imagery.db.models import Sentinel2Patch

    row = Sentinel2Patch(
        cluster_id=cluster_id,
        file_path=getattr(patch, "file_path", None),
        thumbnail_path=getattr(patch, "thumbnail_path", None),
        acquired_at=patch.acquired_at,
        bbox=from_shape(box(*patch.bbox), srid=4326) if patch.bbox else None,
        cloud_percentage=patch.cloud_percentage,
        bands=patch.band_order,
        width_px=patch.width_px,
        height_px=patch.height_px,
        source=patch.source,
    )
    session.add(row)
    session.flush()          # populate row.id for the FK below
    return row.id


def _store_image_prediction(session, cluster_id, patch_id, image_result, smoke, agreement) -> None:
    from imagery.db.models import ImagePrediction

    if image_result is None and smoke is None:
        return

    payload = {
        "patch_id": patch_id,
        "class_probabilities": image_result.class_probabilities if image_result else None,
        "predicted_class": image_result.predicted_class if image_result else None,
        "confidence": image_result.confidence if image_result else None,
        "model_version": image_result.model_version if image_result else None,
        "smoke_detected": bool(smoke.detected) if smoke else False,
        "smoke_bearing_deg": smoke.bearing_deg if smoke else None,
        "smoke_coverage": smoke.coverage if smoke else None,
        "plume_agreement": agreement,
    }

    existing = session.get(ImagePrediction, cluster_id)
    if existing is None:
        session.add(ImagePrediction(cluster_id=cluster_id, **payload))
    else:
        for key, value in payload.items():
            setattr(existing, key, value)


def _update_cluster_features(session, cluster_id: int, thermal, optical) -> None:
    """Write ONLY the four columns M2 reserved for M4.

    M2's tests assert that no member overwrites another's columns, and
    M3/M5 write to this same row concurrently, so this is column-scoped
    rather than a whole-row update.
    """
    try:
        from classifier.db.models import ClusterFeatures
    except ImportError:
        return

    row = session.get(ClusterFeatures, cluster_id)
    if row is None:
        return

    if thermal.ok:
        row.dozier_temp = thermal.fire_temperature_k

    features = (optical or {}).get("features") or {}
    for column in ("ndvi", "ndbi", "smoke_red_blue_ratio"):
        value = features.get(column)
        if value is not None:
            setattr(row, column, value)
