"""Run the whole per-cluster fan-out in-process, without Celery or Redis.

`app/orchestration/tasks.py` fans work out with `.delay()` /
`send_task()`, which needs a running Redis broker and at least one worker
process. That is the right shape for a deployment, but it makes the
platform undemonstrable on a machine with no broker: every dispatch
raises, so clusters land in the database and nothing ever classifies
them.

This module drives the identical task bodies directly, in the order the
queue would have run them, in one process. It is not a reimplementation
— it imports and calls the same functions, so there is exactly one copy
of the logic and no risk of the two paths drifting apart.

The ordering below is the real dependency chain, not an arbitrary one:

    enrich_cluster        population + nearest facility -> fingerprints
    spatial_attribution   facility attribution + WorldCover land cover,
                          overwriting M1's coarser fingerprint values
    build_feature_vector  snapshot context + thermal into the 28-column
                          `cluster_features` row, which must exist before
                          M4 and M5 can fill in their own columns
    imagery_features      Dozier sub-pixel temperature from the VIIRS
                          I4/I5 brightness already in `hotspots`
    rhythm_and_kalman     PTSI, rhythm fingerprint, Kalman forecast
    classify_cluster      XGBoost + SHAP -> `classifications`
    threat_and_plume      Gaussian plume + threat corridor, which needs
                          the predicted class to pick a chemical profile
    wui_threat             Wildland-urban-interface ember-jump proximity
                          check, which needs the same spread-rate physics
                          threat_and_plume already computed

Feature assembly deliberately runs *after* the four enrichment steps
rather than concurrently with them, which is the one behavioural
difference from the queue: under Celery the tasks race and M2 rebuilds
whenever a later one lands. Running them in dependency order instead
means the feature vector is built once, complete, so a cluster is never
classified on a half-populated row.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-cluster work is dominated by network waits (WorldCover COG reads,
# Open-Meteo), so threads help a lot; but every step also talks to
# Postgres, so this must stay well under the SQLAlchemy pool ceiling.
DEFAULT_WORKERS = 6


@dataclass
class CycleReport:
    """What one run actually did — returned to the API and logged."""

    ingested_raw: int = 0
    ingested_deduped: int = 0
    clusters_touched: int = 0
    processed: int = 0
    classified: int = 0
    failed: int = 0
    wui_alerts: int = 0
    crown_fires: int = 0
    rdi_alerts: int = 0
    stage_failures: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ingested_raw": self.ingested_raw,
            "ingested_deduped": self.ingested_deduped,
            "clusters_touched": self.clusters_touched,
            "processed": self.processed,
            "classified": self.classified,
            "failed": self.failed,
            "wui_alerts": self.wui_alerts,
            "crown_fires": self.crown_fires,
            "rdi_alerts": self.rdi_alerts,
            "stage_failures": self.stage_failures,
            # Bounded: a systematic failure would otherwise produce one
            # error string per cluster and a response megabytes wide.
            "errors": self.errors[:10],
        }


@contextmanager
def _dispatch_disabled():
    """Neutralise Celery dispatch for the duration of the block.

    The task bodies call `celery_app.send_task(...)` to chain onward —
    e.g. `spatial_attribution` asks M2 to rebuild its feature vector.
    With no broker those calls raise (or worse, block retrying a TCP
    connect). This driver already performs every one of those follow-ups
    explicitly and in order, so the chaining calls are redundant here,
    not merely inconvenient.

    Patching the one `send_task` method is deliberately narrower than
    stubbing the tasks themselves: the real bodies still run untouched,
    and anything that is *not* a chain-dispatch still fails loudly.
    """
    from app.orchestration.queue import celery_app

    original = celery_app.send_task
    dispatched: List[str] = []

    def _record(name, *args, **kwargs):
        dispatched.append(name)
        logger.debug("Suppressed Celery dispatch of %s (driven in-process instead)", name)
        return None

    celery_app.send_task = _record
    try:
        yield dispatched
    finally:
        celery_app.send_task = original


def process_cluster(cluster_id: int, lon: float, lat: float, optical_available: bool) -> Dict[str, Any]:
    """Run the full per-cluster chain for one cluster, in dependency order.

    Each stage is individually guarded. A cluster whose land cover read
    times out must still get classified on the evidence that did arrive —
    that is precisely what the evidence-weighting engine exists for — so
    a stage failure degrades the result rather than abandoning the
    cluster.
    """
    from app.orchestration.tasks import enrich_cluster
    from classifier.tasks import build_feature_vector, classify_cluster
    from geospatial.tasks import (
        crown_fire,
        resource_demand,
        spatial_attribution,
        threat_and_plume,
        wui_threat,
    )
    from imagery.tasks import imagery_features
    from temporal.tasks import rhythm_and_kalman

    outcome: Dict[str, Any] = {"cluster_id": cluster_id, "stages": {}}

    def _stage(name: str, task, *args):
        """Invoke one task body, recording ok/failure rather than raising.

        `task.run(...)` rather than `task(...)`: calling a Celery task
        object executes `Task.__call__`, which pushes onto a *thread-local*
        request stack that only exists inside a real worker. From this
        driver's thread pool that stack is None and every call dies with
        "'NoneType' object has no attribute 'push'" before reaching any
        project code. `.run` is the undecorated function itself, which is
        exactly what we want here.
        """
        try:
            task.run(*args)
            outcome["stages"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            outcome["stages"][name] = f"failed: {type(exc).__name__}: {exc}"
            logger.warning("Cluster %s: stage %s failed", cluster_id, name, exc_info=True)

    _stage("enrich", enrich_cluster, cluster_id, lon, lat)
    _stage("attribution", spatial_attribution, cluster_id, lon, lat)

    # Must precede M4/M5: both write their own columns into an EXISTING
    # `cluster_features` row and return early if there isn't one
    # (`imagery.tasks._update_cluster_features`, `temporal.tasks`
    # likewise). Under Celery the tasks race and M2 is dispatched first,
    # so the row is normally already there; running them in strict order
    # without this made M4's Dozier temperature and M5's rhythm columns
    # silently vanish — computed, persisted to their own tables, then
    # dropped on the floor on the way into the feature vector.
    _stage("features", build_feature_vector, cluster_id)

    _stage("imagery", imagery_features, cluster_id, lon, lat, optical_available)
    _stage("temporal", rhythm_and_kalman, cluster_id)

    # Classification is the one stage whose result the caller needs back,
    # so it is run outside _stage to capture the returned verdict.
    predicted_class = None
    try:
        result = classify_cluster.run(cluster_id)
        outcome["stages"]["classify"] = result.get("status", "unknown")
        predicted_class = result.get("predicted_class")
        outcome["predicted_class"] = predicted_class
        outcome["confidence_score"] = result.get("confidence_score")
    except Exception as exc:  # noqa: BLE001
        outcome["stages"]["classify"] = f"failed: {type(exc).__name__}: {exc}"
        logger.warning("Cluster %s: classification failed", cluster_id, exc_info=True)

    # Crown-fire FRP threshold detection: independent of WUI and of
    # which class was predicted (an FRP spike is diagnostic for any
    # class), so it is its own guard rather than living inside the
    # predicted-class-dependent block below. Under Celery this is
    # dispatched unconditionally from classify_cluster itself; called
    # explicitly here for the same local-mode reason wui_threat is.
    if predicted_class:
        try:
            crown_result = crown_fire.run(cluster_id)
            outcome["stages"]["crown_fire"] = crown_result.get("status", "unknown")
            outcome["is_crown_fire"] = bool(crown_result.get("is_crown_fire"))
        except Exception as exc:  # noqa: BLE001
            outcome["stages"]["crown_fire"] = f"failed: {type(exc).__name__}: {exc}"
            logger.warning("Cluster %s: crown-fire evaluation failed", cluster_id, exc_info=True)

    # Plume/corridor modelling is downstream of the verdict by design —
    # the chemical profile and whether a corridor is drawn at all both
    # depend on the predicted class.
    if predicted_class:
        _stage("plume", threat_and_plume, cluster_id, predicted_class)

        # Run explicitly rather than through _stage: the caller (report
        # aggregation below) needs the actual wui_threat verdict, not
        # just whether the stage completed without raising. Under Celery
        # this task is ALSO reached by threat_and_plume's own
        # send_task("tasks.m3_wui_threat", ...) call — but that call is
        # silently swallowed for the duration of this function by
        # _dispatch_disabled() (see process_clusters below), so calling
        # it here explicitly is what actually makes it run in local mode.
        try:
            wui_result = wui_threat.run(cluster_id, predicted_class)
            outcome["stages"]["wui"] = wui_result.get("status", "unknown")
            outcome["wui_threat"] = bool(wui_result.get("wui_threat"))
        except Exception as exc:  # noqa: BLE001
            outcome["stages"]["wui"] = f"failed: {type(exc).__name__}: {exc}"
            logger.warning("Cluster %s: WUI evaluation failed", cluster_id, exc_info=True)

        # Resource Demand Index: needs wui_threat's freshly persisted
        # wui_distance_m, so it runs after that stage, same reasoning as
        # above — under Celery this is reached via wui_threat's own
        # send_task("tasks.m3_resource_demand", ...), swallowed here by
        # _dispatch_disabled(), so it is called explicitly instead.
        try:
            rdi_result = resource_demand.run(cluster_id)
            outcome["stages"]["resource_demand"] = rdi_result.get("status", "unknown")
            outcome["rdi_score"] = rdi_result.get("rdi_score")
            outcome["coa_type"] = rdi_result.get("coa_type")
        except Exception as exc:  # noqa: BLE001
            outcome["stages"]["resource_demand"] = f"failed: {type(exc).__name__}: {exc}"
            logger.warning("Cluster %s: resource-demand scoring failed", cluster_id, exc_info=True)

    return outcome


def process_clusters(
    targets: List[tuple],
    workers: int = DEFAULT_WORKERS,
    report: Optional[CycleReport] = None,
) -> CycleReport:
    """Run `process_cluster` over (cluster_id, lon, lat, optical_available)."""
    report = report or CycleReport()

    if not targets:
        return report

    with _dispatch_disabled():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(process_cluster, cid, lon, lat, optical)
                for cid, lon, lat, optical in targets
            ]
            for future in futures:
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001
                    report.failed += 1
                    report.errors.append(f"{type(exc).__name__}: {exc}")
                    logger.warning("Cluster processing raised", exc_info=True)
                    continue

                report.processed += 1
                if outcome.get("predicted_class"):
                    report.classified += 1
                if outcome.get("wui_threat"):
                    report.wui_alerts += 1
                if outcome.get("is_crown_fire"):
                    report.crown_fires += 1
                if outcome.get("coa_type") == "FULL_SUPPRESSION_AIR_TANKERS":
                    report.rdi_alerts += 1

                # Recorded whether or not the verdict came through: a
                # cluster can classify successfully off partial evidence
                # while an enrichment stage is quietly broken, and that
                # must not read as a clean run.
                for stage, status in outcome.get("stages", {}).items():
                    if str(status).startswith("failed"):
                        report.stage_failures[stage] = report.stage_failures.get(stage, 0) + 1
                        report.errors.append(f"cluster {outcome['cluster_id']} {stage} {status}")

    return report


def unprocessed_cluster_targets(limit: Optional[int] = None) -> List[tuple]:
    """Clusters with no classification yet, newest activity first.

    Used to finish a cycle that was interrupted, and to catch clusters
    whose processing failed on an earlier run, without re-doing work that
    already succeeded.
    """
    from geoalchemy2.shape import to_shape
    from sqlalchemy import select

    from app.db.models import Cluster
    from app.db.session import session_scope
    from classifier.db.models import Classification

    with session_scope() as session:
        classified = select(Classification.cluster_id)
        stmt = (
            select(Cluster)
            .where(~Cluster.id.in_(classified))
            .order_by(Cluster.last_seen.desc().nullslast())
        )
        if limit:
            stmt = stmt.limit(limit)

        targets = []
        for cluster in session.execute(stmt).scalars().all():
            point = to_shape(cluster.centroid)
            targets.append((cluster.id, point.x, point.y, bool(cluster.optical_available)))
        return targets


def run_cycle(
    limit: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
    ingest: bool = True,
) -> Dict[str, Any]:
    """Ingest live detections, then classify every cluster that touched.

    `ingest=False` skips the satellite fetch and only works through
    clusters already in the database that have no verdict yet — used by
    the catch-up path and after an interrupted run.
    """
    report = CycleReport()

    if ingest:
        from app.orchestration.pipeline import run_ingestion_cycle

        # dispatch_jobs=False because this module performs the fan-out
        # itself; leaving it True would try to reach a broker that isn't
        # there and raise after the writes had already committed.
        summary = run_ingestion_cycle(dispatch_jobs=False)
        report.ingested_raw = summary.get("raw", 0)
        report.ingested_deduped = summary.get("deduped", 0)
        report.clusters_touched = summary.get("clusters_touched", 0)

    targets = unprocessed_cluster_targets(limit)
    logger.info("Local pipeline: %d clusters to process with %d workers", len(targets), workers)

    process_clusters(targets, workers=workers, report=report)

    logger.info(
        "Local pipeline complete: processed=%d classified=%d failed=%d",
        report.processed, report.classified, report.failed,
    )
    return report.as_dict()
