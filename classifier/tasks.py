"""M2's Celery tasks, registered on M1's Celery app.

The task name `tasks.m2_build_feature_vector` is not arbitrary — M1's
`app/orchestration/tasks.py::dispatch_downstream_jobs` already calls
exactly that name for every cluster it touches. Importing this module in
the worker process is what makes that dispatch land somewhere instead of
sitting unconsumed on the queue.

Run the worker with both packages importable:
    celery -A app.orchestration.queue.celery_app worker \
           --loglevel=info -I classifier.tasks

Ordering note: M1 fires `tasks.enrich_cluster` (its own land-cover /
facility enrichment) and `dispatch_downstream_jobs` back-to-back, so
there's no guarantee M1's fingerprint row exists by the time this task
runs. `assemble_features` handles that by returning whatever it can and
letting the evidence-weighting engine record the gap, rather than
failing or blocking.
"""
import logging
from typing import Any, Dict, Optional

from app.orchestration.queue import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.m2_build_feature_vector")
def build_feature_vector(cluster_id: int) -> Dict[str, Any]:
    """Assemble M2's share of the 28-feature matrix for one cluster.

    Chains into classification automatically so the pipeline runs
    end-to-end from a single M1 dispatch.
    """
    from app.db.session import session_scope

    from classifier.features.assemble import (
        assemble_features,
        feature_group_availability,
        upsert_cluster_features,
    )

    with session_scope() as session:
        features = assemble_features(session, cluster_id)
        if not features:
            logger.warning("Cluster %s produced no features — skipping classification", cluster_id)
            return {"cluster_id": cluster_id, "status": "no_features"}
        upsert_cluster_features(session, cluster_id, features)
        availability = feature_group_availability(features)

    celery_app.send_task("tasks.m2_classify", args=[cluster_id])

    logger.info("Built feature vector for cluster %s (groups: %s)", cluster_id, availability)
    return {"cluster_id": cluster_id, "status": "features_built", "availability": availability}


@celery_app.task(name="tasks.m2_classify")
def classify_cluster(
    cluster_id: int, image_probabilities: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Classify a cluster and persist the verdict + explanation.

    `image_probabilities` is the contract with M4: its task can call this
    with a {class: probability} dict once a Sentinel-2 patch has been
    classified, and the verdict is recomputed with fusion applied. When
    it's absent the tabular model carries full weight.
    """
    from app.db.models import Cluster
    from app.db.session import session_scope

    from classifier.db.models import ClusterFeatures
    from classifier.model.predict import classify_features
    from classifier.model.registry import ModelRegistryError

    with session_scope() as session:
        cluster = session.get(Cluster, cluster_id)
        if cluster is None:
            logger.warning("Cluster %s not found — cannot classify", cluster_id)
            return {"cluster_id": cluster_id, "status": "cluster_not_found"}

        feature_row = session.get(ClusterFeatures, cluster_id)
        if feature_row is None:
            logger.warning("No cluster_features row for %s — run the feature task first", cluster_id)
            return {"cluster_id": cluster_id, "status": "no_features"}

        features = {
            column.name: getattr(feature_row, column.name)
            for column in ClusterFeatures.__table__.columns
        }

        # Image probabilities arrive only as an argument from M4's task —
        # they are not read back from the DB, because `classifications`
        # stores the fused result rather than a pending image verdict.
        try:
            result = classify_features(
                features,
                image_probabilities=image_probabilities,
                optical_available=cluster.optical_available,
                cloud_fraction=cluster.cloud_fraction,
            )
        except ModelRegistryError:
            logger.exception("No trained model available — cluster %s left unclassified", cluster_id)
            return {"cluster_id": cluster_id, "status": "no_model"}

        previous_class = _existing_predicted_class(session, cluster_id)
        _persist_classification(session, cluster_id, result)

    # M3's plume and threat-corridor modelling runs only once the source
    # class is known: you don't project an advance corridor for a gas
    # flare, and the expected chemical profile depends on the verdict.
    # This is the same reason `threat_corridor_present` is excluded from
    # the model's feature matrix — it is computed downstream of the
    # prediction, so feeding it back would be target leakage.
    #
    # Only re-dispatch when the verdict actually changed. A cluster is
    # classified up to three times per cycle (M1's dispatch, M3's Phase-A
    # re-trigger, and M4's fusion callback), and M3's Phase B makes an
    # Open-Meteo request every time. Re-modelling an unchanged verdict is
    # pure waste and risks rate-limiting.
    if previous_class != result.predicted_class:
        celery_app.send_task(
            "tasks.m3_threat_and_plume", args=[cluster_id, result.predicted_class]
        )

    # Crown-fire FRP threshold detection is dispatched unconditionally,
    # unlike threat_and_plume above — an FRP spike is diagnostic on its
    # own and can happen between two classify cycles even when the
    # predicted class itself hasn't changed, so gating it on a class
    # change would miss the case it exists to catch.
    celery_app.send_task("tasks.m3_crown_fire", args=[cluster_id])

    logger.info(
        "Classified cluster %s as %s (%.1f%% confidence, fusion_weight_xgb=%.2f)",
        cluster_id, result.predicted_class, result.confidence_score, result.fusion_weight_xgb,
    )
    return {
        "cluster_id": cluster_id,
        "status": "classified",
        "predicted_class": result.predicted_class,
        "confidence_score": result.confidence_score,
    }


def _existing_predicted_class(session, cluster_id: int):
    """The verdict currently on record, or None if this is the first run."""
    from classifier.db.models import Classification

    existing = session.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    return existing.predicted_class if existing else None


def _persist_classification(session, cluster_id: int, result) -> None:
    """Upsert one classification row per cluster (latest verdict wins)."""
    from classifier.db.models import Classification

    existing = session.query(Classification).filter_by(cluster_id=cluster_id).one_or_none()
    payload = {
        "predicted_class": result.predicted_class,
        "class_probabilities": result.class_probabilities,
        "xgb_probabilities": result.xgb_probabilities,
        "image_probabilities": result.image_probabilities,
        "fusion_weight_xgb": result.fusion_weight_xgb,
        "confidence_score": result.confidence_score,
        "shap_values": result.shap_values,
        "explanation_method": result.explanation_method,
        "reasons": result.reasons,
        "evidence_weight_notes": result.evidence,
        "model_version": result.model_version,
    }

    if existing is None:
        session.add(Classification(cluster_id=cluster_id, **payload))
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
