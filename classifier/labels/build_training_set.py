"""Build the labelled training set by running the weak-supervision rules
over every cluster that has features.

Output goes to `training_labels`. Clusters where rules contradict each
other are written with is_ambiguous=True and excluded from training —
they're the queue for analyst review (M1's incident review service),
which is also how the MLOps loop gets genuinely-verified labels later.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def load_labelable_clusters(session: Session) -> List[Dict[str, Any]]:
    """Every cluster that has an assembled feature row, with its centroid."""
    from geoalchemy2.shape import to_shape

    from app.db.models import Cluster

    from classifier.db.models import ClusterFeatures

    rows = session.execute(
        select(Cluster, ClusterFeatures).join(
            ClusterFeatures, ClusterFeatures.cluster_id == Cluster.id
        )
    ).all()

    result = []
    for cluster, features in rows:
        point = to_shape(cluster.centroid)
        feature_dict = {
            column.name: getattr(features, column.name)
            for column in ClusterFeatures.__table__.columns
        }
        result.append({
            "cluster_id": cluster.id,
            "lon": point.x,
            "lat": point.y,
            "last_seen": cluster.last_seen,
            "features": feature_dict,
            "facility_type": feature_dict.get("facility_type"),
        })
    return result


def label_clusters(
    clusters: List[Dict[str, Any]], fsi_alerts: Optional[List[Dict]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Apply the rules. Returns (resolved_labels, ambiguous_records)."""
    from classifier.labels.fsi_labels import matches_fsi_alert
    from classifier.labels.rules import collect_votes, resolve_label

    resolved: List[Dict] = []
    ambiguous: List[Dict] = []

    for cluster in clusters:
        fsi_matched = False
        if fsi_alerts:
            fsi_matched = matches_fsi_alert(
                cluster["lon"], cluster["lat"], cluster.get("last_seen"), fsi_alerts
            )

        votes = collect_votes(
            cluster["lon"],
            cluster["lat"],
            cluster["features"],
            acq_datetime=cluster.get("last_seen"),
            nearest_facility_type=cluster.get("facility_type"),
            fsi_matched=fsi_matched,
        )
        winner = resolve_label(votes)

        if winner is None:
            if votes:
                ambiguous.append({
                    "cluster_id": cluster["cluster_id"],
                    "votes": [(v.label, v.confidence, v.source) for v in votes],
                })
            continue

        resolved.append({
            "cluster_id": cluster["cluster_id"],
            "label": winner.label,
            "label_source": winner.source,
            "label_confidence": winner.confidence,
            "reason": winner.reason,
        })

    return resolved, ambiguous


def persist_labels(session: Session, resolved: List[Dict], ambiguous: List[Dict]) -> None:
    from classifier.db.models import TrainingLabel

    for record in resolved:
        existing = session.get(TrainingLabel, record["cluster_id"])
        if existing is not None and existing.verified_by_analyst:
            # Never overwrite a human-verified label with a rule guess.
            continue
        if existing is None:
            session.add(TrainingLabel(**record, is_ambiguous=False))
        else:
            for key, value in record.items():
                setattr(existing, key, value)
            existing.is_ambiguous = False

    for record in ambiguous:
        existing = session.get(TrainingLabel, record["cluster_id"])
        if existing is not None and existing.verified_by_analyst:
            continue
        reason = "Contradictory rules: " + ", ".join(
            f"{label}({conf:.2f} via {src})" for label, conf, src in record["votes"]
        )
        if existing is None:
            session.add(
                TrainingLabel(
                    cluster_id=record["cluster_id"],
                    label="unresolved",
                    label_source="AMBIGUOUS",
                    reason=reason,
                    is_ambiguous=True,
                )
            )
        else:
            existing.is_ambiguous = True
            existing.reason = reason


def load_training_matrix(session: Session) -> Tuple[List[Dict], List[str], List[Tuple[float, float]], List[int]]:
    """Read back the labelled set, ready for `train.build_training_matrix`.

    Excludes ambiguous rows and the 'unresolved' placeholder label.
    """
    from geoalchemy2.shape import to_shape

    from app.db.models import Cluster

    from classifier.db.models import ClusterFeatures, TrainingLabel

    rows = session.execute(
        select(ClusterFeatures, TrainingLabel, Cluster)
        .join(TrainingLabel, TrainingLabel.cluster_id == ClusterFeatures.cluster_id)
        .join(Cluster, Cluster.id == ClusterFeatures.cluster_id)
        .where(TrainingLabel.is_ambiguous.is_(False), TrainingLabel.label != "unresolved")
    ).all()

    feature_rows, labels, coordinates, cluster_ids = [], [], [], []
    for features, label, cluster in rows:
        point = to_shape(cluster.centroid)
        feature_rows.append({
            column.name: getattr(features, column.name)
            for column in ClusterFeatures.__table__.columns
        })
        labels.append(label.label)
        coordinates.append((point.x, point.y))
        cluster_ids.append(cluster.id)

    return feature_rows, labels, coordinates, cluster_ids
