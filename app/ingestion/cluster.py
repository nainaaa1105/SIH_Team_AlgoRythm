"""DBSCAN clustering of deduplicated hotspots into physical events.

Per SIH26162_team_task_division.md / the architecture doc: eps=0.005
degrees (~500m at India's latitudes), collapsing a facility's/fire's
multiple satellite pixels into one `cluster_id` that every other
member's feature engineering keys off.

Clustering is done in plain lon/lat degrees (not haversine) because the
tolerance (0.005 deg) is small enough that the equirectangular
approximation's distortion is negligible within India's latitude range,
and it keeps this dependency-light (no extra projection library) and
easy to reason about/test.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from app.config import Settings, get_settings


@dataclass
class ClusterAssignment:
    cluster_id: Optional[int]          # existing DB cluster_id if matched, else None (new cluster)
    temp_label: int                     # DBSCAN's local label for this batch, -1 = noise (own singleton cluster)
    centroid_lon: float
    centroid_lat: float
    record_indices: List[int] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


def dbscan_labels(records: List[Dict], eps_degrees: float, min_samples: int) -> np.ndarray:
    """Run DBSCAN over a batch of canonical records, return an array of
    labels aligned to `records` (-1 = noise, i.e. its own singleton
    cluster since min_samples=1 by default makes every point clusterable).
    """
    if not records:
        return np.array([])
    coords = np.array([[r["lon"], r["lat"]] for r in records])
    model = DBSCAN(eps=eps_degrees, min_samples=min_samples, metric="euclidean")
    return model.fit_predict(coords)


def build_new_clusters(records: List[Dict], labels: np.ndarray) -> List[ClusterAssignment]:
    """Group a labelled batch into per-cluster summaries (centroid,
    first/last seen, member record indices). Noise points (-1) each
    become their own singleton cluster since a lone detection is still a
    real physical event worth tracking.
    """
    groups: Dict[int, List[int]] = {}
    next_singleton_label = (labels.max() + 1) if len(labels) and labels.max() >= 0 else 0

    resolved_labels = labels.copy()
    for i, label in enumerate(resolved_labels):
        if label == -1:
            resolved_labels[i] = next_singleton_label
            next_singleton_label += 1

    for i, label in enumerate(resolved_labels):
        groups.setdefault(int(label), []).append(i)

    assignments = []
    for label, indices in groups.items():
        lons = [records[i]["lon"] for i in indices]
        lats = [records[i]["lat"] for i in indices]
        timestamps = [records[i]["acq_datetime"] for i in indices]
        assignments.append(
            ClusterAssignment(
                cluster_id=None,
                temp_label=label,
                centroid_lon=float(np.mean(lons)),
                centroid_lat=float(np.mean(lats)),
                record_indices=indices,
                first_seen=min(timestamps),
                last_seen=max(timestamps),
            )
        )
    return assignments


def match_to_existing_clusters(
    new_clusters: List[ClusterAssignment],
    existing_clusters: List[Tuple[int, float, float]],  # (cluster_id, centroid_lon, centroid_lat)
    eps_degrees: float,
) -> List[ClusterAssignment]:
    """Match each new cluster to its nearest existing DB cluster, if one
    is within `eps_degrees`.

    Independent per-row nearest-neighbor — NOT a 1-to-1 bijective
    assignment. That used to be scipy's linear_sum_assignment (the
    Hungarian algorithm), which finds the pairing that minimises the
    TOTAL distance summed across every new/existing pair in the whole
    batch. That is the wrong objective here: a live ingestion batch
    spans thousands of clusters across all of India, so the vast
    majority of new/existing pairs are thousands of km apart and cost
    nothing to leave unmatched — but whenever two new clusters both
    legitimately sit near the same existing one (or near each other's
    true matches), the global optimum can trade away an obviously
    correct, well-within-eps match in favour of a lower aggregate cost,
    leaving the discarded new cluster's real match unassigned. That
    reads on the map as the same physical fire (verified live: two real
    clusters 8m apart, active on different days) splitting into separate
    dots across days instead of accumulating into one, because the
    bijection — not the eps threshold — is what rejected the match.
    Nothing here needs a bijection: two different new-batch groups can
    both legitimately belong to the same existing physical fire.
    """
    if not new_clusters or not existing_clusters:
        return new_clusters

    new_coords = np.array([[c.centroid_lon, c.centroid_lat] for c in new_clusters])
    existing_coords = np.array([[c[1], c[2]] for c in existing_clusters])

    # Distance matrix between new_clusters (rows) and existing_clusters (cols)
    diff = new_coords[:, np.newaxis, :] - existing_coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    nearest = np.argmin(dist_matrix, axis=1)
    for r, c in enumerate(nearest):
        if dist_matrix[r, c] <= eps_degrees:
            new_clusters[r].cluster_id = existing_clusters[c][0]

    return new_clusters


def cluster_batch(
    records: List[Dict],
    existing_clusters: Optional[List[Tuple[int, float, float]]] = None,
    settings: Optional[Settings] = None,
) -> List[ClusterAssignment]:
    """End-to-end: DBSCAN a batch of deduplicated records, then reconcile
    against already-known clusters so persistence tracking works across
    scheduler runs.
    """
    settings = settings or get_settings()
    labels = dbscan_labels(records, settings.dbscan_eps_degrees, settings.dbscan_min_samples)
    new_clusters = build_new_clusters(records, labels)
    return match_to_existing_clusters(new_clusters, existing_clusters or [], settings.dbscan_eps_degrees)
