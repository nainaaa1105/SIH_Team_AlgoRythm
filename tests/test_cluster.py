from datetime import datetime, timezone

from app.config import Settings
from app.ingestion.cluster import build_new_clusters, cluster_batch, dbscan_labels, match_to_existing_clusters


def _record(lon, lat, minute=0):
    return {"lon": lon, "lat": lat, "acq_datetime": datetime(2026, 6, 1, 0, minute, tzinfo=timezone.utc)}


def test_dbscan_groups_nearby_points_and_separates_distant_ones():
    records = [
        _record(77.1000, 28.7000),
        _record(77.1001, 28.7001),  # ~15m away -> same cluster
        _record(80.0000, 20.0000),  # far away -> separate cluster
    ]
    labels = dbscan_labels(records, eps_degrees=0.005, min_samples=1)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_build_new_clusters_computes_centroid_as_mean_and_tracks_time_range():
    records = [
        _record(77.100, 28.700, minute=0),
        _record(77.102, 28.702, minute=5),
    ]
    labels = dbscan_labels(records, eps_degrees=0.005, min_samples=1)
    clusters = build_new_clusters(records, labels)

    assert len(clusters) == 1
    c = clusters[0]
    assert c.centroid_lon == (77.100 + 77.102) / 2
    assert c.centroid_lat == (28.700 + 28.702) / 2
    assert c.first_seen == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    assert c.last_seen == datetime(2026, 6, 1, 0, 5, tzinfo=timezone.utc)
    assert set(c.record_indices) == {0, 1}


def test_noise_points_each_become_their_own_singleton_cluster():
    records = [_record(77.1, 28.7), _record(90.0, 10.0), _record(68.0, 35.0)]
    labels = dbscan_labels(records, eps_degrees=0.005, min_samples=1)
    clusters = build_new_clusters(records, labels)
    # every point is far from the others -> 3 distinct clusters, no data lost
    assert len(clusters) == 3
    total_members = sum(len(c.record_indices) for c in clusters)
    assert total_members == 3


def test_match_to_existing_clusters_reuses_cluster_id_within_eps():
    records = [_record(77.1000, 28.7000)]
    labels = dbscan_labels(records, eps_degrees=0.005, min_samples=1)
    new_clusters = build_new_clusters(records, labels)

    existing = [(42, 77.1002, 28.7002)]  # ~25m away, well within 0.005 deg
    matched = match_to_existing_clusters(new_clusters, existing, eps_degrees=0.005)
    assert matched[0].cluster_id == 42


def test_match_to_existing_clusters_leaves_far_points_as_new():
    records = [_record(77.1000, 28.7000)]
    labels = dbscan_labels(records, eps_degrees=0.005, min_samples=1)
    new_clusters = build_new_clusters(records, labels)

    existing = [(42, 90.0, 10.0)]  # far away
    matched = match_to_existing_clusters(new_clusters, existing, eps_degrees=0.005)
    assert matched[0].cluster_id is None


def test_match_to_existing_clusters_lets_two_new_groups_share_one_match():
    """Regression for a real live bug: two different new-batch groups can
    both legitimately be the same existing physical fire (e.g. two
    sensors landing in separate DBSCAN groups this batch, or two nearby
    detections that happened not to cluster together this run). Matching
    used to be a 1-to-1 bijective assignment (scipy's Hungarian
    algorithm), which — with more new clusters than existing ones — can
    only pair off min(new, existing) of them, leaving a genuinely
    within-eps new cluster unmatched purely because another new cluster
    already "used" that existing column, not because it was actually far
    away. That is exactly what was observed live: two real clusters just
    8m apart, active on different days, that never merged into one."""
    records = [_record(77.1000, 28.7000), _record(77.1002, 28.7002)]  # ~25m apart
    labels = dbscan_labels(records, eps_degrees=0.0001, min_samples=1)  # force two separate groups
    new_clusters = build_new_clusters(records, labels)
    assert len(new_clusters) == 2

    existing = [(42, 77.1001, 28.7001)]  # ~15m from both new groups
    matched = match_to_existing_clusters(new_clusters, existing, eps_degrees=0.005)
    assert all(c.cluster_id == 42 for c in matched), \
        "both new groups are within eps of the same existing cluster and must both match it"


def test_cluster_batch_end_to_end_with_settings_defaults():
    settings = Settings(dbscan_eps_degrees=0.005, dbscan_min_samples=1)
    records = [
        _record(77.1000, 28.7000),
        _record(77.1001, 28.7001),
        _record(80.0000, 20.0000),
    ]
    assignments = cluster_batch(records, existing_clusters=None, settings=settings)
    assert len(assignments) == 2  # one merged cluster + one singleton
