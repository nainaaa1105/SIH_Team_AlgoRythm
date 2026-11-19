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


def _assignments_at(coords):
    from app.ingestion.cluster import ClusterAssignment

    return [
        ClusterAssignment(cluster_id=None, temp_label=i, centroid_lon=lon, centroid_lat=lat)
        for i, (lon, lat) in enumerate(coords)
    ]


def test_sliced_matching_gives_the_same_answer_as_the_original_full_matrix(monkeypatch):
    """The matching used to build one (new x existing x 2) array. It now
    works a slice of new clusters at a time to bound memory; the per-row
    answer must be exactly what the full-matrix formula gives, including
    when the slice is forced down to a single row."""
    import numpy as np

    import app.ingestion.cluster as cluster_module

    rng = np.random.default_rng(7)
    new_coords = rng.uniform([68.0, 8.0], [97.0, 35.0], size=(60, 2))
    existing = [(1000 + i, lon, lat) for i, (lon, lat) in enumerate(
        rng.uniform([68.0, 8.0], [97.0, 35.0], size=(400, 2))
    )]
    # Make a handful of new clusters land on top of an existing one so
    # both the matched and the unmatched branches are exercised.
    new_coords[:10] = np.array([[e[1], e[2]] for e in existing[:10]]) + 0.001

    eps = 0.005
    existing_coords = np.array([[e[1], e[2]] for e in existing])
    diff = new_coords[:, np.newaxis, :] - existing_coords[np.newaxis, :, :]
    full = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(full, axis=1)
    expected = [
        existing[c][0] if full[r, c] <= eps else None for r, c in enumerate(nearest)
    ]
    assert any(e is not None for e in expected) and any(e is None for e in expected)

    for cells in (4_000_000, 1):  # default slice size, and one row at a time
        monkeypatch.setattr(cluster_module, "_MATCH_CHUNK_CELLS", cells)
        matched = match_to_existing_clusters(_assignments_at(new_coords), existing, eps)
        assert [m.cluster_id for m in matched] == expected


def test_matching_memory_stays_bounded_with_a_large_stored_cluster_table():
    """Regression for the crash loop: with ~50k stored clusters, a batch of
    a few thousand new ones needed 6 GB+ in the old one-shot array and the
    process was killed. This scale needs ~1.8 GB the old way; the sliced
    version must stay far below that."""
    import tracemalloc

    import numpy as np

    rng = np.random.default_rng(11)
    existing = [(i, lon, lat) for i, (lon, lat) in enumerate(
        rng.uniform([68.0, 8.0], [97.0, 35.0], size=(30_000, 2))
    )]
    new = _assignments_at(rng.uniform([68.0, 8.0], [97.0, 35.0], size=(1_500, 2)))

    tracemalloc.start()
    try:
        match_to_existing_clusters(new, existing, eps_degrees=0.005)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 400 * 1024 * 1024, f"matching peaked at {peak / 1e6:.0f} MB"


def test_cluster_batch_end_to_end_with_settings_defaults():
    settings = Settings(dbscan_eps_degrees=0.005, dbscan_min_samples=1)
    records = [
        _record(77.1000, 28.7000),
        _record(77.1001, 28.7001),
        _record(80.0000, 20.0000),
    ]
    assignments = cluster_batch(records, existing_clusters=None, settings=settings)
    assert len(assignments) == 2  # one merged cluster + one singleton
