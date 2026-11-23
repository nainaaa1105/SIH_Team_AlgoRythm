"""Guards for the memory-related settings added after the Nov 2026 crash
loop (the app was killed mid-ingestion, restarted, and was killed again
until Railway stopped retrying). These pin the cheap, easy-to-undo knobs;
the clustering fix itself is tested in test_cluster.py.
"""
from pathlib import Path


def test_in_process_pipeline_uses_few_workers_by_default():
    from app.orchestration.local_pipeline import DEFAULT_WORKERS

    assert DEFAULT_WORKERS <= 2


def test_container_caps_glibc_malloc_arenas():
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile.api"
    assert "ENV MALLOC_ARENA_MAX=2" in dockerfile.read_text(encoding="utf-8")


def test_cluster_loader_filters_by_window_and_loads_only_needed_columns():
    """The matching loader must only pull recent active clusters, and only
    their id + centroid; window_days=None (backfill) drops the date filter."""
    from datetime import datetime, timezone

    from app.orchestration.pipeline import MATCH_WINDOW_DAYS, load_active_cluster_centroids

    captured = []

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return []

    class _Session:
        def execute(self, stmt):
            captured.append(str(stmt.compile(compile_kwargs={"literal_binds": False})))
            return _Result()

    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    assert MATCH_WINDOW_DAYS == 30
    load_active_cluster_centroids(_Session(), now=now)
    load_active_cluster_centroids(_Session(), window_days=None, now=now)

    windowed, unwindowed = captured
    assert "last_seen" in windowed
    assert "last_seen" not in unwindowed.split("WHERE", 1)[1]
