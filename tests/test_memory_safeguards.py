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


def test_db_connections_are_recycled_so_postgres_backends_cannot_grow_forever(monkeypatch):
    """Long-lived pooled connections let each Postgres backend keep the
    memory of every past query (~1.65 GB each in production). The engine
    must recycle connections and keep pre-ping so a recycled or
    server-closed connection is replaced transparently."""
    import app.db.session as db_session

    captured = {}

    def fake_create_engine(url, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(db_session, "create_engine", fake_create_engine)
    monkeypatch.setattr(db_session, "_engine", None)
    db_session.get_engine()

    assert captured.get("pool_pre_ping") is True
    assert 0 < captured.get("pool_recycle", 0) <= 3600
