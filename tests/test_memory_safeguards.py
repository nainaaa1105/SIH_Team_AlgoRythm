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
