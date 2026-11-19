"""The FIRMS auto-poller and its wiring into the app's own startup.

Real gap found by inspection, not a report: app/orchestration/scheduler.py
has existed the whole time, but nothing ever called start_scheduler() --
app/main.py had no startup hook at all. Every detection up to that point
came from a manual /ingest/run call or a manually-run script; left alone
the dashboard would never pick up a new day's fires on its own. These
tests guard both halves: that start_scheduler() itself is wired correctly
(uses run_cycle_safely, not a raw Celery-dispatch call, and doesn't make
the app wait a full poll interval before doing anything), and that the
FastAPI app's lifespan actually calls it.
"""
from datetime import datetime, timezone

from app.orchestration.scheduler import start_scheduler


def test_scheduler_uses_run_cycle_safely_not_the_raw_celery_dispatch():
    """Regression guard: a scheduler wired to pipeline.run_ingestion_cycle
    directly would hit the exact same silent-no-worker bug /ingest/run
    had -- .delay() calls accepted, nothing ever classified, on a
    deployment with Redis but no live Celery worker."""
    scheduler = start_scheduler(run_soon=False)
    try:
        job = scheduler.get_job("ingestion_cycle")
        assert job is not None
        assert job.func.__name__ == "run_cycle_safely"
    finally:
        scheduler.shutdown(wait=False)


def test_first_cycle_is_delayed_after_startup_but_not_a_full_poll_interval():
    """The first cycle used to fire 5 seconds after boot. When that cycle
    got the process killed, every restart immediately repeated it: a crash
    loop that used up Railway's retries and left the site down for days.
    It now waits STARTUP_DELAY_SECONDS so the server is up and healthy
    first -- but still well short of a full poll interval, so a restart
    doesn't leave the data stale for 30 minutes."""
    from app.orchestration.scheduler import STARTUP_DELAY_SECONDS

    assert STARTUP_DELAY_SECONDS >= 60, "must give the server time to come up before heavy work"
    scheduler = start_scheduler()
    try:
        job = scheduler.get_job("ingestion_cycle")
        assert job.next_run_time is not None
        seconds_until_first_run = (job.next_run_time - datetime.now(timezone.utc)).total_seconds()
        assert STARTUP_DELAY_SECONDS - 10 <= seconds_until_first_run <= STARTUP_DELAY_SECONDS
        assert seconds_until_first_run < 30 * 60
    finally:
        scheduler.shutdown(wait=False)


def test_scheduler_run_soon_false_defers_to_the_normal_interval():
    scheduler = start_scheduler(run_soon=False)
    try:
        job = scheduler.get_job("ingestion_cycle")
        assert job.next_run_time is None
    finally:
        scheduler.shutdown(wait=False)


def test_scheduler_never_overlaps_a_slow_cycle_with_the_next_tick():
    scheduler = start_scheduler(run_soon=False)
    try:
        job = scheduler.get_job("ingestion_cycle")
        assert job.max_instances == 1
    finally:
        scheduler.shutdown(wait=False)


def test_app_lifespan_starts_the_scheduler_on_startup():
    """The actual wiring bug: app/main.py never had a startup hook, so
    start_scheduler() was reachable but never called by anything running
    in production."""
    import inspect

    from app.main import lifespan

    source = inspect.getsource(lifespan)
    assert "start_scheduler" in source
    assert "from app.orchestration.scheduler import start_scheduler" in source


def test_app_is_constructed_with_the_lifespan_hook():
    """FastAPI wraps the passed lifespan in its own merged-context
    function, so identity comparison against `app.router.lifespan_context`
    doesn't work -- TestClient's context manager is what actually drives
    ASGI startup/shutdown events, so that's what has to prove the hook
    fires, not introspection of FastAPI's internals."""
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    import app.orchestration.scheduler as scheduler_module
    from app.main import app

    fake_scheduler = MagicMock()
    original_start_scheduler = scheduler_module.start_scheduler
    try:
        scheduler_module.start_scheduler = MagicMock(return_value=fake_scheduler)
        with TestClient(app):
            pass
        scheduler_module.start_scheduler.assert_called_once()
        fake_scheduler.shutdown.assert_called_once()
    finally:
        scheduler_module.start_scheduler = original_start_scheduler
