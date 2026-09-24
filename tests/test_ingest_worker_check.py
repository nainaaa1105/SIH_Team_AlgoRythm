"""`/ingest/run` and the scheduler's shared Celery-vs-in-process decision.

Real incident: Railway had Redis provisioned (so the broker was
reachable) but no Celery worker service ever deployed (Dockerfile.worker
exists in the repo but was never turned into a running service). The old
check only asked "is the broker reachable", so it picked the Celery path,
`.delay()` accepted every call without error, and nothing was ever
actually classified -- a silent no-op with a 200 response. Locally this
never happened, since there is no Redis to be reachable at all, so the
in-process path was always taken -- which is exactly why the bug never
showed up until deployment. `celery_worker_available` fixes this by
pinging for a live worker, not just a live broker.

Moved from app.main into app.orchestration.pipeline so the scheduler's
automatic poll (app/orchestration/scheduler.py) can share the exact same
decision instead of /ingest/run and the scheduler each risking their own,
possibly-diverging copy.
"""
from app.orchestration.pipeline import broker_is_reachable, celery_worker_available


def test_worker_check_is_false_when_broker_unreachable():
    """No Redis at all (the local setup) -- must not even attempt a ping."""
    assert broker_is_reachable(timeout_seconds=0.01) is False
    assert celery_worker_available(timeout_seconds=0.01) is False


def test_worker_check_pings_for_a_live_worker_not_just_the_broker():
    """A reachable broker with zero responding workers -- the exact
    Railway scenario -- must still resolve to False, not True."""
    import app.orchestration.pipeline as pipeline_module

    calls = {"broker": False, "ping": False}

    def fake_broker_reachable(timeout_seconds=1.0):
        calls["broker"] = True
        return True  # Redis is up

    class FakeControl:
        def ping(self, timeout):
            calls["ping"] = True
            return []  # broker reachable, but no worker answered

    class FakeCeleryApp:
        control = FakeControl()

    import app.orchestration.queue as queue_module

    original_reachable = pipeline_module.broker_is_reachable
    original_celery_app = queue_module.celery_app
    try:
        pipeline_module.broker_is_reachable = fake_broker_reachable
        queue_module.celery_app = FakeCeleryApp()
        result = pipeline_module.celery_worker_available(timeout_seconds=0.01)
    finally:
        pipeline_module.broker_is_reachable = original_reachable
        queue_module.celery_app = original_celery_app

    assert calls["broker"] is True
    assert calls["ping"] is True
    assert result is False


def test_worker_check_is_true_when_a_worker_actually_answers():
    import app.orchestration.pipeline as pipeline_module
    import app.orchestration.queue as queue_module

    class FakeControl:
        def ping(self, timeout):
            return [{"worker@host": "pong"}]

    class FakeCeleryApp:
        control = FakeControl()

    original_reachable = pipeline_module.broker_is_reachable
    original_celery_app = queue_module.celery_app
    try:
        pipeline_module.broker_is_reachable = lambda timeout_seconds=1.0: True
        queue_module.celery_app = FakeCeleryApp()
        result = pipeline_module.celery_worker_available(timeout_seconds=0.01)
    finally:
        pipeline_module.broker_is_reachable = original_reachable
        queue_module.celery_app = original_celery_app

    assert result is True


def test_run_cycle_safely_routes_to_in_process_without_a_live_worker():
    """The function /ingest/run and the scheduler both call -- without a
    live worker it must go through local_pipeline.run_cycle, not
    pipeline.run_ingestion_cycle's Celery .delay() path."""
    import app.orchestration.pipeline as pipeline_module

    calls = {"local_pipeline": False, "celery_path": False}

    def fake_local_run_cycle():
        calls["local_pipeline"] = True
        return {"processed": 0, "classified": 0, "failed": 0}

    def fake_run_ingestion_cycle(*a, **k):
        calls["celery_path"] = True
        return {"raw": 0, "deduped": 0, "clusters_touched": 0}

    import app.orchestration.local_pipeline as local_pipeline_module

    original_worker_available = pipeline_module.celery_worker_available
    original_run_ingestion_cycle = pipeline_module.run_ingestion_cycle
    original_local_run_cycle = local_pipeline_module.run_cycle
    try:
        pipeline_module.celery_worker_available = lambda: False
        pipeline_module.run_ingestion_cycle = fake_run_ingestion_cycle
        local_pipeline_module.run_cycle = fake_local_run_cycle
        summary = pipeline_module.run_cycle_safely()
    finally:
        pipeline_module.celery_worker_available = original_worker_available
        pipeline_module.run_ingestion_cycle = original_run_ingestion_cycle
        local_pipeline_module.run_cycle = original_local_run_cycle

    assert calls["local_pipeline"] is True
    assert calls["celery_path"] is False
    assert summary["mode"] == "in-process"


def test_run_cycle_safely_routes_to_celery_with_a_live_worker():
    import app.orchestration.pipeline as pipeline_module

    calls = {"celery_path": False}

    def fake_run_ingestion_cycle(*a, **k):
        calls["celery_path"] = True
        return {"raw": 0, "deduped": 0, "clusters_touched": 0}

    original_worker_available = pipeline_module.celery_worker_available
    original_run_ingestion_cycle = pipeline_module.run_ingestion_cycle
    try:
        pipeline_module.celery_worker_available = lambda: True
        pipeline_module.run_ingestion_cycle = fake_run_ingestion_cycle
        summary = pipeline_module.run_cycle_safely()
    finally:
        pipeline_module.celery_worker_available = original_worker_available
        pipeline_module.run_ingestion_cycle = original_run_ingestion_cycle

    assert calls["celery_path"] is True
    assert summary["mode"] == "celery"
