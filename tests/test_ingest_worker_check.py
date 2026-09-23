"""`/ingest/run`'s Celery-vs-in-process decision.

Real incident: Railway had Redis provisioned (so the broker was
reachable) but no Celery worker service ever deployed (Dockerfile.worker
exists in the repo but was never turned into a running service). The old
check only asked "is the broker reachable", so it picked the Celery path,
`.delay()` accepted every call without error, and nothing was ever
actually classified -- a silent no-op with a 200 response. Locally this
never happened, since there is no Redis to be reachable at all, so the
in-process path was always taken -- which is exactly why the bug never
showed up until deployment. `_celery_worker_available` fixes this by
pinging for a live worker, not just a live broker.
"""
from app.main import _broker_is_reachable, _celery_worker_available


def test_worker_check_is_false_when_broker_unreachable():
    """No Redis at all (the local setup) -- must not even attempt a ping."""
    assert _broker_is_reachable(timeout_seconds=0.01) is False
    assert _celery_worker_available(timeout_seconds=0.01) is False


def test_worker_check_pings_for_a_live_worker_not_just_the_broker():
    """A reachable broker with zero responding workers -- the exact
    Railway scenario -- must still resolve to False, not True."""
    import app.main as main_module

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

    original_reachable = main_module._broker_is_reachable
    original_celery_app = queue_module.celery_app
    try:
        main_module._broker_is_reachable = fake_broker_reachable
        queue_module.celery_app = FakeCeleryApp()
        result = main_module._celery_worker_available(timeout_seconds=0.01)
    finally:
        main_module._broker_is_reachable = original_reachable
        queue_module.celery_app = original_celery_app

    assert calls["broker"] is True
    assert calls["ping"] is True
    assert result is False


def test_worker_check_is_true_when_a_worker_actually_answers():
    import app.main as main_module
    import app.orchestration.queue as queue_module

    class FakeControl:
        def ping(self, timeout):
            return [{"worker@host": "pong"}]

    class FakeCeleryApp:
        control = FakeControl()

    original_reachable = main_module._broker_is_reachable
    original_celery_app = queue_module.celery_app
    try:
        main_module._broker_is_reachable = lambda timeout_seconds=1.0: True
        queue_module.celery_app = FakeCeleryApp()
        result = main_module._celery_worker_available(timeout_seconds=0.01)
    finally:
        main_module._broker_is_reachable = original_reachable
        queue_module.celery_app = original_celery_app

    assert result is True
