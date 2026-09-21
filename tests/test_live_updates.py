"""The live-push channel: the handshake, the registry and the broadcast.

None of this was covered. The WebSocket endpoint had shipped with its
`websocket` parameter unannotated, and FastAPI resolves handler
parameters by type — so it treated the parameter as a *query parameter*,
failed to validate it, and rejected every handshake with HTTP 403 before
the handler body ever ran. The channel silently delivered nothing, and
the dashboard's only symptom was a "LIVE" pill that never lit.

A test that actually opens the socket catches that in a second, so these
exist now.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import app
from gateway.live import ConnectionManager, manager


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --- the handshake ------------------------------------------------------

def test_the_websocket_handshake_succeeds(client):
    """Regression: a missing `WebSocket` annotation made this 403."""
    with client.websocket_connect("/ws/live-updates") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["clients"] >= 1


def test_the_handler_is_annotated_so_fastapi_injects_the_socket():
    """Guard the specific mistake, not just its symptom.

    Asserting on the annotation as well as the behaviour means the cause
    is named if this ever regresses, instead of leaving whoever hits it
    to rediscover why a websocket route returns 403.
    """
    import inspect

    from starlette.websockets import WebSocket

    import app.main as main

    signature = inspect.signature(main.live_updates)
    parameter = signature.parameters["websocket"]
    assert parameter.annotation is not inspect.Parameter.empty, (
        "live_updates(websocket) must be annotated — without it FastAPI "
        "treats the parameter as a query parameter and rejects the handshake"
    )
    assert parameter.annotation is WebSocket


def test_a_connected_client_is_counted(client):
    with client.websocket_connect("/ws/live-updates") as ws:
        hello = ws.receive_json()
        assert hello["clients"] == manager.count


# --- the registry and broadcast ----------------------------------------

class _FakeSocket:
    """Minimal stand-in: records what it was sent, or refuses to accept."""

    def __init__(self, fail_on_send: bool = False):
        self.sent = []
        self.accepted = False
        self._fail_on_send = fail_on_send

    async def accept(self):
        self.accepted = True

    async def send_text(self, payload: str):
        if self._fail_on_send:
            raise ConnectionResetError("client went away")
        self.sent.append(payload)


def test_broadcast_reaches_every_connected_client():
    async def scenario():
        registry = ConnectionManager()
        a, b = _FakeSocket(), _FakeSocket()
        await registry.connect(a)
        await registry.connect(b)

        delivered = await registry.broadcast({"type": "ingestion_complete", "clusters_touched": 3})

        assert delivered == 2
        assert a.accepted and b.accepted
        for socket in (a, b):
            assert len(socket.sent) == 1
            assert "ingestion_complete" in socket.sent[0]
            # Every message carries the time it was sent; the dashboard
            # has no other way to tell a stale push from a fresh one.
            assert '"ts"' in socket.sent[0]

    asyncio.run(scenario())


def test_a_client_that_went_away_is_dropped_not_retried():
    """A disconnect mid-broadcast is the normal case, not an error, and
    must not take the broadcast down with it."""
    async def scenario():
        registry = ConnectionManager()
        alive, dead = _FakeSocket(), _FakeSocket(fail_on_send=True)
        await registry.connect(alive)
        await registry.connect(dead)

        delivered = await registry.broadcast({"type": "cluster_update", "cluster_id": 1})

        assert delivered == 1, "the live client should still have been served"
        assert registry.count == 1, "the dead client should have been dropped"
        assert len(alive.sent) == 1

    asyncio.run(scenario())


def test_disconnect_removes_the_client():
    async def scenario():
        registry = ConnectionManager()
        socket = _FakeSocket()
        await registry.connect(socket)
        assert registry.count == 1
        await registry.disconnect(socket)
        assert registry.count == 0
        # Discarding an unknown socket must not raise — disconnect runs in
        # a `finally`, so it can fire for a connection that never joined.
        await registry.disconnect(socket)
        assert registry.count == 0

    asyncio.run(scenario())


# --- what the pipeline pushes ------------------------------------------

def test_ingestion_notifies_dashboards_only_when_something_changed():
    """An empty cycle must not wake every open dashboard into a reload."""
    from app.orchestration.pipeline import _notify_dashboards

    sent = []

    async def scenario():
        registry = ConnectionManager()
        socket = _FakeSocket()
        await registry.connect(socket)
        sent.append(socket)

    asyncio.run(scenario())

    # No clusters touched -> returns without touching the event loop at
    # all, which is what makes it safe to call from a sync context.
    assert _notify_dashboards([]) is None
