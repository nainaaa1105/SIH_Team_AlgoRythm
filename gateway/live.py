"""WebSocket channel for live cluster updates.

The architecture puts a "WebSocket channel — live push to clients" in
Layer 3 and M6's plan lists `WS /ws/live-updates`; nothing implemented
it. This does.

Deliberately simple: an in-process connection registry with a broadcast
helper. That is correct for a single-worker deployment, which is what the
hackathon runs. It is *not* correct across multiple Uvicorn workers —
each would hold its own registry and a client would only see events from
the worker it happens to be connected to. The fix is a Redis pub/sub fan
-out (the broker is already there for Celery); noted rather than built,
because it cannot be tested without running the full stack.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks live WebSocket clients and broadcasts to them."""

    def __init__(self) -> None:
        self._connections: Set[Any] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("Dashboard client connected (%d total)", len(self._connections))

    async def disconnect(self, websocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("Dashboard client disconnected (%d remaining)", len(self._connections))

    @property
    def count(self) -> int:
        return len(self._connections)

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Send to every connected client, dropping any that have gone away.

        Failures are collected and removed after the loop rather than
        during it — mutating the set while iterating it would raise, and
        a client disconnecting mid-broadcast is the normal case, not an
        exceptional one.
        """
        payload = json.dumps({**message, "ts": datetime.now(timezone.utc).isoformat()})

        async with self._lock:
            targets = list(self._connections)

        dead: List[Any] = []
        for connection in targets:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)

        if dead:
            async with self._lock:
                for connection in dead:
                    self._connections.discard(connection)

        return len(targets) - len(dead)


manager = ConnectionManager()


async def notify_cluster_update(cluster_id: int, reason: str = "updated") -> int:
    """Called when a cluster changes so open dashboards refresh."""
    return await manager.broadcast(
        {"type": "cluster_update", "cluster_id": cluster_id, "reason": reason}
    )


def broadcast_sync(message: Dict[str, Any]) -> None:
    """Push a message from a sync context (a Celery task, or a
    thread-pool worker under `local_pipeline`).

    `app.orchestration.pipeline._notify_dashboards` already does this
    asyncio-loop-detection dance for the ingestion-complete push; this is
    that same bridge, generalised so a second sync-side alert (WUI, and
    whatever comes after it) does not reimplement it.

    Best-effort: a live push failing must never take down the pipeline
    stage that found something worth reporting, so every failure is
    swallowed here rather than propagated.
    """
    import asyncio

    try:
        async def _broadcast():
            await manager.broadcast(message)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_broadcast())
        except RuntimeError:
            asyncio.run(_broadcast())
    except Exception:
        logger.debug(
            "Sync broadcast failed (non-critical): %s", message.get("type"), exc_info=True,
        )
