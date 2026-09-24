"""FastAPI entry point — the API gateway for the whole platform.

M1 owns the app and its own three routers. Architecture Layer 3 (the
gateway) is M6's, so the mounting of the other members' routers, the
WebSocket channel and the static dashboard all live below under the
Member 6 section.

M1 endpoints:
    GET /hotspots     - bbox/time/source filtered raw detections
    GET /clusters     - clustered events, GET /clusters/{id} for one
    GET /facilities   - industrial infrastructure registry
    POST /ingest/run  - manual trigger for one ingestion cycle (ops/demo use)
    GET /health       - liveness check
    POST /auth/signup, POST /auth/login, GET /auth/me
                      - dashboard operator accounts (app/auth/)

Mounted by M6's gateway composition below (each import is still
individually guarded — the platform is one project now, but a route
whose package fails to import degrades to "unavailable" rather than
taking down the whole app):
    /classify/{id}        M2 - classification, SHAP, evidence notes
    /plume|/threat|
    /attribution/{id}     M3 - dispersion, corridors, facility attribution
    /imagery/{id}         M4 - Dozier retrieval, image verdict, patches
    /ptsi|/forecast|
    /rhythm/{id},
    /escalating           M5 - persistence, escalation, rhythm
    /dashboard/*          M6 - aggregated event, detections, alerts, summary
    WS /ws/live-updates   M6 - live push to open dashboards
    /                     M6 - the dashboard itself
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_clusters, routes_facilities, routes_hotspots
from app.auth.routes import router as auth_router
from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Starts the FIRMS poller in-process on API startup, so the
    dashboard actually refreshes on its own instead of only ever
    reflecting whatever was last ingested by hand. Previously nothing
    called `start_scheduler()` at all -- every detection up to this
    point came from a manual /ingest/run or script invocation; left
    alone, the dashboard would never pick up a new day's fires. Shut
    down cleanly on exit rather than leaving a background thread dangling
    past the ASGI server's own lifetime.
    """
    from app.orchestration.scheduler import start_scheduler

    scheduler = start_scheduler()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="SIH162 FireSight Platform API", version="0.6.0", lifespan=lifespan)

# Allowed origins are driven by the CORS_ORIGINS env var (see app/config.py).
# Defaults to ["*"] for local dev; set CORS_ORIGINS in production, e.g.:
#   CORS_ORIGINS=https://firesight.example.com,https://app.example.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_hotspots.router)
app.include_router(routes_clusters.router)
app.include_router(routes_facilities.router)
app.include_router(auth_router)


# ══════════════════════════════════════════════════════════════════════
# MEMBER 6 — GATEWAY COMPOSITION
#
# Each router is mounted only if its package is importable. Guarding
# them individually is what lets M1's container run on its own, and lets
# the dashboard come up against a partially-deployed stack instead of
# refusing to start.
# ══════════════════════════════════════════════════════════════════════

# Recorded as routers mount. Tracked explicitly rather than introspected
# from app.routes: this FastAPI version wraps included routers in an
# object without a flat `.tags`, so introspection silently returned an
# empty list.
MOUNTED_MODULES: list = ["M1 ingestion"]


def _mount(label: str, import_router):
    try:
        app.include_router(import_router())
        MOUNTED_MODULES.append(label)
        logger.info("Gateway: mounted %s routes", label)
    except ImportError:
        logger.info("Gateway: %s package not installed — its routes are unavailable", label)
    except Exception:
        logger.exception("Gateway: failed to mount %s routes", label)


def _m2():
    from classifier.api.routes_classify import router
    return router


def _m3():
    from geospatial.api.routes_geo import router
    return router


def _m4():
    from imagery.api.routes_imagery import router
    return router


def _m5():
    from temporal.api.routes_temporal import router
    return router


def _m6():
    from gateway.routes_dashboard import router
    return router


_mount("M2 classification", _m2)
_mount("M3 geospatial", _m3)
_mount("M4 imagery", _m4)
_mount("M5 temporal", _m5)
_mount("M6 dashboard", _m6)


@app.exception_handler(Exception)
async def _database_unavailable_handler(request, exc):
    """Turn a missing database into an honest 503, not an opaque 500.

    Every read endpoint depends on Postgres. When it is not running —
    the normal state before `alembic upgrade head` — SQLAlchemy raises
    OperationalError deep inside the handler and FastAPI reports a bare
    500, which tells an operator nothing and looks identical to a bug.
    Anything else is re-raised untouched so real errors stay visible.
    """
    from fastapi.responses import JSONResponse
    from sqlalchemy.exc import InterfaceError, NoSuchModuleError, OperationalError

    # Postgres drivers. A missing one is an environment problem, not a
    # bug — psycopg2 in particular has no Python 3.13 wheel on Windows,
    # so a fresh checkout hits this before it hits a connection error.
    _DB_DRIVERS = {"psycopg2", "psycopg", "psycopg2cffi", "pg8000", "asyncpg"}

    unreachable = isinstance(exc, (OperationalError, InterfaceError, NoSuchModuleError))
    driver_missing = isinstance(exc, ModuleNotFoundError) and exc.name in _DB_DRIVERS

    if unreachable or driver_missing:
        reason = (
            f"PostgreSQL driver '{exc.name}' is not installed."
            if driver_missing else "Database unavailable."
        )
        logger.warning("%s Request: %s", reason, request.url.path)
        return JSONResponse(
            status_code=503,
            content={
                "detail": reason,
                "hint": (
                    "pip install psycopg2-binary (or run inside the Docker image, "
                    "which uses Python 3.11 where a wheel exists)."
                    if driver_missing else
                    "Start Postgres/PostGIS (docker compose up db), then run "
                    "'alembic upgrade head' from the project root."
                ) + " The dashboard falls back to bundled sample data until then.",
                "path": request.url.path,
            },
        )
    raise exc


@app.websocket("/ws/live-updates")
async def live_updates(websocket: WebSocket):
    """Live push channel for open dashboards.

    The client sends nothing meaningful; the receive loop exists purely
    to detect a closed socket. Without it the coroutine would return
    immediately and FastAPI would close a connection the browser still
    believes is open.

    The `WebSocket` annotation is load-bearing, not decoration. FastAPI
    resolves handler parameters by type; with the annotation missing it
    treated `websocket` as a *query parameter*, failed to validate it,
    and rejected every handshake with 403 before this body ever ran. The
    channel silently never delivered a single update.
    """
    from starlette.websockets import WebSocketDisconnect

    from gateway.live import manager

    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "hello", "clients": manager.count})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("Dashboard websocket closed unexpectedly", exc_info=True)
    finally:
        await manager.disconnect(websocket)


def _mount_dashboard() -> None:
    """Serve the dashboard from the gateway so one process runs everything.

    Mounted last: a StaticFiles mount at "/" swallows every unmatched
    path, so registering it before the API routers would shadow them.
    """
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parents[1] / "static"
    if not static_dir.is_dir():
        logger.info("Gateway: dashboard static directory not found at %s", static_dir)
        return
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="dashboard")
    logger.info("Gateway: dashboard served from %s", static_dir)


@app.get("/health")
def health():
    """Liveness plus which members' routes actually came up.

    The dashboard pings this on load, and an operator debugging a
    partial deployment needs to know *which* half is missing.
    """
    return {"status": "ok", "version": app.version, "modules": MOUNTED_MODULES}


@app.post("/ingest/run")
async def trigger_ingestion_cycle(classify: bool = True):
    """Manual trigger for one ingestion cycle — useful for demos/ops
    without waiting for the next scheduled poll. Runs in a thread pool
    so the API event loop stays responsive; returns a summary.

    The scheduler's automatic poll (see `lifespan` below) hits this exact
    same decision via `run_cycle_safely` — not a separate code path.

    `classify=false` ingests only, leaving classification for later.
    """
    import asyncio

    from app.orchestration.pipeline import run_cycle_safely

    return await asyncio.to_thread(run_cycle_safely, classify)


# Registered after every API route so the catch-all static mount cannot
# shadow them.
_mount_dashboard()
