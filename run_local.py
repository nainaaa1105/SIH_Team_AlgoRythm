"""Run the whole FireSight platform on localhost.

    python run_local.py

Single project now: app/, classifier/, geospatial/, imagery/, temporal/,
gateway/ and static/ are all top-level packages/dirs under this same
root, so the only thing needed on sys.path is the root itself.

The API works without a database for /health and the static dashboard;
endpoints that read Postgres return a clean 503 until `alembic upgrade
head` has run and the ingestion pipeline has populated something. The
dashboard detects that and falls back to its bundled sample records, so
it is demonstrable either way.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(ROOT))

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is not installed.\n"
            "  pip install -r requirements.txt"
        )

    print("FireSight — local gateway")
    print("  dashboard:  http://localhost:8000/")
    print("  API docs:   http://localhost:8000/docs")
    print("  health:     http://localhost:8000/health\n")

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
