"""APScheduler entry point — polls all satellite feeds every
POLL_INTERVAL_MINUTES (default 30) and runs the full ingestion cycle.

Run standalone with `python -m app.orchestration.scheduler`, or import
`start_scheduler()` from the FastAPI app's startup hook if you want the
poller and API in one process (fine for a hackathon deploy; split into
separate containers — see docker-compose.yml — for anything longer-lived).
"""
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings
from app.orchestration.pipeline import run_cycle_safely

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_scheduler(run_soon: bool = True) -> BackgroundScheduler:
    """`run_soon=True` (the default) schedules the first cycle a few
    seconds out instead of waiting a full poll_interval_minutes before
    doing anything -- matters for the FastAPI startup hook, where the app
    would otherwise serve a stale/empty dashboard for up to 30 minutes
    after every restart. Runs through run_cycle_safely -- the same
    Celery-vs-in-process decision /ingest/run makes, not a separate path,
    so a deployment with no live Celery worker still gets classified
    results from the scheduled poll instead of silently accumulating an
    unclassified backlog again.
    """
    settings = get_settings()
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_cycle_safely,
        "interval",
        minutes=settings.poll_interval_minutes,
        id="ingestion_cycle",
        next_run_time=(datetime.now(timezone.utc) + timedelta(seconds=5)) if run_soon else None,
        max_instances=1,     # never let a slow cycle overlap the next tick
        coalesce=True,
    )
    scheduler.start()
    logger.info("Scheduler started: ingestion cycle every %d minutes", settings.poll_interval_minutes)
    return scheduler


if __name__ == "__main__":
    import time

    sched = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown()
