"""APScheduler entry point — polls all satellite feeds every
POLL_INTERVAL_MINUTES (default 30) and runs the full ingestion cycle.

Run standalone with `python -m app.orchestration.scheduler`, or import
`start_scheduler()` from the FastAPI app's startup hook if you want the
poller and API in one process (fine for a hackathon deploy; split into
separate containers — see docker-compose.yml — for anything longer-lived).
"""
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings
from app.orchestration.pipeline import run_ingestion_cycle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_scheduler() -> BackgroundScheduler:
    settings = get_settings()
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_ingestion_cycle,
        "interval",
        minutes=settings.poll_interval_minutes,
        id="ingestion_cycle",
        next_run_time=None,  # first run scheduled explicitly below, not immediately on import
        max_instances=1,     # never let a slow cycle overlap the next tick
        coalesce=True,
    )
    scheduler.start()
    logger.info("Scheduler started: ingestion cycle every %d minutes", settings.poll_interval_minutes)
    return scheduler


if __name__ == "__main__":
    import time

    sched = start_scheduler()
    logger.info("Running initial ingestion cycle immediately...")
    run_ingestion_cycle()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown()
