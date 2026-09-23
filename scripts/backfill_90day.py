"""90-day retrospective replay: pulls the FIRMS archive in <=5-day
windows, then runs it through the same dedup -> cluster -> persist path
as the live scheduler, to (a) generate training data for M2 and (b)
validate the pipeline at full scale before the final deployment.

Only FIRMS is replayed here — INSAT-3DS (MOSDAC) and Sentinel-3 FRP don't
have the same simple date-ranged archive endpoint wired up yet (see the
caveats in app/ingestion/insat3ds.py and sentinel3_frp.py); extend this
script once a historical access path for those is confirmed.

Usage: python -m scripts.backfill_90day --days 90 --dispatch-jobs=false
"""
import argparse
import logging
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.db.session import session_scope
from app.ingestion import firms
from app.ingestion.cluster import cluster_batch
from app.ingestion.dedup import full_dedup
from app.orchestration.pipeline import load_active_cluster_centroids, persist_records_and_clusters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW_DAYS = 5  # FIRMS archive query cap — confirmed against the live API


def backfill(total_days: int, dispatch_jobs: bool) -> None:
    settings = get_settings()
    end = datetime.now(timezone.utc).date()
    windows_processed = 0
    total_raw = 0
    total_clusters = 0

    remaining = total_days
    while remaining > 0:
        window = min(WINDOW_DAYS, remaining)
        end_date_str = end.strftime("%Y-%m-%d")
        logger.info("Backfilling window ending %s (%d days)", end_date_str, window)

        records = firms.fetch_historical_window(end_date_str, window, settings)
        total_raw += len(records)

        if records:
            deduped = full_dedup(records)
            with session_scope() as session:
                existing_clusters = load_active_cluster_centroids(session)
                assignments = cluster_batch(deduped, existing_clusters, settings)
                touched = persist_records_and_clusters(session, deduped, assignments)
                total_clusters += len(touched)

            if dispatch_jobs:
                from app.orchestration.tasks import dispatch_downstream_jobs, enrich_cluster
                for cluster_id, lon, lat, optical_available in touched:
                    enrich_cluster.delay(cluster_id, lon, lat)
                    dispatch_downstream_jobs.delay(cluster_id, lon, lat, optical_available)

        end = end - timedelta(days=window)
        remaining -= window
        windows_processed += 1

    logger.info(
        "Backfill complete: %d windows, %d raw records, %d cluster-touch events",
        windows_processed, total_raw, total_clusters,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--dispatch-jobs", type=lambda v: v.lower() == "true", default=False,
                         help="Fan out per-cluster jobs during backfill (usually False — this is for "
                              "generating training data in bulk, not live alerting)")
    args = parser.parse_args()
    backfill(args.days, args.dispatch_jobs)


if __name__ == "__main__":
    main()
