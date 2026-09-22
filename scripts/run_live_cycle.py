"""Run one full live cycle end to end, in this process.

    python -m scripts.run_live_cycle                 # ingest + classify everything new
    python -m scripts.run_live_cycle --no-ingest     # only finish clusters already stored
    python -m scripts.run_live_cycle --limit 50      # cap how many clusters are processed
    python -m scripts.run_live_cycle --reclassify    # redo every cluster, not just new ones

What it does, in order: pull the current FIRMS pass for the configured
bbox, dedup it against what is already stored, DBSCAN it into clusters,
then for every cluster run enrichment, facility attribution, land cover,
the Dozier retrieval, the temporal/Kalman state, the 28-feature vector,
the XGBoost classification with SHAP, and the plume/corridor model.

This exists because the deployment shape of the platform fans that work
out over Celery and Redis, and neither is running in a local setup. See
`app/orchestration/local_pipeline` for how the ordering differs from the
queue's and why.

`--reclassify` is the one to use after bulk-loading more facilities:
facility distance is three of the model's twenty-eight features, so
clusters classified before the facility registry was populated were
decided on genuinely less evidence and are worth re-deciding.
"""
import argparse
import json
import logging
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="skip the satellite fetch; only process clusters already in the database",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process at most this many clusters",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="parallel per-cluster workers (default: 6). Each holds a DB "
             "connection, so keep this under the SQLAlchemy pool ceiling.",
    )
    parser.add_argument(
        "--reclassify", action="store_true",
        help="re-run clusters that already have a verdict (use after loading "
             "more facilities, or after retraining the model)",
    )
    parser.add_argument("--quiet", action="store_true", help="errors only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # rasterio logs a warning per windowed read on some GeoTIFF layouts;
    # it is noise here, not a problem with the read.
    for noisy in ("rasterio", "rasterio._env", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    from app.orchestration.local_pipeline import (
        CycleReport,
        process_clusters,
        run_cycle,
        unprocessed_cluster_targets,
    )

    started = time.time()

    if args.reclassify:
        from geoalchemy2.shape import to_shape
        from sqlalchemy import select

        from app.db.models import Cluster
        from app.db.session import session_scope

        with session_scope() as session:
            stmt = select(Cluster).order_by(Cluster.last_seen.desc().nullslast())
            if args.limit:
                stmt = stmt.limit(args.limit)
            targets = []
            for cluster in session.execute(stmt).scalars().all():
                point = to_shape(cluster.centroid)
                targets.append((cluster.id, point.x, point.y, bool(cluster.optical_available)))

        logging.info("Reclassifying %d clusters", len(targets))
        report = process_clusters(targets, workers=args.workers, report=CycleReport())
        result = report.as_dict()
    else:
        result = run_cycle(
            limit=args.limit, workers=args.workers, ingest=not args.no_ingest
        )

    result["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
    result["still_unprocessed"] = len(unprocessed_cluster_targets())

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
