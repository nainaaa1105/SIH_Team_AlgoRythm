"""Recompute the WUI (wildland-urban interface) verdict for every
classified cluster, using the current geospatial.wui_analysis gating
logic.

Usage:
    python -m scripts.backfill_wui_threat              # every classified cluster
    python -m scripts.backfill_wui_threat --limit 50
    python -m scripts.backfill_wui_threat --spreading-only

Why this exists: `Cluster.wui_threat` is a cached column, written once by
`geospatial.tasks.wui_threat` when a cluster is first evaluated (see that
module's docstring) — it is never recomputed on its own just because the
gating logic in geospatial/wui_analysis.py changes. Run this after any
change to WUI's priority gating (gate_critical_priority, the named-place
requirement, the Active-fire requirement) so already-ingested clusters
reflect the current rule, not whatever was in effect when they first ran
through the pipeline.

Covers every classified cluster by default, not just the current
spreading-class (wildfire/agricultural_burning) ones — a real gap found
running this the first time: 237 clusters read wui_threat=True with none
of them actually spreading-class any more (226 industrial_fire, 7
mining, 4 gas_flare) — stale flags left over from before a cluster was
last reclassified, never cleared because nothing had re-run the WUI
stage for them since. `wui_threat`'s own fast path already handles this
correctly (it persists wui_threat=False for a non-spreading class — see
`_persist_wui`'s docstring) but only for clusters this script, or a live
reclassification, actually revisits. Pass --spreading-only to limit the
run to the slower, network-bound spreading-class path if the non-
spreading fast-path sweep isn't needed. Narrower than
`run_live_cycle.py --reclassify`, which redoes the full classification
pipeline — this only re-runs the one WUI stage, calling the exact same
`wui_threat.run()` the live pipeline uses, not a second implementation.
"""
import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process at most this many clusters")
    parser.add_argument(
        "--spreading-only", action="store_true",
        help="only re-run wildfire/agricultural_burning clusters, skipping the "
             "non-spreading fast-path sweep that clears stale flags",
    )
    parser.add_argument("--quiet", action="store_true", help="errors only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from app.db.models import Cluster
    from app.db.session import session_scope
    from app.orchestration.local_pipeline import _dispatch_disabled
    from classifier.db.models import Classification
    from geospatial.tasks import wui_threat
    from geospatial.threat.corridor import SPREADING_CLASSES

    with session_scope() as session:
        query = session.query(Classification.cluster_id, Classification.predicted_class)
        if args.spreading_only:
            query = query.filter(Classification.predicted_class.in_(SPREADING_CLASSES))
        else:
            query = query.filter(Classification.predicted_class.isnot(None))
        rows = query.all()
        targets = [(cid, cls) for cid, cls in rows]
        if args.limit:
            targets = targets[: args.limit]
        cluster_ids = [cid for cid, _ in targets]
        before_critical = (
            session.query(Cluster)
            .filter(Cluster.id.in_(cluster_ids), Cluster.wui_threat.is_(True))
            .count()
            if cluster_ids else 0
        )

    logger.info("Recomputing WUI for %d clusters (%d currently CRITICAL)",
                len(targets), before_critical)

    after_critical = 0
    # wui_threat's own task body ends with celery_app.send_task(...) to
    # chain onward to resource-demand recompute — with no Redis broker
    # running in this local setup that call raises. This script only
    # needs the WUI verdict itself, so dispatch is neutralised the same
    # way local_pipeline.process_clusters already does for exactly this
    # reason, rather than the caller invoking that unrelated task.
    with _dispatch_disabled():
        for i, (cluster_id, predicted_class) in enumerate(targets, 1):
            result = wui_threat.run(cluster_id, predicted_class)
            if result.get("wui_threat"):
                after_critical += 1
            if i % 100 == 0 or i == len(targets):
                logger.info("%d/%d done", i, len(targets))

    logger.info(
        "Recompute complete: %d -> %d CRITICAL (out of %d clusters)",
        before_critical, after_critical, len(targets),
    )


if __name__ == "__main__":
    main()
