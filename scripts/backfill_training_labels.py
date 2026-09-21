"""Assemble features + apply label rules across every existing cluster.

Run this after M1's `scripts/backfill_90day.py` has populated clusters
and hotspots, to produce the training set.

Usage:
    python -m scripts.backfill_training_labels
    python -m scripts.backfill_training_labels --fsi data/fsi_alerts.csv
    python -m scripts.backfill_training_labels --skip-features   # labels only
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsi", help="Path to an FSI forest-fire alert CSV export")
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Only re-run label rules; assume cluster_features is already built",
    )
    args = parser.parse_args()

    from sqlalchemy import select

    from app.db.models import Cluster
    from app.db.session import session_scope

    from classifier.features.assemble import assemble_features, upsert_cluster_features
    from classifier.labels.build_training_set import (
        label_clusters,
        load_labelable_clusters,
        persist_labels,
    )

    fsi_alerts = None
    if args.fsi:
        from classifier.labels.fsi_labels import load_fsi_alerts

        fsi_alerts = load_fsi_alerts(args.fsi)

    if not args.skip_features:
        with session_scope() as session:
            cluster_ids = session.execute(select(Cluster.id)).scalars().all()
            logger.info("Assembling features for %d clusters", len(cluster_ids))
            for i, cluster_id in enumerate(cluster_ids, start=1):
                features = assemble_features(session, cluster_id)
                if features:
                    upsert_cluster_features(session, cluster_id, features)
                if i % 500 == 0:
                    logger.info("  ...%d/%d", i, len(cluster_ids))

    with session_scope() as session:
        clusters = load_labelable_clusters(session)
        logger.info("Applying label rules to %d clusters with features", len(clusters))
        resolved, ambiguous = label_clusters(clusters, fsi_alerts)
        persist_labels(session, resolved, ambiguous)

    by_label = {}
    for record in resolved:
        by_label[record["label"]] = by_label.get(record["label"], 0) + 1

    logger.info("Labelled %d clusters: %s", len(resolved), by_label)
    logger.info(
        "%d clusters were ambiguous (contradictory rules) and are excluded from "
        "training — these are the analyst-review queue",
        len(ambiguous),
    )


if __name__ == "__main__":
    main()
