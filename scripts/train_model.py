"""Train the XGBoost classifier from the labelled set in the database.

Usage:
    python -m scripts.train_model --version v1
    python -m scripts.train_model --version v2 --trials 60
    python -m scripts.train_model --no-tune            # fast smoke run
"""
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    from app.db.session import session_scope

    from classifier.labels.build_training_set import load_training_matrix
    from classifier.model.train import build_training_matrix, train

    with session_scope() as session:
        rows, labels, coordinates, cluster_ids = load_training_matrix(session)

    if not rows:
        raise SystemExit(
            "No labelled clusters found. Run `python -m scripts.backfill_training_labels` first."
        )

    distribution = {label: labels.count(label) for label in sorted(set(labels))}
    logger.info("Training on %d labelled clusters: %s", len(rows), distribution)

    if len(distribution) < 2:
        raise SystemExit(
            f"Only one class present ({list(distribution)}) — a classifier needs at least two. "
            "Check the label rules and the geographic coverage of the backfill."
        )

    data = build_training_matrix(rows, labels, coordinates, cluster_ids)
    _, metadata = train(
        data,
        version=args.version,
        model_dir=args.model_dir,
        n_trials=args.trials,
        tune=not args.no_tune,
    )

    logger.info("Saved model %s", metadata.version)
    print(json.dumps(metadata.metrics, indent=2))


if __name__ == "__main__":
    main()
