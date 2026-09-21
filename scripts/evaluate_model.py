"""Full evaluation report (per-class precision/recall/F1 + confusion
matrix) for an already-trained, already-saved model — no retraining.

Reuses ingest_and_train.build_dataset() to reproduce the exact same
feature matrix and train/test split the model was actually trained on
(same CSVs, same deterministic clustering, same real facility/land-cover/
population lookups — cached, so this is fast on a second run — same
spatial-tile split seed). Loads the saved model.json and evaluates it on
that reconstructed test set, rather than trusting only the single
macro-F1 number training already logged.

Usage:
    python -m scripts.evaluate_model --version v4
"""
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")

from scripts.ingest_and_train import build_arg_parser, build_dataset


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument(
        "--model-version", default=None,
        help="Which saved model to evaluate (defaults to --version, i.e. the model "
             "this same dataset build would train/overwrite).",
    )
    args = parser.parse_args()
    model_version = args.model_version or args.version

    training_data, sample_weights, split, settings = build_dataset(args)

    from classifier.model.registry import load_metadata, load_model

    model = load_model(version=model_version)
    metadata = load_metadata(version=model_version)
    logger.info("Evaluating model version=%s trained_at=%s", metadata.version, metadata.trained_at)

    if metadata.feature_columns != training_data.feature_columns:
        raise SystemExit(
            "Feature columns on disk don't match this dataset build's columns — "
            "evaluating would silently compare the wrong things. Re-run "
            "ingest_and_train first, or check FEATURE_COLUMNS hasn't changed "
            "since this model was trained."
        )

    X_te = training_data.X[split.test_indices]
    y_te = training_data.y[split.test_indices]

    y_pred = model.predict(X_te)

    # y is encoded to the classes actually present in TRAINING (see
    # encode_labels in classifier/model/train.py) — evaluate against
    # that same encoding, not the full canonical 5-class list, or a
    # class absent from training would print a spurious all-zero row.
    present_classes = training_data.class_names

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    report = classification_report(
        y_te, y_pred, labels=list(range(len(present_classes))),
        target_names=present_classes, digits=3, zero_division=0,
    )
    logger.info("Per-class precision / recall / F1 (test set, n=%d):\n%s", len(y_te), report)

    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(present_classes))))
    header = "true \\ pred".ljust(22) + "".join(c[:12].rjust(14) for c in present_classes)
    lines = [header]
    for i, row in enumerate(cm):
        lines.append(present_classes[i].ljust(22) + "".join(str(v).rjust(14) for v in row))
    logger.info("Confusion matrix (rows=true, cols=predicted):\n%s", "\n".join(lines))

    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    logger.info("macro-F1=%.4f weighted-F1=%.4f (metadata says macro_f1=%.4f)",
                macro_f1, weighted_f1, metadata.metrics.get("macro_f1", float("nan")))


if __name__ == "__main__":
    main()
