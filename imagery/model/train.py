"""Fine-tune EfficientNet-B0 on labelled Sentinel-2 patches.

Two decisions inherited deliberately from M2 rather than reinvented:

  * **The split comes from `classifier.model.split.spatial_group_split`.**
    Writing a second splitter would risk splitting patches randomly,
    which is exactly the spatial-leakage failure the project's research
    notes document: the same physical fire appearing in both train and
    test inflates the score and it collapses under honest evaluation.
    `verify_no_leakage` runs before training, same as M2.
  * **Labels come from M2's `training_labels` table**, so the image model
    and the tabular model are trained against the same ground truth and
    their probabilities are directly fusable.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from imagery.config import get_m4_settings
from imagery.model.registry import ImageModelMetadata, new_metadata, save_model

logger = logging.getLogger(__name__)


@dataclass
class PatchDataset:
    stacks: List[np.ndarray]          # each (C, H, W) reflectance
    labels: List[str]
    coordinates: List[Tuple[float, float]]
    cluster_ids: List[int]
    band_order: List[str]

    def __len__(self) -> int:
        return len(self.stacks)


def encode_labels(labels: Sequence[str], classes: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Contiguous integer encoding over the classes actually present.

    Same constraint M2 hit: indexing into the full five-class tuple gives
    non-contiguous indices like [0, 2, 4] when only three classes are
    labelled, which the loss function cannot use. Encode over the present
    classes, in canonical order, and carry that list into the metadata so
    inference maps outputs back correctly.
    """
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"Unknown label(s) {unknown}; expected one of {list(classes)}")

    observed = set(labels)
    present = [c for c in classes if c in observed]
    index = {c: i for i, c in enumerate(present)}
    return np.array([index[label] for label in labels], dtype=int), present


def train(
    dataset: PatchDataset,
    version: str = "v1",
    model_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    pretrained: bool = True,
) -> Tuple[Any, ImageModelMetadata]:
    """Full training run: split -> leakage guard -> fine-tune -> evaluate -> save."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from classifier.features.schema import CLASSES
    from classifier.model.split import spatial_group_split, verify_no_leakage

    from imagery.model.efficientnet import build_model, normalise_stack

    settings = get_m4_settings()
    model_dir = model_dir or settings.model_dir
    epochs = settings.epochs if epochs is None else epochs

    if len(dataset) == 0:
        raise ValueError("No labelled patches to train on")

    y, class_names = encode_labels(dataset.labels, CLASSES)
    if len(class_names) < 2:
        raise ValueError(
            f"Only one class present ({class_names}) — a classifier needs at least two."
        )

    # M2's splitter, and M2's leakage guard. Do not substitute a random split.
    split = spatial_group_split(
        dataset.coordinates,
        test_fraction=0.2,
        tile_degrees=0.5,
        seed=42,
    )
    verify_no_leakage(split, dataset.coordinates, tile_degrees=0.5, group_ids=dataset.cluster_ids)
    logger.info(
        "Spatial split: %d train / %d test patches across %d/%d tiles",
        len(split.train_indices), len(split.test_indices),
        len(split.train_tiles), len(split.test_tiles),
    )

    stacks = torch.stack([
        normalise_stack(s, settings.image_size_px).squeeze(0) for s in dataset.stacks
    ])
    targets = torch.as_tensor(y, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(stacks[split.train_indices], targets[split.train_indices]),
        batch_size=settings.batch_size, shuffle=True,
    )

    model = build_model(
        n_classes=len(class_names), in_channels=len(dataset.band_order), pretrained=pretrained
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total_loss += float(loss)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, total_loss / max(len(train_loader), 1))

    metrics = evaluate(model, stacks[split.test_indices], targets[split.test_indices], class_names)
    logger.info("Test macro-F1=%.4f", metrics["macro_f1"])

    metadata = new_metadata(
        version=version,
        band_order=dataset.band_order,
        classes=class_names,
        image_size_px=settings.image_size_px,
        metrics=metrics,
        hyperparameters={
            "epochs": epochs, "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate, "pretrained": pretrained,
        },
        n_train_samples=len(split.train_indices),
        n_test_samples=len(split.test_indices),
        notes=(
            "Spatial-tile split with M2's leakage verification. Trained on weakly-"
            "supervised labels from training_labels, so metrics inherit those biases."
        ),
    )
    save_model(model.state_dict(), metadata, model_dir)
    return model, metadata


def evaluate(model, x_test, y_test, class_names: List[str]) -> Dict[str, Any]:
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    if len(x_test) == 0:
        return {"macro_f1": 0.0, "note": "empty test split"}

    model.eval()
    with torch.no_grad():
        predictions = model(x_test).argmax(dim=1).cpu().numpy()
    truth = y_test.cpu().numpy()

    present = sorted(set(truth.tolist()) | set(predictions.tolist()))
    report = classification_report(
        truth, predictions, labels=present,
        target_names=[class_names[i] for i in present],
        output_dict=True, zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "per_class": {
            class_names[i]: report[class_names[i]] for i in present if class_names[i] in report
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=present).tolist(),
        "confusion_matrix_labels": [class_names[i] for i in present],
    }
