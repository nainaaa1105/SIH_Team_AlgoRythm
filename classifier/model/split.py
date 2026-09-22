"""Leakage-safe train/test splitting.

This module exists because of a specific, documented failure mode. From
the project's research notes on "Spatial Leakage in Classifying NASA
FIRMS Thermal Anomalies as Wildfire Incidents": when individual FIRMS
detections are split randomly, detections from the *same physical fire*
land in both train and test, so the model memorises locations and scores
far higher than it deserves. The paper's headline random-split numbers
dropped substantially under event-aware and spatially-controlled
evaluation.

Our unit is already a DBSCAN cluster (one physical event), which fixes
the crudest version of that. But it is not sufficient on its own: two
clusters 800m apart in the same refinery are effectively the same
subject, and a wildfire that M1's pipeline split into two adjacent
clusters would still leak across the boundary. So we group by a spatial
tile and keep whole tiles on one side of the split.

`verify_no_leakage` is the guard rail — training refuses to proceed if
it fails.
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Hashable, List, Optional, Sequence, Tuple


@dataclass
class SplitResult:
    train_indices: List[int]
    test_indices: List[int]
    train_tiles: set
    test_tiles: set

    def __post_init__(self):
        self.train_indices = list(self.train_indices)
        self.test_indices = list(self.test_indices)


class LeakageError(AssertionError):
    """Raised when a split would place related samples on both sides."""


def assign_spatial_tile(lon: float, lat: float, tile_degrees: float = 0.5) -> Tuple[int, int]:
    """Map a coordinate to a coarse grid cell.

    math.floor (not int()) matters here: int() truncates toward zero, so
    -0.3 and +0.3 would both land in tile 0, wrongly merging cells either
    side of the equator/prime meridian. India is entirely in the positive
    quadrant so it wouldn't bite us in practice, but a silently wrong
    helper is not worth keeping.
    """
    return (math.floor(lon / tile_degrees), math.floor(lat / tile_degrees))


def spatial_group_split(
    coordinates: Sequence[Tuple[float, float]],
    test_fraction: float = 0.2,
    tile_degrees: float = 0.5,
    seed: int = 42,
    labels: Optional[Sequence[str]] = None,
) -> SplitResult:
    """Split by spatial tile so no tile appears in both train and test.

    Tiles are assigned to the test set greedily in a seeded-shuffled
    order until the target test fraction is reached, which keeps whole
    tiles intact. When `labels` are supplied, tiles are shuffled such
    that rarer classes are distributed rather than concentrated, but
    tile integrity always wins over class balance — a slightly imbalanced
    split is a real cost, a leaky split is a fake result.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    tile_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (lon, lat) in enumerate(coordinates):
        tile_to_indices[assign_spatial_tile(lon, lat, tile_degrees)].append(idx)

    import random

    rng = random.Random(seed)
    tiles = sorted(tile_to_indices.keys())  # sorted first => deterministic regardless of dict order
    rng.shuffle(tiles)

    target_test_count = max(1, round(len(coordinates) * test_fraction))
    test_tiles: set = set()
    test_count = 0

    for tile in tiles:
        if test_count >= target_test_count:
            break
        test_tiles.add(tile)
        test_count += len(tile_to_indices[tile])

    # Guard the degenerate case where one huge tile swallows everything.
    if len(test_tiles) == len(tiles) and len(tiles) > 1:
        test_tiles.remove(sorted(test_tiles)[0])

    train_tiles = set(tiles) - test_tiles

    train_indices = [i for tile in train_tiles for i in tile_to_indices[tile]]
    test_indices = [i for tile in test_tiles for i in tile_to_indices[tile]]

    return SplitResult(
        train_indices=sorted(train_indices),
        test_indices=sorted(test_indices),
        train_tiles=train_tiles,
        test_tiles=test_tiles,
    )


def temporal_split(
    timestamps: Sequence[datetime], test_fraction: float = 0.2
) -> SplitResult:
    """Hold out the most recent slice of time.

    Complements the spatial split: it is the honest way to check that
    temporal features only use information that would actually have been
    available at prediction time, which the research notes call out
    explicitly.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    cutoff = len(order) - max(1, round(len(order) * test_fraction))
    return SplitResult(
        train_indices=sorted(order[:cutoff]),
        test_indices=sorted(order[cutoff:]),
        train_tiles=set(),
        test_tiles=set(),
    )


def verify_no_leakage(
    split: SplitResult,
    coordinates: Sequence[Tuple[float, float]],
    tile_degrees: float = 0.5,
    group_ids: Optional[Sequence[Hashable]] = None,
) -> None:
    """Fail loudly if train and test share a tile, an index, or a group id.

    Called by `train.py` before fitting anything. If this raises, the
    experiment is invalid — do not "fix" it by relaxing the check.
    """
    overlap_indices = set(split.train_indices) & set(split.test_indices)
    if overlap_indices:
        raise LeakageError(f"{len(overlap_indices)} sample(s) appear in both train and test")

    train_tiles = {assign_spatial_tile(*coordinates[i], tile_degrees) for i in split.train_indices}
    test_tiles = {assign_spatial_tile(*coordinates[i], tile_degrees) for i in split.test_indices}
    shared_tiles = train_tiles & test_tiles
    if shared_tiles:
        raise LeakageError(
            f"{len(shared_tiles)} spatial tile(s) span train and test: {sorted(shared_tiles)[:5]} — "
            "this is the exact leakage mode that inflates FIRMS classification scores"
        )

    if group_ids is not None:
        train_groups = {group_ids[i] for i in split.train_indices}
        test_groups = {group_ids[i] for i in split.test_indices}
        shared_groups = train_groups & test_groups
        if shared_groups:
            raise LeakageError(f"{len(shared_groups)} group id(s) span train and test")
