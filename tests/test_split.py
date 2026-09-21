"""Tests for the leakage guard.

The whole point of this module is preventing the failure mode documented
in the project's research notes: random splits of FIRMS detections let
the same physical fire land in both train and test, producing inflated
scores that collapse under honest evaluation.
"""
import random
from datetime import datetime, timedelta, timezone

import pytest

from classifier.model.split import (
    LeakageError,
    assign_spatial_tile,
    spatial_group_split,
    temporal_split,
    verify_no_leakage,
)


def _clustered_coordinates(n_per_site=10):
    """Several tight clusters of points — i.e. the same physical sites
    detected repeatedly, which is exactly what FIRMS data looks like."""
    sites = [(77.1, 28.7), (72.8, 19.0), (85.3, 23.7), (80.2, 26.8), (75.8, 30.9)]
    coordinates = []
    for lon, lat in sites:
        for i in range(n_per_site):
            coordinates.append((lon + i * 0.001, lat + i * 0.001))
    return coordinates


def test_assign_spatial_tile_groups_nearby_points():
    assert assign_spatial_tile(77.10, 28.70, 0.5) == assign_spatial_tile(77.12, 28.72, 0.5)


def test_assign_spatial_tile_separates_distant_points():
    assert assign_spatial_tile(77.1, 28.7, 0.5) != assign_spatial_tile(80.1, 20.7, 0.5)


def test_assign_spatial_tile_uses_floor_not_truncation():
    """int() truncates toward zero, merging tiles either side of zero."""
    assert assign_spatial_tile(-0.3, -0.3, 0.5) == (-1, -1)
    assert assign_spatial_tile(0.3, 0.3, 0.5) == (0, 0)
    assert assign_spatial_tile(-0.3, -0.3, 0.5) != assign_spatial_tile(0.3, 0.3, 0.5)


def test_spatial_split_keeps_whole_tiles_on_one_side():
    coordinates = _clustered_coordinates()
    split = spatial_group_split(coordinates, test_fraction=0.2, tile_degrees=0.5, seed=42)
    verify_no_leakage(split, coordinates, tile_degrees=0.5)  # must not raise


def test_spatial_split_covers_every_sample_exactly_once():
    coordinates = _clustered_coordinates()
    split = spatial_group_split(coordinates, test_fraction=0.2)
    combined = sorted(split.train_indices + split.test_indices)
    assert combined == list(range(len(coordinates)))


def test_spatial_split_is_deterministic_for_a_given_seed():
    coordinates = _clustered_coordinates()
    first = spatial_group_split(coordinates, seed=7)
    second = spatial_group_split(coordinates, seed=7)
    assert first.test_indices == second.test_indices


def test_a_random_split_would_leak_but_ours_does_not():
    """The contrast this module exists for."""
    coordinates = _clustered_coordinates()

    indices = list(range(len(coordinates)))
    random.Random(0).shuffle(indices)
    cutoff = int(len(indices) * 0.8)
    from classifier.model.split import SplitResult

    naive = SplitResult(
        train_indices=indices[:cutoff], test_indices=indices[cutoff:],
        train_tiles=set(), test_tiles=set(),
    )

    with pytest.raises(LeakageError):
        verify_no_leakage(naive, coordinates, tile_degrees=0.5)

    safe = spatial_group_split(coordinates, test_fraction=0.2, tile_degrees=0.5, seed=42)
    verify_no_leakage(safe, coordinates, tile_degrees=0.5)


def test_verify_no_leakage_catches_duplicated_indices():
    from classifier.model.split import SplitResult

    coordinates = _clustered_coordinates()
    overlapping = SplitResult(
        train_indices=[0, 1, 2], test_indices=[2, 3], train_tiles=set(), test_tiles=set()
    )
    with pytest.raises(LeakageError, match="both train and test"):
        verify_no_leakage(overlapping, coordinates)


def test_verify_no_leakage_catches_shared_group_ids():
    coordinates = [(77.1, 28.7), (90.0, 10.0)]
    group_ids = [42, 42]  # same cluster on both sides
    from classifier.model.split import SplitResult

    split = SplitResult(
        train_indices=[0], test_indices=[1], train_tiles=set(), test_tiles=set()
    )
    with pytest.raises(LeakageError, match="group id"):
        verify_no_leakage(split, coordinates, tile_degrees=0.5, group_ids=group_ids)


def test_spatial_split_rejects_invalid_fractions():
    coordinates = _clustered_coordinates()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            spatial_group_split(coordinates, test_fraction=bad)


def test_spatial_split_never_swallows_every_tile():
    coordinates = _clustered_coordinates()
    split = spatial_group_split(coordinates, test_fraction=0.99, tile_degrees=0.5)
    assert split.train_indices, "train set must not be empty"


def test_temporal_split_holds_out_the_most_recent_slice():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=i) for i in range(100)]
    split = temporal_split(timestamps, test_fraction=0.2)

    latest_train = max(timestamps[i] for i in split.train_indices)
    earliest_test = min(timestamps[i] for i in split.test_indices)
    assert latest_train < earliest_test
    assert len(split.test_indices) == 20
