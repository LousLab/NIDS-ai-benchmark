from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nids.config import ExperimentConfig


def stratified_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    cfg: ExperimentConfig,
) -> dict[str, pd.Index]:
    """Split into train/val/test. Test is isolated from fitting and model selection."""
    idx = np.arange(len(x))
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError("Stratified splitting requires at least 3 examples per class.")
    split_seed = getattr(cfg, "split_seed", cfg.seed)
    idx_tv, idx_test = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=split_seed,
        stratify=y,
    )
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=relative_val,
        random_state=split_seed,
        stratify=y.iloc[idx_tv],
    )
    return {
        "train": pd.Index(idx_train),
        "val": pd.Index(idx_val),
        "test": pd.Index(idx_test),
    }


def stratified_group_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cfg: ExperimentConfig,
) -> dict[str, pd.Index]:
    """Stratify at label level while keeping identical feature rows in one split.

    Groups are shuffled within each class, then assigned by cumulative row count
    to approximate the configured train/validation/test proportions.
    """
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("x, y, and groups must contain the same number of rows.")
    labels = y.to_numpy()
    group_keys, row_group = np.unique(groups.to_numpy(), return_inverse=True)
    group_sizes = np.bincount(row_group)
    first_row = np.full(len(group_keys), len(labels), dtype=np.int64)
    np.minimum.at(first_row, row_group, np.arange(len(labels)))
    group_labels = labels[first_row]
    if not np.array_equal(labels, group_labels[row_group]):
        raise ValueError("Every feature group must contain one consistent label.")

    train_fraction = 1.0 - cfg.val_size - cfg.test_size
    if train_fraction <= 0 or cfg.val_size <= 0 or cfg.test_size <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive.")
    split_by_group = np.full(len(group_keys), -1, dtype=np.int8)
    rng = np.random.default_rng(getattr(cfg, "split_seed", cfg.seed))
    for label in np.unique(group_labels):
        class_groups = np.flatnonzero(group_labels == label)
        if len(class_groups) < 3:
            raise ValueError(
                f"Class {label!r} needs at least 3 distinct feature groups for splitting."
            )
        rng.shuffle(class_groups)
        cumulative = np.cumsum(group_sizes[class_groups])
        class_total = int(cumulative[-1])
        train_target = class_total * train_fraction
        val_end_target = class_total * (1.0 - cfg.test_size)
        first_candidates = np.arange(1, len(class_groups) - 1)
        first_cut = int(
            first_candidates[
                np.argmin(np.abs(cumulative[first_candidates - 1] - train_target))
            ]
        )
        second_candidates = np.arange(first_cut + 1, len(class_groups))
        second_cut = int(
            second_candidates[
                np.argmin(
                    np.abs(cumulative[second_candidates - 1] - val_end_target)
                )
            ]
        )
        split_by_group[class_groups[:first_cut]] = 0
        split_by_group[class_groups[first_cut:second_cut]] = 1
        split_by_group[class_groups[second_cut:]] = 2

    row_split = split_by_group[row_group]
    return {
        "train": pd.Index(np.flatnonzero(row_split == 0)),
        "val": pd.Index(np.flatnonzero(row_split == 1)),
        "test": pd.Index(np.flatnonzero(row_split == 2)),
    }


def time_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    time_values: pd.Series,
    cfg: ExperimentConfig,
) -> dict[str, pd.Index]:
    """Split chronologically at whole time values so ties cannot cross partitions."""
    values = pd.to_numeric(time_values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Time split requires finite numeric time values.")
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    n = len(values)
    if len(unique_values) < 3:
        raise ValueError("Time split requires at least 3 distinct time values.")
    cumulative = np.cumsum(counts)
    first_target = n * (1.0 - cfg.val_size - cfg.test_size)
    second_target = n * (1.0 - cfg.test_size)
    first_cut = int(np.argmin(np.abs(cumulative[:-2] - first_target))) + 1
    remaining_cuts = np.arange(first_cut + 1, len(unique_values))
    if len(remaining_cuts) == 0:
        raise ValueError("Unable to construct non-empty chronological splits.")
    second_cut = int(
        remaining_cuts[np.argmin(np.abs(cumulative[remaining_cuts - 1] - second_target))]
    )
    row_time_group = inverse
    train = np.flatnonzero(row_time_group < first_cut)
    val = np.flatnonzero((row_time_group >= first_cut) & (row_time_group < second_cut))
    test = np.flatnonzero(row_time_group >= second_cut)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return {
        "train": pd.Index(train),
        "val": pd.Index(val),
        "test": pd.Index(test),
    }


def apply_split(x: pd.DataFrame, y: pd.Series, parts: dict[str, pd.Index]) -> dict:
    out = {}
    for name, idx in parts.items():
        out[f"x_{name}"] = x.iloc[idx].reset_index(drop=True)
        out[f"y_{name}"] = y.iloc[idx].reset_index(drop=True)
    out["indices"] = {k: v.to_numpy() for k, v in parts.items()}
    return out


def assert_disjoint_splits(
    parts: dict[str, pd.Index], n_samples: int | None = None
) -> None:
    required = {"train", "val", "test"}
    if set(parts) != required:
        raise AssertionError(f"Expected split keys {sorted(required)}, got {sorted(parts)}.")
    sets = {k: set(v.to_numpy()) for k, v in parts.items()}
    for name, values in parts.items():
        if len(sets[name]) != len(values):
            raise AssertionError(f"{name} split contains repeated indices.")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("Train/val/test indices overlap.")
    if n_samples is not None:
        if any(index < 0 or index >= n_samples for values in sets.values() for index in values):
            raise AssertionError("Split contains an out-of-range index.")
        if set.union(*sets.values()) != set(range(n_samples)):
            raise AssertionError("Train/val/test do not cover every sample exactly once.")
