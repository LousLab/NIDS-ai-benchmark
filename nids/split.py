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
    idx_tv, idx_test = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=relative_val,
        random_state=cfg.seed,
        stratify=y.iloc[idx_tv],
    )
    return {
        "train": pd.Index(idx_train),
        "val": pd.Index(idx_val),
        "test": pd.Index(idx_test),
    }


def time_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    time_values: pd.Series,
    cfg: ExperimentConfig,
) -> dict[str, pd.Index]:
    """Global chronological split. Can hide whole attack days in one split."""
    order = np.argsort(pd.to_numeric(time_values, errors="coerce").fillna(-1).to_numpy())
    n = len(order)
    n_test = int(round(n * cfg.test_size))
    n_val = int(round(n * cfg.val_size))
    n_train = n - n_val - n_test
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
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


def assert_disjoint_splits(parts: dict[str, pd.Index]) -> None:
    sets = {k: set(v.to_numpy()) for k, v in parts.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("Train/val/test indices overlap.")
