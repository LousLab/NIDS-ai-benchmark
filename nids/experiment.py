from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nids.config import ExperimentConfig
from nids.datasets import load_dataset, make_smoke_frame
from nids.evaluate import evaluate_predictions, save_run
from nids.imbalance import maybe_smote
from nids.models import TRAINERS
from nids.plots import (
    plot_class_distribution,
    plot_confusion,
    plot_metrics,
    plot_per_class_f1,
    plot_split_class_distribution,
    plot_training_time,
    plot_training_loss,
)
from nids.preprocess import (
    TrainOnlyPreprocessor,
    coerce_features,
    drop_duplicates,
    group_duplicate_features,
)
from nids.split import (
    apply_split,
    assert_disjoint_splits,
    stratified_group_train_val_test,
    time_train_val_test,
)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _cap_training_indices(
    parts: dict[str, pd.Index], y: pd.Series, cfg: ExperimentConfig
) -> tuple[dict[str, pd.Index], dict]:
    """Apply configured compute caps only after creating the full-data split."""
    original = parts["train"].to_numpy()
    if cfg.split_mode == "time":
        counts = y.iloc[original].value_counts().to_dict()
        return parts, {
            "train_rows_before_cap": int(len(original)),
            "train_rows_after_cap": int(len(original)),
            "train_class_counts_before_cap": counts,
            "train_class_counts_after_cap": counts,
            "caps_applied_to": "none; chronological runs use all training rows",
        }
    selected = []
    rng = np.random.default_rng(cfg.split_seed)
    for label in sorted(y.iloc[original].unique().tolist()):
        class_idx = original[y.iloc[original].to_numpy() == label]
        if cfg.max_per_class is not None and len(class_idx) > cfg.max_per_class:
            class_idx = rng.choice(class_idx, size=cfg.max_per_class, replace=False)
        selected.extend(class_idx.tolist())
    selected = np.asarray(selected, dtype=int)
    if cfg.max_total_rows is not None and len(selected) > cfg.max_total_rows:
        if cfg.max_total_rows < y.iloc[original].nunique():
            raise ValueError("max_total_rows must be at least the number of training classes.")
        selected, _ = train_test_split(
            selected,
            train_size=cfg.max_total_rows,
            random_state=cfg.split_seed,
            stratify=y.iloc[selected],
        )
    capped = dict(parts)
    capped["train"] = pd.Index(np.sort(selected))
    metadata = {
        "train_rows_before_cap": int(len(original)),
        "train_rows_after_cap": int(len(selected)),
        "train_class_counts_before_cap": y.iloc[original].value_counts().to_dict(),
        "train_class_counts_after_cap": y.iloc[selected].value_counts().to_dict(),
        "caps_applied_to": "training partition only",
    }
    return capped, metadata


def _split_class_counts(split: dict) -> dict[str, dict[str, int]]:
    return {
        name: split[f"y_{name}"].value_counts().sort_index().astype(int).to_dict()
        for name in ("train", "val", "test")
    }


def run_experiment(cfg: ExperimentConfig, smoke_frame=None) -> Path:
    if not (
        0 < cfg.test_size < 1
        and 0 < cfg.val_size < 1
        and cfg.test_size + cfg.val_size < 1
    ):
        raise ValueError(
            "test_size and val_size must be positive fractions whose sum is below 1."
        )
    if not cfg.models:
        raise ValueError("At least one model must be selected.")
    if len(set(cfg.models)) != len(cfg.models):
        raise ValueError("Model selections must not contain duplicates.")
    unknown_models = sorted(set(cfg.models) - TRAINERS.keys())
    if unknown_models:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown_models)}. "
            f"Choose from {', '.join(sorted(TRAINERS))}."
        )
    if smoke_frame is not None:
        df = smoke_frame.copy()
        meta = {"path": "synthetic_smoke", "class_counts": df["label"].value_counts().to_dict()}
        if "id" in df.columns:
            df = df.drop(columns=["id", "who CH"], errors="ignore")
    else:
        df, meta = load_dataset(cfg)

    x, y = coerce_features(df)
    time_col = next((c for c in x.columns if str(c).strip().lower() == "time"), None)
    if cfg.split_mode == "time" and time_col is None:
        raise ValueError("Time split requested, but the dataset has no numeric Time feature.")
    if cfg.split_mode == "time":
        order = np.argsort(x[time_col].to_numpy(dtype=float), kind="stable")
        x = x.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        x, y, duplicate_meta = drop_duplicates(
            x,
            y,
            return_metadata=True,
            exclude_columns=(time_col,) if time_col is not None else (),
        )
        time_values = x[time_col].copy()
        x = x.drop(columns=[time_col])
        meta["excluded_predictor_columns"] = [time_col]
        parts = time_train_val_test(x, y, time_values, cfg)
    else:
        x, y, groups, duplicate_meta = group_duplicate_features(
            x, y, exclude_columns=(time_col,) if time_col is not None else ()
        )
        if time_col is not None:
            x = x.drop(columns=[time_col])
            meta["excluded_predictor_columns"] = [time_col]
        parts = stratified_group_train_val_test(x, y, groups, cfg)
    meta["class_counts_after_cleaning"] = y.value_counts().to_dict()
    assert_disjoint_splits(parts, n_samples=len(y))
    full_split_sizes = {name: int(len(indices)) for name, indices in parts.items()}
    parts, train_cap_meta = _cap_training_indices(parts, y, cfg)
    assert_disjoint_splits(parts)
    split = apply_split(x, y, parts)
    split_class_counts = _split_class_counts(split)

    prep = TrainOnlyPreprocessor(cfg)
    prep.fit(split["x_train"], split["y_train"])

    arrays = {
        "x_train": prep.transform_x(split["x_train"]),
        "x_val": prep.transform_x(split["x_val"]),
        "x_test": prep.transform_x(split["x_test"]),
        "y_train": prep.transform_y(split["y_train"]),
        "y_val": prep.transform_y(split["y_val"]),
        "y_test": prep.transform_y(split["y_test"]),
    }

    train_size_before = int(len(arrays["y_train"]))
    if cfg.imbalance == "smote":
        arrays["x_train"], arrays["y_train"] = maybe_smote(
            arrays["x_train"], arrays["y_train"], cfg.seed
        )
    use_weights = cfg.imbalance == "class_weight"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = Path(cfg.output_dir) / f"{stamp}_{cfg.run_name()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(y, out_dir / "class_distribution.png")
    plot_split_class_distribution(
        split_class_counts, out_dir / "split_class_distribution.png"
    )
    metrics = []
    class_reports = {}
    notes = []
    for name in cfg.models:
        trainer = TRAINERS[name]
        try:
            pred, elapsed, model = trainer(cfg, arrays, use_weights)
        except Exception as exc:  # keep other models running
            notes.append({"model": name, "training_error": str(exc)})
            continue
        result = evaluate_predictions(
            name,
            arrays["y_test"],
            pred,
            prep.class_names(),
            elapsed,
        )
        metrics.append(result)
        class_reports[name] = result["classification_report"]
        try:
            plot_confusion(
                result["confusion_matrix"],
                prep.class_names(),
                out_dir / f"cm_{name}.png",
            )
            plot_training_loss(name, model, out_dir / f"loss_{name}.png")
        except Exception as exc:
            notes.append({"model": name, "visualization_error": str(exc)})

    test_dist = (
        split["y_test"].value_counts(normalize=True).round(4).to_dict()
    )
    summary = {
        "config": _jsonable(cfg.__dict__),
        "data": _jsonable(meta),
        "data_cleaning": _jsonable(duplicate_meta),
        "train_constant_features_dropped": prep.dropped_train_constant_features_,
        "n_rows_after_clean": int(len(y)),
        "n_features": int(arrays["x_train"].shape[1]),
        "selected_features": prep.selected_names_,
        "class_names": prep.class_names(),
        "split_seed": int(cfg.split_seed),
        "split_sizes": {
            **full_split_sizes,
            "train": train_size_before,
            "train_after_imbalance": int(len(arrays["y_train"])),
            "val": int(len(arrays["y_val"])),
            "test": int(len(arrays["y_test"])),
        },
        "split_class_counts": _jsonable(split_class_counts),
        "training_cap": _jsonable(train_cap_meta),
        "test_label_distribution": test_dist,
        "test_unsupported_classes": [
            name for name in prep.class_names() if name not in set(split["y_test"])
        ],
        "notes": notes,
        "primary_metric": "f1_macro",
        "leakage_controls": [
            "test indices disjoint from train/val",
            "imputer/scaler/feature selection fit on train only",
            "all-missing and constant feature screening fit on train only",
            "identical feature groups kept in one stratified partition",
            "conflicting labels for identical feature vectors excluded",
            "SMOTE applied to train only",
            "fixed model settings; no validation-based tuning in this baseline",
            "time feature removed from model inputs in chronological runs",
            "Time excluded from WSN-DS model inputs in both split modes",
            "reported metrics computed once on test",
        ],
        "model_selection_policy": (
            "Fixed documented model settings; no model-specific hyperparameter search. "
            "Validation data is retained for a future common tuning protocol."
        ),
    }
    save_run(out_dir, summary, metrics)
    if metrics:
        plot_metrics(out_dir / "metrics.csv", out_dir / "metric_bars.png")
        plot_per_class_f1(class_reports, out_dir / "per_class_f1.png")
        plot_training_time(out_dir / "metrics.csv", out_dir / "training_time.png")
    (out_dir / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return out_dir


def run_smoke(cfg: ExperimentConfig) -> Path:
    return run_experiment(cfg, smoke_frame=make_smoke_frame(cfg.seed))
