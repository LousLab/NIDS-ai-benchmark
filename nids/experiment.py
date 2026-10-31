from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    plot_training_loss,
)
from nids.preprocess import TrainOnlyPreprocessor, coerce_features, drop_duplicates
from nids.split import (
    apply_split,
    assert_disjoint_splits,
    stratified_train_val_test,
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


def run_experiment(cfg: ExperimentConfig, smoke_frame=None) -> Path:
    if cfg.split_mode == "time":
        # Preserve chronology: random class caps would destroy the original ordering.
        cfg.max_per_class = None
        cfg.max_total_rows = None
    if smoke_frame is not None:
        df = smoke_frame.copy()
        meta = {"path": "synthetic_smoke", "class_counts": df["label"].value_counts().to_dict()}
        if "id" in df.columns:
            df = df.drop(columns=["id", "who CH"], errors="ignore")
    else:
        df, meta = load_dataset(cfg)

    x, y = coerce_features(df)
    x, y = drop_duplicates(x, y)

    if cfg.split_mode == "time":
        if "Time" not in x.columns:
            raise ValueError("Time split requested, but the dataset has no numeric Time feature.")
        parts = time_train_val_test(x, y, x["Time"], cfg)
    else:
        parts = stratified_train_val_test(x, y, cfg)
    assert_disjoint_splits(parts)
    split = apply_split(x, y, parts)

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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(cfg.output_dir) / f"{stamp}_{cfg.run_name()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(y, out_dir / "class_distribution.png")
    metrics = []
    class_reports = {}
    notes = []
    for name in cfg.models:
        trainer = TRAINERS[name]
        try:
            pred, elapsed, model = trainer(cfg, arrays, use_weights)
            result = evaluate_predictions(
                name,
                arrays["y_test"],
                pred,
                prep.class_names(),
                elapsed,
            )
            metrics.append(result)
            class_reports[name] = result["classification_report"]
            plot_confusion(
                result["confusion_matrix"],
                prep.class_names(),
                out_dir / f"cm_{name}.png",
            )
            plot_training_loss(name, model, out_dir / f"loss_{name}.png")
        except Exception as exc:  # keep other models running
            notes.append({"model": name, "skipped": True, "reason": str(exc)})

    test_dist = (
        split["y_test"].value_counts(normalize=True).round(4).to_dict()
    )
    summary = {
        "config": _jsonable(cfg.__dict__),
        "data": _jsonable(meta),
        "n_rows_after_clean": int(len(y)),
        "n_features": int(arrays["x_train"].shape[1]),
        "selected_features": prep.selected_names_,
        "class_names": prep.class_names(),
        "split_sizes": {
            "train": train_size_before,
            "train_after_imbalance": int(len(arrays["y_train"])),
            "val": int(len(arrays["y_val"])),
            "test": int(len(arrays["y_test"])),
        },
        "test_label_distribution": test_dist,
        "notes": notes,
        "primary_metric": "f1_macro",
        "leakage_controls": [
            "test indices disjoint from train/val",
            "imputer/scaler/feature selection fit on train only",
            "SMOTE applied to train only",
            "early stopping uses val, never test",
            "reported metrics computed once on test",
        ],
    }
    save_run(out_dir, summary, metrics)
    if metrics:
        plot_metrics(out_dir / "metrics.csv", out_dir / "metric_bars.png")
        plot_per_class_f1(class_reports, out_dir / "per_class_f1.png")
    (out_dir / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return out_dir


def run_smoke(cfg: ExperimentConfig) -> Path:
    return run_experiment(cfg, smoke_frame=make_smoke_frame(cfg.seed))
