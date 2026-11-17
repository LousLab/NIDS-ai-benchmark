from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    train_seconds: float,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional label arrays.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate predictions on an empty test set.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of labels.")
    if not class_names:
        raise ValueError("At least one class name is required for evaluation.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique.")
    if not math.isfinite(train_seconds) or train_seconds < 0:
        raise ValueError("train_seconds must be finite and non-negative.")
    labels = list(range(len(class_names)))
    valid_labels = set(labels)
    if not set(np.unique(y_true)).issubset(valid_labels) or not set(
        np.unique(y_pred)
    ).issubset(valid_labels):
        raise ValueError("Evaluation labels must be encoded as class-name indices.")
    present_labels = sorted(np.unique(y_true).tolist())
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    supported_scores = [report[class_names[i]] for i in present_labels]
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean([score["precision"] for score in supported_scores])),
        "recall_macro": float(np.mean([score["recall"] for score in supported_scores])),
        # Primary macro-F1 averages only classes with test support. Keep a
        # second all-label score so missing classes are visible in stress tests.
        "f1_macro": float(np.mean([score["f1-score"] for score in supported_scores])),
        "f1_macro_all_labels": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "train_seconds": float(train_seconds),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "classification_report": report,
        "n_test": int(len(y_true)),
        "test_supported_classes": [class_names[i] for i in present_labels],
        "test_unsupported_classes": [
            class_names[i] for i in labels if i not in present_labels
        ],
    }


def save_run(output_dir: Path, summary: dict, metrics: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for item in metrics:
        rows.append(
            {
                "model": item["model"],
                "accuracy": item["accuracy"],
                "balanced_accuracy": item["balanced_accuracy"],
                "precision_macro": item["precision_macro"],
                "recall_macro": item["recall_macro"],
                "f1_macro": item["f1_macro"],
                "f1_macro_all_labels": item["f1_macro_all_labels"],
                "f1_weighted": item["f1_weighted"],
                "mcc": item["mcc"],
                "train_seconds": item["train_seconds"],
                "n_test": item["n_test"],
                "test_supported_classes": "|".join(item["test_supported_classes"]),
            }
        )
        (output_dir / f"{item['model']}_report.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )
    metric_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_macro_all_labels",
        "f1_weighted",
        "mcc",
        "train_seconds",
        "n_test",
        "test_supported_classes",
    ]
    pd.DataFrame(rows, columns=metric_columns).to_csv(
        output_dir / "metrics.csv", index=False
    )
