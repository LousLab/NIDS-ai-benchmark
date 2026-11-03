from __future__ import annotations

import json
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
    labels = list(range(len(class_names)))
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
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
