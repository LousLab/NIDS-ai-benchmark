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
    precision_score,
    recall_score,
)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    train_seconds: float,
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "train_seconds": float(train_seconds),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "classification_report": report,
        "n_test": int(len(y_true)),
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
                "f1_weighted": item["f1_weighted"],
                "mcc": item["mcc"],
                "train_seconds": item["train_seconds"],
            }
        )
        (output_dir / f"{item['model']}_report.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
