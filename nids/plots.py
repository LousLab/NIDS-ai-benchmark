from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    cols = ["balanced_accuracy", "f1_macro", "mcc"]
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 4), sharex=True)
    for ax, col in zip(axes, cols):
        ax.bar(df["model"], df[col], color="#4C78A8")
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        if col == "mcc":
            ax.set_ylim(-1, 1)
        else:
            ax.set_ylim(0, 1)
    fig.suptitle("Held-out test metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    cm = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_class_distribution(y: pd.Series, output_path: Path) -> None:
    counts = y.value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Cleaned dataset class distribution")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_split_class_distribution(
    split_counts: dict[str, dict[str, int]], output_path: Path
) -> None:
    frame = pd.DataFrame(split_counts).fillna(0)
    proportions = frame.div(frame.sum(axis=0), axis=1)
    ax = proportions.plot(kind="bar", figsize=(9, 4.5), rot=35)
    ax.set_title("Class prevalence by split")
    ax.set_ylabel("Share of split")
    ax.set_xlabel("Class")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_training_time(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    ax = df.set_index("model")["train_seconds"].plot(
        kind="bar", figsize=(7, 4), color="#F58518", rot=0
    )
    ax.set_title("Model training time")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_per_class_f1(reports: dict[str, dict], output_path: Path) -> None:
    rows = {
        model: {name: scores["f1-score"] for name, scores in report.items()
                if name not in {"accuracy", "macro avg", "weighted avg"}}
        for model, report in reports.items()
    }
    frame = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.65 * len(frame))))
    im = ax.imshow(frame.to_numpy(), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    ax.set_title("Per-class F1 by model")
    for i in range(len(frame.index)):
        for j in range(len(frame.columns)):
            ax.text(j, i, f"{frame.iloc[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_training_loss(model_name: str, model, output_path: Path) -> None:
    values = getattr(model, "loss_curve_", None)
    if values is None:
        history = getattr(model, "history", None)
        if isinstance(history, dict):
            values = history.get("loss")
        elif history is not None:
            nested = getattr(history, "history", None)
            if isinstance(nested, dict):
                values = nested.get("loss")
            elif hasattr(history, "__getitem__"):
                try:
                    values = history["loss"]
                except (KeyError, TypeError, AttributeError):
                    values = None
    if values is None or len(values) == 0:
        return
    values = np.asarray(values, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(values) + 1), values, color="#F58518")
    ax.set(title=f"{model_name.upper()} training loss", xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
