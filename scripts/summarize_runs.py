#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "f1_macro",
    "f1_macro_all_labels",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_weighted",
    "mcc",
    "train_seconds",
]


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Expected summary.json and metrics.csv in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("config"), dict):
        raise ValueError(f"Run summary has no valid config object: {summary_path}")
    if not {"seed", "models"}.issubset(summary["config"]):
        raise ValueError(f"Run summary is missing seed or model settings: {summary_path}")
    metrics = pd.read_csv(metrics_path)
    required_columns = {"model", *METRICS}
    missing_columns = sorted(required_columns - set(metrics.columns))
    if missing_columns:
        raise ValueError(
            f"Metrics file is missing required columns {missing_columns}: {metrics_path}"
        )
    metrics["seed"] = int(summary["config"]["seed"])
    return summary, metrics


def _validate_runs(runs: list[tuple[dict, pd.DataFrame]]) -> None:
    if len(runs) < 2:
        raise ValueError("At least two seed runs are required to report variability.")
    first_summary, first_metrics = runs[0]
    reference = first_summary
    reference_models = set(first_metrics["model"])
    configured_models = set(reference["config"].get("models", []))
    if not reference_models:
        raise ValueError("Run has no successful model metrics to aggregate.")
    if reference_models != configured_models:
        raise ValueError(
            "Run metrics do not include every configured model; "
            "resolve model errors before aggregating seeds."
        )
    reference_config = {
        key: value
        for key, value in reference["config"].items()
        if key not in {"seed", "output_dir"}
    }
    seen_seeds = set()
    for summary, metrics in runs:
        config = summary["config"]
        comparable_config = {
            key: value for key, value in config.items() if key not in {"seed", "output_dir"}
        }
        comparable = (
            comparable_config == reference_config
            and summary["split_seed"] == reference["split_seed"]
            and summary["split_class_counts"] == reference["split_class_counts"]
            and summary["training_cap"]["train_class_counts_after_cap"]
            == reference["training_cap"]["train_class_counts_after_cap"]
        )
        if not comparable:
            raise ValueError("Runs do not share the same dataset, split, and training sample.")
        seed = int(config["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate model seed {seed}.")
        seen_seeds.add(seed)
        if set(metrics["model"]) != reference_models:
            raise ValueError("Every seed run must contain the same set of successful models.")


def _aggregate_per_class(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        seed = int(summary["config"]["seed"])
        for report_path in run_dir.glob("*_report.json"):
            report_file = json.loads(report_path.read_text(encoding="utf-8"))
            model = report_file["model"]
            report = report_file["classification_report"]
            for class_name in summary["class_names"]:
                score = report[class_name]
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "class": class_name,
                        "support": score["support"],
                        "precision": score["precision"],
                        "recall": score["recall"],
                        "f1": score["f1-score"],
                    }
                )
    return pd.DataFrame(rows)


def _plot_summary(aggregate: pd.DataFrame, class_summary: pd.DataFrame, out: Path) -> None:
    models = aggregate["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(
        models,
        aggregate["f1_macro_mean"],
        yerr=aggregate["f1_macro_sd"],
        capsize=4,
        color="#4C78A8",
    )
    axes[0].set_title("Test macro-F1 across model seeds")
    axes[0].set_ylabel("Mean ± sample SD")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        models,
        aggregate["train_seconds_mean"],
        yerr=aggregate["train_seconds_sd"],
        capsize=4,
        color="#F58518",
    )
    axes[1].set_title("Training time across model seeds")
    axes[1].set_ylabel("Seconds, mean ± sample SD")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "seed_variability.png", dpi=150)
    plt.close(fig)

    pivot = class_summary.pivot(index="model", columns="class", values="f1_mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(pivot.to_numpy(), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Mean per-class F1 across model seeds")
    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(out / "mean_per_class_f1.png", dpi=150)
    plt.close(fig)


def summarize(run_dirs: list[Path], output_dir: Path) -> None:
    resolved = [path.resolve() for path in run_dirs]
    runs = [_load_run(path) for path in resolved]
    _validate_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat([metrics for _, metrics in runs], ignore_index=True)
    summary_rows = []
    for model, group in raw.groupby("model", sort=True):
        row = {"model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    aggregate = pd.DataFrame(summary_rows)
    aggregate.to_csv(output_dir / "seed_summary.csv", index=False)

    per_class = _aggregate_per_class(resolved)
    class_summary = (
        per_class.groupby(["model", "class"], sort=True)
        .agg(
            support_mean=("support", "mean"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
        )
        .reset_index()
    )
    class_summary.to_csv(output_dir / "per_class_seed_summary.csv", index=False)
    _plot_summary(aggregate, class_summary, output_dir)
    print(f"Saved aggregate results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated runs that share the same holdout and training sample."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/stratified_seed_analysis")
    )
    args = parser.parse_args()
    summarize(args.run_dirs, args.output_dir)


if __name__ == "__main__":
    main()
