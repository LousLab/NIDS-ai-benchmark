from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


ImbalanceMethod = Literal["none", "class_weight", "smote"]
SplitMode = Literal["stratified", "time"]
TaskType = Literal["multiclass", "binary"]
Preset = Literal["smoke", "fast", "full"]


@dataclass
class ExperimentConfig:
    dataset: Literal["wsn_ds", "cicids2018"]
    preset: Preset = "fast"
    seed: int = 42
    # Keep the holdout fixed across model seeds to isolate training randomness.
    split_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    split_mode: SplitMode = "stratified"
    task: TaskType = "multiclass"
    imbalance: ImbalanceMethod = "class_weight"
    feature_selection: Literal["none", "mutual_info"] = "none"
    k_features: int = 20
    models: Sequence[str] = field(
        default_factory=lambda: ("rf", "xgb", "mlp", "tabnet")
    )
    # Caps apply to training only; validation and test keep observed prevalence.
    max_per_class: int | None = 15000
    max_total_rows: int | None = 200000
    skip_large_cicids_files_mb: float = 1024.0
    mlp_epochs: int = 25
    tabnet_epochs: int = 20
    rf_estimators: int = 200
    xgb_estimators: int = 200
    output_dir: Path = Path("results")

    def run_name(self) -> str:
        return f"{self.dataset}_{self.task}_{self.split_mode}_{self.preset}_{self.imbalance}"


def preset_overrides(preset: Preset) -> dict:
    if preset == "smoke":
        return {
            "max_per_class": 80,
            "max_total_rows": 400,
            "models": ("rf", "mlp"),
            "mlp_epochs": 3,
            "rf_estimators": 20,
            "xgb_estimators": 20,
            "feature_selection": "none",
        }
    if preset == "fast":
        return {
            "max_per_class": 8000,
            "max_total_rows": 80000,
            "mlp_epochs": 15,
            "tabnet_epochs": 12,
            "rf_estimators": 150,
            "xgb_estimators": 150,
        }
    if preset == "full":
        return {
            "max_per_class": 30000,
            "max_total_rows": 400000,
            "skip_large_cicids_files_mb": 5000.0,
            "mlp_epochs": 30,
            "tabnet_epochs": 25,
        }
    raise ValueError(f"Unknown preset: {preset}")


def build_config(**kwargs) -> ExperimentConfig:
    preset = kwargs.get("preset", "fast")
    merged = preset_overrides(preset)
    merged.update({k: v for k, v in kwargs.items() if v is not None})
    return ExperimentConfig(**merged)
