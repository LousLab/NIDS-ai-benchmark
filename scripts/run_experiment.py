#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nids.config import build_config
from nids.experiment import run_experiment, run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RF/XGBoost/MLP/TabNet for NIDS with a frozen test split."
    )
    parser.add_argument("--dataset", choices=["wsn_ds", "cicids2018"], default="wsn_ds")
    parser.add_argument("--preset", choices=["smoke", "fast", "full"], default="fast")
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument(
        "--split-mode", choices=["stratified", "time"], default="stratified"
    )
    parser.add_argument(
        "--imbalance",
        choices=["none", "class_weight", "smote"],
        default="class_weight",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "mutual_info"],
        default="none",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of rf,xgb,mlp,tabnet",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=None)
    parser.add_argument("--xgb-estimators", type=int, default=None)
    parser.add_argument("--mlp-epochs", type=int, default=None)
    parser.add_argument("--tabnet-epochs", type=int, default=None)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the holdout split and training-only sample caps.",
    )
    args = parser.parse_args()
    models = None
    if args.models is not None:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        if not models:
            parser.error("--models must include at least one model name")
        supported = {"rf", "xgb", "mlp", "tabnet"}
        unknown = sorted(set(models) - supported)
        if unknown:
            parser.error(
                f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(sorted(supported))}"
            )
    cfg = build_config(
        dataset=args.dataset,
        preset=args.preset,
        task=args.task,
        split_mode=args.split_mode,
        imbalance=args.imbalance,
        feature_selection=args.feature_selection,
        models=models,
        seed=args.seed,
        split_seed=args.split_seed,
        rf_estimators=args.rf_estimators,
        xgb_estimators=args.xgb_estimators,
        mlp_epochs=args.mlp_epochs,
        tabnet_epochs=args.tabnet_epochs,
    )
    if args.preset == "smoke":
        out = run_smoke(cfg)
    else:
        out = run_experiment(cfg)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
