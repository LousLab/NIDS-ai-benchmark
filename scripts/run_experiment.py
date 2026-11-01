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
    args = parser.parse_args()
    models = None
    if args.models:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    cfg = build_config(
        dataset=args.dataset,
        preset=args.preset,
        task=args.task,
        imbalance=args.imbalance,
        feature_selection=args.feature_selection,
        models=models,
        seed=args.seed,
    )
    if args.preset == "smoke":
        out = run_smoke(cfg)
    else:
        out = run_experiment(cfg)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
