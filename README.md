# NIDS AI Benchmark

An evaluation project comparing machine-learning and deep-learning models for network intrusion detection. The goal is to identify models that suit different NIDS datasets and tasks using consistent splits and metrics.

## Models

- Random Forest
- XGBoost
- MLP
- TabNet (optional dependency)

## Datasets

The pipeline supports WSN-DS and CICIDS2018. Dataset files are not included in this repository. Place WSN-DS at `data/raw/WSN-DS.csv`, or set `WSN_DS_PATH`. Place CICIDS2018 CSV files under `data/raw/cicids2018/`, or set `CICIDS2018_DIR`.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

To include TabNet, install the optional dependencies:

```bash
python -m pip install -r requirements-tabnet.txt
```

## Run an experiment

```bash
python scripts/run_experiment.py --dataset wsn_ds --preset fast --task multiclass --imbalance class_weight --models rf,xgb,mlp,tabnet --seed 42
```

Presets are `smoke`, `fast`, and `full`. Use `--help` to see the available datasets, tasks, imbalance options, feature-selection options, and model choices. The smoke preset uses synthetic data to check that the pipeline runs; it is not a model benchmark.

## Outputs and evaluation

Each run writes a timestamped folder under `results/` with aggregate metrics, per-class reports, confusion matrices, and comparison plots. The primary metric is macro-F1; accuracy, balanced accuracy, macro precision and recall, weighted F1, MCC, and training time are also recorded. Preprocessing is fitted on training data, and the held-out test split is reserved for final evaluation.

Raw data, local environments, and generated run outputs are excluded from Git. Reports intended for publication should be added separately with the run configuration, sampling details, and limitations.
