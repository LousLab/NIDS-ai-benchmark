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
python scripts/run_experiment.py --dataset wsn_ds --preset fast --task multiclass --split-mode stratified --imbalance class_weight --models rf,xgb,mlp,tabnet --seed 42 --split-seed 42 --mlp-epochs 45 --tabnet-epochs 20
```

Presets are `smoke`, `fast`, and `full`. Use `--help` to see the available datasets, tasks, imbalance options, feature-selection options, and model choices. The smoke preset uses synthetic data to check that the pipeline runs; it is not a model benchmark.

## Outputs and evaluation

Each run writes a timestamped folder under `results/` with aggregate metrics, per-class reports, confusion matrices, class prevalence by split, and comparison plots. For stratified WSN-DS runs, identical feature vectors are grouped into the same partition, conflicting feature vectors with different labels are excluded, and compute caps are applied to training only. Validation and test retain the cleaned dataset's observed class prevalence. The holdout uses `--split-seed` (42 by default), while `--seed` controls model randomness. Repeating model seeds with one split isolates training variability from test-set variation.

The first comparison is a fixed-configuration baseline: no model-specific hyperparameter search is performed, and validation is reserved for a later shared tuning design. It uses 150 trees for RF and XGBoost, 45 MLP epochs, and 20 TabNet epochs. The primary metric is macro-F1 over classes present in the test set; reports also include all-label macro-F1, accuracy, balanced accuracy, macro precision and recall, weighted F1, MCC, per-class support, and training time. The WSN-DS `Time` column is excluded from model inputs in both protocols because it can act as a simulation-scenario shortcut; chronological mode uses it only to order whole-time-value boundaries and collapses repeated feature vectors to their earliest occurrence. That result is a temporal/scenario-shift stress test; attack classes absent from the final window are explicitly reported as unsupported.

Example reduced-budget chronological stress run (caps are intentionally disabled for this split mode):

```bash
python scripts/run_experiment.py --dataset wsn_ds --preset fast --task multiclass --split-mode time --imbalance class_weight --models rf,xgb,mlp,tabnet --seed 42 --split-seed 42 --rf-estimators 50 --xgb-estimators 50 --mlp-epochs 5 --tabnet-epochs 4
```

CICIDS2018 currently uses class-capped sampling before splitting because of its size. Its resulting test distribution is not representative of raw traffic prevalence, so those results are exploratory and should not be presented as deployment estimates. WSN-DS is the prevalence-aware primary evaluation in this repository. WSN-DS was generated from simulated LEACH traffic; it is not a substitute for validation on operational network captures ([original dataset paper](https://doi.org/10.1155/2016/4731953)).

The current CICIDS2018 comparison repeats each of the four models across three model seeds on one fixed split. XGBoost has the highest macro-F1 (0.8199), MLP has the highest balanced accuracy (0.8723) and MCC (0.9161), and Random Forest trains fastest. No model leads on every metric. See [the CICIDS2018 findings and reproduction instructions](docs/cicids2018_findings.md) for the full results, per-class failures, protocol, and limitations. For the WSN-DS comparison, see [the WSN-DS findings](docs/benchmark_findings.md).

Raw data, local environments, and generated run outputs are excluded from Git. Reports intended for publication should be added separately with the run configuration, sampling details, and limitations.

## Summarize repeated runs

To combine runs that use the same split and training sample, pass their result folders to the summary script:

```bash
python scripts/summarize_runs.py results/run_seed_42 results/run_seed_43 results/run_seed_44 --output-dir results/stratified_seed_analysis
```

The script checks run comparability, then writes mean and standard deviation metric tables and seed variability and per-class F1 plots.


