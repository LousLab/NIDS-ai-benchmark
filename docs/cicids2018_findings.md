# CICIDS2018 model comparison

## Scope

This is a multiclass comparison of Random Forest (RF), XGBoost, an MLP, and TabNet on the Kaggle copy of CSE-CIC-IDS2018 [`towhidultonmoy/cicids2018`](https://www.kaggle.com/datasets/towhidultonmoy/cicids2018). The original dataset is described by the [Canadian Institute for Cybersecurity at UNB](https://www.unb.ca/cic/datasets/ids-2018.html). These results apply to this dataset version and protocol.

## Evaluation protocol

- All ten daily CSV files were included, including the 3.87 GB file. CICIDS2018's class-capped working sample was formed before splitting: up to 8,000 rows per class, followed by an 80,000-row total cap. The sampling seed was fixed at 42 across model seeds.
- The grouped stratified split used `split_seed=42` and a 70/15/15 train/validation/test ratio. Identical feature vectors stayed in one partition. Fifty-two feature groups had conflicting labels; the 9,894 sampled rows in those groups were excluded. The cleaned sample contained **70,106 rows** and **70 numeric input features**.
- After the training cap, the split contained 49,860 training rows, 10,827 validation rows, and 9,419 test rows. All 15 classes had test support, though support varied substantially by class.
- RF and XGBoost each used 150 trees; MLP used 45 epochs; TabNet used 20 epochs. Class weights were used. Preprocessing was fitted on training data. The repeated model seeds 42, 43, and 44 reused the same sample and split.
- These are fixed-configuration results without hyperparameter tuning. The MLP reached its 45-iteration limit without convergence; TabNet used the final epoch without early stopping. Training times cover model fitting, not data loading or inference.

## Results

Metrics are mean ± sample standard deviation across three model seeds on the same 9,419-row test set.

| Model | Macro-F1 | Balanced accuracy | MCC | Mean training time |
|---|---:|---:|---:|---:|
| XGBoost | **0.8199 ± 0.0001** | 0.8301 ± 0.0003 | 0.9054 ± 0.0006 | 11.5 s |
| MLP | 0.8136 ± 0.0183 | **0.8723 ± 0.0069** | **0.9161 ± 0.0025** | 17.4 s |
| Random Forest | 0.8006 ± 0.0021 | 0.8526 ± 0.0010 | 0.9041 ± 0.0007 | **1.4 s** |
| TabNet | 0.7929 ± 0.0087 | 0.8632 ± 0.0056 | 0.9097 ± 0.0002 | 38.5 s |

XGBoost had the highest macro-F1, with MLP close behind. MLP had the highest balanced accuracy and MCC, and more seed variation in macro-F1. RF trained fastest. The ranking depends on the metric, so the results do not support a single best model for every objective.

Class-level scores show the main weaknesses. DoS attacks-SlowHTTPTest had only 46 test examples and F1 ranged from 0.044 for XGBoost to 0.173 for MLP. SQL Injection had 11 test examples, making its scores highly uncertain. RF and XGBoost had zero FTP-BruteForce F1 despite 266 test examples; MLP and TabNet scored 0.875 and 0.898 on that class.

## Relation to WSN-DS

The grouped stratified WSN-DS report records macro-F1 of 0.9548 for RF, 0.9538 for XGBoost, 0.9432 for MLP, and 0.8642 for TabNet. On CICIDS2018, XGBoost led macro-F1 and MLP led balanced accuracy and MCC. The models were trained and evaluated separately on each dataset, which has a different label set and sampling protocol. Treat these as dataset-specific comparisons; the scores do not measure cross-dataset transfer.

## Limitations

1. CICIDS2018 sampling happens before the split and caps classes, so the test distribution is not raw traffic prevalence. Do not present these metrics as deployment estimates.
2. The 9,894 rows removed for conflicting duplicate feature vectors are a substantial part of the 80,000-row working sample. Their effect on class composition should be considered when interpreting results.
3. The three seeds reuse one sample and one holdout. Their standard deviations measure model-seed variability, not uncertainty across independently sampled test sets.
4. Some classes have small test support, especially SQL Injection, Brute Force-XSS, and DoS attacks-SlowHTTPTest. Per-class estimates for these labels are unstable.
5. The pipeline drops CICIDS2018 timestamps, so this run is stratified and does not test chronological generalization.
6. These fixed settings were not tuned. The MLP convergence warning and lack of TabNet early stopping limit claims about their best attainable performance.

## Reproducibility artifacts

The three timestamped run folders and aggregate CSVs are under `results/cicids2018_grouped_stratified_seed_analysis/`. `results/` is excluded from Git, so copy the required metrics, configurations, and reports into the internship submission package if it needs to be reproducible outside this checkout.

To repeat the same fixed-budget protocol after placing all ten CSVs in the configured CICIDS2018 data directory, run this command once per model seed (`42`, `43`, and `44`), keeping `--split-seed 42` unchanged:

```bash
python scripts/run_experiment.py --dataset cicids2018 --preset fast --task multiclass --split-mode stratified --imbalance class_weight --models rf,xgb,mlp,tabnet --seed 42 --split-seed 42 --rf-estimators 150 --xgb-estimators 150 --mlp-epochs 45 --tabnet-epochs 20 --skip-large-cicids-files-mb 5000
```

The 5,000 MB threshold includes the largest daily CSV. After recording the three timestamped output paths, aggregate them with:

```bash
python scripts/summarize_runs.py <seed-42-run> <seed-43-run> <seed-44-run> --output-dir results/cicids2018_grouped_stratified_seed_analysis
```
