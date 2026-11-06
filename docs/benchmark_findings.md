# WSN-DS model comparison: protocol and findings

## Scope

This report evaluates Random Forest (RF), XGBoost, a scikit-learn MLP, and TabNet on the five-class WSN-DS task. WSN-DS contains simulated LEACH wireless-sensor-network records, not operational network captures; the dataset paper reports 374,661 records covering normal behavior and four attack types ([Almomani et al., 2016](https://doi.org/10.1155/2016/4731953)). The results therefore describe this dataset and protocol only.

## Evaluation protocol

- The primary evaluation uses one grouped, stratified 70/15/15 train/validation/test split with `split_seed=42`. Model seeds 42, 43, and 44 reuse the same split and training sample, so their standard deviations measure model-seed variability, not holdout-sampling uncertainty.
- Before splitting, identical model-input feature vectors are grouped together, with `Time` excluded from the grouping and from all model inputs. Groups remain wholly within train, validation, or test. Sixty-nine feature groups had conflicting labels; their 916 records were excluded as ambiguous. The remaining **373,745** records retain their observed prevalence.
- The full dataset was split before the compute cap. The training partition contained 261,637 records; the fast preset's 8,000-per-class cap reduced it to 29,491. Validation (56,046) and test (56,062) were left uncapped. Every class appears in all three partitions.
- Imputation, scaling, constant-feature screening, and any optional feature selection are fitted on training data only. Class weighting is used for model training. `Time` is not a model feature.
- These are fixed-configuration baselines with **no hyperparameter search**. The validation set is reserved for a future common search procedure. Settings were 150 trees for RF and XGBoost, 45 epochs for MLP, and 20 epochs for TabNet. In this environment MLP used `sklearn.neural_network.MLPClassifier`; its 45-epoch run emitted a maximum-iterations convergence warning, although the loss curve flattened toward the end.
- Reported training times cover model fitting, not data preparation or inference.

The main run artifacts are in `results/grouped_stratified_seed_analysis/` and the three timestamped run folders beginning `20261103T105410Z`, `20261103T105808Z`, and `20261103T110143Z`.

## Primary results

Values are mean ± sample standard deviation across the three model seeds on the same 56,062-row test set.

| Model | Macro-F1 | Balanced accuracy | MCC | Training time (s) |
|---|---:|---:|---:|---:|
| RF | **0.95475 ± 0.00078** | **0.98215 ± 0.00022** | 0.94654 ± 0.00154 | **2.46 ± 0.17** |
| XGBoost | 0.95376 ± 0.00085 | 0.98175 ± 0.00028 | **0.95147 ± 0.00113** | 4.41 ± 0.25 |
| MLP | 0.94317 ± 0.00537 | 0.96995 ± 0.00344 | 0.94613 ± 0.00543 | 76.77 ± 7.10 |
| TabNet | 0.86423 ± 0.01010 | 0.91476 ± 0.00430 | 0.83632 ± 0.00628 | 109.42 ± 20.90 |

RF and XGBoost are effectively tied on macro-F1: the difference is about 0.001, similar in size to their seed-to-seed standard deviations. RF is faster in this run; XGBoost has the strongest MCC and weighted F1. These results do not establish a statistically significant winner. MLP is within about 0.012 macro-F1 of the tree models but requires substantially more training time. TabNet trails the other models under these fixed settings and has more seed variability.

### Per-class F1

The support column is the count in each fixed test set. Scores are mean ± sample standard deviation across model seeds.

| Class (support) | RF | XGBoost | MLP | TabNet |
|---|---:|---:|---:|---:|
| Blackhole (1,396) | 0.9849 ± 0.0005 | 0.9849 ± 0.0004 | 0.9496 ± 0.0130 | 0.7929 ± 0.0181 |
| Flooding (497) | 0.9376 ± 0.0000 | 0.9361 ± 0.0005 | 0.9224 ± 0.0010 | 0.9283 ± 0.0088 |
| Grayhole (2,166) | 0.9305 ± 0.0035 | **0.9541 ± 0.0016** | 0.9430 ± 0.0097 | 0.6927 ± 0.0188 |
| Normal (51,008) | 0.9952 ± 0.0002 | **0.9958 ± 0.0001** | **0.9963 ± 0.0009** | 0.9891 ± 0.0030 |
| TDMA (995) | **0.9256 ± 0.0009** | 0.8980 ± 0.0019 | 0.9046 ± 0.0446 | 0.9181 ± 0.0152 |

Normal traffic is about 91% of the test set, so accuracy and weighted F1 alone hide the differences on attacks. Grayhole and TDMA are the most model-dependent classes: XGBoost is strongest on Grayhole, while RF is strongest on TDMA. TabNet's main weaknesses are Grayhole and Blackhole.

## Limitations and next work

1. The three primary repetitions vary model seeds on one fixed holdout. They do not measure uncertainty across independently sampled test partitions; repeated grouped holdouts or grouped cross-validation would strengthen ranking claims.
2. Hyperparameters were fixed rather than tuned. A fair tuning study should give every model the same validation-based search budget and select settings using the same primary metric, macro-F1.
3. MLP reached its 45-epoch cap with a convergence warning. Its reported result is a fixed-budget score, not a demonstrated optimum. TabNet was also trained for a fixed epoch budget without validation early stopping.
4. Removing the 916 records from conflicting feature/label groups avoids asking a model to assign different labels to identical inputs, but it changes class proportions, especially for Blackhole: 743 of the 10,049 raw Blackhole records were in those conflicts. This cleaning decision should be included in any reproduction and should be examined against the dataset's generation process.
5. WSN-DS is simulated, only this dataset was available for these runs, and CICIDS2018 was not evaluated. There is no evidence here for cross-dataset or operational-network generalization.
6. The reports and plots under `results/` are excluded by `.gitignore`; the code and this findings document do not by themselves publish those artifacts.

## Conclusion

On this grouped, prevalence-preserving WSN-DS holdout, RF and XGBoost perform similarly and clearly lead the fixed-budget TabNet run. RF is the faster choice in the measured training loop; XGBoost has somewhat better MCC and weighted F1. MLP performs competitively after increasing its epoch budget, but remains slower and hit its iteration cap. No model can be declared a general NIDS winner from these experiments: conclusions are limited by one dataset, one holdout, no hyperparameter search, ambiguous-label cleaning. See [the separate chronological stress test](temporal_stress_test.md) for the temporal results.

