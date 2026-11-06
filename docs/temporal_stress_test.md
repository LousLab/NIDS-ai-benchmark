# WSN-DS chronological stress test

## Chronological stress test

A separate one-seed run uses whole timestamp boundaries and the reduced settings 50 trees for each tree model, 5 MLP epochs, and 4 TabNet epochs. It trains on 234,771 rows, validates on 49,254, and tests on 49,358 rows covering `Time=1750–3600`. The test contains no Flooding examples because Flooding records end earlier in this dataset. The test supports Blackhole, Grayhole, Normal, and TDMA only.

| Model | Macro-F1 on supported classes | All-label macro-F1* | Blackhole recall |
|---|---:|---:|---:|
| RF | 0.9347 | 0.7478 | 0.7163 |
| XGBoost | **0.9395** | 0.7516 | 0.7245 |
| MLP | 0.9021 | 0.7217 | 0.6583 |
| TabNet | 0.9117 | 0.7294 | 0.6735 |

\*The all-label score assigns zero to Flooding, which has no test support; it is a coverage warning, not a complete five-class performance estimate. This temporal stress test has lower training budgets and a different class distribution, so its scores should not be compared directly with the stratified results. It does show a marked drop in Blackhole recall and illustrates scenario shift.
