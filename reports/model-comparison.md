# Formal Development-Set Model Comparison

Final hold-out status: **sealed and not accessed**.

All values below are from the same 10-fold SHA1-group-aware CV on 38,122 development rows.

| Rank | Model | Selected candidate | AUC mean ± std | Accuracy mean ± std | CV runtime |
|---:|---|---|---:|---:|---:|
| 1 | xgboost | `xgb-depth6` | 0.996657 ± 0.001488 | 0.976339 ± 0.009205 | 51.9s |
| 2 | lightgbm | `lgbm-leaves31` | 0.996593 ± 0.002102 | 0.976890 ± 0.009554 | 39.9s |
| 3 | random_forest | `rf-depth24` | 0.995607 ± 0.001768 | 0.967210 ± 0.010550 | 113.8s |
| 4 | catboost | `cat-depth8` | 0.995086 ± 0.001350 | 0.970174 ± 0.010118 | 268.5s |
| 5 | decision_tree | `dt-unbounded-balanced` | 0.967801 ± 0.008805 | 0.966895 ± 0.009693 | 100.6s |
| 6 | pytorch_mlp | `mlp-128x64` | 0.960621 ± 0.010555 | 0.888046 ± 0.030562 | 38.1s |
| 7 | logistic_regression | `lr-c1-balanced` | 0.813705 ± 0.020730 | 0.821967 ± 0.014244 | 26.5s |

## Feature/dimensionality ablation

| Configuration | AUC mean ± std | Accuracy mean ± std |
|---|---:|---:|
| `hash256_reference` | 0.996593 ± 0.002102 | 0.976890 ± 0.009554 |
| `lgbm-text128` | 0.996525 ± 0.002647 | 0.976549 ± 0.008472 |
| `lgbm-no-text` | 0.996193 ± 0.001199 | 0.968706 ± 0.013899 |

The 256-feature hashing configuration is retained because it has the highest AUC and Accuracy. Dropping import-list text reduces Accuracy materially.

## Production selection

LightGBM is selected. XGBoost's mean AUC is higher by only 0.000064, far smaller than either model's fold standard deviation. LightGBM has slightly higher mean Accuracy and lower selected-candidate CV runtime. These are the predeclared secondary/production tie-breakers for close AUC results.

Threshold remains fixed at 0.5. No hold-out result was used in this decision.
