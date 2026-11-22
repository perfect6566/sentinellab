# Evaluation and system design

## Data and evaluation contract

The source is the Brazilian malware structured PE feature dataset documented in `data/README.md`. No executable samples are stored or run. The audited source has 50,181 rows and 27 source attributes plus `Label`; `SHA1` and `FirstSeenDate` are excluded from model input. The 25 production features include the import-list text field. Conflicting SHA1 labels and exact duplicate rows are handled before splitting. The final split is group-aware by SHA1: 38,122 development rows and 9,531 sealed test rows, with no SHA1 overlap. Learned preprocessing is fitted inside each CV fold.

## Seven-model comparison

The formal comparison uses the same 10-fold SHA1-group-aware CV over the 38,122-row development partition. Results below are copied from `reports/model-comparison.md`.

| Model | ROC AUC mean ± std | Accuracy mean ± std | CV runtime |
|---|---:|---:|---:|
| XGBoost | 0.996657 ± 0.001488 | 0.976339 ± 0.009205 | 51.9 s |
| LightGBM | 0.996593 ± 0.002102 | 0.976890 ± 0.009554 | 39.9 s |
| Random Forest | 0.995607 ± 0.001768 | 0.967210 ± 0.010550 | 113.8 s |
| CatBoost | 0.995086 ± 0.001350 | 0.970174 ± 0.010118 | 268.5 s |
| Decision Tree | 0.967801 ± 0.008805 | 0.966895 ± 0.009693 | 100.6 s |
| PyTorch MLP | 0.960621 ± 0.010555 | 0.888046 ± 0.030562 | 38.1 s |
| Logistic Regression | 0.813705 ± 0.020730 | 0.821967 ± 0.014244 | 26.5 s |

LightGBM (`lgbm-leaves31`) was selected before hold-out access. XGBoost's AUC advantage is 0.000064, below fold variability; LightGBM has slightly higher mean accuracy and lower runtime. The frozen threshold is 0.5. Import-text ablation lowers accuracy from 0.976890 to 0.968706, supporting retention of the hashed text representation.

## One-time hold-out

For frozen model `malware-lightgbm-20261028-v1`, the test set has 9,531 rows and 8,742 SHA1 groups. ROC AUC is 0.995296 and accuracy is 0.971042 at threshold 0.5. Confusion matrix in true-row/predicted-column order `[0,1]`: TN 4,167, FP 53, FN 223, TP 5,088. These results are descriptive only and were not used to retune the model.

## Application architecture

Flask serves the browser UI and JSON endpoints. At startup, it verifies and loads the trusted joblib artifact once. `predict_single`, `predict_batch`, and `evaluate_labeled` share the production schema and model wrapper. Batch files are bounded CSV uploads referenced by session-bound opaque IDs; results preserve row IDs and invalid rows and can be downloaded. `evaluate_labeled` computes metrics in Python and uses confusion-matrix order `[0,1]`. The Agent may select only the three registered tools. The model decides labels and metrics; the LLM routes tools and communicates their outputs. Conditional prediction is enforced in the server flow after evaluation. Session results and uploaded files are isolated and expire.

## Validation status (current implementation)

- Deterministic automated tests: 100 passed locally (`pytest --collect-only -q` reports 100 tests); `ruff check` and `mypy` pass.
- Public deployment: live at `https://sentinellab.onrender.com` (native Python 3.11, verified `/health` 2026-11-21).
- Public Agent evaluation: 9/9 passed with DeepSeek `deepseek-chat` against the deployed service (evidence [`public-agent-evaluation-2026-11-21.json`](public-agent-evaluation-2026-11-21.json), harness `scripts/run_public_evaluation.py`).
- Tool activity panel shows tool name, `arguments` (`file_reference` / `minimum_accuracy` / `record:<25 PE features>`), and `result_id`.
- Record the GitHub Actions run link for the current `main` commit in `reports/deployed.md` once confirmed green in the GitHub UI.

## Limits

The artifact produces a binary goodware/malware risk classification from structured features. It does not identify malware family, establish feature causality, or prove a file safe to execute. Hold-out results are from this dataset and may not generalize to later malware distributions. The deployed service uses the `deepseek` provider (`deepseek-chat`, server-side key in Render secrets); CI uses the `mock` provider.
