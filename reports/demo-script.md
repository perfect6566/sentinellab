# 8–10 minute project demonstration

Target duration: about 9 minutes. Use the public site at `https://sentinellab.onrender.com`, the checked-in sample fixtures, and the reports referenced below. The public DeepSeek nine-scenario evaluation passed 9/9 (see `reports/deployed.md` and `reports/public-agent-evaluation-2026-11-21.json`).

| Time | Demonstration | Talking points |
|---|---|---|
| 0:00–0:40 | State the problem and safety boundary | Structured PE features only; no executable malware is run. Output is a model risk classification, not proof of safety. |
| 0:40–1:50 | Data contract and leakage controls | Show `reports/data-audit.md`, SHA1 grouping and sealed 20% hold-out (38,122 development / 9,531 test); explain fold-local preprocessing and exclusion of `SHA1`, `FirstSeenDate`, and `Label` from inputs. |
| 1:50–2:50 | Compare models | Show `reports/model-comparison.md`: seven required models, 10-fold group-aware CV, AUC/Accuracy mean ± std. Name all seven, then read the top three and bottom two CV numbers: Top3 = XGBoost (AUC 0.996657 ± 0.001488, Acc 0.976339 ± 0.009205), LightGBM (0.996593 ± 0.002102, 0.976890 ± 0.009554), Random Forest (0.995607 ± 0.001768, 0.967210 ± 0.010550); Bottom2 = PyTorch MLP (0.960621 ± 0.010555, 0.888046 ± 0.030562), Logistic Regression (0.813705 ± 0.020730, 0.821967 ± 0.014244). LightGBM selected on near-tied XGBoost (ΔAUC 0.000064 < fold std) via higher accuracy + faster runtime (39.9s vs 51.9s). |
| 2:50–3:30 | Final hold-out | Show `reports/holdout-evaluation.md`: AUC 0.995296, Accuracy 0.971042, TN/FP/FN/TP 4167/53/223/5088; clarify it was accessed once after freeze. |
| 3:30–4:15 | Architecture and tools | Describe model loader, schema validation, Agent loop and the exact tools: `predict_single`, `predict_batch`, `evaluate_labeled`. LLM chooses/routes; Python/model computes labels and metrics. Open one assistant reply's `N tool calls` expander and read tool name + `arguments` (`file_reference` / `minimum_accuracy` / `record:<25 PE features>`) + `result_id`, then the result card. |
| 4:15–5:00 | Single-record prediction | Use `samples/single-record.json` and the quick task; point to malware score, threshold 0.5, model version `malware-lightgbm-20261028-v1`, and grounded explanation. |
| 5:00–5:50 | Batch prediction | Upload `samples/batch-sample.csv`; show every row receives a result/status and download the full output. |
| 5:50–6:40 | Labeled evaluation | Upload `samples/labeled-sample.csv`; explain AUC, accuracy, `[0,1]` confusion matrix and FN/FP semantics. |
| 6:40–7:30 | Conditional branch and follow-up | Evaluate the labeled sample at 98%; it achieves 100% on this small workflow fixture, so the server permits a single prediction next. Ask for FN count as a result-grounded follow-up. State fixture metrics are flow demonstration, not model-quality evidence. |
| 7:30–8:20 | Failure and security behavior | Show `samples/evaluation/single-class.csv` or invalid-feature fixture; discuss `auc=null` for one class, invalid-row reporting, opaque session-bound references, and cross-session denial. |
| 8:20–9:00 | Live Agent evaluation, tests and status | Show the public DeepSeek 9/9 (`deepseek-chat`, `reports/public-agent-evaluation-2026-11-21.json`). Show 100 local tests, Ruff/mypy checks, and the public `/health` at `https://sentinellab.onrender.com` (`ready:true`, model + `deepseek` agent ready). |

## Evidence already captured for the recording

The following captures were taken from the live service and are supplied with the presentation material, so the recording does not need to be re-shot:

- Tool activity — expanded activity record with tool names, arguments, `minimum_accuracy`, and `result_id`.
- Single-record and batch classification — both results on the live deployment.
- Conditional task — both branches of the conditional task.
- False-negative follow-up — answered from the stored evaluation result.
- Invalid input — invalid-feature rejection.
- CI release run — the test job and the deploy job both successful.
- Render deployment — the deployment entry for the same commit, live.
- CI gate blocked — the deploy job skipped after an intentionally failed test on the gate-demonstration commit.

## Demo preparation

1. Open the public site at `https://sentinellab.onrender.com` and wait for `/health` to report model and Agent ready (model `malware-lightgbm-20261028-v1`, provider `deepseek`, model `deepseek-chat`).
2. Allow for a 10–20 second first response (free-plan cold start).
3. Keep a terminal ready with the same Python's `pytest -q` and the CI run page. Keep the deterministic sample files downloaded locally.
4. If the live LLM service is unavailable, demonstrate deterministic tests and the preserved public evaluation JSON as dated evidence; do not present mock output as a new live run.
5. Stop the recording near nine minutes; use the spare minute for questions rather than making unsupported deployment claims.
