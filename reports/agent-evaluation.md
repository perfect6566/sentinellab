# Agent evaluation record

## Scope

Evaluation against the release code with the live DeepSeek provider `deepseek-chat`, model `malware-lightgbm-20261028-v1`. Harness `scripts/run_public_evaluation.py` calls `/api/upload`, `/api/chat`, and `/api/results` directly. Full prompts, expected/actual tool sequences, HTTP status, structured results, final answers, and check outcomes are preserved in [`public-agent-evaluation-2026-11-21.json`](public-agent-evaluation-2026-11-21.json). Created `2026-11-22T10:09:54Z`, 9/9 passed.

| ID | Scenario | Expected tools | Actual tools | HTTP | Audited result | Pass |
|---|---|---|---|---:|---|---|
| AE-01 | Single prediction | `predict_single` | `predict_single` | 200 | Label 1, score 0.9978823131027981, threshold 0.5; answer matches tool result | Yes |
| AE-02 | Batch prediction | `predict_batch` | `predict_batch` | 200 | 5/5 valid; 5 goodware, 0 malware; per-row scores and counts match | Yes |
| AE-03 | Conditional threshold met | `evaluate_labeled`, `predict_single` | `evaluate_labeled`, `predict_single` | 200 | Accuracy 1.0 on the 20-row workflow fixture; 98% gate met so the prediction followed | Yes |
| AE-04 | Conditional threshold not met | `evaluate_labeled` | `evaluate_labeled` | 200 | Accuracy 0.0; 10 FP and 10 FN; no `predict_single` call | Yes |
| AE-05 | False-negative follow-up | (none) | (none) | 200 | Returned 0 FN from the stored AE-03 result via `context_result_id` | Yes |
| AE-06 | Invalid feature input | `predict_batch` | `predict_batch` | 422 | 1 row, 0 valid; `tool_failure` surfaced, no silent drop | Yes |
| AE-07 | Missing labels | `evaluate_labeled` | `evaluate_labeled` | 422 | `tool_failure` surfaced for missing `Label` column | Yes |
| AE-08 | Single-class evaluation | `evaluate_labeled` | `evaluate_labeled` | 200 | 5 samples, accuracy 1.0, AUC null with reason, matrix `[[5,0],[0,0]]` | Yes |
| AE-09 | Unavailable file failure | `evaluate_labeled` | `evaluate_labeled` | 422 | Missing session-owned file reference returned `invalid_file`; Agent reported failure and stopped | Yes |

## Input fixtures

- AE-01 / AE-03 record: `samples/single-record.json` (complete 25-feature record).
- AE-02: `samples/batch-sample.csv` (5 rows).
- AE-03: `samples/labeled-sample.csv` (20 rows, 10 goodware / 10 malware) with condition `minimum_accuracy: 98`.
- AE-04: `samples/evaluation/labels-flipped.csv` with condition `minimum_accuracy: 98`.
- AE-05: `context_result_id` from the stored AE-03 evaluation result.
- AE-06: `samples/evaluation/invalid-feature.csv`.
- AE-07: `samples/evaluation/missing-label.csv`.
- AE-08: `samples/evaluation/single-class.csv`.
- AE-09: randomly generated absent file reference in the same session.

Scenario definitions are in `evaluation/scenarios.json`.

## Numeric grounding

Every final answer was checked against the structured tool output: predicted labels, malware scores, accuracy/AUC, confusion-matrix counts, and row counts. The response guard rejects rounded-score mismatches with half-unit tolerance and rejects uncalibrated "confidence" claims; a unit regression covers this behavior.

The guard applies numeric grounding to the numbers that answer to a metric word (accuracy, AUC, score, threshold, sample and row counts, confusion-matrix counts). A number in a sentence without a metric word is presentation detail rather than a metric claim, so a step heading such as `Step 2 - Gate check` and a file reference that begins with digits are not treated as invented metrics. A stated confidence limitation, for example "this is an uncalibrated score, not a confidence measure", is allowed, while a confidence claim is rejected.

## Defects found and fixed during this evaluation cycle

The conditional scenario was the weakest point of the earlier run. The three defects below were found while auditing it and are fixed in the code this run was taken against.

- The guard treated every number in an answer as a metric claim, so presentation numbers such as a `Step 2` heading were rejected. Grounding now applies to the numbers in a sentence that answers to a metric word. The guard also rejected a correct statement of the confidence limitation while letting an actual confidence claim through; both are now distinguished.
- The validated feature count of a single prediction was stored only in the activity record. A follow-up receives the stored result data as its trusted context, so a correct answer stating that count could not be grounded, and the model was observed stating 24 where the value was 25. The count is now part of the result data, which grounds it in both the main and the follow-up path.
- The gate only stopped a prediction that the condition did not allow. Nothing required the prediction to run when the condition was satisfied, so the Agent could end the turn after the evaluation and report the task as complete without ever predicting. The Agent is now asked to run the outstanding prediction before it reports the result.

The browser additionally restates the outcome from the tool results when the guard rejects the wording, so the interface shows the authoritative metrics next to the successful tool activity instead of the guard's internal message.

With these fixes the conditional scenario returned HTTP 200 with both tool calls, and a follow-up asking for the validated feature count is answered from the stored result. Repeated runs of the conditional scenario and of the feature-count follow-up completed without a rejection.

## Presentation evidence captured on the live deployment

The narrated demonstration uses captures taken from the running service on 2026-11-22, not from a mock or a replay. The captures are supplied with the presentation material:

| Capture | Scenario | What it shows |
|---|---|---|
| Tool activity | Conditional task, criterion met | The expanded activity record with `evaluate_labeled: succeeded` and `predict_single: succeeded`, each entry showing `arguments`, `minimum_accuracy: 0.98`, and `result_id` |
| False-negative follow-up | False-negative follow-up | The follow-up answer reports `fn: 0` from the stored evaluation result and defines a false negative as a true-malware record predicted as goodware, without recomputing from the CSV |
| CI release run | Gated CI/CD | The release run with the test job and the deploy job both successful |
| Render deployment | Gated CI/CD | The Render deployment entry for the same commit, live, so the deployed commit and the tested commit are identical |

The follow-up was issued in the same conversation with `context_result_id` pointing at the stored evaluation result, so the Agent answered from the retained result object rather than re-deriving metrics.

## Limits

The 20-row and 5-row fixtures demonstrate routing, gating, and response grounding only. Their perfect metrics are not evidence of generalization. Production model quality is established only by the 10-fold development CV in `reports/model-comparison.md` and the one-time hold-out in `reports/holdout-evaluation.md` (AUC 0.995296, Accuracy 0.971042).
