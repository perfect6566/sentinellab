# Assignment Compliance Matrix

Source: `Machine.pdf` — *Machine Learning and AI Agent Project Prompt and Rubric*  
Reviewed: 2026-10-28

This matrix maps every material assignment requirement to the engineering tasks and the evidence that must exist before the requirement can be considered satisfied. A planned file is not evidence until it contains real, verified results.

## Dataset and evaluation protocol

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| Use the supplied structured PE dataset and verify schema and label mapping (`0 = goodware`, `1 = malware`) | S1-T05, S1-T06 | Data provenance, actual file hash, dimensions, schema, label distribution and retained/excluded column decision in `data/README.md` and `reports/data-audit.md` |
| Do not collect, upload or execute actual malware | S1-T05, S1-T06, S6-T04 | Repository contents and upload controls show that only structured data is used; safety statement in README |
| Hold out a stratified 20% test set before learned preprocessing, feature selection or tuning | S2-T02, S2-T03 | Reproducible split metadata/IDs, class distributions and overlap checks; sealed-test marker |
| Inspect missing values, imbalance, duplicates, related records and potential leakage | S2-T02, S2-T04 | Development-only audit with handling decisions and impact on sample counts |
| Fit imputation, encoding, scaling and feature selection only within training folds | S2-T05, S7-T01 | Pipeline implementation and leakage-focused tests proving fold-local fitting |
| Explore feature selection or dimensionality reduction and justify the final choice | S2-T04, S2-T05, S3-T05 | Experiment/config record and written keep/drop decision in design report |
| Use stratified 10-fold cross-validation within the 80% development set | S3-T01–S3-T05 | Per-fold machine-readable results and mean ± standard deviation summaries for every model |
| Primary metric AUC; secondary metric Accuracy; calculate AUC from continuous malware scores | S3-T01–S3-T06 | Metric implementation/tests and result tables showing both metrics and score source |

## Required models and production artifact

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| Train Logistic Regression, Decision Tree and Random Forest | S3-T02, S3-T03 | Ten-fold per-model result files, selected parameters and timing |
| Train an MLP implemented with PyTorch | S3-T02, S3-T03 | PyTorch source, per-fold reinitialization test/evidence and ten-fold results |
| Train XGBoost and LightGBM plus at least one other high-performing/model-family alternative | S3-T02, S3-T04 | XGBoost, LightGBM and CatBoost (or approved alternative) ten-fold results |
| Describe tuning scope and selected hyperparameters for every model | S3-T01, S3-T03, S3-T04 | Search configuration, trial budget/count and selected parameter records |
| Select production model using CV AUC, Accuracy and production considerations for close results | S3-T05 | Frozen selection report produced before final-test evaluation |
| Freeze preprocessing, hyperparameters and threshold before opening final test | S3-T05 | Timestamped/versioned frozen configuration and hash |
| Refit on full development set and evaluate once on untouched hold-out | S3-T06 | Run log tied to frozen config; hold-out AUC, Accuracy and confusion matrix |
| Do not retune/reselect using test results | S3-T05, S3-T06 | Immutable config/hash and documented test-use policy |
| Save model, fitted preprocessing, schema, label mapping, version and threshold | S3-T07 | Reloadable artifact, manifest and save/reload parity test |

## Agent and three required tools

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| One AI model-based Agent interprets requests, selects tools and explains returned results | S5-T01, S5-T02, S5-T05 | Tool-calling implementation and deterministic dispatch tests |
| Trained ML model produces every classification; LLM does not invent predictions, metrics or causes | S4-T01–S4-T05, S5-T01, S5-T05 | Prompt/tool boundaries, source code and negative tests |
| Single-record tool validates a full record and returns class, score/probability and model version | S4-T01, S4-T02 | Tool schema, valid example and known-result tests |
| Provide a pre-filled single-record example | S4-T02, S6-T02 | Sample record that succeeds against the production schema |
| Batch tool accepts a server-side file reference, preserves stable row IDs and returns per-row/downloadable results plus class counts | S4-T03, S6-T03 | Integration tests and UI/download evidence |
| Batch tool reports invalid rows and never silently drops them | S4-T03, S7-T02 | Mixed-validity fixture with input/output row reconciliation |
| Labeled evaluation tool returns sample counts, mapping, AUC, Accuracy and confusion matrix | S4-T04 | Known-answer metric tests and structured response evidence |
| Missing labels produce a request/error; single-class input returns undefined AUC plus applicable metrics | S4-T04, S7-T01, S7-T03 | Deterministic missing-label and single-class tests |
| Conditional task evaluates labeled CSV first and predicts one record only if returned Accuracy meets the user threshold | S5-T03, S7-T03, S9-T02 | Both branches, call order, service-side gate and actual-LLM activity records |
| If evaluation fails or Accuracy is unavailable, do not predict | S5-T03, S7-T03 | Negative call assertions and activity logs |
| Support result follow-ups such as false-negative count | S5-T04, S6-T03, S7-T03 | Session result test and actual interaction record |
| Use code for metrics/counts; do not send whole CSV to LLM | S4-T03–S4-T05, S5-T02, S5-T04 | Bounded context implementation and tests/log inspection |
| Keep predictions separate from labels and explain FP/FN correctly | S4-T04, S5-T05 | Known matrix tests and response tests |
| Do not call uncalibrated score calibrated confidence | S4-T02, S5-T05 | Output terminology test/review |
| Only give feature-level causes if a supporting explanation tool exists | S5-T05 | Prompt/response tests that refuse unsupported explanations |
| Show concise tool activity without hidden chain-of-thought | S4-T05, S6-T03 | Activity schema and rendered panel |

## Flask deployment and security

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| Flask application with chat and CSV upload supports all three tools | S6-T01–S6-T03 | Browser/integration evidence for all workflows |
| Public shareable URL; grader supplies no personal API key | S8-T04, S8-T05 | Live URL and successful grader-style access |
| Credentials stay in server secrets and never enter repository/browser | S1-T02, S5-T01, S8-T01, S8-T05 | Secret scan, server config and browser/network inspection |
| Validate upload type, schema and size; uploaded content is data, not instructions | S4-T03, S6-T04, S7-T04 | Upload-limit, injection and malformed-file tests |
| Restrict Agent to project tools; no arbitrary shell or uploaded-file execution | S5-T02, S6-T04, S7-T03, S7-T04 | Allowlist implementation and rejection tests |
| Set request, upload and tool-call limits; handle timeouts/failures clearly | S5-T02, S6-T01, S6-T02 | Config and boundary/failure tests |
| Isolate uploaded data/results by user session | S5-T04, S6-T04, S7-T04 | Cross-session rejection tests |
| Provide `/health` and safe tool/error logging | S4-T05, S6-T01, S8-T05 | Health integration/smoke tests and log review |
| Document model loading and application startup | S6-T01, S8-T03, S10-T01 | Verified README/deployment commands |

## Automated testing and gated CI/CD

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| Unit tests for preprocessing/schema, model wrapper, class mapping and metrics | S7-T01 | Passing deterministic unit suite with known expected values |
| Integration tests for predict, batch, evaluation and Agent application boundary | S7-T02 | Passing integration suite |
| Mocked Agent tests for dispatch, validation, result/failure handling | S7-T03 | Passing deterministic Agent suite without a live LLM |
| Cover malformed/missing features, missing labels, single class and tool failure | S7-T01–S7-T03 | Named tests for each case |
| Never report success after failed tool call | S5-T02, S7-T03 | Negative assertion on final response and activity status |
| GitHub Actions runs required tests before deployment | S8-T01, S8-T02 | Successful and intentionally failing workflow evidence |
| Deployment occurs only after tests pass for the same commit; no bypassing platform auto-deploy | S8-T04 | Commit SHA linkage and hosting configuration evidence |
| Post-deployment `/health` smoke test | S8-T05 | CI log tied to deployed commit and URL |

## Actual-LLM evaluation

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| Run at least eight required actual-model scenarios | S9-T01–S9-T03 | Nine planned scenarios: single, batch, conditional pass/fail, FN follow-up, invalid input, missing labels, single class and controlled failure |
| Record request, expected behavior, actual tools, observed result and pass/fail | S9-T02, S9-T03 | Complete `agent-evaluation.md` entries and captured activity |
| Verify natural-language numbers against tool output | S9-T03 | Per-scenario numerical audit |
| Report pass rate, failure patterns, AI model and evaluation date | S9-T03 | Recomputable summary in `agent-evaluation.md` |

## Submission documentation and presentation

| Requirement | Task IDs | Required completion evidence |
|---|---|---|
| Repository contains source, tests, CI/CD and model artifact or reproducible build/retrieval | S8-T01–S8-T03, S10-T01 | Repository audit and clean-environment reproduction |
| README covers setup, reproducible training/evaluation, startup and samples | S10-T01 | Commands executed successfully from a clean environment |
| Include secret-free environment template | S1-T02, S10-T01 | `.env.example` scan/review |
| Include `deployed.md` with URL and startup delay | S8-T05, S10-T02 | Real tested URL and measured limitation |
| Include `evaluation-and-design.md` with data, CV, hold-out, selection, architecture, tools, LLM and limitations | S10-T02 | Content audit against required sections |
| Include `agent-evaluation.md` | S9-T03, S10-T02 | Nine complete scenarios and summary |
| Include `ai-tooling.md` describing development AI usage and limitations | S10-T02 | Accurate record distinguishing Codex from runtime LLM |
| Include pinned dependencies and small sample CSVs | S1-T03, S4-T02–S4-T04, S10-T01 | Lockfile and validated samples |
| 8–10 minute narrated screen-share covers required model results, architecture, tool calls, workflows, error case, tests, gate and smoke test | S10-T03 | Timed rehearsal checklist and final accessible video |
| Add `quantic-grader` collaborator and submit video plus GitHub links | S10-T05 | Access check and course submission confirmation |
| Document AI assistance and external sources; maintain academic integrity | S10-T02, S10-T05 | `ai-tooling.md`, citations and final review |

## Minimum passing gate

The submission cannot pass unless all of the following are demonstrably true:

1. A student-trained malware/goodware model exists.
2. A hosted AI Agent calls that trained model for classification.
3. Repository code, accessible deployment and demonstration evidence prove the integration.

A conventional form-only predictor or an LLM that invents classifications does not satisfy the assignment.
