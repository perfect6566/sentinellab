# Malware Classification ML + AI Agent

A reproducible malware/goodware classification project built from structured PE-file features. It
compares seven model families, freezes a production LightGBM pipeline, exposes three model-backed
Agent tools, and provides a Flask website for chat, CSV upload, evaluation and result download.

The classifier produces every malware/goodware decision. The LLM only interprets requests, selects
tools and explains structured tool results. A prediction is not proof that software is safe to run.

- **Live site**: <https://sentinellab.onrender.com>
- **Backend**: Flask (Python 3.11), served by Gunicorn on Render
- **Agent model**: DeepSeek `deepseek-chat` via the OpenAI-compatible API
- **Agent tools**: `predict_single`, `predict_batch`, `evaluate_labeled`
- **Production model**: LightGBM, 25 features, threshold 0.5

## Technology stack

| Layer | Technology | Role |
|---|---|---|
| **Backend** | **Flask** (`app/`) | Serves the chat UI, CSV upload, result download and `/health`. Runs under **Gunicorn** on Render. |
| **Agent model** | **DeepSeek `deepseek-chat`** | Interprets the request and selects tools. Reached over HTTPS at `https://api.deepseek.com/v1`. The API key is a server-side secret and never reaches the browser. |
| **Agent loop** | `app/agent.py`, `app/agent_protocol.py` | Bounded loop: max 8 turns, max 6 tool calls. |
| **Tools** | `app/tools.py` | The only three callable tools. |
| **ML model** | LightGBM | The classifier behind every decision. |
| **Inference** | `app/inference.py` | Loads the hash-verified artifact with frozen preprocessing. |
| **Storage** | none (filesystem) | Session state and uploads live in the Flask instance directory, bound to the session cookie, expired by TTL. |
| **Serving** | Render | Native Python web service, free tier, auto-deploy disabled. |

### The two-model division of labour

The system runs **two different models**, and the split is deliberate:

| | DeepSeek `deepseek-chat` | LightGBM |
|---|---|---|
| Role | Orchestration | Classification |
| Decides | Which tool to call, with which arguments | Whether a record is malware or goodware |
| Produces | The natural-language answer | Every score, AUC, accuracy and count |
| Called | Once per Agent turn, up to 8 turns | Once per tool invocation, up to 6 |
| Never does | Classify, compute metrics, or invent predictions | Talk to the user |

DeepSeek never classifies anything. It calls a tool and then explains the numbers the tool returned.
Every metric stated in a chat answer is checked against the tool output by `app/response_guard.py`
before it reaches the browser. The guard compares the numbers that answer to a metric word
(accuracy, AUC, score, threshold, sample and row counts, confusion-matrix counts) against the tool
results and the trusted metadata. Presentation numbers such as section or step numbering are not
metric claims and are not rejected. The guard also rejects an uncalibrated score presented as a
confidence level, and rejects a confidence claim while allowing a stated confidence limitation.

Provider selection is by environment variable:

| `LLM_PROVIDER` | Behaviour |
|---|---|
| `deepseek` | Live DeepSeek `deepseek-chat`. Used by the hosted deployment and local live runs. |
| `openai_compatible` | Any other OpenAI-compatible Chat Completions endpoint. |
| `codex_cli` | Local-only structured access through an authenticated Codex CLI login. |
| `mock` | Deterministic scripted provider with no network. Used by the CI test suite. |

`mock` is rejected when `REQUIRE_LIVE_LLM=true`, so a production instance cannot silently start
without a real provider.

## Contents

| Section | What it covers |
|---|---|
| [Technology stack](#technology-stack) | Flask backend, DeepSeek Agent model, LightGBM classifier, and how the two models divide the work |
| [Verified status](#verified-status-current-implementation) | Measured results and live deployment state |
| [Model comparison](#model-comparison) | The seven-model cross-validation table and selection rationale |
| [Architecture](#architecture) | How the Agent, tools and model fit together |
| [Repository layout](#repository-layout) | Directory map |
| [Local setup](#local-setup) | Environment, dependencies and configuration |
| [Dataset](#dataset) | Source, provenance and split |
| [Training and final evaluation](#training-and-final-evaluation) | Reproducing the model |
| [Agent tools](#agent-tools) | Tool contracts and the public evaluation harness |
| [Website](#website) | Running and checking the site |
| [Tests and checks](#tests-and-checks) | Local and CI verification |
| [Production runtime](#production-runtime) | Deployment configuration |
| [Security and limitations](#security-and-limitations) | What the system does not do |

## Verified status (current implementation)

- **Dataset**: 50,181 structured records; no malware binaries are downloaded or executed.
- **Split**: group-safe 80/20 development/test split by SHA1 identity (38,122 development /
  9,531 sealed test, no SHA1 overlap).
- **Comparison**: 10-fold `StratifiedGroupKFold` on the development partition across Logistic
  Regression, Decision Tree, Random Forest, PyTorch MLP, XGBoost, LightGBM and CatBoost.
- **Production model**: `malware-lightgbm-20261028-v1` (LightGBM, 25 features, threshold 0.5,
  artifact SHA-256 `2e0dd9f4...b6d4791`), selected before opening the final hold-out.
- **Hold-out** (one-time, frozen config): AUC `0.995296`, Accuracy `0.971042`; TN `4167`, FP `53`,
  FN `223`, TP `5088`.
- **Public Agent evaluation**: 9/9 passed against `https://sentinellab.onrender.com` with DeepSeek
  `deepseek-chat`; evidence in
  [`reports/public-agent-evaluation-2026-11-21.json`](reports/public-agent-evaluation-2026-11-21.json),
  harness `scripts/run_public_evaluation.py`.
- **Automated tests**: 100 passing locally (`pytest --collect-only -q` reports 100 tests); `ruff check`
  and `mypy` pass.
- **Public deployment**: live at `https://sentinellab.onrender.com` (native Python 3.11,
  `runtime: python`, `ready:true`, model `malware-lightgbm-20261028-v1`, Agent
  `deepseek`/`deepseek-chat` ready).

Requirement traceability is in
[`reports/requirements-matrix.md`](reports/requirements-matrix.md); system design and model evidence
are in [`reports/evaluation-and-design.md`](reports/evaluation-and-design.md).

Render deploys from [`render.yaml`](render.yaml) with the native Python runtime. CI deploys the
tested `main` commit only after the test job succeeds, then verifies the public `/health` endpoint
reports the same commit SHA, expected model version and a ready Agent. GitHub `production`
environment needs `RENDER_DEPLOY_HOOK_URL` / `RENDER_PUBLIC_URL` (`https://sentinellab.onrender.com`)
for the gated deploy job to run automatically.

## Model comparison

All values come from the same 10-fold SHA1-group-aware cross-validation on the 38,122 development
rows. The final hold-out was still sealed when this ranking was made.

| Rank | Model | Selected candidate | AUC mean ± std | Accuracy mean ± std | CV runtime |
|---:|---|---|---:|---:|---:|
| 1 | XGBoost | `xgb-depth6` | 0.996657 ± 0.001488 | 0.976339 ± 0.009205 | 51.9s |
| 2 | **LightGBM** | `lgbm-leaves31` | 0.996593 ± 0.002102 | **0.976890** ± 0.009554 | **39.9s** |
| 3 | Random Forest | `rf-depth24` | 0.995607 ± 0.001768 | 0.967210 ± 0.010550 | 113.8s |
| 4 | CatBoost | `cat-depth8` | 0.995086 ± 0.001350 | 0.970174 ± 0.010118 | 268.5s |
| 5 | Decision Tree | `dt-unbounded-balanced` | 0.967801 ± 0.008805 | 0.966895 ± 0.009693 | 100.6s |
| 6 | PyTorch MLP | `mlp-128x64` | 0.960621 ± 0.010555 | 0.888046 ± 0.030562 | 38.1s |
| 7 | Logistic Regression | `lr-c1-balanced` | 0.813705 ± 0.020730 | 0.821967 ± 0.014244 | 26.5s |

**Why LightGBM and not XGBoost.** XGBoost's mean AUC is higher by 0.000064, but either model's
fold-to-fold standard deviation is roughly 0.002 — about thirty times the gap. The ranking is
therefore inside the noise, not a real difference. The tie-breakers were declared in
[`config/tuning.json`](config/tuning.json) before the runs: AUC first, then mean accuracy, then
production cost. LightGBM wins on accuracy (0.976890 vs 0.976339) and trains in 39.9s against
51.9s, so it is the production model. The threshold stays fixed at 0.5.

Each family has two deliberately chosen candidates evaluated under the identical protocol. This is a
bounded stepwise search, not an exhaustive grid. Full records, including the feature ablation that
justified keeping the 256-dimension text hashing, are in
[`reports/model-comparison.md`](reports/model-comparison.md).

## Architecture

```text
Browser website
    |
    v
Flask backend (app/routes.py, session-owned file references)
    |
    v
Bounded Agent loop (app/agent.py, max 8 turns / 6 tool calls)
    |
    +-- turn 1 --> DeepSeek deepseek-chat  <-->  https://api.deepseek.com/v1
    |               request + the three tool schemas
    |               <-- returns a tool name and arguments
    |
    +-- Flask executes the tool (app/tools.py)
    |       predict_single | predict_batch | evaluate_labeled
    |           |
    |           v
    |       LightGBM artifact (app/inference.py)
    |       frozen preprocessing and production classifier
    |           |
    |           v
    |       tool result JSON: scores, AUC, accuracy, confusion matrix
    |
    +-- the tool result is appended to the conversation
    |
    +-- turn 2 --> DeepSeek deepseek-chat
                    request + the tool result
                    <-- returns the final natural-language answer
                        |
                        v
              response_guard.py checks every number in that
              answer against the tool outputs
                        |
                        v
              Flask response -> Browser website
```

**Flask** is the only backend. It serves the chat interface, accepts CSV uploads, owns the session
file store, returns tool results and exposes `/health`. It never classifies anything itself.

**DeepSeek `deepseek-chat`** is the only AI model in the request path. It receives the user's
message plus the JSON schemas of the three tools, and replies with either text or a tool call.
When it asks for a tool, Flask executes that tool against the LightGBM artifact and feeds the
result back for the next turn.

The model is called again after a tool result arrives, because the tool output is structured data
rather than an answer. A single-record request typically costs two model calls: one to select the
tool, one to turn the tool result into the final reply. The bounded loop allows up to eight turns
and six tool calls, so multi-step requests cost more.

That second call is also where the conditional task is decided. After `evaluate_labeled` returns,
the accuracy is placed in the conversation and the model chooses whether to call
`predict_single` or to explain that the threshold was not met.

The LLM cannot access arbitrary server paths or application tools. CSV contents are treated as
untrusted data. Python code calculates every prediction, metric and count.

Training and official hold-out sizes are loaded from the verified production manifest and supplied
as trusted Agent context, so the chat can answer model-dataset questions without mistaking an
uploaded evaluation CSV for the official test set.

## Repository layout

```text
app/                 Flask website, Agent, tools, inference and session isolation
app/templates/       Website HTML
app/static/          Website CSS and JavaScript
ml/                  audit, split, preprocessing, training, tuning and finalization
artifacts/           frozen model, manifest and selection records
config/              seeds, search spaces and production configuration
data/metadata/       source and model schema metadata
data/raw/            local source CSV; ignored by Git
samples/             safe development-derived examples
evaluation/          nine-scenario Agent evaluation specification
reports/             actual experiment and validation evidence
scripts/             generation and validation utilities
tests/               unit, integration and deterministic Agent tests
```

## Local setup

Requirements: Python 3.11 (see `.python-version`: `3.11.9`), `pip`, and compatible ML wheels.
macOS LightGBM may also require:

```bash
brew install libomp
```

Create the isolated project environment:

```bash
python -m pip install -r requirements-runtime.txt -r requirements-dev.txt
cp .env.example .env
```

For DeepSeek (used for hosted and local live runs):

```dotenv
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_BASE=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-deepseek-key
AGENT_TIMEOUT_SECONDS=120
```

`deepseek` defaults to `LLM_MODEL=deepseek-chat` and `LLM_API_BASE=https://api.deepseek.com/v1`
when left empty, so only `LLM_API_KEY` is required. Use `deepseek-chat` for tool calling; do not use
`deepseek-reasoner` for the Agent loop.

`LLM_PROVIDER=mock` is the offline default and needs no key. It is rejected when
`REQUIRE_LIVE_LLM=true`, so production cannot silently start without a real provider.

## Dataset

Source: <https://github.com/fabriciojoc/brazilian-malware-dataset>

Place the structured file at `data/raw/brazilian-malware.csv`. Expected SHA-256:

```text
d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242
```

Verified provenance, labels, column roles and safety policy are in
[`data/README.md`](data/README.md). The source has 27 non-label columns; `SHA1` and `FirstSeenDate`
are excluded from model inputs, leaving 25 production features.

```bash
python -m ml.data_audit data/raw/brazilian-malware.csv \
  --output reports/data-validation.json
python -m ml.split_data
python -m ml.eda
```

The hold-out is created before learned preprocessing or tuning. SHA1 groups do not cross evaluation
partitions, and learned transformations are fitted inside each training fold.

## Training and final evaluation

The steps are ordered and depend on each other: audit → split → train → tune → finalize.

Run a bounded smoke test before formal training:

```bash
python -m ml.train --smoke --folds 2 --sample-rows 2000
```

Run the configured seven-model comparison, then tune every family:

```bash
python -m ml.train
python -m ml.tune --models logistic_regression decision_tree random_forest \
    pytorch_mlp xgboost lightgbm catboost
```

Each `ml.tune` invocation re-runs the same 10-fold protocol for every candidate of the named
families and records the winner under `reports/training/tuning/`.

Verify the saved artifact in a new process:

```bash
python -m ml.finalize --verify-artifact
```

Actual CV results are in [`reports/model-comparison.md`](reports/model-comparison.md). The one-time
hold-out is in [`reports/holdout-evaluation.md`](reports/holdout-evaluation.md) and must not trigger
retuning or model reselection; `ml.finalize` refuses to re-evaluate once that report exists.

## Agent tools

- `predict_single`: predicts one complete 25-feature record.
- `predict_batch`: predicts a session-owned CSV while preserving row IDs and invalid rows.
- `evaluate_labeled`: calculates AUC, Accuracy and a `[0, 1]` confusion matrix.

Confusion-matrix rows are true classes and columns are predicted classes. `FN` is actual malware
predicted as goodware; `FP` is actual goodware predicted as malware. With one true class, AUC is
`null` with a reason while other metrics remain available.

Ready-made inputs for each tool are in [`samples/`](samples/README.md): `single-record.json`,
`batch-sample.csv` and `labeled-sample.csv`. All three come from the development partition and are
for tool-flow testing, not performance estimates.

```bash
pytest tests/agent -q
python scripts/run_public_evaluation.py
```

The public harness exercises AE-01 through AE-09 directly against the deployed service and writes
`reports/public-agent-evaluation-2026-11-21.json`.

## Website

The website is a **Flask** application (`wsgi.py` → `app/create_app`). It serves the HTML chat
interface from `app/templates/`, the front-end assets from `app/static/`, and exposes:

| Route | Purpose |
|---|---|
| `GET /` | Chat interface with example inputs |
| `POST /api/chat` | Sends the user request to the Agent loop and returns the answer plus the tool activity record |
| `POST /api/upload` | Accepts a CSV upload into the session-owned file store |
| `GET /api/results/<result_id>` | Returns a stored result by its `result_id` |
| `GET /api/download/<file_reference>` | Downloads a batch result file |
| `GET /samples/<name>` | Serves the bundled example inputs |
| `DELETE /api/sessions/<conversation_id>` | Ends a conversation and releases its session-bound files |
| `GET /health` | Readiness check: model version, Agent provider/model readiness, service name |

Live public site: <https://sentinellab.onrender.com>. The website supports Agent chat, CSV upload,
example inputs, all three tool workflows, threshold-gated prediction, result cards, batch
preview/download, metrics, confusion matrix, expandable tool activities (tool name + `arguments` +
`result_id`), and session-isolated files.

Local fallback for rehearsal:

```powershell
python -m flask --app wsgi run --host 127.0.0.1 --port 5001
```

Open <http://127.0.0.1:5001> only when rehearsing locally.

```powershell
curl.exe --fail https://sentinellab.onrender.com/health
```

Local fallback: `curl.exe --fail http://127.0.0.1:5001/health`.

## Tests and checks

```bash
pytest -q
ruff check app ml tests scripts
mypy app ml
node --check app/static/app.js
```

CI tests use scripted model responses, not a real LLM, so tool dispatch and failure behavior remain
deterministic. Live-provider quality is measured separately by the public harness.

## Production runtime

`requirements-runtime.txt` excludes training-only Torch, XGBoost and CatBoost dependencies. Render
runs the service with the native Python runtime (no Docker): it installs `requirements-runtime.txt`
and starts `gunicorn --config gunicorn.conf.py wsgi:app`. `/health` is the platform health check.
There is no `Dockerfile`: the repo deploys with the native Python runtime declared in `render.yaml`.

Production mode rejects the mock provider and local development secret. The single Gunicorn worker
is intentional because the current session store is process-local. Verify artifact/application
readiness without binding a port using `python -m scripts.verify_runtime`.

## Evaluation fixtures

```bash
python -m scripts.prepare_agent_evaluation
```

The nine scenarios are defined in [`evaluation/scenarios.json`](evaluation/scenarios.json). Flipped,
single-class and invalid fixtures test control flow only; they are not model-quality evidence.

## Security and limitations

- Only structured feature files are used; no binaries are executed.
- Upload references are UUIDs bound to a server session; arbitrary paths are rejected.
- Uploads are UTF-8 CSV, bounded by bytes and rows, with a TTL cleanup policy.
- API keys are server-only and excluded from logs, Git and health responses.
- Uploaded CSV content is treated as data, not instructions.
- Multi-worker deployment requires shared session/file metadata storage first.
- The model does not provide malware-family attribution or feature-level causality.
- Scores are not calibrated probabilities and must not be described as confidence.
