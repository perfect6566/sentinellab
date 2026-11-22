# Deployment status

## Gated CI/CD evidence

The gate was exercised in both directions. The failing run proves the deploy job is skipped when the test job does not succeed; the passing run proves the same job promotes the tested commit.

| Run | Event | Test job | Deploy job |
|---|---|---|---|
| Release | `push` | success | success |
| Gate demonstration | `workflow_dispatch` (`simulate_failure=true`) | **failure** at the `Controlled failure injection` step | **skipped** (0 s) |
| Gate demonstration | `push` | success | success (1 m 28 s) |

The controlled failure injects `exit 1` after the deterministic suite, the model check, and the Gunicorn config check have all passed, so the run demonstrates the gate itself rather than a broken test. Because the deploy job declares `needs: test`, it was skipped and no release reached the hosted service. The two gate-demonstration runs are on the same commit, so the only difference between them is whether the test suite passed.

The release run is the current release. The deploy job triggered the Render deploy hook for that commit and then waited until `/health` reported the same commit SHA, the frozen model version, and a ready Agent. The report pairs the run page with the Render deployment entry so the two commit SHAs can be compared directly.

## Current state

Public deployment is live at `https://sentinellab.onrender.com` (native Python 3.11, `runtime: python`, no Docker, no `uv`).

Verified 2026-11-22 via `GET https://sentinellab.onrender.com/health`:

- `ready: true`, `service: malware-ml-agent`
- model `malware-lightgbm-20261028-v1` (`ready: true`, 25 features, threshold 0.5, artifact SHA-256 `2e0dd9f4f57554efca6d0dcdc0af5af79f9a7abfc383f70569e1662afb6d4791`)
- Agent `deepseek` / `deepseek-chat` (`mode: deepseek`, `ready: true`)
- `commit_sha`: reports the deployed commit SHA (verified to match the tested commit)

The public nine-scenario DeepSeek evaluation passed 9/9 against this deployment on 2026-11-22 (evidence [`public-agent-evaluation-2026-11-21.json`](public-agent-evaluation-2026-11-21.json), harness `scripts/run_public_evaluation.py`).

The repository contains the native-Python Render Blueprint (`runtime: python`) and the CI-gated deploy job (`.github/workflows/ci.yml` runs only after `Test Python 3.11` passes on `main`, deploys the exact tested SHA, then verifies `/health` commit/model/Agent).

## Render configuration

`render.yaml` defines a native Python web service with auto-deploy disabled and `/health` as its platform check. Render installs `requirements-runtime.txt` via the Blueprint `buildCommand` and starts `gunicorn --config gunicorn.conf.py wsgi:app` via `startCommand`. `.github/workflows/ci.yml` has a deployment job that runs only after the test job passes on `main`, deploys the exact tested SHA through a secret deploy hook, and waits for the public health response to report the same commit, the expected model, and a ready Agent. The Blueprint also sets `autoDeployTrigger: off` ([Render deployment docs](https://render.com/docs/deploys), [Blueprint reference](https://render.com/docs/blueprint-spec)).

To enable the gated deploy job, add the Render service deploy-hook URL as secret `RENDER_DEPLOY_HOOK_URL` and `https://sentinellab.onrender.com` as variable `RENDER_PUBLIC_URL` in the GitHub `production` environment. `SECRET_KEY`, `LLM_MODEL`, `LLM_API_BASE`, and `LLM_API_KEY` are entered as Render secret values. Keep `autoDeployTrigger` off in Render.

The Blueprint uses Render's free web-service plan for a no-cost course preview. Free web services use 0.1 CPU and 512 MB RAM and spin down after 15 minutes idle, so the first request can have a cold start ([free plan limitations](https://render.com/docs/free), [compute plans](https://render.com/docs/compute-plans)). Session uploads and follow-up results are instance-local and can be lost after a restart; the free service suits a demo flow, not durable file storage. `Dockerfile`, `.dockerignore`, and `uv.lock` are intentionally absent.

## Local run

The supported local runtime is native Windows Python 3.11 with `pip` (no Docker, no `uv`):

```powershell
python -m flask --app wsgi run --host 127.0.0.1 --port 5001
```

Local `.env` selects the live DeepSeek provider:

```dotenv
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_BASE=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-own-key
AGENT_TIMEOUT_SECONDS=120
```

`deepseek` defaults to `LLM_MODEL=deepseek-chat` and `LLM_API_BASE=https://api.deepseek.com/v1` when left empty; only `LLM_API_KEY` is required. Use `deepseek-chat` for the Agent loop, not `deepseek-reasoner`.

Check readiness with `curl.exe --fail http://127.0.0.1:5001/health`. For a production-like local check, set `REQUIRE_LIVE_LLM=true` with a server-side provider secret and verify `gunicorn --check-config --config gunicorn.conf.py wsgi:app` plus `python -m scripts.verify_runtime`; do not put API keys in browser code or source control.

## Production artifact

The runtime model is `artifacts/production-model.joblib`, SHA-256 `2e0dd9f4f57554efca6d0dcdc0af5af79f9a7abfc383f70569e1662afb6d4791`, version `malware-lightgbm-20261028-v1`, size 400,354 bytes. The manifest is `artifacts/production-manifest.json`. Joblib artifacts must be treated as trusted code-bearing files; verify the manifest hash before loading.

## Remaining submitter actions — status 2026-11-22

1. Add `RENDER_DEPLOY_HOOK_URL` and `RENDER_PUBLIC_URL` (`https://sentinellab.onrender.com`) to GitHub's `production` environment if the gated deploy job is to promote new commits automatically.
2. Record the demonstration video and complete the course dashboard submission (see `reports/submission-checklist.md`).
3. Add `quantic-grader` as a repository collaborator.
