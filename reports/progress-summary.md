# Management Progress Summary

Updated: 2026-11-21 (DeepSeek provider live; public Render deploy live at `https://sentinellab.onrender.com`)

## Delivery phases and current state

| Phase | Scope | Current state |
|---|---|---|
| 1. Data preparation and model training | Data acquisition/audit, leakage-safe split and preprocessing, seven-model comparison, frozen selection, independent hold-out and production artifact | Complete. 50,181 source rows; 38,122 development / 9,531 sealed test with no SHA1 overlap; LightGBM `malware-lightgbm-20261028-v1` frozen before hold-out; hold-out AUC 0.995296, Accuracy 0.971042. |
| 2. Agent development | Three production-model tools, LLM tool loop, conditional prediction, follow-ups and failures | Complete. Public DeepSeek (`deepseek-chat`) evaluation passed 9/9 on `https://sentinellab.onrender.com` (evidence `reports/public-agent-evaluation-2026-11-21.json`). |
| 3. Testing and validation | Unit/integration checks, deterministic Agent scenarios, actual-LLM evaluation | Complete. 100 automated tests pass; `ruff check` and `mypy` pass. |
| 4. Website development | Flask chat/upload/results carrier, file/session security | Complete. Chat, upload, result cards, download, metrics, confusion matrix, tool activity (`arguments` + `result_id`), session isolation verified. |
| 5. Deployment and delivery | Gated CI/CD, public deployment, documentation and demo | Public deployment live at `https://sentinellab.onrender.com` (native Python 3.11). CI gate + exact-SHA deploy/health workflow implemented. Recording and dashboard submission remain with the submitter. |

## Verified implementation facts

- Native Python 3.11 with `pip` is the only runtime; `Dockerfile`, `.dockerignore`, and `uv.lock` are intentionally absent.
- Secret-safe `.gitignore`, `.env.example`, pinned `requirements-runtime.txt` / `requirements-dev.txt`, and safety/provenance documentation exist.
- The fold-local preprocessing pipeline transforms numeric, categorical and import-list text features without using sealed test rows.
- The trusted production artifact reloads with zero probability difference and has a recorded SHA-256 manifest.
- `predict_single`, `predict_batch` and `evaluate_labeled` are implemented with schema validation, stable row IDs, invalid-row reporting, single-class AUC handling and session-owned file references.

## Remaining submitter actions

- Add `RENDER_DEPLOY_HOOK_URL` and `RENDER_PUBLIC_URL` to GitHub's `production` environment if the gated deploy job is not yet enabled.
- Redeploy Render once the gated deploy job is enabled, so the hosted service runs the same commit that the test job approved.
- Record the 8–10 minute demonstration and complete grader access plus dashboard submission (see `reports/submission-checklist.md`).
