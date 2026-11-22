# Environment Report

Checked: 2026-11-21. The supported toolchain is native Python 3.11 + `pip` (no Docker, no `uv`).

## Current supported toolchain

| Item | Result |
|---|---|
| Runtime | Native Python 3.11.9 (`requires-python >=3.11,<3.12`, `.python-version`: `3.11.9`) |
| Package tool | `pip` only (`requirements-runtime.txt` for deploy, `requirements-dev.txt` for training/dev) |
| Service definition | `render.yaml` uses `runtime: python`; `Dockerfile`, `.dockerignore`, `uv.lock` deleted |
| Local run | `python -m flask --app wsgi run --host 127.0.0.1 --port 5001` on Windows Python 3.11 |
| CI run | `actions/setup-python@v5` with Python 3.11 on `ubuntu-24.04` |
| Live deploy | `https://sentinellab.onrender.com` (`runtime: python`, `ready:true`) |

## Reproducibility notes

- Keep raw data, caches and artifacts controlled and excluded from Git unless explicitly approved (see `.gitignore`).
- Verify artifact/application readiness without binding a port using `python -m scripts.verify_runtime`.
- CI installs `requirements-runtime.txt` plus the `dev` extra, compiles sources, runs `pytest -q`, verifies the production model and `/health` wiring, and checks the Gunicorn config.
