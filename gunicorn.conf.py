"""Conservative production defaults for a model-loaded Flask process."""

from __future__ import annotations

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("WEB_THREADS", "2"))
timeout = int(os.environ.get("WEB_TIMEOUT_SECONDS", "120"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True
preload_app = True
max_requests = 1000
max_requests_jitter = 50
