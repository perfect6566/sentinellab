"""Flask application package for model-backed Agent workflows."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask

from app.agent import AgentRunner
from app.agent_config import AgentConfigurationError, load_agent_config
from app.agent_protocol import build_provider
from app.files import SessionFileStore
from app.inference import ModelService, get_model_service
from app.routes import main
from app.sessions import SessionResultStore
from app.tools import ProjectTools


def create_app(
    test_config: dict[str, Any] | None = None,
    *,
    model_loader: Callable[[], ModelService] = get_model_service,
) -> Flask:
    """Create the application and initialize trusted runtime services once."""
    load_dotenv()
    require_live_llm = os.environ.get("REQUIRE_LIVE_LLM", "").lower() in {"1", "true", "yes"}
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "local-development-only"),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024,
        MAX_CSV_ROWS=int(os.environ.get("MAX_CSV_ROWS", "10000")),
        SESSION_FILE_TTL_SECONDS=int(os.environ.get("SESSION_FILE_TTL_SECONDS", "3600")),
        SESSION_FILE_ROOT=str(Path(app.instance_path) / "session-files"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower()
        in {"1", "true", "yes"},
    )
    if test_config:
        app.config.update(test_config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    runtime: dict[str, Any] = {"model": None, "agent": None, "errors": {}}
    if require_live_llm and str(app.config["SECRET_KEY"]).startswith("local-development"):
        runtime["errors"]["configuration"] = "A production SECRET_KEY is required"
    try:
        model = model_loader()
        runtime["model"] = model
    except (OSError, ValueError, RuntimeError) as exc:
        runtime["errors"]["model"] = f"{type(exc).__name__}: {exc}"

    try:
        agent_config = load_agent_config(require_live=require_live_llm)
        runtime["agent_config"] = agent_config
        if runtime["model"] is not None:
            file_store = SessionFileStore(
                Path(app.config["SESSION_FILE_ROOT"]),
                max_bytes=int(app.config["MAX_CONTENT_LENGTH"]),
                max_rows=int(app.config["MAX_CSV_ROWS"]),
            )
            tools = ProjectTools(
                model=runtime["model"],
                files=file_store,
                max_rows=int(app.config["MAX_CSV_ROWS"]),
            )
            runtime["files"] = file_store
            runtime["results"] = SessionResultStore()
            runtime["tools"] = tools
            runtime["agent"] = AgentRunner(
                agent_config,
                build_provider(agent_config),
                tools,
                runtime["results"],
            )
    except (AgentConfigurationError, OSError, ValueError, RuntimeError) as exc:
        runtime["errors"]["agent"] = f"{type(exc).__name__}: {exc}"

    app.extensions["malware_runtime"] = runtime
    app.register_blueprint(main)
    return app
