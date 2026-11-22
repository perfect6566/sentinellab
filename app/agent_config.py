"""Environment-backed configuration for the runtime LLM Agent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


class AgentConfigurationError(ValueError):
    """Raised when runtime Agent configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str | None
    api_base: str | None
    api_key: str | None
    max_turns: int
    max_tool_calls: int
    timeout_seconds: float

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @property
    def is_codex_cli(self) -> bool:
        return self.provider == "codex_cli"

    @property
    def is_deepseek(self) -> bool:
        return self.provider == "deepseek"

    @property
    def is_openai_compatible(self) -> bool:
        return self.provider == "openai_compatible"

    @property
    def is_live_provider(self) -> bool:
        return self.is_openai_compatible or self.is_deepseek

    @property
    def is_ready(self) -> bool:
        return (
            self.is_mock
            or self.is_codex_cli
            or bool(self.model and self.api_base and self.api_key)
        )

    def public_status(self) -> dict[str, object]:
        """Return diagnostics that are safe for logs and health endpoints."""
        if self.is_mock:
            mode = "mock"
        elif self.is_codex_cli:
            mode = "local_codex"
        elif self.is_deepseek:
            mode = "deepseek"
        else:
            mode = "live"
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": mode,
            "ready": self.is_ready,
        }


DEEPSEEK_DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise AgentConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise AgentConfigurationError(f"{name} must be greater than zero")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise AgentConfigurationError(f"{name} must be numeric") from exc
    if value <= 0:
        raise AgentConfigurationError(f"{name} must be greater than zero")
    return value


def load_agent_config(
    environ: Mapping[str, str] | None = None,
    *,
    require_live: bool = False,
) -> AgentConfig:
    """Load Agent settings without ever logging or persisting the API key."""
    values = os.environ if environ is None else environ
    provider = values.get("LLM_PROVIDER", "mock").strip().lower()
    if provider not in {"mock", "codex_cli", "openai_compatible", "deepseek"}:
        raise AgentConfigurationError(
            "LLM_PROVIDER must be 'mock', 'codex_cli', 'openai_compatible', or 'deepseek'"
        )
    if provider == "deepseek":
        api_base = values.get("LLM_API_BASE", "").strip().rstrip("/") or DEEPSEEK_DEFAULT_API_BASE
        model = values.get("LLM_MODEL", "").strip() or DEEPSEEK_DEFAULT_MODEL
        api_key = values.get("LLM_API_KEY", "").strip() or None
    elif provider == "mock":
        api_base = None
        model = None
        api_key = None
    else:
        api_base = values.get("LLM_API_BASE", "").strip().rstrip("/") or None
        model = values.get("LLM_MODEL", "").strip() or None
        api_key = values.get("LLM_API_KEY", "").strip() or None
    config = AgentConfig(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_turns=_positive_int(values, "AGENT_MAX_TURNS", 8),
        max_tool_calls=_positive_int(values, "AGENT_MAX_TOOL_CALLS", 6),
        timeout_seconds=_positive_float(values, "AGENT_TIMEOUT_SECONDS", 60.0),
    )
    if require_live and config.is_mock:
        raise AgentConfigurationError("A real model provider is required for this validation")
    if config.is_live_provider and not config.is_ready:
        missing = [
            name
            for name, value in (
                ("LLM_MODEL", config.model),
                ("LLM_API_BASE", config.api_base),
                ("LLM_API_KEY", config.api_key),
            )
            if not value
        ]
        raise AgentConfigurationError(
            "Live LLM configuration is incomplete; missing " + ", ".join(missing)
        )
    return config
