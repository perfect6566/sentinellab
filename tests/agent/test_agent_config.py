from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent_config import AgentConfigurationError, load_agent_config
from app.agent_protocol import (
    AgentMessage,
    CodexCLIProvider,
    ModelTurn,
    ScriptedMockProvider,
    agent_bootstrap_messages,
    registered_tools,
)


def test_mock_configuration_is_ready_without_credentials() -> None:
    config = load_agent_config({"LLM_PROVIDER": "mock"})

    assert config.is_mock
    assert config.is_ready
    assert config.public_status() == {
        "provider": "mock",
        "model": None,
        "mode": "mock",
        "ready": True,
    }
    assert "api_key" not in config.public_status()


def test_live_configuration_reports_missing_values_without_exposing_secrets() -> None:
    with pytest.raises(AgentConfigurationError, match="LLM_MODEL, LLM_API_BASE, LLM_API_KEY"):
        load_agent_config({"LLM_PROVIDER": "openai_compatible"})


def test_live_configuration_is_externalized_and_status_is_safe() -> None:
    config = load_agent_config(
        {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": "test-model",
            "LLM_API_BASE": "https://example.invalid/v1/",
            "LLM_API_KEY": "super-secret",
        }
    )

    assert config.api_base == "https://example.invalid/v1"
    assert config.api_key == "super-secret"
    assert "super-secret" not in repr(config.public_status())


def test_deepseek_configuration_uses_defaults_and_needs_only_api_key() -> None:
    from app.agent_config import DEEPSEEK_DEFAULT_API_BASE, DEEPSEEK_DEFAULT_MODEL

    config = load_agent_config({"LLM_PROVIDER": "deepseek", "LLM_API_KEY": "sk-test"})

    assert config.is_deepseek
    assert config.is_live_provider
    assert config.is_ready
    assert config.api_base == DEEPSEEK_DEFAULT_API_BASE
    assert config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.public_status()["mode"] == "deepseek"
    assert "sk-test" not in repr(config.public_status())


def test_deepseek_configuration_requires_api_key() -> None:
    with pytest.raises(AgentConfigurationError, match="LLM_API_KEY"):
        load_agent_config({"LLM_PROVIDER": "deepseek"})


def test_codex_cli_configuration_needs_no_api_key(monkeypatch) -> None:
    monkeypatch.setattr("app.agent_protocol.shutil.which", lambda _: "/usr/bin/codex")
    config = load_agent_config({"LLM_PROVIDER": "codex_cli"}, require_live=True)

    assert config.is_codex_cli
    assert config.is_ready
    assert config.public_status()["mode"] == "local_codex"
    assert CodexCLIProvider(config).executable


def test_codex_cli_provider_parses_structured_tool_call(monkeypatch) -> None:
    monkeypatch.setattr("app.agent_protocol.shutil.which", lambda _: "/usr/bin/codex")
    config = load_agent_config({"LLM_PROVIDER": "codex_cli"})
    provider = CodexCLIProvider(config)

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "text": None,
                    "tool_calls": [
                        {
                            "call_id": "call-1",
                            "name": "predict_batch",
                            "arguments_json": '{"file_reference":"safe-ref"}',
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert "--sandbox" in command and "read-only" in command
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("app.agent_protocol.subprocess.run", fake_run)
    turn = provider.complete([AgentMessage("user", "process it")], registered_tools())

    assert turn.tool_calls[0].name == "predict_batch"
    assert turn.tool_calls[0].arguments == {"file_reference": "safe-ref"}


def test_mock_provider_receives_exactly_three_registered_tools() -> None:
    expected = ModelTurn(text="mock response")
    provider = ScriptedMockProvider([expected])
    messages = agent_bootstrap_messages("Evaluate my uploaded file")
    tools = registered_tools()

    assert provider.complete(messages, tools) == expected
    assert {tool["name"] for tool in tools} == {
        "predict_single",
        "predict_batch",
        "evaluate_labeled",
    }
    assert "never invent" in messages[0].content
    assert "untrusted data" in messages[0].content


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AGENT_MAX_TURNS", "0"),
        ("AGENT_MAX_TOOL_CALLS", "not-an-int"),
        ("AGENT_TIMEOUT_SECONDS", "-1"),
    ],
)
def test_agent_limits_must_be_positive(name: str, value: str) -> None:
    with pytest.raises(AgentConfigurationError):
        load_agent_config({"LLM_PROVIDER": "mock", name: value})
