from __future__ import annotations

from typing import Any

from app.agent import AgentRunner
from app.agent_config import load_agent_config
from app.agent_protocol import ModelTurn, ScriptedMockProvider, ToolCall


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def predict_single(self, record: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("predict_single", (record,)))
        return {
            "ok": True,
            "data": {"predicted_label": 1},
            "activity": {"tool": "predict_single", "status": "succeeded"},
        }

    def predict_batch(self, session_id: str, reference: str) -> dict[str, Any]:
        self.calls.append(("predict_batch", (session_id, reference)))
        return {
            "ok": True,
            "data": {"summary": {"input_rows": 2}},
            "activity": {"tool": "predict_batch", "status": "succeeded"},
        }

    def evaluate_labeled(self, session_id: str, reference: str) -> dict[str, Any]:
        self.calls.append(("evaluate_labeled", (session_id, reference)))
        return {
            "ok": False,
            "error": {"code": "missing_labels", "message": "Label is missing"},
            "activity": {
                "tool": "evaluate_labeled",
                "status": "failed",
                "error_code": "missing_labels",
            },
        }


def make_runner(
    turns: list[ModelTurn], *, max_tool_calls: int = 6
) -> tuple[AgentRunner, FakeTools]:
    config = load_agent_config(
        {"LLM_PROVIDER": "mock", "AGENT_MAX_TOOL_CALLS": str(max_tool_calls)}
    )
    tools = FakeTools()
    agent = AgentRunner(config, ScriptedMockProvider(turns), tools)  # type: ignore[arg-type]
    return agent, tools


def test_loop_executes_tool_and_returns_final_answer() -> None:
    agent, tools = make_runner(
        [
            ModelTurn(tool_calls=(ToolCall("call-1", "predict_batch", {"file_reference": "f"}),)),
            ModelTurn(text="Two rows were processed."),
        ]
    )
    result = agent.run("session", "Process my file")

    assert result.ok
    assert result.answer == "Two rows were processed."
    assert tools.calls == [("predict_batch", ("session", "f"))]
    assert result.tool_calls == 1


def test_unknown_tool_is_rejected_without_dispatch() -> None:
    agent, tools = make_runner(
        [ModelTurn(tool_calls=(ToolCall("call-1", "shell", {"command": "whoami"}),))]
    )
    result = agent.run("session", "Run a command")

    assert not result.ok
    assert result.error_code == "tool_failure"
    assert tools.calls == []


def test_invalid_arguments_are_rejected() -> None:
    agent, tools = make_runner(
        [ModelTurn(tool_calls=(ToolCall("call-1", "predict_batch", {"path": "/tmp/a"}),))]
    )
    result = agent.run("session", "Process a path")

    assert not result.ok
    assert result.error_code == "tool_failure"
    assert tools.calls == []


def test_tool_failure_stops_without_success_answer() -> None:
    provider = ScriptedMockProvider(
        [
            ModelTurn(
                tool_calls=(ToolCall("call-1", "evaluate_labeled", {"file_reference": "bad"}),)
            ),
            ModelTurn(text="Everything succeeded"),
        ]
    )
    config = load_agent_config({"LLM_PROVIDER": "mock"})
    tools = FakeTools()
    agent = AgentRunner(config, provider, tools)  # type: ignore[arg-type]
    result = agent.run("session", "Evaluate")

    assert not result.ok
    assert result.error_code == "tool_failure"
    assert "failed" in result.answer
    assert len(provider.requests) == 1


def test_tool_call_limit_is_enforced() -> None:
    agent, tools = make_runner(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("call-1", "predict_single", {"record": {}}),
                    ToolCall("call-2", "predict_single", {"record": {}}),
                )
            )
        ],
        max_tool_calls=1,
    )
    result = agent.run("session", "Predict twice")

    assert not result.ok
    assert result.error_code == "tool_limit"
    assert len(tools.calls) == 1
