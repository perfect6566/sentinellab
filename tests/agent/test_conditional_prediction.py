from __future__ import annotations

from typing import Any

from app.agent import AgentRunner, ConditionalPrediction
from app.agent_config import load_agent_config
from app.agent_protocol import ModelTurn, ScriptedMockProvider, ToolCall
from app.sessions import SessionResultStore


class ConditionalTools:
    def __init__(self, accuracy: float) -> None:
        self.accuracy = accuracy
        self.calls: list[str] = []

    def evaluate_labeled(self, session_id: str, reference: str) -> dict[str, Any]:
        self.calls.append("evaluate_labeled")
        return {
            "ok": True,
            "data": {
                "accuracy": self.accuracy,
                "auc": None,
                "auc_reason": "AUC is undefined because only one true class is present",
            },
            "activity": {
                "tool": "evaluate_labeled",
                "status": "succeeded",
                "accuracy": self.accuracy,
            },
        }

    def predict_single(self, record: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("predict_single")
        return {
            "ok": True,
            "data": {"predicted_label": 1},
            "activity": {"tool": "predict_single", "status": "succeeded"},
        }

    def predict_batch(self, session_id: str, reference: str) -> dict[str, Any]:
        raise AssertionError("predict_batch is not expected")


def run_condition(accuracy: float, turns: list[ModelTurn]):
    config = load_agent_config({"LLM_PROVIDER": "mock"})
    provider = ScriptedMockProvider(turns)
    tools = ConditionalTools(accuracy)
    agent = AgentRunner(config, provider, tools)  # type: ignore[arg-type]
    condition = ConditionalPrediction(0.98, "labeled-file")
    result = agent.run("session", "Evaluate, then predict if accuracy is 98%", condition=condition)
    return result, tools


def test_threshold_met_allows_prediction_after_evaluation() -> None:
    result, tools = run_condition(
        0.99,
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("eval-1", "evaluate_labeled", {"file_reference": "labeled-file"}),
                )
            ),
            ModelTurn(tool_calls=(ToolCall("pred-1", "predict_single", {"record": {}}),)),
            ModelTurn(text="Evaluation passed and the prediction is malware."),
        ],
    )

    assert result.ok
    assert tools.calls == ["evaluate_labeled", "predict_single"]


def test_threshold_met_requires_the_prediction_even_if_the_model_answers_early() -> None:
    """Ending the turn before the conditional prediction is not accepted.

    The gate already blocks a prediction that the condition does not allow.
    This covers the opposite case: the evaluation met the threshold, so the
    prediction is outstanding and the Agent is asked to run it before it
    reports the result.
    """
    result, tools = run_condition(
        0.99,
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("eval-1", "evaluate_labeled", {"file_reference": "labeled-file"}),
                )
            ),
            ModelTurn(text="The evaluation passed at 0.99 accuracy, so the file is goodware."),
            ModelTurn(tool_calls=(ToolCall("pred-1", "predict_single", {"record": {}}),)),
            ModelTurn(text="Evaluation passed and the prediction is malware."),
        ],
    )

    assert result.ok
    assert tools.calls == ["evaluate_labeled", "predict_single"]


def test_threshold_not_met_blocks_prediction_even_if_model_requests_it() -> None:
    result, tools = run_condition(
        0.97,
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("eval-1", "evaluate_labeled", {"file_reference": "labeled-file"}),
                )
            ),
            ModelTurn(tool_calls=(ToolCall("pred-1", "predict_single", {"record": {}}),)),
        ],
    )

    assert not result.ok
    assert result.error_code == "tool_failure"
    assert tools.calls == ["evaluate_labeled"]


def test_prediction_cannot_bypass_required_evaluation() -> None:
    result, tools = run_condition(
        1.0,
        [ModelTurn(tool_calls=(ToolCall("pred-1", "predict_single", {"record": {}}),))],
    )

    assert not result.ok
    assert tools.calls == []


def test_prior_matching_evaluation_context_allows_follow_up_prediction() -> None:
    config = load_agent_config({"LLM_PROVIDER": "mock"})
    provider = ScriptedMockProvider(
        [
            ModelTurn(tool_calls=(ToolCall("pred-1", "predict_single", {"record": {}}),)),
            ModelTurn(text="The prediction is malware."),
        ]
    )
    tools = ConditionalTools(1.0)
    results = SessionResultStore()
    prior = results.put(
        "session",
        "evaluate_labeled",
        {"accuracy": 1.0, "file_reference": "labeled-file"},
    )
    agent = AgentRunner(config, provider, tools, results)  # type: ignore[arg-type]

    result = agent.run(
        "session",
        "Use the previous evaluation and predict this record",
        condition=ConditionalPrediction(0.98, "labeled-file"),
        context_result_id=prior.result_id,
    )

    assert result.ok
    assert tools.calls == ["predict_single"]


def test_prior_evaluation_for_another_file_cannot_open_condition_gate() -> None:
    config = load_agent_config({"LLM_PROVIDER": "mock"})
    provider = ScriptedMockProvider(
        [ModelTurn(tool_calls=(ToolCall("pred-1", "predict_single", {"record": {}}),))]
    )
    tools = ConditionalTools(1.0)
    results = SessionResultStore()
    prior = results.put(
        "session",
        "evaluate_labeled",
        {"accuracy": 1.0, "file_reference": "other-file"},
    )
    agent = AgentRunner(config, provider, tools, results)  # type: ignore[arg-type]

    result = agent.run(
        "session",
        "Predict this record",
        condition=ConditionalPrediction(0.98, "labeled-file"),
        context_result_id=prior.result_id,
    )

    assert not result.ok
    assert tools.calls == []


def test_wrong_evaluation_file_is_rejected() -> None:
    result, tools = run_condition(
        1.0,
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("eval-1", "evaluate_labeled", {"file_reference": "other-file"}),
                )
            )
        ],
    )

    assert not result.ok
    assert tools.calls == []


def test_single_class_evaluation_can_report_undefined_auc_without_prediction() -> None:
    result, tools = run_condition(
        0.97,
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("eval-1", "evaluate_labeled", {"file_reference": "labeled-file"}),
                )
            ),
            ModelTurn(text="AUC is undefined because the evaluation has one true class."),
        ],
    )

    assert result.ok
    assert tools.calls == ["evaluate_labeled"]
