"""Bounded Agent loop that exposes only the three production model tools."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from app.agent_config import AgentConfig
from app.agent_protocol import AgentMessage, LLMProvider, agent_bootstrap_messages, registered_tools
from app.response_guard import validate_final_answer
from app.sessions import SessionResultError, SessionResultStore
from app.tools import ProjectTools


@dataclass(frozen=True)
class AgentRunResult:
    ok: bool
    answer: str
    activities: tuple[dict[str, Any], ...]
    turns: int
    tool_calls: int
    error_code: str | None = None
    result_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConditionalPrediction:
    minimum_accuracy: float
    evaluation_file_reference: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_accuracy <= 1.0:
            raise ValueError("minimum_accuracy must be between 0 and 1")
        if not self.evaluation_file_reference:
            raise ValueError("evaluation_file_reference is required")


class AgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tools: ProjectTools,
        results: SessionResultStore | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.results = results or SessionResultStore()
        self.schemas = registered_tools()
        self.allowed_tools = {str(schema["name"]) for schema in self.schemas}

    @staticmethod
    def _invalid_arguments(name: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": name,
            "error": {"code": "invalid_arguments", "message": message, "details": {}},
            "activity": {
                "tool": name,
                "status": "failed",
                "error_code": "invalid_arguments",
            },
        }

    def _execute(
        self,
        session_id: str,
        name: str,
        arguments: dict[str, Any],
        condition: ConditionalPrediction | None,
        evaluated_accuracy: float | None,
    ) -> dict[str, Any]:
        if name not in self.allowed_tools:
            return self._invalid_arguments(name, "Unknown or unavailable tool")
        if name == "predict_single":
            if set(arguments) != {"record"} or not isinstance(arguments["record"], dict):
                return self._invalid_arguments(name, "Expected exactly one object argument: record")
            if condition is not None and (
                evaluated_accuracy is None or evaluated_accuracy < condition.minimum_accuracy
            ):
                return self._invalid_arguments(
                    name,
                    "Conditional prediction is not allowed until evaluation succeeds "
                    "and reaches the required accuracy",
                )
            return self.tools.predict_single(arguments["record"])
        if set(arguments) != {"file_reference"} or not isinstance(arguments["file_reference"], str):
            return self._invalid_arguments(
                name, "Expected exactly one string argument: file_reference"
            )
        if (
            name == "evaluate_labeled"
            and condition is not None
            and arguments["file_reference"] != condition.evaluation_file_reference
        ):
            return self._invalid_arguments(
                name, "The conditional task must evaluate the designated labeled file"
            )
        method = getattr(self.tools, name)
        return method(session_id, arguments["file_reference"])

    @staticmethod
    def _safe_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return a compact, display-safe copy of tool-call arguments."""
        if name == "predict_single":
            record = arguments.get("record")
            count = len(record) if isinstance(record, dict) else 0
            return {"record": f"<{count} PE features>"}
        if name in {"predict_batch", "evaluate_labeled"}:
            reference = arguments.get("file_reference")
            return {"file_reference": str(reference)} if reference is not None else {}
        return dict(arguments)

    @staticmethod
    def _failure(
        code: str,
        message: str,
        activities: list[dict[str, Any]],
        turns: int,
        tool_calls: int,
        result_ids: list[str] | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            ok=False,
            answer=message,
            activities=tuple(activities),
            turns=turns,
            tool_calls=tool_calls,
            error_code=code,
            result_ids=tuple(result_ids or ()),
        )

    def run(
        self,
        session_id: str,
        user_request: str,
        *,
        condition: ConditionalPrediction | None = None,
        context_result_id: str | None = None,
    ) -> AgentRunResult:
        messages = agent_bootstrap_messages(user_request)
        grounded_results: list[dict[str, Any]] = []
        trusted_metadata: dict[str, Any] | None = getattr(
            getattr(self.tools, "model", None), "dataset_metadata", None
        )
        if trusted_metadata:
            messages.insert(
                1,
                AgentMessage(
                    role="system",
                    content=(
                        "Trusted, versioned training and official hold-out metadata for the "
                        "currently loaded production model. Use these facts to answer questions "
                        "about dataset sizes and split design. Do not call them metrics for a "
                        "user-uploaded CSV. Metadata: "
                        + json.dumps(trusted_metadata, separators=(",", ":"))
                    ),
                ),
            )
        evaluated_accuracy: float | None = None
        if condition is not None:
            grounded_results.append({"minimum_accuracy": condition.minimum_accuracy})
        if context_result_id:
            try:
                previous = self.results.get(session_id, context_result_id)
            except SessionResultError:
                return self._failure(
                    "missing_context",
                    "The saved server-side result has expired or was cleared by a restart. "
                    "Please rerun that analysis before asking a follow-up.",
                    [],
                    0,
                    0,
                )
            messages.insert(
                1,
                AgentMessage(
                    role="system",
                    content=(
                        "The following is a trusted, server-retrieved result object for this "
                        f"session. Answer follow-ups only from it: {json.dumps(previous.data)}"
                    ),
                ),
            )
            grounded_results.append(previous.data)
            if (
                condition is not None
                and previous.tool == "evaluate_labeled"
                and previous.data.get("file_reference") == condition.evaluation_file_reference
            ):
                accuracy = previous.data.get("accuracy")
                if isinstance(accuracy, (int, float)):
                    evaluated_accuracy = float(accuracy)
        activities: list[dict[str, Any]] = []
        result_ids: list[str] = []
        tool_call_count = 0
        started = time.monotonic()
        for turn_number in range(1, self.config.max_turns + 1):
            if time.monotonic() - started >= self.config.timeout_seconds:
                return self._failure(
                    "timeout",
                    "The Agent timed out before completing the request.",
                    activities,
                    turn_number - 1,
                    tool_call_count,
                    result_ids,
                )
            try:
                model_turn = self.provider.complete(messages, self.schemas)
            except RuntimeError as exc:
                return self._failure(
                    "provider_failure",
                    f"The language-model service failed: {exc}",
                    activities,
                    turn_number,
                    tool_call_count,
                    result_ids,
                )
            messages.append(
                AgentMessage(
                    role="assistant",
                    content=model_turn.text,
                    tool_calls=model_turn.tool_calls,
                )
            )
            if not model_turn.tool_calls:
                if model_turn.text:
                    # A conditional task is only complete when the prediction ran
                    # after the evaluation met its threshold. The gate already
                    # blocks a prediction that is not allowed; this stops the
                    # opposite case, where the model ends the turn without
                    # running the prediction the condition calls for.
                    if (
                        condition is not None
                        and evaluated_accuracy is not None
                        and evaluated_accuracy >= condition.minimum_accuracy
                        and not any(
                            activity.get("tool") == "predict_single" for activity in activities
                        )
                    ):
                        messages.append(
                            AgentMessage(
                                role="user",
                                content=(
                                    "The evaluation met the required accuracy, so the "
                                    "conditional prediction is still outstanding. Call "
                                    "predict_single with the supplied record before "
                                    "answering, then report the result."
                                ),
                            )
                        )
                        continue
                    guard_error = validate_final_answer(
                        model_turn.text, grounded_results, trusted_metadata=trusted_metadata
                    )
                    if guard_error:
                        return self._failure(
                            "ungrounded_response",
                            guard_error,
                            activities,
                            turn_number,
                            tool_call_count,
                            result_ids,
                        )
                    return AgentRunResult(
                        ok=True,
                        answer=model_turn.text,
                        activities=tuple(activities),
                        turns=turn_number,
                        tool_calls=tool_call_count,
                        result_ids=tuple(result_ids),
                    )
                return self._failure(
                    "empty_response",
                    "The Agent returned no answer or tool call.",
                    activities,
                    turn_number,
                    tool_call_count,
                    result_ids,
                )
            for call in model_turn.tool_calls:
                if tool_call_count >= self.config.max_tool_calls:
                    return self._failure(
                        "tool_limit",
                        "The Agent exceeded the tool-call limit.",
                        activities,
                        turn_number,
                        tool_call_count,
                        result_ids,
                    )
                tool_call_count += 1
                result = self._execute(
                    session_id,
                    call.name,
                    call.arguments,
                    condition,
                    evaluated_accuracy,
                )
                activity_entry = dict(result["activity"])
                activity_entry["arguments"] = self._safe_arguments(call.name, call.arguments)
                if (
                    condition is not None
                    and call.name == "evaluate_labeled"
                    and call.arguments.get("file_reference") == condition.evaluation_file_reference
                ):
                    activity_entry["minimum_accuracy"] = condition.minimum_accuracy
                activities.append(activity_entry)
                if result.get("ok") and call.name == "evaluate_labeled":
                    accuracy = result.get("data", {}).get("accuracy")
                    if isinstance(accuracy, (int, float)):
                        evaluated_accuracy = float(accuracy)
                if result.get("ok") and isinstance(result.get("data"), dict):
                    grounded_results.append(result["data"])
                    grounded_results.append(dict(result.get("activity") or {}))
                    stored = self.results.put(session_id, call.name, result["data"])
                    result_ids.append(stored.result_id)
                    activities[-1]["result_id"] = stored.result_id
                messages.append(
                    AgentMessage(
                        role="tool",
                        content=json.dumps(result, separators=(",", ":"), default=str),
                        tool_call_id=call.call_id,
                        name=call.name,
                    )
                )
                if not result.get("ok"):
                    return self._failure(
                        "tool_failure",
                        f"Tool {call.name} failed: {result['error']['message']}",
                        activities,
                        turn_number,
                        tool_call_count,
                        result_ids,
                    )
        return self._failure(
            "turn_limit",
            "The Agent exceeded the maximum number of turns.",
            activities,
            self.config.max_turns,
            tool_call_count,
            result_ids,
        )
