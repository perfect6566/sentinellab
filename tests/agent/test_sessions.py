from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent import AgentRunner
from app.agent_config import load_agent_config
from app.agent_protocol import ModelTurn, ScriptedMockProvider
from app.files import SessionFileStore
from app.inference import ModelService
from app.sessions import SessionResultError, SessionResultStore
from app.tools import ProjectTools


def test_result_is_session_owned() -> None:
    store = SessionResultStore()
    result = store.put("session-a", "evaluate_labeled", {"accuracy": 0.99})

    assert store.get("session-a", result.result_id).data["accuracy"] == 0.99
    with pytest.raises(SessionResultError):
        store.get("session-b", result.result_id)


def test_expired_result_is_unavailable() -> None:
    store = SessionResultStore(ttl_seconds=1)
    result = store.put("session", "predict_single", {"predicted_label": 1})
    store.purge_expired(now=result.created_at + 1)

    with pytest.raises(SessionResultError):
        store.get("session", result.result_id)


def test_batch_storage_omits_row_preview() -> None:
    store = SessionResultStore()
    result = store.put(
        "session",
        "predict_batch",
        {
            "summary": {"input_rows": 2},
            "rows": [{"large": "content"}],
            "result_file_reference": "result-id",
            "model_version": "v1",
        },
    )

    assert "rows" not in result.data
    assert result.data["summary"] == {"input_rows": 2}


def test_oldest_result_is_evicted_at_session_limit() -> None:
    store = SessionResultStore(max_results_per_session=1)
    first = store.put("session", "predict_single", {"predicted_label": 0})
    second = store.put("session", "predict_single", {"predicted_label": 1})

    with pytest.raises(SessionResultError):
        store.get("session", first.result_id)
    assert store.get("session", second.result_id).data["predicted_label"] == 1


def test_follow_up_receives_session_result_with_false_negatives() -> None:
    store = SessionResultStore()
    stored = store.put(
        "session",
        "evaluate_labeled",
        {"accuracy": 0.98, "confusion_matrix": {"fn": 3, "fp": 1}},
    )
    provider = ScriptedMockProvider([ModelTurn(text="There were 3 false negatives.")])
    agent = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        object(),  # type: ignore[arg-type]
        store,
    )

    result = agent.run("session", "How many false negatives?", context_result_id=stored.result_id)

    assert result.ok
    assert result.answer == "There were 3 false negatives."
    context = provider.requests[0][0][1].content
    assert context is not None and '"fn": 3' in context


def test_follow_up_rejects_cross_session_result() -> None:
    store = SessionResultStore()
    stored = store.put("session-a", "predict_single", {"predicted_label": 1})
    provider = ScriptedMockProvider([ModelTurn(text="should not be used")])
    agent = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        object(),  # type: ignore[arg-type]
        store,
    )

    result = agent.run("session-b", "What was it?", context_result_id=stored.result_id)

    assert not result.ok
    assert result.error_code == "missing_context"
    assert provider.requests == []


def test_follow_up_can_ground_the_validated_feature_count() -> None:
    """A follow-up that states the validated feature count must not be rejected.

    The count reaches the follow-up only through the stored result data, so the
    response guard has to treat it as grounded. The result is produced by the
    real tool, because that is where the count used to be missing.
    """
    tools = ProjectTools(ModelService(), SessionFileStore(), max_rows=100)
    record = json.loads(Path("samples/single-record.json").read_text(encoding="utf-8"))["record"]
    tool_result = tools.predict_single(record)
    assert tool_result["ok"]
    feature_count = tool_result["data"]["validated_feature_count"]

    store = SessionResultStore()
    stored = store.put("session", "predict_single", tool_result["data"])
    provider = ScriptedMockProvider(
        [ModelTurn(text=f"The prediction validated {feature_count} PE features.")]
    )
    agent = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        tools,
        store,
    )

    result = agent.run(
        "session",
        "How many PE features were validated for that prediction?",
        context_result_id=stored.result_id,
    )

    assert result.ok
    assert result.error_code is None
    assert str(feature_count) in result.answer
