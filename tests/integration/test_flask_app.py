from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from app import create_app
from app.agent import AgentRunner
from app.agent_config import load_agent_config
from app.agent_protocol import ModelTurn, ScriptedMockProvider, ToolCall


class ReadyModel:
    schema = SimpleNamespace(feature_names=("x",))
    model_version = "test-model-v1"

    @staticmethod
    def readiness():
        return {
            "ready": True,
            "model_version": "test-model-v1",
            "artifact_sha256": "test-hash",
            "feature_count": 1,
            "threshold": 0.5,
        }


def test_health_reports_model_and_agent_readiness(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("RENDER_GIT_COMMIT", "0123456789abcdef")
    app = create_app(
        {"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )

    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.json["ready"] is True
    assert response.json["model"]["model_version"] == "test-model-v1"
    assert response.json["agent"]["mode"] == "mock"
    assert response.json["commit_sha"] == "0123456789abcdef"
    assert "api_key" not in response.get_data(as_text=True).lower()


def test_health_omits_commit_identity_when_not_deployed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "ci-runner-commit-sha")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )

    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert "commit_sha" not in response.json


def test_health_is_not_ready_when_model_load_fails(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    def fail_model():
        raise RuntimeError("artifact is damaged at /secret/internal/path")

    app = create_app({"TESTING": True}, model_loader=fail_model)
    response = app.test_client().get("/health")

    assert response.status_code == 503
    assert response.json == {
        "ready": False,
        "service": "malware-ml-agent",
        "model": {"ready": False},
        "agent": {"provider": "mock", "model": None, "mode": "mock", "ready": True},
        "components_with_errors": ["model"],
    }
    assert "/secret/internal/path" not in response.get_data(as_text=True)


def test_production_mode_rejects_mock_llm_and_local_secret(monkeypatch) -> None:
    monkeypatch.setenv("REQUIRE_LIVE_LLM", "true")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SECRET_KEY", "local-development-only")
    app = create_app(
        {"TESTING": True},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )

    response = app.test_client().get("/health")

    assert response.status_code == 503
    assert response.json["ready"] is False
    assert response.json["components_with_errors"] == ["agent", "configuration"]
    assert "SECRET_KEY" not in response.get_data(as_text=True)


def test_root_has_safe_service_discovery(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )

    response = app.test_client().get("/")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Sentinel Lab" in page
    assert "Agent conversation" in page
    assert "/static/app.css" in page


def test_upload_returns_session_owned_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )
    client = app.test_client()

    response = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"x,Label\n1,0\n"), "evaluation.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["ok"] is True
    assert response.json["file"]["name"] == "evaluation.csv"
    assert response.json["file"]["reference"]


def test_upload_rejects_non_csv_and_malformed_content(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )
    client = app.test_client()

    wrong_type = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"not csv"), "payload.exe", "application/octet-stream")},
        content_type="multipart/form-data",
    )
    malformed = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"a,b\n1\n"), "broken.csv", "text/csv")},
        content_type="multipart/form-data",
    )

    assert wrong_type.status_code == 400
    assert malformed.status_code == 400


def test_chat_endpoint_returns_agent_answer(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )
    provider = ScriptedMockProvider([ModelTurn(text="Which uploaded file should I evaluate?")])
    runtime = app.extensions["malware_runtime"]
    runtime["agent"] = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        object(),  # type: ignore[arg-type]
        runtime["results"],
    )

    response = app.test_client().post("/api/chat", json={"message": "Evaluate it"})

    assert response.status_code == 200
    assert response.json["ok"] is True
    assert response.json["answer"] == "Which uploaded file should I evaluate?"


def test_chat_rejects_bad_condition_without_calling_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )

    response = app.test_client().post(
        "/api/chat",
        json={"message": "Evaluate then predict", "condition": {"minimum_accuracy": "bad"}},
    )

    assert response.status_code == 400
    assert response.json == {"ok": False, "error": "Invalid conditional prediction"}


def test_sample_downloads_and_unknown_sample(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )
    client = app.test_client()

    assert client.get("/samples/single").status_code == 200
    assert client.get("/samples/batch").status_code == 200
    assert client.get("/samples/not-real").status_code == 404


def test_result_detail_and_download_are_session_owned(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )
    runtime = app.extensions["malware_runtime"]
    client = app.test_client()
    with client.session_transaction() as browser_session:
        browser_session["session_id"] = "11111111-1111-4111-8111-111111111111"
    stored_file = runtime["files"].put_csv(
        "11111111-1111-4111-8111-111111111111",
        "predictions.csv",
        b"row_id,status\n1,predicted\n",
        kind="batch_result",
    )
    stored_result = runtime["results"].put(
        "11111111-1111-4111-8111-111111111111",
        "predict_batch",
        {
            "summary": {"input_rows": 1, "valid_rows": 1, "invalid_rows": 0},
            "rows": [{"row_id": "1", "status": "predicted"}],
            "result_file_reference": stored_file.reference,
        },
    )

    detail = client.get(f"/api/results/{stored_result.result_id}")
    download = client.get(f"/api/download/{stored_file.reference}")

    assert detail.status_code == 200
    assert detail.json["result"]["data"]["rows"][0]["status"] == "predicted"
    assert download.status_code == 200
    assert b"predicted" in download.data

    other_client = app.test_client()
    assert other_client.get(f"/api/results/{stored_result.result_id}").status_code == 404
    assert other_client.get(f"/api/download/{stored_file.reference}").status_code == 404


def _install_scripted_agent(app, turns: list[ModelTurn]) -> None:
    runtime = app.extensions["malware_runtime"]
    runtime["agent"] = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        ScriptedMockProvider(turns),
        runtime["tools"],
        runtime["results"],
    )


def test_batch_workflow_through_flask_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app({"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)})
    client = app.test_client()
    upload = client.post(
        "/api/upload",
        data={
            "file": (
                io.BytesIO(Path("samples/batch-sample.csv").read_bytes()),
                "batch.csv",
                "text/csv",
            )
        },
        content_type="multipart/form-data",
    )
    reference = upload.json["file"]["reference"]
    _install_scripted_agent(
        app,
        [
            ModelTurn(
                tool_calls=(ToolCall("batch-1", "predict_batch", {"file_reference": reference}),)
            ),
            ModelTurn(text="Processed 5 valid rows with 0 malware predictions."),
        ],
    )

    chat = client.post("/api/chat", json={"message": "Run batch prediction"})
    detail = client.get(f"/api/results/{chat.json['result_ids'][0]}")
    download_reference = detail.json["result"]["data"]["result_file_reference"]
    download = client.get(f"/api/download/{download_reference}")

    assert chat.status_code == 200
    assert [item["tool"] for item in chat.json["activities"]] == ["predict_batch"]
    assert detail.json["result"]["data"]["summary"]["input_rows"] == 5
    assert download.status_code == 200


def test_labeled_evaluation_through_flask_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app({"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)})
    client = app.test_client()
    upload = client.post(
        "/api/upload",
        data={
            "file": (
                io.BytesIO(Path("samples/labeled-sample.csv").read_bytes()),
                "labeled.csv",
                "text/csv",
            )
        },
        content_type="multipart/form-data",
    )
    reference = upload.json["file"]["reference"]
    _install_scripted_agent(
        app,
        [
            ModelTurn(
                tool_calls=(ToolCall("eval-1", "evaluate_labeled", {"file_reference": reference}),)
            ),
            ModelTurn(text="Accuracy was 100% and AUC was 1.0."),
        ],
    )

    chat = client.post("/api/chat", json={"message": "Evaluate labeled data"})
    detail = client.get(f"/api/results/{chat.json['result_ids'][0]}")

    assert chat.status_code == 200
    assert detail.json["result"]["data"]["accuracy"] == 1.0
    assert detail.json["result"]["data"]["confusion_matrix"]["fn"] == 0


def test_single_prediction_through_flask_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app({"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)})
    record = json.loads(Path("samples/single-record.json").read_text(encoding="utf-8"))["record"]
    _install_scripted_agent(
        app,
        [
            ModelTurn(tool_calls=(ToolCall("single-1", "predict_single", {"record": record}),)),
            ModelTurn(text="The production model classified the record as malware."),
        ],
    )

    chat = app.test_client().post("/api/chat", json={"message": "Predict this record"})

    assert chat.status_code == 200
    assert chat.json["activities"][0]["tool"] == "predict_single"
    assert chat.json["result_ids"]


def test_conversation_ids_isolate_uploads_and_can_be_deleted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = create_app(
        {"TESTING": True, "SESSION_FILE_ROOT": str(tmp_path)},
        model_loader=lambda: ReadyModel(),  # type: ignore[arg-type,return-value]
    )
    client = app.test_client()
    first = "11111111-1111-4111-8111-111111111111"
    second = "22222222-2222-4222-8222-222222222222"
    uploaded = client.post(
        "/api/upload",
        headers={"X-Conversation-ID": first},
        data={"file": (io.BytesIO(b"x\n1\n"), "record.csv")},
        content_type="multipart/form-data",
    )
    reference = uploaded.json["file"]["reference"]

    cross_conversation = client.get(f"/api/download/{reference}?conversation_id={second}")
    deleted = client.delete(f"/api/sessions/{first}")
    after_delete = client.get(f"/api/download/{reference}?conversation_id={first}")

    assert uploaded.status_code == 200
    assert cross_conversation.status_code == 404
    assert deleted.status_code == 200
    assert after_delete.status_code == 404
