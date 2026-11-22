"""HTTP routes shared by the local and deployed Flask application."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    session,
)

from app.agent import ConditionalPrediction
from app.files import FileStoreError, new_session_id

main = Blueprint("main", __name__)
MAX_CONVERSATIONS_PER_BROWSER = 12


@main.get("/")
def index():
    return render_template("index.html")


def _session_id() -> str:
    conversation_id = request.headers.get("X-Conversation-ID") or request.args.get(
        "conversation_id"
    )
    if conversation_id:
        try:
            conversation_id = str(uuid.UUID(conversation_id))
        except ValueError:
            abort(400, description="Invalid conversation ID")
        conversations = session.setdefault("conversation_sessions", {})
        session_id = conversations.get(conversation_id)
        if session_id is None:
            if len(conversations) >= MAX_CONVERSATIONS_PER_BROWSER:
                abort(400, description="This browser has reached its conversation limit")
            session_id = new_session_id()
            conversations[conversation_id] = session_id
            session.modified = True
        return session_id
    if "session_id" not in session:
        session["session_id"] = new_session_id()
    return session["session_id"]


@main.delete("/api/sessions/<conversation_id>")
def delete_conversation_session(conversation_id: str):
    try:
        normalized_id = str(uuid.UUID(conversation_id))
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid conversation ID"}), 400
    conversations = session.get("conversation_sessions", {})
    if normalized_id not in conversations:
        return jsonify({"ok": True}), 200
    del conversations[normalized_id]
    session["conversation_sessions"] = conversations
    session.modified = True
    return jsonify({"ok": True})


@main.post("/api/upload")
def upload_csv():
    runtime = current_app.extensions["malware_runtime"]
    if runtime.get("files") is None:
        return jsonify({"ok": False, "error": "File service is not ready"}), 503
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "Choose a CSV file to upload"}), 400
    allowed_types = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }
    if uploaded.mimetype not in allowed_types:
        return jsonify({"ok": False, "error": "Only CSV uploads are accepted"}), 400
    try:
        runtime["files"].purge_expired(current_app.config["SESSION_FILE_TTL_SECONDS"])
        stored = runtime["files"].put_csv(
            _session_id(),
            uploaded.filename,
            uploaded.read(),
        )
    except FileStoreError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify(
        {
            "ok": True,
            "file": {
                "reference": stored.reference,
                "name": stored.original_name,
                "byte_size": stored.byte_size,
            },
        }
    )


@main.post("/api/chat")
def chat():
    runtime = current_app.extensions["malware_runtime"]
    agent = runtime.get("agent")
    if agent is None:
        return jsonify({"ok": False, "error": "Agent is not ready"}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Expected a JSON request"}), 400
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"ok": False, "error": "Message is required"}), 400
    context_result_id = payload.get("context_result_id")
    if context_result_id is not None and not isinstance(context_result_id, str):
        return jsonify({"ok": False, "error": "Invalid result context"}), 400
    condition = None
    condition_payload = payload.get("condition")
    if condition_payload is not None:
        try:
            threshold = float(condition_payload["minimum_accuracy"])
            if threshold > 1:
                threshold /= 100
            condition = ConditionalPrediction(
                threshold,
                str(condition_payload["evaluation_file_reference"]),
            )
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid conditional prediction"}), 400
    result = agent.run(
        _session_id(),
        message.strip(),
        condition=condition,
        context_result_id=context_result_id,
    )
    status = 200 if result.ok else 422
    return jsonify(asdict(result)), status


@main.get("/api/results/<result_id>")
def result_detail(result_id: str):
    runtime = current_app.extensions["malware_runtime"]
    results = runtime.get("results")
    if results is None:
        return jsonify({"ok": False, "error": "Result service is not ready"}), 503
    try:
        stored = results.get(_session_id(), result_id)
    except ValueError:
        return jsonify({"ok": False, "error": "Result is unavailable"}), 404
    return jsonify(
        {
            "ok": True,
            "result": {
                "result_id": stored.result_id,
                "tool": stored.tool,
                "data": stored.display_data,
            },
        }
    )


@main.get("/api/download/<file_reference>")
def download_result(file_reference: str):
    runtime = current_app.extensions["malware_runtime"]
    files = runtime.get("files")
    if files is None:
        return jsonify({"ok": False, "error": "File service is not ready"}), 503
    try:
        stored = files.resolve(_session_id(), file_reference)
    except FileStoreError:
        return jsonify({"ok": False, "error": "Result file is unavailable"}), 404
    if stored.kind != "batch_result":
        return jsonify({"ok": False, "error": "File is not a downloadable result"}), 403
    return send_file(stored.path, as_attachment=True, download_name="batch-predictions.csv")


@main.get("/samples/<name>")
def sample_file(name: str):
    allowed = {
        "batch": "batch-sample.csv",
        "labeled": "labeled-sample.csv",
        "single": "single-record.json",
    }
    filename = allowed.get(name)
    if filename is None:
        return jsonify({"ok": False, "error": "Unknown sample"}), 404
    return send_file(Path(current_app.root_path).parent / "samples" / filename, as_attachment=True)


@main.get("/health")
def health():
    runtime = current_app.extensions["malware_runtime"]
    model = runtime.get("model")
    agent = runtime.get("agent")
    agent_config = runtime.get("agent_config")
    model_status = model.readiness() if model is not None else {"ready": False}
    agent_status = (
        agent_config.public_status()
        if agent_config is not None
        else {"provider": None, "model": None, "mode": None, "ready": False}
    )
    ready = bool(
        model_status["ready"]
        and agent is not None
        and agent_status["ready"]
        and not runtime["errors"]
    )
    payload = {
        "ready": ready,
        "service": "malware-ml-agent",
        "model": model_status,
        "agent": agent_status,
    }
    commit_sha = os.environ.get("RENDER_GIT_COMMIT")
    if commit_sha:
        payload["commit_sha"] = commit_sha
    if runtime["errors"]:
        payload["components_with_errors"] = sorted(runtime["errors"])
    return jsonify(payload), 200 if ready else 503
