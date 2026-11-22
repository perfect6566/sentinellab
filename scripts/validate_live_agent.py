"""Run local, real-LLM Agent prevalidation before web development."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.agent import AgentRunner, AgentRunResult, ConditionalPrediction
from app.agent_config import AgentConfigurationError, load_agent_config
from app.agent_protocol import build_provider
from app.files import SessionFileStore, new_session_id
from app.sessions import SessionResultStore
from app.tools import ProjectTools

ROOT = Path(__file__).resolve().parents[1]


def _result_summary(name: str, result: AgentRunResult) -> dict[str, Any]:
    actual_tools = [activity.get("tool") for activity in result.activities]
    expected_tools = {
        "predict_single": ["predict_single"],
        "predict_batch": ["predict_batch"],
        "evaluate_labeled": ["evaluate_labeled"],
        "conditional_threshold_met": ["evaluate_labeled", "predict_single"],
        "conditional_threshold_not_met": ["evaluate_labeled"],
    }[name]
    return {
        "scenario": name,
        "ok": result.ok,
        "behavior_ok": result.ok and actual_tools == expected_tools,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "answer": result.answer,
        "activities": list(result.activities),
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "error_code": result.error_code,
    }


def _flip_labels(source: Path) -> bytes:
    rows = list(csv.DictReader(io.StringIO(source.read_text(encoding="utf-8-sig"))))
    if not rows or "Label" not in rows[0]:
        raise ValueError("Labeled sample is empty or has no Label column")
    for row in rows:
        row["Label"] = "0" if row["Label"] == "1" else "1"
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def run_validation() -> list[dict[str, Any]]:
    load_dotenv(ROOT / ".env")
    config = load_agent_config(require_live=True)
    provider = build_provider(config)
    session_id = new_session_id()
    results = SessionResultStore()
    scenarios: list[dict[str, Any]] = []
    record = json.loads((ROOT / "samples/single-record.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="malware-agent-live-") as directory:
        files = SessionFileStore(Path(directory))
        tools = ProjectTools(files=files)
        runner = AgentRunner(config, provider, tools, results)
        batch = files.put_csv(
            session_id,
            "batch-sample.csv",
            (ROOT / "samples/batch-sample.csv").read_bytes(),
        )
        labeled = files.put_csv(
            session_id,
            "labeled-sample.csv",
            (ROOT / "samples/labeled-sample.csv").read_bytes(),
        )
        low_accuracy = files.put_csv(
            session_id,
            "labels-flipped-for-flow-test.csv",
            _flip_labels(ROOT / "samples/labeled-sample.csv"),
        )

        scenarios.append(
            _result_summary(
                "predict_single",
                runner.run(
                    session_id,
                    "Use predict_single for this record: " + json.dumps(record),
                ),
            )
        )
        scenarios.append(
            _result_summary(
                "predict_batch",
                runner.run(
                    session_id,
                    f"Use predict_batch for uploaded file reference {batch.reference}.",
                ),
            )
        )
        scenarios.append(
            _result_summary(
                "evaluate_labeled",
                runner.run(
                    session_id,
                    f"Use evaluate_labeled for uploaded file reference {labeled.reference}.",
                ),
            )
        )
        scenarios.append(
            _result_summary(
                "conditional_threshold_met",
                runner.run(
                    session_id,
                    "Evaluate the labeled file, then predict the supplied record only if accuracy "
                    f"is at least 98%. File reference: {labeled.reference}. Record: "
                    + json.dumps(record),
                    condition=ConditionalPrediction(0.98, labeled.reference),
                ),
            )
        )
        scenarios.append(
            _result_summary(
                "conditional_threshold_not_met",
                runner.run(
                    session_id,
                    "This is controlled flow-test data with deliberately flipped labels. Evaluate "
                    "it, then predict the supplied record only if accuracy is at least 98%. "
                    f"File reference: {low_accuracy.reference}. Record: " + json.dumps(record),
                    condition=ConditionalPrediction(0.98, low_accuracy.reference),
                ),
            )
        )
    return scenarios


def main() -> int:
    try:
        scenarios = run_validation()
    except AgentConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(scenarios, indent=2, ensure_ascii=False))
    return 0 if all(item["behavior_ok"] for item in scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
