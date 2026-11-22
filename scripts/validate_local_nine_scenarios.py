"""Run the nine real-provider Agent scenarios against the local project runtime.

The controlled failure is injected only by this script's test adapter. It does not add a
failure switch to the production application or website.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
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


class AuditedProjectTools(ProjectTools):
    """Capture structured outputs for an auditable local scenario record."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.structured_results: list[dict[str, Any]] = []

    def _capture(self, result: dict[str, Any]) -> dict[str, Any]:
        self.structured_results.append(result)
        return result

    def predict_single(self, record: Any) -> dict[str, Any]:
        return self._capture(super().predict_single(record))

    def predict_batch(self, session_id: str, file_reference: str) -> dict[str, Any]:
        return self._capture(super().predict_batch(session_id, file_reference))

    def evaluate_labeled(self, session_id: str, file_reference: str) -> dict[str, Any]:
        return self._capture(super().evaluate_labeled(session_id, file_reference))


class ControlledFailureTools(AuditedProjectTools):
    """A test-only adapter that fails a valid evaluation call on demand."""

    def evaluate_labeled(self, session_id: str, file_reference: str) -> dict[str, Any]:
        result = self._failure(
            "evaluate_labeled",
            "controlled_failure",
            "Controlled local evaluation-service failure for Agent behavior testing.",
        )
        return self._capture(result)


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


def _scenario(
    name: str,
    prompt: str,
    expected_tools: list[str],
    result: AgentRunResult,
    structured_results: list[dict[str, Any]],
    *,
    expected_success: bool = True,
    expected_tool_error: str | None = None,
) -> dict[str, Any]:
    actual_tools = [item.get("tool") for item in result.activities]
    tool_results = structured_results[-result.tool_calls :] if result.tool_calls else []
    behavior_ok = actual_tools == expected_tools and result.ok is expected_success
    if expected_tool_error:
        behavior_ok = behavior_ok and any(
            item.get("error", {}).get("code") == expected_tool_error for item in tool_results
        )
    return {
        "scenario": name,
        "prompt": prompt,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "must_not_call": sorted(
            set(
                tool
                for tool in ("predict_single", "predict_batch", "evaluate_labeled")
                if tool not in expected_tools
            )
        ),
        "expected_success": expected_success,
        "ok": result.ok,
        "behavior_ok": behavior_ok,
        "error_code": result.error_code,
        "final_answer": result.answer,
        "activities": list(result.activities),
        "structured_tool_results": tool_results,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
    }


def run_validation() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    config = load_agent_config(require_live=True)
    provider = build_provider(config)
    effective_model = config.model
    if effective_model is None and config.is_codex_cli:
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        try:
            codex_settings = tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            codex_settings = {}
        configured_model = codex_settings.get("model")
        if isinstance(configured_model, str):
            effective_model = configured_model
    session_id = new_session_id()
    result_store = SessionResultStore()
    record = json.loads((ROOT / "samples/single-record.json").read_text(encoding="utf-8"))
    scenarios: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="malware-agent-nine-") as directory:
        files = SessionFileStore(Path(directory))
        tools = AuditedProjectTools(files=files)
        runner = AgentRunner(config, provider, tools, result_store)
        batch = files.put_csv(
            session_id, "batch-sample.csv", (ROOT / "samples/batch-sample.csv").read_bytes()
        )
        labeled = files.put_csv(
            session_id, "labeled-sample.csv", (ROOT / "samples/labeled-sample.csv").read_bytes()
        )
        flipped = files.put_csv(
            session_id,
            "labels-flipped-flow-fixture.csv",
            _flip_labels(ROOT / "samples/labeled-sample.csv"),
        )
        invalid = files.put_csv(
            session_id,
            "invalid-feature.csv",
            (ROOT / "samples/evaluation/invalid-feature.csv").read_bytes(),
        )
        missing = files.put_csv(
            session_id,
            "missing-label.csv",
            (ROOT / "samples/evaluation/missing-label.csv").read_bytes(),
        )
        single_class = files.put_csv(
            session_id,
            "single-class.csv",
            (ROOT / "samples/evaluation/single-class.csv").read_bytes(),
        )

        prompt = "Use predict_single to classify this complete feature record: " + json.dumps(
            record
        )
        start = len(tools.structured_results)
        single_result = runner.run(session_id, prompt)
        scenarios.append(
            _scenario(
                "single prediction",
                prompt,
                ["predict_single"],
                single_result,
                tools.structured_results[start:],
            )
        )

        prompt = f"Use predict_batch on uploaded CSV reference {batch.reference}."
        start = len(tools.structured_results)
        batch_result = runner.run(session_id, prompt)
        scenarios.append(
            _scenario(
                "batch prediction",
                prompt,
                ["predict_batch"],
                batch_result,
                tools.structured_results[start:],
            )
        )

        prompt = f"Evaluate this labeled CSV reference: {labeled.reference}"
        start = len(tools.structured_results)
        eval_result = runner.run(session_id, prompt)
        scenarios.append(
            _scenario(
                "labeled evaluation",
                prompt,
                ["evaluate_labeled"],
                eval_result,
                tools.structured_results[start:],
            )
        )

        prompt = (
            "Evaluate this file and predict the supplied record only if accuracy is at least 98%. "
            f"File reference: {labeled.reference}. Record: {json.dumps(record)}"
        )
        start = len(tools.structured_results)
        met_result = runner.run(
            session_id,
            prompt,
            condition=ConditionalPrediction(0.98, labeled.reference),
        )
        scenarios.append(
            _scenario(
                "conditional threshold met",
                prompt,
                ["evaluate_labeled", "predict_single"],
                met_result,
                tools.structured_results[start:],
            )
        )

        prompt = (
            "This fixture has deliberately flipped labels. Evaluate it and predict the supplied "
            "record only if accuracy is at least 98%. "
            f"File reference: {flipped.reference}. Record: {json.dumps(record)}"
        )
        start = len(tools.structured_results)
        not_met_result = runner.run(
            session_id,
            prompt,
            condition=ConditionalPrediction(0.98, flipped.reference),
        )
        scenarios.append(
            _scenario(
                "conditional threshold not met",
                prompt,
                ["evaluate_labeled"],
                not_met_result,
                tools.structured_results[start:],
            )
        )

        if eval_result.result_ids:
            prompt = "How many false negatives were there in that evaluation?"
            start = len(tools.structured_results)
            context_result_id = eval_result.result_ids[-1]
            followup = runner.run(session_id, prompt, context_result_id=context_result_id)
            followup_case = _scenario(
                "false-negative follow-up",
                prompt,
                [],
                followup,
                tools.structured_results[start:],
            )
            followup_case["context_result_id"] = context_result_id
            followup_case["context_result"] = result_store.get(session_id, context_result_id).data
            scenarios.append(followup_case)
        else:
            scenarios.append(
                {
                    "scenario": "false-negative follow-up",
                    "behavior_ok": False,
                    "error": "evaluation produced no stored result ID",
                }
            )

        prompt = f"Use predict_batch on uploaded CSV reference {invalid.reference}."
        start = len(tools.structured_results)
        invalid_result = runner.run(session_id, prompt)
        scenarios.append(
            _scenario(
                "invalid feature input",
                prompt,
                ["predict_batch"],
                invalid_result,
                tools.structured_results[start:],
                expected_success=False,
                expected_tool_error="no_valid_rows",
            )
        )

        prompt = f"Evaluate uploaded CSV reference {missing.reference} using evaluate_labeled."
        start = len(tools.structured_results)
        missing_result = runner.run(session_id, prompt)
        scenarios.append(
            _scenario(
                "missing labels",
                prompt,
                ["evaluate_labeled"],
                missing_result,
                tools.structured_results[start:],
                expected_success=False,
                expected_tool_error="missing_labels",
            )
        )

        prompt = f"Evaluate this one-class labeled CSV reference: {single_class.reference}"
        start = len(tools.structured_results)
        one_class_result = runner.run(session_id, prompt)
        one_class_case = _scenario(
            "single-class evaluation",
            prompt,
            ["evaluate_labeled"],
            one_class_result,
            tools.structured_results[start:],
        )
        one_class_data: dict[str, Any] = next(
            (
                item.get("data", {})
                for item in one_class_case["structured_tool_results"]
                if item.get("ok")
            ),
            {},
        )
        one_class_case["behavior_ok"] = bool(
            one_class_case["behavior_ok"] and one_class_data.get("auc") is None
        )
        one_class_case["expected_auc"] = None
        scenarios.append(one_class_case)

        failing_tools = ControlledFailureTools(files=files)
        failing_runner = AgentRunner(config, provider, failing_tools, result_store)
        prompt = f"Evaluate labeled CSV reference {labeled.reference}."
        failure_result = failing_runner.run(session_id, prompt)
        scenarios.append(
            _scenario(
                "controlled tool failure",
                prompt,
                ["evaluate_labeled"],
                failure_result,
                failing_tools.structured_results,
                expected_success=False,
                expected_tool_error="controlled_failure",
            )
        )

    scenario_ids = {
        "single prediction": "AE-01",
        "batch prediction": "AE-02",
        "conditional threshold met": "AE-03",
        "conditional threshold not met": "AE-04",
        "false-negative follow-up": "AE-05",
        "invalid feature input": "AE-06",
        "missing labels": "AE-07",
        "single-class evaluation": "AE-08",
        "controlled tool failure": "AE-09",
        "labeled evaluation": "LOCAL-01",
    }
    for item in scenarios:
        item["scenario_id"] = scenario_ids[item["scenario"]]
    required_scenarios = [item for item in scenarios if item["scenario_id"].startswith("AE-")]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "local real-provider prevalidation; not final-deployment acceptance",
        "provider": config.public_status(),
        "model_record": effective_model or "provider default could not be read from local config",
        "scenario_count": len(scenarios),
        "passed": sum(bool(item.get("behavior_ok")) for item in scenarios),
        "required_scenario_count": len(required_scenarios),
        "required_scenarios_passed": sum(
            bool(item.get("behavior_ok")) for item in required_scenarios
        ),
        "additional_local_scenario_count": len(scenarios) - len(required_scenarios),
        "failed": [item["scenario"] for item in scenarios if not item.get("behavior_ok")],
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, help="Optional JSON path for the complete audit record"
    )
    args = parser.parse_args()
    try:
        report = run_validation()
    except AgentConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 3
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote local scenario evidence to {args.output}")
    else:
        print(rendered, end="")
    return 0 if report["passed"] == report["scenario_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
