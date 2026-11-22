"""Versioned JSON schemas exposed to the runtime LLM."""

from __future__ import annotations

from ml.contracts import load_model_schema


def tool_schemas() -> list[dict[str, object]]:
    schema = load_model_schema()
    properties = {}
    required = []
    for feature in schema.features:
        properties[feature.name] = {
            "type": ["number", "string", "null"]
            if feature.missing_allowed
            else ["number", "string"],
            "description": f"{feature.kind} PE feature",
        }
        required.append(feature.name)
    return [
        {
            "name": "predict_single",
            "description": "Classify one complete PE feature record using the production ML model.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "record": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    }
                },
                "required": ["record"],
                "additionalProperties": False,
            },
        },
        {
            "name": "predict_batch",
            "description": "Classify a session-owned uploaded CSV by server-side file reference.",
            "input_schema": {
                "type": "object",
                "properties": {"file_reference": {"type": "string"}},
                "required": ["file_reference"],
                "additionalProperties": False,
            },
        },
        {
            "name": "evaluate_labeled",
            "description": (
                "Evaluate the production model on a session-owned CSV containing true Label values."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"file_reference": {"type": "string"}},
                "required": ["file_reference"],
                "additionalProperties": False,
            },
        },
    ]
