"""Shared project configuration and record-schema contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "config" / "project.json"
DEFAULT_MODEL_SCHEMA = PROJECT_ROOT / "data" / "metadata" / "model-schema.json"
MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "?"}


class ContractError(ValueError):
    """Raised when configuration or record input violates the shared contract."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    missing_allowed: bool


@dataclass(frozen=True)
class ProjectConfig:
    project_version: str
    random_seed: int
    label_column: str
    label_mapping: dict[int, str]
    group_column: str
    excluded_metadata_columns: tuple[str, ...]
    holdout_fraction: float
    cv_folds: int
    primary_metric: str
    secondary_metric: str
    default_classification_threshold: float
    confusion_matrix_label_order: tuple[int, ...]


@dataclass(frozen=True)
class ModelSchema:
    schema_version: str
    source_dataset_sha256: str
    target_name: str
    row_identifier_name: str
    excluded_metadata: tuple[str, ...]
    features: tuple[FeatureSpec, ...]
    unknown_field_policy: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features if feature.kind == "numeric")

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features if feature.kind == "categorical")

    @property
    def text_features(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features if feature.kind == "text")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"Contract file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Contract file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"Contract root must be an object: {path}")
    return payload


def load_project_config(path: Path = DEFAULT_PROJECT_CONFIG) -> ProjectConfig:
    data = _read_json(path)
    mapping = {int(key): str(value) for key, value in data["label_mapping"].items()}
    config = ProjectConfig(
        project_version=str(data["project_version"]),
        random_seed=int(data["random_seed"]),
        label_column=str(data["label_column"]),
        label_mapping=mapping,
        group_column=str(data["group_column"]),
        excluded_metadata_columns=tuple(data["excluded_metadata_columns"]),
        holdout_fraction=float(data["holdout_fraction"]),
        cv_folds=int(data["cv_folds"]),
        primary_metric=str(data["primary_metric"]),
        secondary_metric=str(data["secondary_metric"]),
        default_classification_threshold=float(data["default_classification_threshold"]),
        confusion_matrix_label_order=tuple(
            int(item) for item in data["confusion_matrix_label_order"]
        ),
    )
    if not 0 < config.holdout_fraction < 1:
        raise ContractError("holdout_fraction must be between 0 and 1")
    if config.cv_folds < 2:
        raise ContractError("cv_folds must be at least 2")
    if set(config.label_mapping) != {0, 1}:
        raise ContractError("label_mapping must define exactly classes 0 and 1")
    return config


def load_model_schema(path: Path = DEFAULT_MODEL_SCHEMA) -> ModelSchema:
    data = _read_json(path)
    features = tuple(
        FeatureSpec(
            name=str(item["name"]),
            kind=str(item["kind"]),
            missing_allowed=bool(item["missing_allowed"]),
        )
        for item in data["features"]
    )
    names = [feature.name for feature in features]
    if len(names) != len(set(names)):
        raise ContractError("Model schema contains duplicate feature names")
    invalid_kinds = sorted(
        {feature.kind for feature in features} - {"numeric", "categorical", "text"}
    )
    if invalid_kinds:
        raise ContractError(f"Unsupported feature kinds: {invalid_kinds}")
    schema = ModelSchema(
        schema_version=str(data["schema_version"]),
        source_dataset_sha256=str(data["source_dataset_sha256"]),
        target_name=str(data["target"]["name"]),
        row_identifier_name=str(data["row_identifier"]["name"]),
        excluded_metadata=tuple(data["excluded_metadata"]),
        features=features,
        unknown_field_policy=str(data["unknown_field_policy"]),
    )
    if schema.target_name in schema.feature_names:
        raise ContractError("Target column may not appear in model features")
    if schema.row_identifier_name in schema.feature_names:
        raise ContractError("Row identifier may not appear in model features")
    return schema


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in MISSING_STRINGS)


def validate_inference_record(
    record: Mapping[str, Any], schema: ModelSchema | None = None
) -> dict[str, Any]:
    """Validate and normalize one model-input record by field name.

    The target, grouping ID and excluded metadata are rejected rather than silently
    entering the model. Numeric strings are normalized to finite floats. Text and
    categorical values are normalized to strings.
    """

    schema = schema or load_model_schema()
    supplied = set(record)
    expected = set(schema.feature_names)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    if missing:
        raise ContractError(f"Missing required feature fields: {missing}")
    if unexpected:
        raise ContractError(f"Unexpected or non-model fields: {unexpected}")

    normalized: dict[str, Any] = {}
    for feature in schema.features:
        value = record[feature.name]
        if _missing(value):
            if not feature.missing_allowed:
                raise ContractError(f"Feature {feature.name!r} may not be missing")
            # The frozen preprocessing pipeline was fitted with empty strings as
            # the explicit missing categorical value.
            normalized[feature.name] = ""
            continue
        if feature.kind == "numeric":
            if isinstance(value, bool):
                raise ContractError(f"Feature {feature.name!r} must be numeric, not boolean")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ContractError(f"Feature {feature.name!r} must be numeric") from exc
            if not math.isfinite(numeric):
                raise ContractError(f"Feature {feature.name!r} must be finite")
            normalized[feature.name] = numeric
        else:
            normalized[feature.name] = str(value).strip()
    return normalized
