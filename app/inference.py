"""Trusted production-artifact loading and schema-validated inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.contracts import ModelSchema, load_model_schema, validate_inference_record


class ModelLoadError(RuntimeError):
    """Raised when the trusted production bundle fails integrity validation."""


@dataclass(frozen=True)
class Prediction:
    predicted_label: int
    class_name: str
    malware_score: float
    threshold: float
    model_version: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ModelService:
    def __init__(
        self,
        artifact_path: Path = Path("artifacts/production-model.joblib"),
        manifest_path: Path = Path("artifacts/production-manifest.json"),
        schema_path: Path = Path("data/metadata/model-schema.json"),
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256(artifact_path) != manifest["artifact_sha256"]:
            raise ModelLoadError("Production artifact hash mismatch")
        if _sha256(schema_path) != manifest["model_schema_sha256"]:
            raise ModelLoadError("Production schema hash mismatch")
        bundle = joblib.load(artifact_path)
        required = {"pipeline", "model_version", "threshold", "class_mapping", "feature_names"}
        if not required <= set(bundle):
            raise ModelLoadError("Production bundle is incomplete")
        self.schema: ModelSchema = load_model_schema(schema_path)
        if tuple(bundle["feature_names"]) != self.schema.feature_names:
            raise ModelLoadError("Artifact feature names differ from saved schema")
        self._pipeline = bundle["pipeline"]
        self.model_version = str(bundle["model_version"])
        self.threshold = float(bundle["threshold"])
        self.class_mapping = {
            int(key): str(value) for key, value in bundle["class_mapping"].items()
        }
        self.artifact_sha256 = manifest["artifact_sha256"]
        metadata = manifest.get("dataset_metadata")
        self.dataset_metadata = dict(metadata) if isinstance(metadata, dict) else None

    def normalize_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return validate_inference_record(record, self.schema)

    def predict_many(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        if not records:
            return []
        normalized = [self.normalize_record(record) for record in records]
        frame = pd.DataFrame.from_records(normalized, columns=self.schema.feature_names)
        scores = np.asarray(self._pipeline.predict_proba(frame)[:, 1], dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("Production model returned invalid malware scores")
        labels = (scores >= self.threshold).astype(np.int8)
        return [
            Prediction(
                predicted_label=int(label),
                class_name=self.class_mapping[int(label)],
                malware_score=float(score),
                threshold=self.threshold,
                model_version=self.model_version,
            )
            for label, score in zip(labels, scores, strict=True)
        ]

    def predict_one(self, record: Mapping[str, Any]) -> Prediction:
        return self.predict_many([record])[0]

    def readiness(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model_version": self.model_version,
            "artifact_sha256": self.artifact_sha256,
            "feature_count": len(self.schema.feature_names),
            "threshold": self.threshold,
            "dataset_metadata": self.dataset_metadata,
        }


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Load and cache the trusted model once per application process."""

    return ModelService()
