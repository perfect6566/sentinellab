"""Load only rows authorized by the sealed split manifest."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ml.contracts import ModelSchema, load_model_schema
from ml.split_data import prepare_rows, sha256_file

Partition = Literal["development", "test"]


class DatasetError(ValueError):
    """Raised when source data and the sealed split cannot be reconciled."""


@dataclass(frozen=True)
class PartitionData:
    features: pd.DataFrame
    labels: np.ndarray
    groups: np.ndarray
    row_ids: np.ndarray


def _load_manifest(
    path: Path, expected_hash: str, partition: Partition
) -> dict[str, tuple[str, int]]:
    if sha256_file(path) != expected_hash:
        raise DatasetError("Split manifest does not match its sealed hash")
    selected: dict[str, tuple[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["partition"] == partition:
                selected[row["row_id"]] = (row["sha1"], int(row["label"]))
    if not selected:
        raise DatasetError(f"No rows found for partition {partition!r}")
    return selected


def load_partition(
    partition: Partition,
    *,
    source_path: Path = Path("data/raw/brazilian-malware.csv"),
    manifest_path: Path = Path("data/splits/holdout-manifest.csv"),
    split_summary_path: Path = Path("reports/split-summary.json"),
    schema: ModelSchema | None = None,
) -> PartitionData:
    """Load a named sealed partition and return model inputs only.

    Callers used for model development must request ``development``. Loading
    ``test`` is reserved for the final evaluation command after configuration
    freeze; this function makes the boundary explicit rather than silently
    mixing partitions.
    """

    if partition not in {"development", "test"}:
        raise DatasetError(f"Unknown partition: {partition!r}")
    summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    if not summary.get("sealed"):
        raise DatasetError("Split summary is not sealed")
    if sha256_file(source_path) != summary.get("source_sha256"):
        raise DatasetError("Raw dataset hash no longer matches the sealed split")

    manifest = _load_manifest(manifest_path, summary["manifest_sha256"], partition)
    header, prepared_rows, _, _ = prepare_rows(source_path)
    column_index = {name: header.index(name) for name in header}
    schema = schema or load_model_schema()

    records: list[dict[str, str]] = []
    labels: list[int] = []
    groups: list[str] = []
    row_ids: list[str] = []
    for row in prepared_rows:
        manifest_item = manifest.get(row.row_id)
        if manifest_item is None:
            continue
        expected_group, expected_label = manifest_item
        if row.sha1 != expected_group or row.label != expected_label:
            raise DatasetError(f"Manifest mismatch for row {row.row_id}")
        records.append({name: row.values[column_index[name]] for name in schema.feature_names})
        labels.append(row.label)
        groups.append(row.sha1)
        row_ids.append(row.row_id)

    if len(records) != len(manifest):
        raise DatasetError(
            f"Mapped {len(records)} source rows but manifest expected {len(manifest)}"
        )
    frame = pd.DataFrame.from_records(records, columns=schema.feature_names)
    for name in schema.numeric_features:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    for name in (*schema.categorical_features, *schema.text_features):
        frame[name] = frame[name].astype("string")

    forbidden = {schema.target_name, schema.row_identifier_name, *schema.excluded_metadata}
    if forbidden & set(frame.columns):
        raise DatasetError(
            f"Forbidden model columns present: {sorted(forbidden & set(frame.columns))}"
        )
    return PartitionData(
        features=frame,
        labels=np.asarray(labels, dtype=np.int8),
        groups=np.asarray(groups, dtype=object),
        row_ids=np.asarray(row_ids, dtype=object),
    )
