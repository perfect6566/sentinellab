"""Development-partition-only exploratory data analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.contracts import ModelSchema, load_model_schema
from ml.split_data import PreparedRow, prepare_rows, sha256_file


class EDAError(ValueError):
    """Raised when EDA cannot prove it is using the sealed development partition."""


def load_development_ids(manifest_path: Path) -> tuple[set[str], dict[str, int]]:
    development: set[str] = set()
    counts: Counter[str] = Counter()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"row_id", "sha1", "label", "partition", "source_line"}
        if set(reader.fieldnames or []) != expected:
            raise EDAError("Split manifest has an unexpected schema")
        for row in reader:
            partition = row["partition"]
            if partition not in {"development", "test"}:
                raise EDAError(f"Unknown partition {partition!r}")
            counts[partition] += 1
            if partition == "development":
                development.add(row["row_id"])
    if not development or not counts["test"]:
        raise EDAError("Manifest must contain both development and test rows")
    return development, dict(counts)


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise EDAError("Cannot calculate a quantile of an empty feature")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile_development(
    header: list[str],
    rows: list[PreparedRow],
    development_ids: set[str],
    schema: ModelSchema,
) -> dict[str, Any]:
    index = {name: header.index(name) for name in schema.feature_names}
    development_rows = [row for row in rows if row.row_id in development_ids]
    if len(development_rows) != len(development_ids):
        raise EDAError("Development IDs do not map one-to-one to prepared rows")

    label_counts = Counter(row.label for row in development_rows)
    numeric_values: dict[str, list[float]] = {name: [] for name in schema.numeric_features}
    numeric_missing: Counter[str] = Counter()
    numeric_non_finite: Counter[str] = Counter()
    categorical_values: dict[str, Counter[str]] = {
        name: Counter() for name in schema.categorical_features
    }
    text_lengths: dict[str, list[int]] = {name: [] for name in schema.text_features}
    text_token_counts: dict[str, list[int]] = {name: [] for name in schema.text_features}
    text_unique_hashes: dict[str, set[str]] = {name: set() for name in schema.text_features}
    text_missing: Counter[str] = Counter()

    for row in development_rows:
        for name in schema.numeric_features:
            raw = row.values[index[name]].strip()
            if not raw:
                numeric_missing[name] += 1
                continue
            try:
                value = float(raw)
            except ValueError:
                numeric_non_finite[name] += 1
                continue
            if not math.isfinite(value):
                numeric_non_finite[name] += 1
                continue
            numeric_values[name].append(value)
        for name in schema.categorical_features:
            raw = row.values[index[name]].strip()
            categorical_values[name][raw if raw else "<MISSING>"] += 1
        for name in schema.text_features:
            raw = row.values[index[name]].strip()
            if not raw:
                text_missing[name] += 1
                continue
            text_lengths[name].append(len(raw))
            text_token_counts[name].append(len(raw.split()))
            text_unique_hashes[name].add(_hash_value(raw))

    numeric_profiles: dict[str, dict[str, Any]] = {}
    for name, values in numeric_values.items():
        ordered = sorted(values)
        if not ordered:
            numeric_profiles[name] = {
                "missing_count": numeric_missing[name],
                "non_finite_or_parse_error_count": numeric_non_finite[name],
                "valid_count": 0,
            }
            continue
        q1 = _quantile(ordered, 0.25)
        median = _quantile(ordered, 0.5)
        q3 = _quantile(ordered, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        numeric_profiles[name] = {
            "missing_count": numeric_missing[name],
            "non_finite_or_parse_error_count": numeric_non_finite[name],
            "valid_count": len(ordered),
            "unique_count": len(set(ordered)),
            "min": ordered[0],
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": ordered[-1],
            "mean": statistics.fmean(ordered),
            "standard_deviation": statistics.pstdev(ordered),
            "zero_count": sum(value == 0 for value in ordered),
            "iqr_outlier_count": sum(value < lower or value > upper for value in ordered),
            "constant": ordered[0] == ordered[-1],
        }

    categorical_profiles = {
        name: {
            "missing_count": values["<MISSING>"],
            "unique_count_including_missing": len(values),
            "top_values": [
                {"value": value[:120], "count": count} for value, count in values.most_common(10)
            ],
        }
        for name, values in categorical_values.items()
    }
    text_profiles = {}
    for name in schema.text_features:
        lengths = text_lengths[name]
        tokens = text_token_counts[name]
        text_profiles[name] = {
            "missing_count": text_missing[name],
            "unique_count": len(text_unique_hashes[name]),
            "character_length_min": min(lengths) if lengths else None,
            "character_length_median": _quantile(sorted(lengths), 0.5) if lengths else None,
            "character_length_max": max(lengths) if lengths else None,
            "token_count_min": min(tokens) if tokens else None,
            "token_count_median": _quantile(sorted(tokens), 0.5) if tokens else None,
            "token_count_max": max(tokens) if tokens else None,
        }

    return {
        "development_rows_profiled": len(development_rows),
        "test_rows_profiled": 0,
        "label_counts": {str(label): label_counts[label] for label in (0, 1)},
        "label_fractions": {
            str(label): label_counts[label] / len(development_rows) for label in (0, 1)
        },
        "numeric": numeric_profiles,
        "categorical": categorical_profiles,
        "text": text_profiles,
        "constant_numeric_features": sorted(
            name for name, profile in numeric_profiles.items() if profile.get("constant")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/brazilian-malware.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/holdout-manifest.csv"))
    parser.add_argument("--split-summary", type=Path, default=Path("reports/split-summary.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/development-eda.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    if not split_summary.get("sealed"):
        raise EDAError("Test split is not marked sealed")
    actual_manifest_hash = sha256_file(args.manifest)
    if actual_manifest_hash != split_summary.get("manifest_sha256"):
        raise EDAError("Split manifest hash does not match the sealed summary")

    header, rows, _, _ = prepare_rows(args.input)
    development_ids, manifest_counts = load_development_ids(args.manifest)
    profile = profile_development(header, rows, development_ids, load_model_schema())
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(args.input),
        "source_sha256": sha256_file(args.input),
        "manifest_path": str(args.manifest),
        "manifest_sha256": actual_manifest_hash,
        "manifest_partition_counts": manifest_counts,
        "scope": "development partition only; final test features were not profiled",
        **profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
