"""Audit a structured CSV before any model-development work begins.

This module deliberately uses only the Python standard library so provenance and
schema checks can run before the project's scientific dependencies are installed.
It never executes or interprets uploaded content as code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}


@dataclass(frozen=True)
class ColumnAudit:
    name: str
    inferred_type: str
    missing_count: int
    unique_count: int
    sample_values: list[str]


@dataclass(frozen=True)
class DatasetAudit:
    source_path: str
    filename: str
    byte_size: int
    sha256: str
    audited_at: str
    row_count: int
    column_count: int
    feature_count: int
    label_column: str
    label_mapping_expected: dict[str, str]
    label_counts: dict[str, int]
    header: list[str]
    duplicate_row_count: int
    columns: list[ColumnAudit]
    differences: list[str]
    passed: bool


class AuditError(ValueError):
    """Raised when a CSV cannot be safely or meaningfully audited."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def _infer_type(values: Iterable[str]) -> str:
    present = [value.strip() for value in values if not _is_missing(value)]
    if not present:
        return "unknown"

    integers = True
    numbers = True
    booleans = True
    for value in present:
        lower = value.lower()
        if lower not in {"true", "false", "0", "1"}:
            booleans = False
        try:
            integer = int(value)
            if str(integer) != value and str(integer) != value.lstrip("+"):
                integers = False
        except ValueError:
            integers = False
        try:
            number = float(value)
            if not math.isfinite(number):
                numbers = False
        except ValueError:
            numbers = False

    if booleans:
        return "boolean"
    if integers:
        return "integer"
    if numbers:
        return "number"
    return "string"


def audit_csv(
    path: Path,
    *,
    label_column: str = "Label",
    expected_feature_count: int = 27,
    encoding: str = "utf-8-sig",
) -> DatasetAudit:
    """Read and audit a CSV without mutating it.

    The audit fails its compliance flag when the expected label, feature count or
    binary label values differ from the assignment specification. It still returns
    the observed facts so mismatches are visible rather than hidden.
    """

    supplied_path = path
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AuditError(f"CSV file does not exist: {path}")
    if resolved.suffix.lower() != ".csv":
        raise AuditError(f"Expected a .csv file, received: {resolved.name}")

    with resolved.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise AuditError("CSV is empty") from exc

        header = [name.strip() for name in header]
        if not header or any(not name for name in header):
            raise AuditError("CSV header contains an empty column name")
        if len(set(header)) != len(header):
            duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
            raise AuditError(f"CSV header contains duplicate columns: {duplicates}")

        values_by_column: dict[str, list[str]] = {name: [] for name in header}
        row_fingerprints: Counter[tuple[str, ...]] = Counter()
        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise AuditError(f"Row {line_number} has {len(row)} values; expected {len(header)}")
            normalized = tuple(value.strip() for value in row)
            row_fingerprints[normalized] += 1
            row_count += 1
            for name, value in zip(header, normalized, strict=True):
                values_by_column[name].append(value)

    if row_count == 0:
        raise AuditError("CSV has a header but no data rows")

    differences: list[str] = []
    if label_column not in header:
        differences.append(f"Missing required label column {label_column!r}")
        label_values: list[str] = []
        feature_count = len(header)
    else:
        label_values = [value for value in values_by_column[label_column] if not _is_missing(value)]
        feature_count = len(header) - 1

    if feature_count != expected_feature_count:
        differences.append(
            f"Observed {feature_count} non-label columns; assignment expects "
            f"{expected_feature_count}. ID/hash/helper columns may still need exclusion."
        )

    label_counts = Counter(label_values)
    unexpected_labels = sorted(set(label_counts) - {"0", "1"})
    if unexpected_labels:
        differences.append(f"Unexpected non-missing label values: {unexpected_labels}")
    if label_column in header and any(
        _is_missing(value) for value in values_by_column[label_column]
    ):
        differences.append("Label column contains missing values")
    if label_column in header and not label_counts:
        differences.append("Label column has no non-missing values")

    columns = []
    for name in header:
        values = values_by_column[name]
        columns.append(
            ColumnAudit(
                name=name,
                inferred_type=_infer_type(values),
                missing_count=sum(_is_missing(value) for value in values),
                unique_count=len(set(values)),
                sample_values=[value[:120] for value in list(dict.fromkeys(values))[:5]],
            )
        )

    duplicate_row_count = sum(count - 1 for count in row_fingerprints.values() if count > 1)
    return DatasetAudit(
        source_path=str(supplied_path),
        filename=resolved.name,
        byte_size=resolved.stat().st_size,
        sha256=sha256_file(resolved),
        audited_at=datetime.now(UTC).isoformat(),
        row_count=row_count,
        column_count=len(header),
        feature_count=feature_count,
        label_column=label_column,
        label_mapping_expected={"0": "goodware", "1": "malware"},
        label_counts=dict(sorted(label_counts.items())),
        header=header,
        duplicate_row_count=duplicate_row_count,
        columns=columns,
        differences=differences,
        passed=not differences,
    )


def audit_to_dict(audit: DatasetAudit) -> dict[str, Any]:
    return asdict(audit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Structured CSV file to audit")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--expected-feature-count", type=int, default=27)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_csv(
            args.csv_path,
            label_column=args.label_column,
            expected_feature_count=args.expected_feature_count,
        )
    except (AuditError, OSError, UnicodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 2

    payload = audit_to_dict(result)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
