"""Create a reproducible, SHA1-grouped, stratified development/test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.contracts import DEFAULT_MODEL_SCHEMA, DEFAULT_PROJECT_CONFIG, load_project_config


@dataclass(frozen=True)
class PreparedRow:
    row_id: str
    sha1: str
    label: int
    source_line: int
    values: tuple[str, ...]


@dataclass(frozen=True)
class PreparationStats:
    input_rows: int
    conflicting_sha1_groups: int
    conflicting_rows_removed: int
    exact_duplicate_rows_removed: int
    eligible_rows: int
    eligible_sha1_groups: int


class SplitError(ValueError):
    """Raised when safe grouped splitting cannot be completed."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(header: Iterable[str], values: Iterable[str]) -> str:
    canonical = json.dumps(
        dict(zip(header, values, strict=True)), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_rows(
    path: Path, *, label_column: str = "Label", group_column: str = "SHA1"
) -> tuple[list[str], list[PreparedRow], PreparationStats, list[str]]:
    """Remove ambiguous identity groups and exact duplicates deterministically."""

    raw_rows: list[tuple[int, tuple[str, ...]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [value.strip() for value in next(reader)]
        except StopIteration as exc:
            raise SplitError("Dataset is empty") from exc
        if label_column not in header or group_column not in header:
            raise SplitError(f"Dataset must contain {label_column!r} and {group_column!r}")
        if len(header) != len(set(header)):
            raise SplitError("Dataset contains duplicate header names")
        for source_line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise SplitError(f"Row {source_line} has the wrong number of fields")
            raw_rows.append((source_line, tuple(value.strip() for value in row)))

    label_index = header.index(label_column)
    group_index = header.index(group_column)
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for source_line, values in raw_rows:
        sha1 = values[group_index]
        if not sha1:
            raise SplitError(f"Row {source_line} has an empty {group_column}")
        try:
            label = int(values[label_index])
        except ValueError as exc:
            raise SplitError(f"Row {source_line} has a non-integer label") from exc
        if label not in {0, 1}:
            raise SplitError(f"Row {source_line} has unsupported label {label}")
        labels_by_group[sha1].add(label)

    conflicting_groups = sorted(
        group for group, labels in labels_by_group.items() if len(labels) > 1
    )
    conflicting_set = set(conflicting_groups)
    eligible_before_dedup = [
        item for item in raw_rows if item[1][group_index] not in conflicting_set
    ]
    conflicting_rows_removed = len(raw_rows) - len(eligible_before_dedup)

    seen_rows: set[tuple[str, ...]] = set()
    prepared: list[PreparedRow] = []
    duplicate_count = 0
    for source_line, values in eligible_before_dedup:
        if values in seen_rows:
            duplicate_count += 1
            continue
        seen_rows.add(values)
        fingerprint = _fingerprint(header, values)
        prepared.append(
            PreparedRow(
                row_id=fingerprint,
                sha1=values[group_index],
                label=int(values[label_index]),
                source_line=source_line,
                values=values,
            )
        )

    stats = PreparationStats(
        input_rows=len(raw_rows),
        conflicting_sha1_groups=len(conflicting_groups),
        conflicting_rows_removed=conflicting_rows_removed,
        exact_duplicate_rows_removed=duplicate_count,
        eligible_rows=len(prepared),
        eligible_sha1_groups=len({row.sha1 for row in prepared}),
    )
    return header, prepared, stats, conflicting_groups


def _select_test_groups(groups: list[tuple[str, int]], target_rows: float, seed: int) -> set[str]:
    """Choose whole groups close to the requested row count, reproducibly."""

    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    selected: set[str] = set()
    count = 0
    for group, size in ordered:
        current_error = abs(count - target_rows)
        candidate_error = abs((count + size) - target_rows)
        if count < target_rows or candidate_error < current_error:
            selected.add(group)
            count += size
    return selected


def grouped_stratified_split(
    rows: list[PreparedRow], *, test_fraction: float, seed: int
) -> dict[str, str]:
    if not 0 < test_fraction < 1:
        raise SplitError("test_fraction must be between 0 and 1")
    groups_by_label: dict[int, dict[str, int]] = {0: Counter(), 1: Counter()}
    for row in rows:
        groups_by_label[row.label][row.sha1] += 1

    test_groups: set[str] = set()
    for label in (0, 1):
        groups = list(groups_by_label[label].items())
        if len(groups) < 2:
            raise SplitError(f"Class {label} needs at least two identity groups")
        target = sum(size for _, size in groups) * test_fraction
        test_groups.update(_select_test_groups(groups, target, seed + label))

    assignment = {
        row.row_id: ("test" if row.sha1 in test_groups else "development") for row in rows
    }
    if set(assignment) != {row.row_id for row in rows}:
        raise SplitError("Row identifiers are not unique after deduplication")
    return assignment


def write_manifest(path: Path, rows: list[PreparedRow], assignment: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["row_id", "sha1", "label", "partition", "source_line"]
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.source_line):
            writer.writerow(
                {
                    "row_id": row.row_id,
                    "sha1": row.sha1,
                    "label": row.label,
                    "partition": assignment[row.row_id],
                    "source_line": row.source_line,
                }
            )


def build_summary(
    rows: list[PreparedRow],
    assignment: dict[str, str],
    stats: PreparationStats,
    *,
    seed: int,
    requested_test_fraction: float,
) -> dict[str, Any]:
    counts: dict[str, Counter[int]] = {
        "development": Counter(),
        "test": Counter(),
    }
    groups: dict[str, set[str]] = {"development": set(), "test": set()}
    for row in rows:
        partition = assignment[row.row_id]
        counts[partition][row.label] += 1
        groups[partition].add(row.sha1)
    overlap = groups["development"] & groups["test"]
    total = len(rows)
    test_rows = sum(counts["test"].values())
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "random_seed": seed,
        "requested_test_fraction": requested_test_fraction,
        "preparation": {
            "input_rows": stats.input_rows,
            "conflicting_sha1_groups_removed": stats.conflicting_sha1_groups,
            "conflicting_rows_removed": stats.conflicting_rows_removed,
            "exact_duplicate_rows_removed": stats.exact_duplicate_rows_removed,
            "eligible_rows": stats.eligible_rows,
            "eligible_sha1_groups": stats.eligible_sha1_groups,
        },
        "partitions": {
            partition: {
                "rows": sum(counts[partition].values()),
                "groups": len(groups[partition]),
                "label_counts": {str(label): counts[partition][label] for label in (0, 1)},
            }
            for partition in ("development", "test")
        },
        "actual_test_fraction": test_rows / total,
        "sha1_group_overlap_count": len(overlap),
        "sealed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/brazilian-malware.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/holdout-manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/split-summary.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_project_config()
    header, rows, stats, conflicting = prepare_rows(
        args.input, label_column=config.label_column, group_column=config.group_column
    )
    assignment = grouped_stratified_split(
        rows, test_fraction=config.holdout_fraction, seed=config.random_seed
    )
    write_manifest(args.manifest, rows, assignment)
    summary = build_summary(
        rows,
        assignment,
        stats,
        seed=config.random_seed,
        requested_test_fraction=config.holdout_fraction,
    )
    summary.update(
        {
            "source_path": str(args.input),
            "source_sha256": sha256_file(args.input),
            "source_columns": header,
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "project_config_path": str(
                DEFAULT_PROJECT_CONFIG.relative_to(DEFAULT_PROJECT_CONFIG.parents[1])
            ),
            "project_config_sha256": sha256_file(DEFAULT_PROJECT_CONFIG),
            "model_schema_path": str(
                DEFAULT_MODEL_SCHEMA.relative_to(DEFAULT_MODEL_SCHEMA.parents[2])
            ),
            "model_schema_sha256": sha256_file(DEFAULT_MODEL_SCHEMA),
            "conflicting_sha1_ids_path": "not persisted in tracked reports",
            "conflicting_sha1_count": len(conflicting),
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["sha1_group_overlap_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
