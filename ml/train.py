"""Run resumable group-aware cross-validation on the sealed development set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.contracts import load_project_config
from ml.dataset import PartitionData, load_partition
from ml.model_registry import MODEL_REGISTRY, build_model_pipeline


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subsample(data: PartitionData, maximum_rows: int, seed: int) -> PartitionData:
    if len(data.labels) <= maximum_rows:
        return data
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(data.labels), size=maximum_rows, replace=False))
    return PartitionData(
        features=data.features.iloc[selected].reset_index(drop=True),
        labels=data.labels[selected],
        groups=data.groups[selected],
        row_ids=data.row_ids[selected],
    )


def run_model_cv(
    model_name: str,
    data: PartitionData,
    *,
    folds: int,
    seed: int,
    smoke: bool,
    output_directory: Path,
    candidate_id: str | None = None,
    pipeline_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline_params = pipeline_params or {}
    run_config = {
        "model": model_name,
        "folds": folds,
        "seed": seed,
        "smoke": smoke,
        "rows": len(data.labels),
        "groups": len(set(data.groups)),
        "candidate_id": candidate_id,
        "pipeline_params": pipeline_params,
        "row_ids_sha256": hashlib.sha256("\n".join(data.row_ids).encode()).hexdigest(),
    }
    config_hash = _stable_hash(run_config)
    model_directory = output_directory / model_name
    if candidate_id:
        model_directory = model_directory / candidate_id
    model_directory.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, Any]] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(data.features, data.labels, data.groups), start=1
    ):
        result_path = model_directory / f"fold-{fold_index:02d}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") == config_hash and existing.get("status") == "passed":
                fold_results.append(existing)
                continue

        pipeline = build_model_pipeline(model_name, seed=seed + fold_index, smoke=smoke)
        if pipeline_params:
            pipeline.set_params(**pipeline_params)
        started = time.perf_counter()
        try:
            pipeline.fit(data.features.iloc[train_indices], data.labels[train_indices])
            scores = pipeline.predict_proba(data.features.iloc[validation_indices])[:, 1]
            predictions = (scores >= 0.5).astype(np.int8)
            result = {
                "status": "passed",
                "config_hash": config_hash,
                "model": model_name,
                "fold": fold_index,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "train_groups": len(set(data.groups[train_indices])),
                "validation_groups": len(set(data.groups[validation_indices])),
                "group_overlap": len(
                    set(data.groups[train_indices]) & set(data.groups[validation_indices])
                ),
                "auc": float(roc_auc_score(data.labels[validation_indices], scores)),
                "accuracy": float(accuracy_score(data.labels[validation_indices], predictions)),
                "elapsed_seconds": time.perf_counter() - started,
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "config_hash": config_hash,
                "model": model_name,
                "fold": fold_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            raise
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        fold_results.append(result)

    auc = np.asarray([item["auc"] for item in fold_results])
    accuracy = np.asarray([item["accuracy"] for item in fold_results])
    summary = {
        "status": "passed",
        "run_type": "smoke" if smoke else "formal_cv",
        "config": run_config,
        "config_hash": config_hash,
        "model": model_name,
        "display_name": MODEL_REGISTRY[model_name].display_name,
        "candidate_id": candidate_id,
        "pipeline_params": pipeline_params,
        "auc_mean": float(auc.mean()),
        "auc_std": float(auc.std(ddof=0)),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=0)),
        "elapsed_seconds": float(sum(item["elapsed_seconds"] for item in fold_results)),
        "fold_results": fold_results,
    }
    (model_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_REGISTRY), default=sorted(MODEL_REGISTRY)
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", type=int)
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_project_config()
    folds = args.folds or (2 if args.smoke else config.cv_folds)
    if not args.smoke and folds != config.cv_folds:
        raise ValueError("Formal evaluation must use the configured 10 folds")
    data = load_partition("development")
    if args.smoke:
        data = _subsample(data, args.sample_rows, config.random_seed)
    output = args.output or Path("reports/training") / ("smoke" if args.smoke else "formal")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "torch": importlib.metadata.version("torch"),
        "models": args.models,
        "folds": folds,
        "rows": len(data.labels),
        "smoke": args.smoke,
        "warning": "Smoke results are engineering checks and must not be reported as formal "
        "model results."
        if args.smoke
        else None,
    }
    (output / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    summaries = []
    for model_name in args.models:
        summaries.append(
            run_model_cv(
                model_name,
                data,
                folds=folds,
                seed=config.random_seed,
                smoke=args.smoke,
                output_directory=output,
            )
        )
    # A subset run must not erase valid summaries produced by earlier subset
    # runs. Consolidate all model summaries matching this run scope.
    consolidated: list[dict[str, Any]] = []
    for model_name in MODEL_REGISTRY:
        summary_path = output / model_name / "summary.json"
        if not summary_path.exists():
            continue
        candidate = json.loads(summary_path.read_text(encoding="utf-8"))
        candidate_config = candidate.get("config", {})
        if (
            candidate.get("run_type") == ("smoke" if args.smoke else "formal_cv")
            and candidate_config.get("folds") == folds
            and candidate_config.get("seed") == config.random_seed
            and candidate_config.get("rows") == len(data.labels)
        ):
            consolidated.append(candidate)
    metadata["completed_models"] = [item["model"] for item in consolidated]
    (output / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(consolidated, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
