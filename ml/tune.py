"""Budgeted 10-fold tuning on the sealed development partition."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.contracts import load_project_config
from ml.dataset import load_partition
from ml.model_registry import MODEL_REGISTRY
from ml.train import run_model_cv

DEFAULT_TUNING_CONFIG = Path("config/tuning.json")


def load_tuning_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("cv_folds") != 10:
        raise ValueError("Formal tuning configuration must use 10 folds")
    configured = set(data.get("models", {}))
    if configured != set(MODEL_REGISTRY):
        raise ValueError(f"Tuning config model set mismatch: {configured ^ set(MODEL_REGISTRY)}")
    for model, candidates in data["models"].items():
        if not candidates:
            raise ValueError(f"Model {model} has no tuning candidates")
        ids = [item["id"] for item in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Model {model} has duplicate candidate IDs")
    return data


def select_best(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No candidate summaries supplied")
    return max(summaries, key=lambda item: (item["auc_mean"], item["accuracy_mean"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TUNING_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("reports/training/tuning"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tuning = load_tuning_config(args.config)
    project = load_project_config()
    if project.cv_folds != tuning["cv_folds"]:
        raise ValueError("Project and tuning fold counts differ")
    data = load_partition("development")
    selected: list[dict[str, Any]] = []
    for model_name in args.models:
        candidate_summaries = []
        for candidate in tuning["models"][model_name]:
            candidate_summaries.append(
                run_model_cv(
                    model_name,
                    data,
                    folds=project.cv_folds,
                    seed=project.random_seed,
                    smoke=False,
                    output_directory=args.output,
                    candidate_id=candidate["id"],
                    pipeline_params=candidate["params"],
                )
            )
        best = select_best(candidate_summaries)
        selection = {
            "model": model_name,
            "selected_candidate_id": best["candidate_id"],
            "selected_params": best["pipeline_params"],
            "auc_mean": best["auc_mean"],
            "auc_std": best["auc_std"],
            "accuracy_mean": best["accuracy_mean"],
            "accuracy_std": best["accuracy_std"],
            "candidate_count": len(candidate_summaries),
            "candidates": candidate_summaries,
        }
        model_directory = args.output / model_name
        model_directory.mkdir(parents=True, exist_ok=True)
        (model_directory / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        selected.append(selection)
    run_summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "run_type": "budgeted_10_fold_tuning",
        "models_requested": args.models,
        "development_rows": len(data.labels),
        "development_groups": len(set(data.groups)),
        "cv_folds": project.cv_folds,
        "selection_metric": tuning["selection_metric"],
        "tie_breaker": tuning["tie_breaker"],
        "policy": tuning["policy"],
        "selections": selected,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latest-run.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
