"""Create the auditable model comparison and freeze production selection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "pytorch_mlp",
    "xgboost",
    "lightgbm",
    "catboost",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    selections = []
    for model in MODELS:
        path = Path("reports/training/tuning") / model / "selection.json"
        selections.append(json.loads(path.read_text(encoding="utf-8")))
    ranked = sorted(selections, key=lambda item: (-item["auc_mean"], -item["accuracy_mean"]))
    by_model = {item["model"]: item for item in selections}
    xgb = by_model["xgboost"]
    lgbm = by_model["lightgbm"]
    auc_gap = xgb["auc_mean"] - lgbm["auc_mean"]
    if not (0 <= auc_gap < 0.001):
        raise ValueError("Expected XGBoost/LightGBM to be within the documented close-result band")
    if lgbm["accuracy_mean"] <= xgb["accuracy_mean"]:
        raise ValueError("LightGBM no longer has the documented accuracy advantage")

    ablation = json.loads(
        Path("reports/training/ablation/summary.json").read_text(encoding="utf-8")
    )
    ablation_rows = [
        {
            "configuration": "hash256_reference",
            "auc_mean": lgbm["auc_mean"],
            "auc_std": lgbm["auc_std"],
            "accuracy_mean": lgbm["accuracy_mean"],
            "accuracy_std": lgbm["accuracy_std"],
        }
    ]
    for result in ablation["results"]:
        ablation_rows.append(
            {
                "configuration": result["candidate_id"],
                "auc_mean": result["auc_mean"],
                "auc_std": result["auc_std"],
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
            }
        )
    if (
        max(ablation_rows, key=lambda item: item["auc_mean"])["configuration"]
        != "hash256_reference"
    ):
        raise ValueError("256-dimensional text configuration is no longer the AUC winner")

    comparison: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "sealed development partition only; final hold-out not accessed",
        "development_rows": 38122,
        "development_groups": 34651,
        "cv": "10-fold StratifiedGroupKFold, seed 20261028",
        "primary_metric": "ROC AUC from malware-class probabilities",
        "secondary_metric": "Accuracy at threshold 0.5",
        "ranked_models": [
            {
                "rank": index,
                "model": item["model"],
                "candidate": item["selected_candidate_id"],
                "auc_mean": item["auc_mean"],
                "auc_std": item["auc_std"],
                "accuracy_mean": item["accuracy_mean"],
                "accuracy_std": item["accuracy_std"],
                "elapsed_seconds": max(
                    candidate["elapsed_seconds"]
                    for candidate in item["candidates"]
                    if candidate["candidate_id"] == item["selected_candidate_id"]
                ),
                "selected_params": item["selected_params"],
            }
            for index, item in enumerate(ranked, start=1)
        ],
        "feature_ablation": ablation_rows,
        "production_selection": {
            "model": "lightgbm",
            "candidate": lgbm["selected_candidate_id"],
            "auc_gap_to_xgboost": auc_gap,
            "reason": (
                "XGBoost has a 0.000064 AUC edge, far below fold variability. "
                "LightGBM has higher mean Accuracy and lower selected-candidate CV runtime, "
                "so it wins the documented close-result tie-break."
            ),
        },
    }
    reports = Path("reports")
    (reports / "model-comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Formal Development-Set Model Comparison",
        "",
        "Final hold-out status: **sealed and not accessed**.",
        "",
        "All values below are from the same 10-fold SHA1-group-aware CV on 38,122 development "
        "rows.",
        "",
        "| Rank | Model | Selected candidate | AUC mean ± std | Accuracy mean ± std | CV runtime |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in comparison["ranked_models"]:
        lines.append(
            f"| {row['rank']} | {row['model']} | `{row['candidate']}` | "
            f"{row['auc_mean']:.6f} ± {row['auc_std']:.6f} | "
            f"{row['accuracy_mean']:.6f} ± {row['accuracy_std']:.6f} | "
            f"{row['elapsed_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "## Feature/dimensionality ablation",
            "",
            "| Configuration | AUC mean ± std | Accuracy mean ± std |",
            "|---|---:|---:|",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| `{row['configuration']}` | {row['auc_mean']:.6f} ± {row['auc_std']:.6f} | "
            f"{row['accuracy_mean']:.6f} ± {row['accuracy_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The 256-feature hashing configuration is retained because it has the highest AUC and "
            "Accuracy. Dropping import-list text reduces Accuracy materially.",
            "",
            "## Production selection",
            "",
            "LightGBM is selected. XGBoost's mean AUC is higher by only "
            f"{auc_gap:.6f}, far smaller than either model's fold standard deviation. "
            "LightGBM has slightly higher mean Accuracy and lower selected-candidate CV runtime. "
            "These are the predeclared secondary/production tie-breakers for close AUC results.",
            "",
            "Threshold remains fixed at 0.5. No hold-out result was used in this decision.",
            "",
        ]
    )
    (reports / "model-comparison.md").write_text("\n".join(lines), encoding="utf-8")

    production = {
        "schema_version": "1.0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_holdout_evaluation": True,
        "model_version": "malware-lightgbm-20261028-v1",
        "model_name": "lightgbm",
        "selected_candidate": lgbm["selected_candidate_id"],
        "pipeline_params": lgbm["selected_params"],
        "preprocessing": {
            "numeric": "median imputation, fold-fitted variance filter, no scaling for tree model",
            "Identify": "explicit missing category and one-hot encoding",
            "ImportedDlls_hash_features": 256,
            "ImportedSymbols_hash_features": 256,
            "excluded_columns": ["Label", "SHA1", "FirstSeenDate"],
            "candidate_model_feature_count": 25,
        },
        "classification_threshold": 0.5,
        "class_mapping": {"0": "goodware", "1": "malware"},
        "cv_metrics": {
            "auc_mean": lgbm["auc_mean"],
            "auc_std": lgbm["auc_std"],
            "accuracy_mean": lgbm["accuracy_mean"],
            "accuracy_std": lgbm["accuracy_std"],
        },
        "random_seed": 20261028,
        "source_dataset_sha256": "d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242",
        "split_manifest_sha256": "fa843af3135bc656d2663a3012ea3d7e4148a676f688f330106a5659e7655f51",
        "model_schema_sha256": sha256_file(Path("data/metadata/model-schema.json")),
        "project_config_sha256": sha256_file(Path("config/project.json")),
        "comparison_report_sha256": sha256_file(Path("reports/model-comparison.json")),
        "holdout_metrics": None,
        "rule": "Do not change this configuration based on final hold-out results.",
    }
    production_path = Path("config/production.json")
    production_path.write_text(json.dumps(production, indent=2) + "\n", encoding="utf-8")
    freeze_manifest = {
        "production_config_path": str(production_path),
        "production_config_sha256": sha256_file(production_path),
        "frozen_at": production["frozen_at"],
        "holdout_accessed": False,
    }
    Path("artifacts/selection-freeze.json").write_text(
        json.dumps(freeze_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
