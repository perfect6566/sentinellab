"""Fit the frozen production pipeline and perform the one-time hold-out evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from ml.dataset import load_partition
from ml.model_registry import build_model_pipeline

PRODUCTION_CONFIG = Path("config/production.json")
FREEZE_MANIFEST = Path("artifacts/selection-freeze.json")
ARTIFACT_PATH = Path("artifacts/production-model.joblib")
ARTIFACT_MANIFEST = Path("artifacts/production-manifest.json")
HOLDOUT_REPORT = Path("reports/holdout-evaluation.json")
HOLDOUT_MARKER = Path("artifacts/holdout-evaluation.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_config() -> tuple[dict[str, Any], str]:
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256_file(PRODUCTION_CONFIG)
    if actual_hash != freeze["production_config_sha256"]:
        raise RuntimeError("Frozen production config hash mismatch")
    if freeze.get("holdout_accessed") is not False:
        raise RuntimeError("Selection freeze does not prove pre-holdout status")
    config = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    if (
        not config.get("frozen_before_holdout_evaluation")
        or config.get("holdout_metrics") is not None
    ):
        raise RuntimeError("Production config is not a clean pre-holdout freeze")
    return config, actual_hash


def verify_existing_artifact() -> dict[str, Any]:
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(ARTIFACT_PATH) != manifest["artifact_sha256"]:
        raise RuntimeError("Production artifact hash mismatch")
    if sha256_file(PRODUCTION_CONFIG) != manifest["production_config_sha256"]:
        raise RuntimeError("Production config no longer matches artifact manifest")
    bundle = joblib.load(ARTIFACT_PATH)
    required = {"pipeline", "model_version", "threshold", "class_mapping", "feature_names"}
    if not required <= set(bundle):
        raise RuntimeError("Production bundle is missing required fields")
    development = load_partition("development")
    scores = bundle["pipeline"].predict_proba(development.features.iloc[:10])[:, 1]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("Reloaded artifact produced invalid scores")
    return {
        "status": "passed",
        "artifact_sha256": manifest["artifact_sha256"],
        "model_version": bundle["model_version"],
        "sample_scores": scores.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-artifact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_artifact:
        print(json.dumps(verify_existing_artifact(), indent=2))
        return 0
    if HOLDOUT_REPORT.exists() or HOLDOUT_MARKER.exists():
        raise RuntimeError(
            "Final hold-out evaluation already exists. Use --verify-artifact; "
            "do not rerun or retune."
        )

    config, config_hash = verify_frozen_config()
    development = load_partition("development")
    pipeline = build_model_pipeline(config["model_name"], seed=config["random_seed"], smoke=False)
    pipeline.set_params(**config["pipeline_params"])
    fit_started = time.perf_counter()
    pipeline.fit(development.features, development.labels)
    fit_seconds = time.perf_counter() - fit_started

    # This is the first feature access to the sealed test partition after freeze.
    test = load_partition("test")
    inference_started = time.perf_counter()
    scores = pipeline.predict_proba(test.features)[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    threshold = float(config["classification_threshold"])
    predictions = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(test.labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "auc": float(roc_auc_score(test.labels, scores)),
        "accuracy": float(accuracy_score(test.labels, predictions)),
        "confusion_matrix": {
            "label_order": [0, 1],
            "rows": "true labels",
            "columns": "predicted labels",
            "matrix": matrix.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    split_summary = json.loads(Path("reports/split-summary.json").read_text(encoding="utf-8"))
    evaluated_at = datetime.now(UTC).isoformat()
    report = {
        "status": "passed",
        "evaluation_type": "one-time untouched final hold-out",
        "evaluated_at": evaluated_at,
        "production_config_sha256": config_hash,
        "model_name": config["model_name"],
        "model_version": config["model_version"],
        "threshold": threshold,
        "class_mapping": config["class_mapping"],
        "development_rows_fitted": len(development.labels),
        "development_groups_fitted": len(set(development.groups)),
        "test_rows": len(test.labels),
        "test_groups": len(set(test.groups)),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "metrics": metrics,
        "selection_policy": (
            "Configuration was frozen before test access; results must not trigger "
            "retuning or reselection."
        ),
    }
    HOLDOUT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    predictions_path = Path("reports/generated/holdout-predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "true_label", "malware_score", "predicted_label"])
        writer.writerows(zip(test.row_ids, test.labels, scores, predictions, strict=True))

    bundle = {
        "pipeline": pipeline,
        "model_version": config["model_version"],
        "model_name": config["model_name"],
        "threshold": threshold,
        "class_mapping": {0: "goodware", 1: "malware"},
        "feature_names": list(test.features.columns),
        "schema_version": "1.0",
        "production_config_sha256": config_hash,
        "trained_at": evaluated_at,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    before_reload = pipeline.predict_proba(development.features.iloc[:100])[:, 1]
    reloaded = joblib.load(ARTIFACT_PATH)
    after_reload = reloaded["pipeline"].predict_proba(development.features.iloc[:100])[:, 1]
    max_difference = float(np.max(np.abs(before_reload - after_reload)))
    if max_difference > 1e-12:
        raise RuntimeError(f"Artifact reload changed probabilities: {max_difference}")

    manifest = {
        "schema_version": "1.0",
        "model_version": config["model_version"],
        "artifact_path": str(ARTIFACT_PATH),
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
        "artifact_byte_size": ARTIFACT_PATH.stat().st_size,
        "production_config_path": str(PRODUCTION_CONFIG),
        "production_config_sha256": config_hash,
        "model_schema_path": "data/metadata/model-schema.json",
        "model_schema_sha256": config["model_schema_sha256"],
        "source_dataset_sha256": config["source_dataset_sha256"],
        "split_manifest_sha256": config["split_manifest_sha256"],
        "threshold": threshold,
        "class_mapping": config["class_mapping"],
        "dataset_metadata": {
            "source_rows": split_summary["preparation"]["input_rows"],
            "eligible_rows": split_summary["preparation"]["eligible_rows"],
            "training_rows_fitted": len(development.labels),
            "training_groups": len(set(development.groups)),
            "test_rows": len(test.labels),
            "test_groups": len(set(test.groups)),
            "training_fraction": 1.0 - float(split_summary["requested_test_fraction"]),
            "test_fraction": float(split_summary["requested_test_fraction"]),
            "test_label_counts": {
                "goodware": int(np.count_nonzero(test.labels == 0)),
                "malware": int(np.count_nonzero(test.labels == 1)),
            },
            "split_method": "Stratified 80/20 split grouped by SHA1 file identity",
            "cross_validation": {
                "folds": 10,
                "scope": "development partition only",
                "grouped_by": "SHA1",
            },
            "holdout_evaluation": {
                "type": "one-time untouched final test set",
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
            },
            "interpretation": (
                "Uploaded labeled CSV evaluations are separate from this official holdout. "
                "The built-in 20-row labeled workflow fixture comes from the development "
                "partition and is not an unseen-data performance estimate."
            ),
        },
        "holdout_report_path": str(HOLDOUT_REPORT),
        "holdout_report_sha256": sha256_file(HOLDOUT_REPORT),
        "reload_max_probability_difference": max_difference,
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "python-dotenv",
                "joblib",
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
            ]
        },
        "python_version": platform.python_version(),
        "warning": (
            "Load only this trusted, hash-verified joblib artifact. "
            "Joblib is unsafe for untrusted files."
        ),
    }
    ARTIFACT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    marker = {
        "evaluated_at": evaluated_at,
        "production_config_sha256": config_hash,
        "holdout_report_sha256": manifest["holdout_report_sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "do_not_repeat_for_model_selection": True,
    }
    HOLDOUT_MARKER.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")

    markdown = f"""# Final Hold-out Evaluation

This is the one-time evaluation of the frozen `{config["model_version"]}` pipeline. The
configuration hash was `{config_hash}` before test access. No test result was used for model
selection or tuning.

| Metric | Result |
|---|---:|
| Test rows | {len(test.labels):,} |
| Test SHA1 groups | {len(set(test.groups)):,} |
| ROC AUC | {metrics["auc"]:.6f} |
| Accuracy | {metrics["accuracy"]:.6f} |
| Threshold | {threshold:.2f} |
| TN | {tn:,} |
| FP | {fp:,} |
| FN | {fn:,} |
| TP | {tp:,} |

Confusion-matrix rows are true labels and columns are predicted labels, both in `[0, 1]` order.
`FN` means actual malware predicted as goodware; `FP` means actual goodware predicted as malware.

The result must not be used to alter the selected model, preprocessing, hyperparameters or
threshold.
"""
    Path("reports/holdout-evaluation.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
