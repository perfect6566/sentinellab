"""Run the required preprocessing/feature-dimensionality ablation."""

from __future__ import annotations

import json
from pathlib import Path

from ml.contracts import load_project_config
from ml.dataset import load_partition
from ml.train import run_model_cv


def main() -> int:
    config = json.loads(Path("config/feature-ablation.json").read_text(encoding="utf-8"))
    project = load_project_config()
    data = load_partition("development")
    output = Path("reports/training/ablation")
    results = []
    for candidate in config["candidates"]:
        params = {**config["base_classifier_params"], **candidate["params"]}
        results.append(
            run_model_cv(
                config["model"],
                data,
                folds=project.cv_folds,
                seed=project.random_seed,
                smoke=False,
                output_directory=output,
                candidate_id=candidate["id"],
                pipeline_params=params,
            )
        )
    payload = {
        "status": "passed",
        "scope": "development partition only",
        "reference_candidate": config["reference_candidate"],
        "reference_description": config["reference_description"],
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
