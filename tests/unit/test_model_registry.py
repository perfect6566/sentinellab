from __future__ import annotations

import subprocess
import sys
import unittest

from ml.model_registry import MODEL_REGISTRY, build_model_pipeline


class ModelRegistryTests(unittest.TestCase):
    def test_all_required_model_families_are_registered(self) -> None:
        self.assertEqual(
            set(MODEL_REGISTRY),
            {
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "pytorch_mlp",
                "xgboost",
                "lightgbm",
                "catboost",
            },
        )

    def test_baseline_pipelines_contain_unfitted_preprocessing(self) -> None:
        for name in ("logistic_regression", "decision_tree", "random_forest"):
            pipeline = build_model_pipeline(name, seed=123, smoke=True)
            self.assertEqual(list(pipeline.named_steps), ["preprocessing", "classifier"])
            self.assertFalse(hasattr(pipeline.named_steps["preprocessing"], "transformers_"))

    def test_pytorch_pipeline_builds_in_isolated_process(self) -> None:
        code = (
            "from ml.model_registry import build_model_pipeline; "
            "p=build_model_pipeline('pytorch_mlp',seed=123,smoke=True); "
            "assert list(p.named_steps)==['preprocessing','classifier']"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
