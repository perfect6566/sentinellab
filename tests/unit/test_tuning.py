from __future__ import annotations

import unittest

from ml.tune import DEFAULT_TUNING_CONFIG, load_tuning_config, select_best


class TuningTests(unittest.TestCase):
    def test_config_has_two_candidates_for_every_required_model(self) -> None:
        config = load_tuning_config(DEFAULT_TUNING_CONFIG)
        self.assertEqual(config["cv_folds"], 10)
        self.assertTrue(all(len(candidates) == 2 for candidates in config["models"].values()))

    def test_selection_prefers_auc_then_accuracy(self) -> None:
        candidates = [
            {"candidate_id": "a", "auc_mean": 0.9, "accuracy_mean": 0.95},
            {"candidate_id": "b", "auc_mean": 0.91, "accuracy_mean": 0.8},
        ]
        self.assertEqual(select_best(candidates)["candidate_id"], "b")
        tied = [
            {"candidate_id": "a", "auc_mean": 0.9, "accuracy_mean": 0.8},
            {"candidate_id": "b", "auc_mean": 0.9, "accuracy_mean": 0.85},
        ]
        self.assertEqual(select_best(tied)["candidate_id"], "b")


if __name__ == "__main__":
    unittest.main()
