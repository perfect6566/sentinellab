from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from app.inference import ModelService, get_model_service
from ml.contracts import ContractError


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(Path("samples/single-record.json").read_text(encoding="utf-8"))
        cls.record = payload["record"]

    def test_singleton_loads_once_and_reports_readiness(self) -> None:
        first = get_model_service()
        second = get_model_service()
        self.assertIs(first, second)
        readiness = first.readiness()
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["model_version"], "malware-lightgbm-20261028-v1")
        self.assertEqual(readiness["feature_count"], 25)
        self.assertEqual(readiness["dataset_metadata"]["training_rows_fitted"], 38122)
        self.assertEqual(readiness["dataset_metadata"]["test_rows"], 9531)

    def test_prediction_is_order_independent_and_grounded_in_pipeline(self) -> None:
        service = ModelService()
        first = service.predict_one(self.record)
        reversed_record = dict(reversed(list(self.record.items())))
        second = service.predict_one(reversed_record)
        direct = service._pipeline.predict_proba(pd.DataFrame([self.record]))[0, 1]

        self.assertAlmostEqual(first.malware_score, direct, places=12)
        self.assertEqual(first, second)
        self.assertEqual(first.predicted_label, int(first.malware_score >= first.threshold))

    def test_target_and_bad_numeric_values_are_rejected(self) -> None:
        service = ModelService()
        record = dict(self.record)
        record["Label"] = 1
        with self.assertRaises(ContractError):
            service.predict_one(record)
        record = dict(self.record)
        record["Entropy"] = "not-a-number"
        with self.assertRaises(ContractError):
            service.predict_one(record)


if __name__ == "__main__":
    unittest.main()
