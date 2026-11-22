from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.files import SessionFileStore, new_session_id
from app.inference import ModelService
from app.tools import ProjectTools


class ToolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = ModelService()
        cls.batch = pd.read_csv("samples/batch-sample.csv", keep_default_na=False)
        cls.labeled = pd.read_csv("samples/labeled-sample.csv", keep_default_na=False)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.files = SessionFileStore(Path(self.temporary_directory.name), max_bytes=2_000_000)
        self.session = new_session_id()
        self.tools = ProjectTools(self.model, self.files, max_rows=100)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def csv_bytes(self, rows: list[dict[str, object]], include_label: bool = False) -> bytes:
        fields = list(self.model.schema.feature_names)
        if include_label:
            fields.append("Label")
        fields.append("row_id")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode()

    def test_predict_single_returns_model_result(self) -> None:
        record = self.batch.iloc[0].drop(labels=["row_id"]).to_dict()
        result = self.tools.predict_single(record)
        direct = self.model.predict_one(record)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["predicted_label"], direct.predicted_label)
        self.assertAlmostEqual(result["data"]["malware_score"], direct.malware_score)

    def test_predict_single_data_carries_the_validated_feature_count(self) -> None:
        """The feature count must live in the result data, not only in the activity.

        A follow-up question receives the stored result data as its trusted
        context. When the count existed only in the activity record, a correct
        answer stating it could not be grounded and the response guard rejected
        an accurate reply.
        """
        record = self.batch.iloc[0].drop(labels=["row_id"]).to_dict()
        result = self.tools.predict_single(record)

        self.assertEqual(result["data"]["validated_feature_count"], len(record))
        self.assertEqual(
            result["data"]["validated_feature_count"],
            result["activity"]["validated_feature_count"],
        )

    def test_batch_reports_valid_and_invalid_rows_and_download(self) -> None:
        valid = self.batch.iloc[0].to_dict() | {"row_id": "valid"}
        invalid = self.batch.iloc[1].to_dict() | {"row_id": "invalid", "Entropy": "bad"}
        stored = self.files.put_csv(self.session, "batch.csv", self.csv_bytes([valid, invalid]))

        result = self.tools.predict_batch(self.session, stored.reference)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["summary"]["input_rows"], 2)
        self.assertEqual(result["data"]["summary"]["valid_rows"], 1)
        self.assertEqual(result["data"]["summary"]["invalid_rows"], 1)
        downloaded = self.files.resolve(self.session, result["data"]["result_file_reference"])
        self.assertIn("predicted", downloaded.path.read_text())
        self.assertIn("invalid", downloaded.path.read_text())

    def test_labeled_evaluation_and_single_class_auc(self) -> None:
        rows = self.labeled.to_dict(orient="records")
        stored = self.files.put_csv(
            self.session, "labeled.csv", self.csv_bytes(rows, include_label=True)
        )
        result = self.tools.evaluate_labeled(self.session, stored.reference)
        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["data"]["auc"])
        matrix = result["data"]["confusion_matrix"]
        self.assertEqual(matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"], len(rows))
        self.assertIn("not verified", result["data"]["interpretation_note"])

        fixture = self.files.put_csv(
            self.session,
            "labeled-sample (1).csv",
            Path("samples/labeled-sample.csv").read_bytes(),
        )
        fixture_result = self.tools.evaluate_labeled(self.session, fixture.reference)
        self.assertIn("not a valid estimate", fixture_result["data"]["interpretation_note"])

        single_class = [row for row in rows if int(row["Label"]) == 0][:5]
        stored = self.files.put_csv(
            self.session, "single-class.csv", self.csv_bytes(single_class, include_label=True)
        )
        result = self.tools.evaluate_labeled(self.session, stored.reference)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["data"]["auc"])
        self.assertIn("one true class", result["data"]["auc_reason"])

    def test_missing_label_and_cross_session_reference_fail(self) -> None:
        row = self.batch.iloc[0].to_dict() | {"row_id": "one"}
        stored = self.files.put_csv(self.session, "unlabeled.csv", self.csv_bytes([row]))
        result = self.tools.evaluate_labeled(self.session, stored.reference)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "missing_labels")

        result = self.tools.predict_batch(new_session_id(), stored.reference)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_file")


if __name__ == "__main__":
    unittest.main()
