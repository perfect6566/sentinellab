from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ml.data_audit import AuditError, audit_csv


class DataAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_csv(self, name: str, header: list[str], rows: list[list[str]]) -> Path:
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_compliant_binary_dataset_passes(self) -> None:
        path = self.write_csv(
            "dataset.csv",
            ["feature_a", "feature_b", "Label"],
            [["1", "2.5", "0"], ["2", "3.5", "1"]],
        )

        result = audit_csv(path, expected_feature_count=2)

        self.assertTrue(result.passed)
        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.feature_count, 2)
        self.assertEqual(result.label_counts, {"0": 1, "1": 1})
        self.assertEqual(len(result.sha256), 64)
        self.assertEqual(result.columns[0].inferred_type, "integer")
        self.assertEqual(result.source_path, str(path))

    def test_sample_values_are_bounded(self) -> None:
        long_value = "x" * 500
        path = self.write_csv("dataset.csv", ["feature", "Label"], [[long_value, "0"]])

        result = audit_csv(path, expected_feature_count=1)

        self.assertEqual(len(result.columns[0].sample_values[0]), 120)

    def test_differences_are_reported_without_fabricating_compliance(self) -> None:
        path = self.write_csv(
            "dataset.csv",
            ["feature_a", "Label"],
            [["1", "2"], ["1", ""]],
        )

        result = audit_csv(path, expected_feature_count=2)

        self.assertFalse(result.passed)
        self.assertTrue(any("expects 2" in item for item in result.differences))
        self.assertTrue(any("Unexpected" in item for item in result.differences))
        self.assertTrue(any("missing" in item.lower() for item in result.differences))

    def test_duplicate_rows_are_counted(self) -> None:
        path = self.write_csv(
            "dataset.csv",
            ["feature", "Label"],
            [["1", "0"], ["1", "0"], ["2", "1"]],
        )

        result = audit_csv(path, expected_feature_count=1)

        self.assertEqual(result.duplicate_row_count, 1)

    def test_malformed_width_raises_clear_error(self) -> None:
        path = self.directory / "broken.csv"
        path.write_text("a,Label\n1,0,extra\n", encoding="utf-8")

        with self.assertRaisesRegex(AuditError, "Row 2"):
            audit_csv(path, expected_feature_count=1)

    def test_duplicate_header_raises_clear_error(self) -> None:
        path = self.write_csv("duplicate.csv", ["feature", "feature", "Label"], [["1", "2", "0"]])

        with self.assertRaisesRegex(AuditError, "duplicate columns"):
            audit_csv(path, expected_feature_count=2)

    def test_non_csv_file_is_rejected(self) -> None:
        path = self.directory / "dataset.txt"
        path.write_text("feature,Label\n1,0\n", encoding="utf-8")

        with self.assertRaisesRegex(AuditError, "Expected a .csv"):
            audit_csv(path, expected_feature_count=1)


if __name__ == "__main__":
    unittest.main()
