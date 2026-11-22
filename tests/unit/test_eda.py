from __future__ import annotations

import unittest

from ml.contracts import FeatureSpec, ModelSchema
from ml.eda import profile_development
from ml.split_data import PreparedRow


class EDATests(unittest.TestCase):
    def test_only_development_rows_are_profiled(self) -> None:
        schema = ModelSchema(
            schema_version="test",
            source_dataset_sha256="x",
            target_name="Label",
            row_identifier_name="SHA1",
            excluded_metadata=(),
            features=(
                FeatureSpec("number", "numeric", False),
                FeatureSpec("category", "categorical", True),
                FeatureSpec("text", "text", False),
            ),
            unknown_field_policy="reject",
        )
        header = ["number", "category", "text", "SHA1", "Label"]
        rows = [
            PreparedRow("dev", "a", 0, 2, ("1", "", "one two", "a", "0")),
            PreparedRow("test", "b", 1, 3, ("999999", "secret", "hidden", "b", "1")),
        ]

        result = profile_development(header, rows, {"dev"}, schema)

        self.assertEqual(result["development_rows_profiled"], 1)
        self.assertEqual(result["test_rows_profiled"], 0)
        self.assertEqual(result["numeric"]["number"]["max"], 1)
        self.assertEqual(result["categorical"]["category"]["missing_count"], 1)
        self.assertEqual(result["text"]["text"]["token_count_max"], 2)


if __name__ == "__main__":
    unittest.main()
