from __future__ import annotations

import unittest

from ml.contracts import (
    ContractError,
    load_model_schema,
    load_project_config,
    validate_inference_record,
)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_model_schema()

    def valid_record(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for feature in self.schema.features:
            result[feature.name] = "category" if feature.kind != "numeric" else "1.5"
        return result

    def test_project_config_has_single_seed_and_binary_mapping(self) -> None:
        config = load_project_config()

        self.assertEqual(config.random_seed, 20261028)
        self.assertEqual(config.label_mapping, {0: "goodware", 1: "malware"})
        self.assertEqual(config.cv_folds, 10)
        self.assertEqual(config.holdout_fraction, 0.2)

    def test_schema_excludes_target_identifier_and_collection_date(self) -> None:
        self.assertEqual(len(self.schema.feature_names), 25)
        self.assertNotIn("Label", self.schema.feature_names)
        self.assertNotIn("SHA1", self.schema.feature_names)
        self.assertNotIn("FirstSeenDate", self.schema.feature_names)

    def test_valid_record_is_matched_by_name_and_normalized(self) -> None:
        record = self.valid_record()
        reversed_record = dict(reversed(list(record.items())))

        normalized = validate_inference_record(reversed_record, self.schema)

        self.assertEqual(tuple(normalized), self.schema.feature_names)
        self.assertEqual(normalized["Entropy"], 1.5)

    def test_target_is_rejected_from_inference_input(self) -> None:
        record = self.valid_record()
        record["Label"] = 1

        with self.assertRaisesRegex(ContractError, "non-model"):
            validate_inference_record(record, self.schema)

    def test_missing_feature_is_rejected(self) -> None:
        record = self.valid_record()
        record.pop("Entropy")

        with self.assertRaisesRegex(ContractError, "Missing"):
            validate_inference_record(record, self.schema)

    def test_non_finite_numeric_value_is_rejected(self) -> None:
        record = self.valid_record()
        record["Entropy"] = "inf"

        with self.assertRaisesRegex(ContractError, "finite"):
            validate_inference_record(record, self.schema)

    def test_only_identify_may_be_missing(self) -> None:
        record = self.valid_record()
        record["Identify"] = ""
        normalized = validate_inference_record(record, self.schema)
        self.assertEqual(normalized["Identify"], "")

        record = self.valid_record()
        record["ImportedDlls"] = ""
        with self.assertRaisesRegex(ContractError, "may not be missing"):
            validate_inference_record(record, self.schema)


if __name__ == "__main__":
    unittest.main()
