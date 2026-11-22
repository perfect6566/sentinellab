from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ml.contracts import load_model_schema
from ml.preprocessing import (
    PreprocessingConfig,
    assert_finite_transformed,
    build_preprocessor,
    validate_feature_frame,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_model_schema()

    def frame(self, rows: int = 4) -> pd.DataFrame:
        data: dict[str, list[object]] = {}
        for feature in self.schema.features:
            if feature.kind == "numeric":
                data[feature.name] = [float(index + 1) for index in range(rows)]
            elif feature.name == "Identify":
                data[feature.name] = ["compiler-a", "", "compiler-b", "compiler-a"][:rows]
            elif feature.name == "ImportedDlls":
                data[feature.name] = ["kernel32.dll user32.dll"] * rows
            else:
                data[feature.name] = ["createfilea readfile writefile"] * rows
        # Real data has these constants; the fold-local variance filter should remove them.
        for name, value in {"Magic": 267, "PE_TYPE": 267, "SizeOfOptionalHeader": 224}.items():
            data[name] = [value] * rows
        return pd.DataFrame(data)

    def test_transformer_handles_missing_category_and_text(self) -> None:
        frame = self.frame()
        transformer = build_preprocessor(
            PreprocessingConfig(scale_numeric=True, text_hash_features=32), self.schema
        )

        transformed = transformer.fit_transform(frame)

        self.assertEqual(transformed.shape[0], len(frame))
        self.assertGreater(transformed.shape[1], 32)
        assert_finite_transformed(transformed)

    def test_column_order_does_not_change_output(self) -> None:
        frame = self.frame()
        transformer = build_preprocessor(
            PreprocessingConfig(scale_numeric=False, text_hash_features=32), self.schema
        )
        transformer.fit(frame)

        normal = transformer.transform(frame)
        reversed_columns = transformer.transform(frame[list(reversed(frame.columns))])

        np.testing.assert_allclose(normal.toarray(), reversed_columns.toarray())

    def test_validation_rejects_label_and_missing_features(self) -> None:
        frame = self.frame()
        frame["Label"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_feature_frame(frame, self.schema)

        frame = self.frame().drop(columns=["Entropy"])
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_feature_frame(frame, self.schema)

    def test_scaler_is_fitted_only_on_training_values(self) -> None:
        train = self.frame()
        validation = self.frame(1)
        validation.loc[:, "Entropy"] = 1_000_000_000
        transformer = build_preprocessor(
            PreprocessingConfig(scale_numeric=True, include_text=False), self.schema
        )

        transformer.fit(train)
        numeric_pipeline = transformer.named_transformers_["numeric"]
        scaler = numeric_pipeline.named_steps["scaler"]

        self.assertNotEqual(
            float(scaler.center_[self.schema.numeric_features.index("Entropy")]), 1_000_000_000
        )
        assert_finite_transformed(transformer.transform(validation))


if __name__ == "__main__":
    unittest.main()
