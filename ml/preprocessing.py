"""Leakage-safe preprocessing factories for model pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ml.contracts import ModelSchema, load_model_schema


@dataclass(frozen=True)
class PreprocessingConfig:
    scale_numeric: bool
    include_text: bool = True
    text_hash_features: int = 256
    categorical_min_frequency: int = 5

    def __post_init__(self) -> None:
        if self.text_hash_features < 16 or self.text_hash_features & (self.text_hash_features - 1):
            raise ValueError("text_hash_features must be a power of two and at least 16")
        if self.categorical_min_frequency < 1:
            raise ValueError("categorical_min_frequency must be positive")


def _numeric_pipeline(scale: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
    ]
    if scale:
        steps.append(("scaler", RobustScaler()))
    return Pipeline(steps)


def build_preprocessor(
    config: PreprocessingConfig, schema: ModelSchema | None = None
) -> ColumnTransformer:
    """Build an unfitted transformer to be placed inside every model pipeline.

    Returning an unfitted object is intentional: cross-validation clones and fits
    it separately on each training fold. No caller should fit this transformer on
    the complete dataset before cross-validation.
    """

    schema = schema or load_model_schema()
    transformers: list[tuple[str, object, object]] = [
        ("numeric", _numeric_pipeline(config.scale_numeric), list(schema.numeric_features)),
        (
            "categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            missing_values="", strategy="constant", fill_value="<MISSING>"
                        ),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=config.categorical_min_frequency,
                            sparse_output=True,
                        ),
                    ),
                ]
            ),
            list(schema.categorical_features),
        ),
    ]
    if config.include_text:
        for name in schema.text_features:
            transformers.append(
                (
                    f"text_{name}",
                    HashingVectorizer(
                        n_features=config.text_hash_features,
                        alternate_sign=False,
                        norm="l2",
                        lowercase=True,
                        token_pattern=r"(?u)\b[\w.]+\b",
                    ),
                    name,
                )
            )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def validate_feature_frame(frame: pd.DataFrame, schema: ModelSchema | None = None) -> None:
    schema = schema or load_model_schema()
    actual = set(frame.columns)
    expected = set(schema.feature_names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"Feature frame mismatch; missing={missing}, unexpected={unexpected}")


def assert_finite_transformed(matrix: Any) -> None:
    """Raise if a transformed dense or sparse matrix contains NaN/Inf."""

    import numpy as np

    values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(values).all():
        raise ValueError("Preprocessing produced non-finite values")
