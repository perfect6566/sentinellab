"""Factories for all seven assignment-required model families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.preprocessing import PreprocessingConfig, build_preprocessor

EstimatorFactory = Callable[[int, bool], Any]


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    display_name: str
    scale_numeric: bool
    factory: EstimatorFactory


def _logistic(seed: int, smoke: bool) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=300 if smoke else 1000,
        solver="liblinear",
        random_state=seed,
    )


def _decision_tree(seed: int, smoke: bool) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=8 if smoke else 16, random_state=seed)


def _random_forest(seed: int, smoke: bool) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=30 if smoke else 300,
        max_depth=10 if smoke else None,
        n_jobs=4,
        random_state=seed,
    )


def _pytorch_mlp(seed: int, smoke: bool) -> Any:
    # Import lazily. On macOS, eagerly loading PyTorch's OpenMP runtime in a
    # process that later loads XGBoost/LightGBM can cause a native crash.
    from ml.torch_mlp import TorchMLPClassifier

    return TorchMLPClassifier(
        hidden_dims=(64, 32) if smoke else (128, 64),
        max_epochs=2 if smoke else 20,
        batch_size=128 if smoke else 256,
        random_state=seed,
        device="cpu",
    )


def _xgboost(seed: int, smoke: bool) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=30 if smoke else 300,
        max_depth=4 if smoke else 8,
        learning_rate=0.1 if smoke else 0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )


def _lightgbm(seed: int, smoke: bool) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=30 if smoke else 300,
        num_leaves=31,
        learning_rate=0.1 if smoke else 0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=seed,
        verbosity=-1,
    )


def _catboost(seed: int, smoke: bool) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=30 if smoke else 300,
        depth=6,
        learning_rate=0.1 if smoke else 0.05,
        random_seed=seed,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        "logistic_regression", "Logistic Regression", True, _logistic
    ),
    "decision_tree": ModelDefinition("decision_tree", "Decision Tree", False, _decision_tree),
    "random_forest": ModelDefinition("random_forest", "Random Forest", False, _random_forest),
    "pytorch_mlp": ModelDefinition("pytorch_mlp", "PyTorch MLP", True, _pytorch_mlp),
    "xgboost": ModelDefinition("xgboost", "XGBoost", False, _xgboost),
    "lightgbm": ModelDefinition("lightgbm", "LightGBM", False, _lightgbm),
    "catboost": ModelDefinition("catboost", "CatBoost", False, _catboost),
}


def build_model_pipeline(name: str, *, seed: int, smoke: bool = False) -> Pipeline:
    try:
        definition = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}; expected one of {sorted(MODEL_REGISTRY)}"
        ) from exc
    preprocessor = build_preprocessor(
        PreprocessingConfig(
            scale_numeric=definition.scale_numeric,
            include_text=True,
            text_hash_features=64 if smoke else 256,
        )
    )
    return Pipeline(
        [("preprocessing", preprocessor), ("classifier", definition.factory(seed, smoke))]
    )
