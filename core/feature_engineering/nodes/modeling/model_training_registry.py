"""Model registry for the modeling training node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass(frozen=True)
class ModelSpec:
    key: str
    problem_type: str  # "classification" or "regression"
    factory: Callable[..., Any]
    default_params: Dict[str, Any]


_MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        problem_type="classification",
        factory=LogisticRegression,
        default_params={
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "auto",
        },
    ),
    "random_forest_classifier": ModelSpec(
        key="random_forest_classifier",
        problem_type="classification",
        factory=RandomForestClassifier,
        default_params={
            "n_estimators": 50,  # Reduced from 200 to decrease model size
            "max_depth": 10,  # Limit tree depth to reduce complexity and size
            "min_samples_split": 5,  # Prevent overfitting and reduce tree size
            "min_samples_leaf": 2,  # Further reduce tree complexity
            "n_jobs": -1,
            "random_state": 42,
        },
    ),
    "random_forest_regressor": ModelSpec(
        key="random_forest_regressor",
        problem_type="regression",
        factory=RandomForestRegressor,
        default_params={
            "n_estimators": 50,  # Reduced from 200 to decrease model size
            "max_depth": 10,  # Limit tree depth to reduce complexity and size
            "min_samples_split": 5,  # Prevent overfitting and reduce tree size
            "min_samples_leaf": 2,  # Further reduce tree complexity
            "n_jobs": -1,
            "random_state": 42,
        },
    ),
    "ridge_regression": ModelSpec(
        key="ridge_regression",
        problem_type="regression",
        factory=Ridge,
        default_params={
            "alpha": 1.0,
            "solver": "auto",
        },
    ),
}


def get_model_spec(model_type: str) -> ModelSpec:
    """Return the registered model spec, raising if unavailable."""

    if model_type not in _MODEL_REGISTRY:
        raise KeyError(f"Unsupported model_type '{model_type}'")
    return _MODEL_REGISTRY[model_type]


def list_registered_models() -> Dict[str, ModelSpec]:
    """Return the registry mapping (read-only copy)."""

    return dict(_MODEL_REGISTRY)
