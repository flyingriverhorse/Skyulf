"""Registry that ties model types to their hyperparameter definitions."""

from typing import Any, Dict, List

from .base import HyperparameterField
from .logistic_regression import LOGISTIC_REGRESSION_PARAMS
from .random_forest_classifier import RANDOM_FOREST_CLASSIFIER_PARAMS
from .random_forest_regressor import RANDOM_FOREST_REGRESSOR_PARAMS
from .ridge_regression import RIDGE_REGRESSION_PARAMS

MODEL_HYPERPARAMETERS: Dict[str, List[HyperparameterField]] = {
    "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
    "random_forest_classifier": RANDOM_FOREST_CLASSIFIER_PARAMS,
    "random_forest_regressor": RANDOM_FOREST_REGRESSOR_PARAMS,
    "ridge_regression": RIDGE_REGRESSION_PARAMS,
}


def get_hyperparameters_for_model(model_type: str) -> List[Dict[str, Any]]:
    fields = MODEL_HYPERPARAMETERS.get(model_type, [])
    return [field.to_dict() for field in fields]


def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    fields = MODEL_HYPERPARAMETERS.get(model_type, [])
    return {field.name: field.default for field in fields}
