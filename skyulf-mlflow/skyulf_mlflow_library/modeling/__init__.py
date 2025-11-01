"""
Modeling module for Skyulf MLflow.

This module provides tools for model training, evaluation, metrics calculation,
and model registry with versioning.
"""

from skyulf_mlflow_library.modeling.metrics import (
    MetricsCalculator,
    calculate_metrics,
)
from skyulf_mlflow_library.modeling.registry import ModelRegistry

# Import classification models
from skyulf_mlflow_library.modeling.classifiers import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    SupportVectorClassifier,
    DecisionTreeClassifier,
    KNeighborsClassifier,
)

# Import regression models
from skyulf_mlflow_library.modeling.regressors import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    DecisionTreeRegressor,
    KNeighborsRegressor,
)

__all__ = [
    # Metrics and registry
    'MetricsCalculator',
    'calculate_metrics',
    'ModelRegistry',
    # Classification models
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'LogisticRegression',
    'SupportVectorClassifier',
    'DecisionTreeClassifier',
    'KNeighborsClassifier',
    # Regression models
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'DecisionTreeRegressor',
    'KNeighborsRegressor',
]
