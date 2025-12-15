"""Regression models."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .sklearn_wrapper import SklearnApplier, SklearnCalculator


# --- Ridge Regression ---
class RidgeRegressionCalculator(SklearnCalculator):
    """Ridge Regression Calculator."""

    def __init__(self):
        super().__init__(
            model_class=Ridge,
            default_params={
                "alpha": 1.0,
                "solver": "auto",
                "random_state": 42,
            },
            problem_type="regression",
        )


class RidgeRegressionApplier(SklearnApplier):
    """Ridge Regression Applier."""

    pass


# --- Random Forest Regressor ---
class RandomForestRegressorCalculator(SklearnCalculator):
    """Random Forest Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=RandomForestRegressor,
            default_params={
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": 42,
            },
            problem_type="regression",
        )


class RandomForestRegressorApplier(SklearnApplier):
    """Random Forest Regressor Applier."""

    pass
