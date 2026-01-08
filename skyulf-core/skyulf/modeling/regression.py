"""Regression models."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator


# --- Ridge Regression ---
class RidgeRegressionApplier(SklearnApplier):
    """Ridge Regression Applier."""

    pass


@NodeRegistry.register("ridge_regression", RidgeRegressionApplier)
@node_meta(
    id="ridge_regression",
    name="Ridge Regression",
    category="Modeling",
    description="Linear least squares with l2 regularization.",
    params={"alpha": 1.0, "solver": "auto", "random_state": 42}
)
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


# --- Random Forest Regressor ---
class RandomForestRegressorApplier(SklearnApplier):
    """Random Forest Regressor Applier."""

    pass


@NodeRegistry.register("random_forest_regressor", RandomForestRegressorApplier)
@node_meta(
    id="random_forest_regressor",
    name="Random Forest Regressor",
    category="Modeling",
    description="Ensemble of decision trees for regression.",
    params={"n_estimators": 50, "max_depth": 10, "min_samples_split": 5}
)
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
