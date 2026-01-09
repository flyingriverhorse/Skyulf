"""Regression models."""

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
    params={"alpha": 1.0, "solver": "auto", "random_state": 42},
    tags=["requires_scaling"],
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


# --- Lasso ---
class LassoRegressionApplier(SklearnApplier):
    """Lasso Regression Applier."""

    pass


@NodeRegistry.register("lasso_regression", LassoRegressionApplier)
@node_meta(
    id="lasso_regression",
    name="Lasso Regression",
    category="Modeling",
    description="Linear Model trained with L1 prior as regularizer.",
    params={"alpha": 1.0, "selection": "cyclic"},
    tags=["requires_scaling"],
)
class LassoRegressionCalculator(SklearnCalculator):
    """Lasso Regression Calculator."""

    def __init__(self):
        super().__init__(
            model_class=Lasso,
            default_params={"alpha": 1.0, "selection": "cyclic", "random_state": 42},
            problem_type="regression",
        )


# --- ElasticNet ---
class ElasticNetRegressionApplier(SklearnApplier):
    """ElasticNet Regression Applier."""

    pass


@NodeRegistry.register("elasticnet_regression", ElasticNetRegressionApplier)
@node_meta(
    id="elasticnet_regression",
    name="ElasticNet Regression",
    category="Modeling",
    description="Linear regression with combined L1 and L2 priors.",
    params={"alpha": 1.0, "l1_ratio": 0.5, "selection": "cyclic"},
    tags=["requires_scaling"],
)
class ElasticNetRegressionCalculator(SklearnCalculator):
    """ElasticNet Regression Calculator."""

    def __init__(self):
        super().__init__(
            model_class=ElasticNet,
            default_params={
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "selection": "cyclic",
                "random_state": 42,
            },
            problem_type="regression",
        )


# --- SVR ---
class SVRApplier(SklearnApplier):
    """SVR Applier."""

    pass


@NodeRegistry.register("svr", SVRApplier)
@node_meta(
    id="svr",
    name="Support Vector Regressor",
    category="Modeling",
    description="Epsilon-Support Vector Regression.",
    params={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    tags=["requires_scaling"],
)
class SVRCalculator(SklearnCalculator):
    """SVR Calculator."""

    def __init__(self):
        super().__init__(
            model_class=SVR,
            default_params={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            problem_type="regression",
        )


# --- K-Neighbors ---
class KNeighborsRegressorApplier(SklearnApplier):
    """K-Neighbors Regressor Applier."""

    pass


@NodeRegistry.register("k_neighbors_regressor", KNeighborsRegressorApplier)
@node_meta(
    id="k_neighbors_regressor",
    name="K-Neighbors Regressor",
    category="Modeling",
    description="Regression based on k-nearest neighbors.",
    params={"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    tags=["requires_scaling"],
)
class KNeighborsRegressorCalculator(SklearnCalculator):
    """K-Neighbors Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=KNeighborsRegressor,
            default_params={
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto",
                "n_jobs": -1,
            },
            problem_type="regression",
        )


# --- Decision Tree ---
class DecisionTreeRegressorApplier(SklearnApplier):
    """Decision Tree Regressor Applier."""

    pass


@NodeRegistry.register("decision_tree_regressor", DecisionTreeRegressorApplier)
@node_meta(
    id="decision_tree_regressor",
    name="Decision Tree Regressor",
    category="Modeling",
    description="A decision tree regressor.",
    params={"max_depth": None, "min_samples_split": 2, "criterion": "squared_error"},
)
class DecisionTreeRegressorCalculator(SklearnCalculator):
    """Decision Tree Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=DecisionTreeRegressor,
            default_params={
                "max_depth": None,
                "min_samples_split": 2,
                "criterion": "squared_error",
                "random_state": 42,
            },
            problem_type="regression",
        )


# --- Gradient Boosting ---
class GradientBoostingRegressorApplier(SklearnApplier):
    """Gradient Boosting Regressor Applier."""

    pass


@NodeRegistry.register("gradient_boosting_regressor", GradientBoostingRegressorApplier)
@node_meta(
    id="gradient_boosting_regressor",
    name="Gradient Boosting Regressor",
    category="Modeling",
    description="Gradient Boosting for regression.",
    params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
)
class GradientBoostingRegressorCalculator(SklearnCalculator):
    """Gradient Boosting Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=GradientBoostingRegressor,
            default_params={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
            problem_type="regression",
        )


# --- AdaBoost ---
class AdaBoostRegressorApplier(SklearnApplier):
    """AdaBoost Regressor Applier."""

    pass


@NodeRegistry.register("adaboost_regressor", AdaBoostRegressorApplier)
@node_meta(
    id="adaboost_regressor",
    name="AdaBoost Regressor",
    category="Modeling",
    description="An AdaBoost regressor.",
    params={"n_estimators": 50, "learning_rate": 1.0},
)
class AdaBoostRegressorCalculator(SklearnCalculator):
    """AdaBoost Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=AdaBoostRegressor,
            default_params={
                "n_estimators": 50,
                "learning_rate": 1.0,
                "random_state": 42,
            },
            problem_type="regression",
        )


# --- XGBoost ---
class XGBRegressorApplier(SklearnApplier):
    """XGBoost Regressor Applier."""

    pass


@NodeRegistry.register("xgboost_regressor", XGBRegressorApplier)
@node_meta(
    id="xgboost_regressor",
    name="XGBoost Regressor",
    category="Modeling",
    description="Extreme Gradient Boosting regressor.",
    params={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3},
)
class XGBRegressorCalculator(SklearnCalculator):
    """XGBoost Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=XGBRegressor,
            default_params={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.3,
                "n_jobs": -1,
                "random_state": 42,
                "eval_metric": "rmse",
            },
            problem_type="regression",
        )
