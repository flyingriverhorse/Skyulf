"""Regression models."""

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor  # ty: ignore[unresolved-import]

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as _lgb  # ty: ignore[unresolved-import]
    from lightgbm import LGBMRegressor  # ty: ignore[unresolved-import]

    LIGHTGBM_AVAILABLE = True

    # LightGBM 4.x emits C++ stderr warnings ("No further splits with positive
    # gain", auto col-wise info, etc.) that the `verbose=-1` constructor param
    # does not always silence. Register a no-op logger so all native messages
    # are intercepted by Python and dropped. Safe to call multiple times.
    class _SilentLgbmLogger:
        def info(self, msg: str) -> None:  # noqa: D401
            pass

        def warning(self, msg: str) -> None:  # noqa: D401
            pass

    _lgb.register_logger(_SilentLgbmLogger())
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator


# --- Linear Regression ---
class LinearRegressionApplier(SklearnApplier):
    """Linear Regression Applier."""


@NodeRegistry.register("linear_regression", LinearRegressionApplier)
@node_meta(
    id="linear_regression",
    name="Linear Regression",
    category="Modeling",
    description="Ordinary least squares Linear Regression.",
    params={"fit_intercept": True, "copy_X": True, "n_jobs": -1},
    tags=["requires_scaling"],
)
class LinearRegressionCalculator(SklearnCalculator):
    """Linear Regression Calculator."""

    def __init__(self):
        super().__init__(
            model_class=LinearRegression,
            default_params={
                "fit_intercept": True,
                "copy_X": True,
                "n_jobs": -1,
            },
            problem_type="regression",
        )


# --- Ridge Regression ---
class RidgeRegressionApplier(SklearnApplier):
    """Ridge Regression Applier."""


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


@NodeRegistry.register("random_forest_regressor", RandomForestRegressorApplier)
@node_meta(
    id="random_forest_regressor",
    name="Random Forest Regressor",
    category="Modeling",
    description="Ensemble of decision trees for regression.",
    params={"n_estimators": 50, "max_depth": 10, "min_samples_split": 5},
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


# --- Extra Trees Regressor ---
class ExtraTreesRegressorApplier(SklearnApplier):
    """Extra Trees Regressor Applier."""


@NodeRegistry.register("extra_trees_regressor", ExtraTreesRegressorApplier)
@node_meta(
    id="extra_trees_regressor",
    name="Extra Trees Regressor",
    category="Modeling",
    description="Extremely randomised trees — faster than Random Forest, often comparably accurate.",
    params={"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
)
class ExtraTreesRegressorCalculator(SklearnCalculator):
    """Extra Trees Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=ExtraTreesRegressor,
            default_params={
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "criterion": "squared_error",
                "bootstrap": False,
                "n_jobs": -1,
                "random_state": 42,
            },
            problem_type="regression",
        )


# --- HistGradientBoosting Regressor ---
class HistGradientBoostingRegressorApplier(SklearnApplier):
    """HistGradientBoosting Regressor Applier."""


@NodeRegistry.register(
    "hist_gradient_boosting_regressor", HistGradientBoostingRegressorApplier
)
@node_meta(
    id="hist_gradient_boosting_regressor",
    name="Hist Gradient Boosting Regressor",
    category="Modeling",
    description="Histogram-based gradient boosting — sklearn's fast LightGBM-style implementation. Handles NaN natively.",
    params={"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
)
class HistGradientBoostingRegressorCalculator(SklearnCalculator):
    """HistGradientBoosting Regressor Calculator."""

    def __init__(self):
        super().__init__(
            model_class=HistGradientBoostingRegressor,
            default_params={
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "max_depth": None,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
                "max_bins": 255,
                "random_state": 42,
            },
            problem_type="regression",
        )


# --- LightGBM Regressor (optional) ---
if LIGHTGBM_AVAILABLE:

    class LGBMRegressorApplier(SklearnApplier):
        """LightGBM Regressor Applier.

        LightGBM 4.x sets ``feature_names_in_`` to auto-generated names
        (``Column_0``, ``Column_1``...) even when fit with numpy arrays, and the
        property's deleter is intentionally a no-op (see upstream source). That
        triggers sklearn's ``UserWarning: X does not have valid feature names``
        on every predict call. We suppress it locally here so the warning never
        leaks out of the applier boundary.
        """

        def predict(self, df, model_artifact):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*valid feature names.*"
                )
                return super().predict(df, model_artifact)

    @NodeRegistry.register("lgbm_regressor", LGBMRegressorApplier)
    @node_meta(
        id="lgbm_regressor",
        name="LightGBM Regressor",
        category="Modeling",
        description="Microsoft LightGBM: leaf-wise gradient boosting, fast and memory-efficient with categorical support.",
        params={"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
    )
    class LGBMRegressorCalculator(SklearnCalculator):
        """LightGBM Regressor Calculator."""

        def __init__(self):
            super().__init__(
                model_class=LGBMRegressor,
                default_params={
                    "n_estimators": 100,
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "min_child_samples": 20,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                    "boosting_type": "gbdt",
                    "n_jobs": -1,
                    "random_state": 42,
                    "verbose": -1,
                    "verbosity": -1,
                },
                problem_type="regression",
            )

        def fit(self, X, y, config, **kwargs):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*valid feature names.*"
                )
                return super().fit(X, y, config, **kwargs)


# --- XGBoost ---
if XGBOOST_AVAILABLE:

    class XGBRegressorApplier(SklearnApplier):
        """XGBoost Regressor Applier."""

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
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                problem_type="regression",
            )
