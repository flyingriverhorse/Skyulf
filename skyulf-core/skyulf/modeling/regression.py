"""Regression models."""

import warnings
from typing import Any

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
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
        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

    _lgb.register_logger(_SilentLgbmLogger())  # ty: ignore[unresolved-attribute]
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ._boosting_progress import LightGBMIterationAdapter, XgboostIterationAdapter
from ._lightgbm import resolve_sampling_frequency
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
    tags=["requires_scaling", "regression"],
    learns_from_data=True,
)
class LinearRegressionCalculator(SklearnCalculator):
    """Linear Regression Calculator."""

    def __init__(self):
        """Bind ``LinearRegression`` with sklearn's defaults and every core.

        ``fit_intercept=True`` and ``copy_X=True`` are sklearn's own, restated
        so the node's tunable surface is explicit. ``n_jobs=-1`` departs from
        sklearn's ``None``: the solve is dispatched through joblib, which per
        sklearn only pays off on extremely large problems.
        """
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
    params={"alpha": 1.0, "solver": "auto"},
    tags=["requires_scaling", "regression"],
    learns_from_data=True,
)
class RidgeRegressionCalculator(SklearnCalculator):
    """Ridge Regression Calculator."""

    def __init__(self):
        """Bind ``Ridge`` with sklearn's own defaults.

        ``alpha=1.0`` is the L2 penalty strength and ``solver="auto"`` lets
        sklearn pick the solver from the data's shape and dtype. Both are
        restated so the node's tunable surface is explicit; ``fit``'s config
        overrides either.
        """
        super().__init__(
            model_class=Ridge,
            default_params={
                "alpha": 1.0,
                "solver": "auto",
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
    tags=["regression"],
    learns_from_data=True,
)
class RandomForestRegressorCalculator(SklearnCalculator):
    """Random Forest Regressor Calculator."""

    def __init__(self):
        """Bind ``RandomForestRegressor`` with 50 shallow, constrained trees.

        ``max_depth=10``, ``min_samples_split=5`` and ``min_samples_leaf=2``
        restrict tree growth that sklearn leaves open by default, and the
        forest is half sklearn's default size. ``n_jobs=-1`` fits across every
        core. All are overridable from ``fit``'s config.
        """
        super().__init__(
            model_class=RandomForestRegressor,
            default_params={
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
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
    tags=["requires_scaling", "regression"],
    learns_from_data=True,
)
class LassoRegressionCalculator(SklearnCalculator):
    """Lasso Regression Calculator."""

    def __init__(self):
        """Bind ``Lasso`` with sklearn's own defaults.

        ``alpha=1.0`` is the L1 penalty strength, and ``selection="cyclic"``
        walks the features in fixed order during coordinate descent rather
        than randomly. Both are restated so the node's tunable surface is
        explicit; ``fit``'s config overrides either.
        """
        super().__init__(
            model_class=Lasso,
            default_params={"alpha": 1.0, "selection": "cyclic"},
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
    tags=["requires_scaling", "regression"],
    learns_from_data=True,
)
class ElasticNetRegressionCalculator(SklearnCalculator):
    """ElasticNet Regression Calculator."""

    def __init__(self):
        """Bind ``ElasticNet`` with sklearn's own defaults.

        ``alpha=1.0`` is the total penalty strength and ``l1_ratio=0.5``
        splits it evenly between the L1 and L2 priors; ``selection="cyclic"``
        orders the coordinate-descent updates. All three are restated so the
        node's tunable surface is explicit; ``fit``'s config overrides any of
        them.
        """
        super().__init__(
            model_class=ElasticNet,
            default_params={
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "selection": "cyclic",
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
    tags=["requires_scaling", "regression"],
    learns_from_data=True,
)
class SVRCalculator(SklearnCalculator):
    """SVR Calculator."""

    def __init__(self):
        """Bind ``SVR`` with sklearn's own defaults.

        ``C=1.0``, ``kernel="rbf"`` and ``gamma="scale"`` are all sklearn's,
        restated so the node's tunable surface is explicit. The width of the
        epsilon-insensitive tube stays at sklearn's ``epsilon=0.1`` unless
        ``fit``'s config overrides it.
        """
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
    tags=["requires_scaling", "regression"],
    learns_from_data=True,
)
class KNeighborsRegressorCalculator(SklearnCalculator):
    """K-Neighbors Regressor Calculator."""

    def __init__(self):
        """Bind ``KNeighborsRegressor`` with an unweighted 5-neighbour mean.

        ``algorithm="auto"`` leaves sklearn to pick the neighbour-search
        structure from the data's shape and size, and ``n_jobs=-1``
        parallelizes the queries.
        """
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
    tags=["regression"],
    learns_from_data=True,
)
class DecisionTreeRegressorCalculator(SklearnCalculator):
    """Decision Tree Regressor Calculator."""

    def __init__(self):
        """Bind ``DecisionTreeRegressor`` with sklearn's own unpruned defaults.

        ``max_depth=None`` grows the tree until its leaves are pure or fall
        below ``min_samples_split=2``, so unlike the forest and boosting nodes
        in this module this one ships no regularization of its own.
        ``criterion="squared_error"`` is the regression analogue of the
        classifier node's ``gini`` default.
        """
        super().__init__(
            model_class=DecisionTreeRegressor,
            default_params={
                "max_depth": None,
                "min_samples_split": 2,
                "criterion": "squared_error",
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
    tags=["regression"],
    learns_from_data=True,
)
class GradientBoostingRegressorCalculator(SklearnCalculator):
    """Gradient Boosting Regressor Calculator."""

    def __init__(self):
        """Bind ``GradientBoostingRegressor`` with 100 depth-3 trees at rate 0.1.

        All three are exactly sklearn's own defaults, restated so the node's
        tunable surface is explicit; ``fit``'s config overrides any of them.
        """
        super().__init__(
            model_class=GradientBoostingRegressor,
            default_params={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
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
    tags=["regression"],
    learns_from_data=True,
)
class AdaBoostRegressorCalculator(SklearnCalculator):
    """AdaBoost Regressor Calculator."""

    def __init__(self):
        """Bind ``AdaBoostRegressor`` with 50 weak learners at rate 1.0.

        Both are sklearn's defaults. No base estimator is pinned here, so
        ``fit``'s config may supply one; otherwise sklearn falls back to
        ``DecisionTreeRegressor(max_depth=3)`` — a deeper weak learner than
        the ``AdaBoostClassifier`` node's depth-1 stump.
        """
        super().__init__(
            model_class=AdaBoostRegressor,
            default_params={
                "n_estimators": 50,
                "learning_rate": 1.0,
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
    tags=["regression"],
    learns_from_data=True,
)
class ExtraTreesRegressorCalculator(SklearnCalculator):
    """Extra Trees Regressor Calculator."""

    def __init__(self):
        """Bind ``ExtraTreesRegressor`` with 100 unpruned, non-bootstrapped trees.

        ``bootstrap=False`` is the distinguishing default: extra-trees
        randomizes split thresholds instead of resampling rows, so every tree
        sees the whole sample. The remaining settings match sklearn's defaults
        (``criterion="squared_error"`` being the regression analogue of
        ``gini``), and ``n_jobs=-1`` fits across every core.
        """
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
            },
            problem_type="regression",
        )


# --- HistGradientBoosting Regressor ---
class HistGradientBoostingRegressorApplier(SklearnApplier):
    """HistGradientBoosting Regressor Applier."""


@NodeRegistry.register("hist_gradient_boosting_regressor", HistGradientBoostingRegressorApplier)
@node_meta(
    id="hist_gradient_boosting_regressor",
    name="Hist Gradient Boosting Regressor",
    category="Modeling",
    description="Histogram-based gradient boosting — sklearn's fast LightGBM-style implementation.",
    params={"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
    tags=["regression"],
    learns_from_data=True,
)
class HistGradientBoostingRegressorCalculator(SklearnCalculator):
    """HistGradientBoosting Regressor Calculator."""

    def __init__(self):
        """Bind ``HistGradientBoostingRegressor`` with sklearn's own defaults.

        Tree shape is governed leaf-wise — ``max_leaf_nodes=31`` with
        ``max_depth=None`` — because this estimator bins features into
        ``max_bins=255`` buckets and grows by best leaf rather than level by
        level. That is the opposite of the level-wise boosting nodes elsewhere
        in this module.
        """
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
            },
            problem_type="regression",
        )


# --- LightGBM Regressor (optional) ---
if LIGHTGBM_AVAILABLE:

    class _SamplingLGBMRegressor(LGBMRegressor):
        """Resolve automatic row sampling after tuning applies candidate parameters."""

        def _process_params(self, stage: str) -> dict[str, Any]:
            """Apply the sampling policy without mutating cloneable constructor state."""
            return resolve_sampling_frequency(super()._process_params(stage))

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
            """Delegate to the base predict with the feature-name warning suppressed."""
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().predict(df, model_artifact)

    @NodeRegistry.register("lgbm_regressor", LGBMRegressorApplier)
    @node_meta(
        id="lgbm_regressor",
        name="LightGBM Regressor",
        category="Modeling",
        description="LightGBM: leaf-wise gradient boosting, fast and memory-efficient with categorical support.",
        params={"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
        tags=["regression"],
        learns_from_data=True,
    )
    class LGBMRegressorCalculator(SklearnCalculator):
        """LightGBM Regressor Calculator."""

        def __init__(self):
            """Bind ``LGBMRegressor`` with an unregularized leaf-wise configuration.

            ``max_depth=-1`` leaves depth unbounded, so ``num_leaves=31`` alone
            governs tree shape, and both subsampling rates and both L1/L2
            penalties start at their neutral values. Automatic ``subsample_freq``
            enables per-round bagging when the fraction is below one, except
            for GOSS, which keeps its gradient sampling. ``verbose`` and
            ``verbosity`` are pinned to -1 to quiet LightGBM's native logging
            alongside the no-op logger registered at import time.
            """
            super().__init__(
                model_class=_SamplingLGBMRegressor,
                default_params={
                    "n_estimators": 100,
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "min_child_samples": 20,
                    "subsample": 1.0,
                    "subsample_freq": None,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                    "boosting_type": "gbdt",
                    "n_jobs": -1,
                    "verbose": -1,
                    "verbosity": -1,
                },
                problem_type="regression",
            )

        def fit(
            self,
            X,
            y,
            config,
            progress_callback=None,
            log_callback=None,
            validation_data=None,
            iteration_callback=None,
        ):
            """Delegate to the base fit with the feature-name warning suppressed."""
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().fit(
                    X,
                    y,
                    config,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    validation_data=validation_data,
                    iteration_callback=iteration_callback,
                )

        def _boosting_fit_kwargs(self, model, X_np, y_np, iteration_callback):
            if iteration_callback is None:
                return {}
            return {
                "eval_set": [(X_np, y_np)],
                "callbacks": [LightGBMIterationAdapter(iteration_callback)],
            }


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
        tags=["regression"],
        learns_from_data=True,
    )
    class XGBRegressorCalculator(SklearnCalculator):
        """XGBoost Regressor Calculator."""

        def __init__(self):
            """Bind ``XGBRegressor`` with XGBoost's defaults plus row/column subsampling.

            ``n_estimators=100``, ``max_depth=6`` and ``learning_rate=0.3`` are
            XGBoost's own. ``subsample`` and ``colsample_bytree`` are pulled
            down from XGBoost's 1.0 to 0.8, and neither appears in the node's
            ``params`` metadata, so they are only reachable through ``fit``'s
            config.
            """
            super().__init__(
                model_class=XGBRegressor,
                default_params={
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_jobs": -1,
                },
                problem_type="regression",
            )

        def _boosting_fit_kwargs(self, model, X_np, y_np, iteration_callback):
            if iteration_callback is None or XgboostIterationAdapter is None:
                return {}
            # XGBoost 3.x reads callbacks from the estimator itself (they were
            # removed from fit()). eval_set is display-only (no early
            # stopping), so the trained model is identical to a plain fit — it
            # just streams per-round training loss for the live chart.
            model.callbacks = [
                XgboostIterationAdapter(iteration_callback, total=int(model.n_estimators))
            ]
            return {
                "eval_set": [(X_np, y_np)],
                "verbose": False,
                "_detach_callbacks": True,
            }
