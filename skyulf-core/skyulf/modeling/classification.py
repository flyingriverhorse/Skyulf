"""Classification models."""

import logging
from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier  # ty: ignore[unresolved-import]

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as _lgb  # ty: ignore[unresolved-import]
    from lightgbm import LGBMClassifier  # ty: ignore[unresolved-import]

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

    _lgb.register_logger(_SilentLgbmLogger())  # ty: ignore[unresolved-attribute]
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator

logger = logging.getLogger(__name__)


# --- Logistic Regression ---
class LogisticRegressionApplier(SklearnApplier):
    """Logistic Regression Applier."""


@NodeRegistry.register("logistic_regression", LogisticRegressionApplier)
@node_meta(
    id="logistic_regression",
    name="Logistic Regression",
    category="Modeling",
    description="Linear model for classification.",
    params={"max_iter": 1000, "solver": "lbfgs", "random_state": 42},
    tags=["requires_scaling"],
)
class LogisticRegressionCalculator(SklearnCalculator):
    """Logistic Regression Calculator."""

    # sklearn solver -> penalties it actually supports. Manual/UI configuration
    # allows selecting solver and penalty independently (unlike the tuner's own
    # search space, which restricts solver to "saga" whenever penalty is
    # varied), so an incompatible combination reaches `fit()` unchecked and
    # would otherwise surface as an opaque sklearn ValueError at model-fit time.
    _SOLVER_PENALTIES: dict[str, set[Any]] = {
        "lbfgs": {"l2", None},
        "liblinear": {"l1", "l2"},
        "newton-cg": {"l2", None},
        "newton-cholesky": {"l2", None},
        "sag": {"l2", None},
        "saga": {"l1", "l2", "elasticnet", None},
    }

    def __init__(self):
        super().__init__(
            model_class=LogisticRegression,
            default_params={
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
            },
            problem_type="classification",
        )

    def fit(
        self,
        X: Any,
        y: Any,
        config: dict[str, Any],
        progress_callback: Callable[..., Any] | None = None,
        log_callback: Callable[..., Any] | None = None,
        validation_data: Any = None,
    ) -> Any:
        self._validate_solver_penalty(config)
        return super().fit(X, y, config, progress_callback, log_callback, validation_data)

    @classmethod
    def _extract_solver_penalty_params(cls, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """Returns the params dict from config, or None if unavailable/not a dict."""
        if not config:
            return None
        params = config.get("params", config)
        if not isinstance(params, dict):
            return None
        return params

    @classmethod
    def _raise_incompatible_solver_penalty(cls, solver: Any, penalty: Any) -> None:
        """Raises a ValueError listing solvers compatible with the requested penalty."""
        compatible_solvers = sorted(
            s for s, penalties in cls._SOLVER_PENALTIES.items() if penalty in penalties
        )
        raise ValueError(
            f"Logistic Regression: solver={solver!r} does not support "
            f"penalty={penalty!r}. Solvers compatible with this penalty: "
            f"{compatible_solvers or 'none'}."
        )

    @classmethod
    def _validate_solver_penalty(cls, config: dict[str, Any] | None) -> None:
        """Raise a clear, actionable error for an invalid solver/penalty pair.

        sklearn's own error for this (e.g. "Solver lbfgs supports only 'l2' or
        None penalties") is only raised deep inside `LogisticRegression.fit`,
        after data has already been split/validated upstream. Failing fast
        here with the full list of compatible solvers is more actionable.
        """
        params = cls._extract_solver_penalty_params(config)
        if params is None:
            return
        solver = params.get("solver")
        penalty = params.get("penalty")
        if solver is None or "penalty" not in params:
            return
        compatible = cls._SOLVER_PENALTIES.get(solver)
        if compatible is not None and penalty not in compatible:
            cls._raise_incompatible_solver_penalty(solver, penalty)


# --- Calibrated Classifier ---
class CalibratedClassifierApplier(SklearnApplier):
    """Calibrated Classifier Applier (well-calibrated predict_proba)."""


@NodeRegistry.register("calibrated_classifier", CalibratedClassifierApplier)
@node_meta(
    id="calibrated_classifier",
    name="Calibrated Classifier",
    category="Modeling",
    description=(
        "Wraps a base classifier with CalibratedClassifierCV so predicted "
        "probabilities are well-calibrated (Platt/sigmoid or isotonic)."
    ),
    params={"base_estimator": "logistic_regression", "method": "sigmoid", "cv": 5},
    tags=["requires_scaling"],
)
class CalibratedClassifierCalculator(SklearnCalculator):
    """Calibrated Classifier Calculator with a selectable base estimator.

    The frontend sends ``base_estimator`` as a string key (e.g.
    ``"random_forest"``); it is resolved here into a fresh estimator instance
    before ``CalibratedClassifierCV`` is constructed. Defaults to logistic
    regression for backward compatibility.
    """

    # Map of selectable base estimators → factory. Each must support
    # ``predict_proba`` (or ``decision_function``) so calibration is meaningful.
    BASE_ESTIMATORS: dict[str, Callable[[], BaseEstimator]] = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
        "gaussian_nb": lambda: GaussianNB(),
        "svc": lambda: SVC(probability=True, random_state=42),
    }

    def __init__(self):
        super().__init__(
            model_class=CalibratedClassifierCV,
            default_params={
                "estimator": LogisticRegression(max_iter=1000),
                "method": "sigmoid",
                "cv": 5,
            },
            problem_type="classification",
        )

    def fit(
        self,
        X: Any,
        y: Any,
        config: dict[str, Any],
        progress_callback: Callable[..., Any] | None = None,
        log_callback: Callable[..., Any] | None = None,
        validation_data: Any = None,
    ) -> Any:
        config = self._resolve_base_estimator(config)
        return super().fit(X, y, config, progress_callback, log_callback, validation_data)

    @classmethod
    def _resolve_base_estimator(cls, config: dict[str, Any] | None) -> dict[str, Any]:
        """Translate a ``base_estimator`` string key into an estimator instance.

        Supports both the flat config shape and the nested ``{"params": {...}}``
        shape used by the model-training payload. Unknown keys fall back to
        logistic regression with a warning.
        """
        if not config:
            return config or {}
        resolved = dict(config)
        nested = isinstance(resolved.get("params"), dict)
        bucket = dict(resolved["params"]) if nested else resolved
        key = bucket.pop("base_estimator", None)
        if isinstance(key, str):
            factory = cls.BASE_ESTIMATORS.get(key)
            if factory is None:
                logger.warning(
                    "Unknown base_estimator '%s'; falling back to logistic_regression.", key
                )
                factory = cls.BASE_ESTIMATORS["logistic_regression"]
            bucket["estimator"] = factory()
        if nested:
            resolved["params"] = bucket
            return resolved
        return bucket


# --- Random Forest Classifier ---
class RandomForestClassifierApplier(SklearnApplier):
    """Random Forest Classifier Applier."""


@NodeRegistry.register("random_forest_classifier", RandomForestClassifierApplier)
@node_meta(
    id="random_forest_classifier",
    name="Random Forest Classifier",
    category="Modeling",
    description="Ensemble of decision trees.",
    params={"n_estimators": 50, "max_depth": 10, "min_samples_split": 5},
)
class RandomForestClassifierCalculator(SklearnCalculator):
    """Random Forest Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=RandomForestClassifier,
            default_params={
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": 42,
            },
            problem_type="classification",
        )


# --- SVC ---
class SVCApplier(SklearnApplier):
    """SVC Applier."""


@NodeRegistry.register("svc", SVCApplier)
@node_meta(
    id="svc",
    name="Support Vector Classifier",
    category="Modeling",
    description="C-Support Vector Classification.",
    params={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    tags=["requires_scaling"],
)
class SVCCalculator(SklearnCalculator):
    """SVC Calculator."""

    def __init__(self):
        super().__init__(
            model_class=SVC,
            default_params={
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "probability": True,
                "random_state": 42,
            },
            problem_type="classification",
        )


# --- K-Neighbors ---
class KNeighborsClassifierApplier(SklearnApplier):
    """K-Neighbors Classifier Applier."""


@NodeRegistry.register("k_neighbors_classifier", KNeighborsClassifierApplier)
@node_meta(
    id="k_neighbors_classifier",
    name="K-Neighbors Classifier",
    category="Modeling",
    description="Classifier implementing the k-nearest neighbors vote.",
    params={"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    tags=["requires_scaling"],
)
class KNeighborsClassifierCalculator(SklearnCalculator):
    """K-Neighbors Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=KNeighborsClassifier,
            default_params={
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto",
                "n_jobs": -1,
            },
            problem_type="classification",
        )


# --- Decision Tree ---
class DecisionTreeClassifierApplier(SklearnApplier):
    """Decision Tree Classifier Applier."""


@NodeRegistry.register("decision_tree_classifier", DecisionTreeClassifierApplier)
@node_meta(
    id="decision_tree_classifier",
    name="Decision Tree Classifier",
    category="Modeling",
    description="A non-parametric supervised learning method used for classification.",
    params={"max_depth": None, "min_samples_split": 2, "criterion": "gini"},
)
class DecisionTreeClassifierCalculator(SklearnCalculator):
    """Decision Tree Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=DecisionTreeClassifier,
            default_params={
                "max_depth": None,
                "min_samples_split": 2,
                "criterion": "gini",
                "random_state": 42,
            },
            problem_type="classification",
        )


# --- Gradient Boosting ---
class GradientBoostingClassifierApplier(SklearnApplier):
    """Gradient Boosting Classifier Applier."""


@NodeRegistry.register("gradient_boosting_classifier", GradientBoostingClassifierApplier)
@node_meta(
    id="gradient_boosting_classifier",
    name="Gradient Boosting Classifier",
    category="Modeling",
    description="Gradient Boosting for classification.",
    params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
)
class GradientBoostingClassifierCalculator(SklearnCalculator):
    """Gradient Boosting Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=GradientBoostingClassifier,
            default_params={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
            problem_type="classification",
        )


# --- AdaBoost ---
class AdaBoostClassifierApplier(SklearnApplier):
    """AdaBoost Classifier Applier."""


@NodeRegistry.register("adaboost_classifier", AdaBoostClassifierApplier)
@node_meta(
    id="adaboost_classifier",
    name="AdaBoost Classifier",
    category="Modeling",
    description="An AdaBoost classifier.",
    params={"n_estimators": 50, "learning_rate": 1.0},
)
class AdaBoostClassifierCalculator(SklearnCalculator):
    """AdaBoost Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=AdaBoostClassifier,
            default_params={
                "n_estimators": 50,
                "learning_rate": 1.0,
                "random_state": 42,
            },
            problem_type="classification",
        )


# --- XGBoost ---
if XGBOOST_AVAILABLE:

    class XGBClassifierApplier(SklearnApplier):
        """XGBoost Classifier Applier."""

    @NodeRegistry.register("xgboost_classifier", XGBClassifierApplier)
    @node_meta(
        id="xgboost_classifier",
        name="XGBoost Classifier",
        category="Modeling",
        description="Extreme Gradient Boosting classifier.",
        params={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3},
    )
    class XGBClassifierCalculator(SklearnCalculator):
        """XGBoost Classifier Calculator."""

        def __init__(self):
            super().__init__(
                model_class=XGBClassifier,
                default_params={
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                problem_type="classification",
            )


# --- Extra Trees Classifier ---
class ExtraTreesClassifierApplier(SklearnApplier):
    """Extra Trees Classifier Applier."""


@NodeRegistry.register("extra_trees_classifier", ExtraTreesClassifierApplier)
@node_meta(
    id="extra_trees_classifier",
    name="Extra Trees Classifier",
    category="Modeling",
    description="Extremely randomised trees — faster than Random Forest, often comparably accurate.",
    params={"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
)
class ExtraTreesClassifierCalculator(SklearnCalculator):
    """Extra Trees Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=ExtraTreesClassifier,
            default_params={
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "criterion": "gini",
                "bootstrap": False,
                "n_jobs": -1,
                "random_state": 42,
            },
            problem_type="classification",
        )


# --- HistGradientBoosting Classifier ---
class HistGradientBoostingClassifierApplier(SklearnApplier):
    """HistGradientBoosting Classifier Applier."""


@NodeRegistry.register("hist_gradient_boosting_classifier", HistGradientBoostingClassifierApplier)
@node_meta(
    id="hist_gradient_boosting_classifier",
    name="Hist Gradient Boosting Classifier",
    category="Modeling",
    description="Histogram-based gradient boosting — sklearn's fast LightGBM-style implementation.",
    params={"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
)
class HistGradientBoostingClassifierCalculator(SklearnCalculator):
    """HistGradientBoosting Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=HistGradientBoostingClassifier,
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
            problem_type="classification",
        )


# --- LightGBM Classifier (optional) ---
if LIGHTGBM_AVAILABLE:

    class LGBMClassifierApplier(SklearnApplier):
        """LightGBM Classifier Applier.

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
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().predict(df, model_artifact)

        def predict_proba(self, df, model_artifact):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().predict_proba(df, model_artifact)

    @NodeRegistry.register("lgbm_classifier", LGBMClassifierApplier)
    @node_meta(
        id="lgbm_classifier",
        name="LightGBM Classifier",
        category="Modeling",
        description="LightGBM: leaf-wise gradient boosting, fast and memory-efficient with categorical support.",
        params={"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
    )
    class LGBMClassifierCalculator(SklearnCalculator):
        """LightGBM Classifier Calculator."""

        def __init__(self):
            super().__init__(
                model_class=LGBMClassifier,
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
                problem_type="classification",
            )

        def fit(
            self, X, y, config, progress_callback=None, log_callback=None, validation_data=None
        ):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return super().fit(
                    X,
                    y,
                    config,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    validation_data=validation_data,
                )


# --- Gaussian NB ---
class GaussianNBApplier(SklearnApplier):
    """Gaussian Naive Bayes Applier."""


@NodeRegistry.register("gaussian_nb", GaussianNBApplier)
@node_meta(
    id="gaussian_nb",
    name="Gaussian Naive Bayes",
    category="Modeling",
    description="Gaussian Naive Bayes (GaussianNB).",
    params={"var_smoothing": 1e-9},
)
class GaussianNBCalculator(SklearnCalculator):
    """Gaussian Naive Bayes Calculator."""

    def __init__(self):
        super().__init__(
            model_class=GaussianNB,
            default_params={"var_smoothing": 1e-9},
            problem_type="classification",
        )


# --- SGD Classifier ---
class SGDClassifierApplier(SklearnApplier):
    """Stochastic Gradient Descent Classifier Applier."""


@NodeRegistry.register("sgd_classifier", SGDClassifierApplier)
@node_meta(
    id="sgd_classifier",
    name="SGD Classifier (text / linear)",
    category="Modeling",
    description=(
        "Linear classifiers (SVM, logistic regression, etc.) with SGD training. "
        "Highly efficient for high-dimensional sparse/dense text representations "
        "and large datasets."
    ),
    params={
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "max_iter": 1000,
        "random_state": 42,
    },
    tags=["text", "nlp", "classification", "linear", "requires_scaling"],
)
class SGDClassifierCalculator(SklearnCalculator):
    """SGD Classifier Calculator."""

    def __init__(self):
        super().__init__(
            model_class=SGDClassifier,
            default_params={
                "loss": "log_loss",
                "penalty": "l2",
                "alpha": 0.0001,
                "l1_ratio": 0.15,
                "max_iter": 1000,
                "random_state": 42,
            },
            problem_type="classification",
        )
