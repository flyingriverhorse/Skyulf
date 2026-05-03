"""Classification models."""

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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

    _lgb.register_logger(_SilentLgbmLogger())
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator


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
