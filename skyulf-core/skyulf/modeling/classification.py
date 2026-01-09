"""Classification models."""

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator


# --- Logistic Regression ---
class LogisticRegressionApplier(SklearnApplier):
    """Logistic Regression Applier."""

    pass


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

    pass


@NodeRegistry.register("random_forest_classifier", RandomForestClassifierApplier)
@node_meta(
    id="random_forest_classifier",
    name="Random Forest Classifier",
    category="Modeling",
    description="Ensemble of decision trees.",
    params={"n_estimators": 50, "max_depth": 10, "min_samples_split": 5}
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

    pass


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

    pass


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

    pass


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

    pass


@NodeRegistry.register(
    "gradient_boosting_classifier", GradientBoostingClassifierApplier
)
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

    pass


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
class XGBClassifierApplier(SklearnApplier):
    """XGBoost Classifier Applier."""

    pass


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
                "n_jobs": -1,
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            },
            problem_type="classification",
        )


# --- Gaussian NB ---
class GaussianNBApplier(SklearnApplier):
    """Gaussian Naive Bayes Applier."""

    pass


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
