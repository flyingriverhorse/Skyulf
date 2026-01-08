"""Classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
    params={"max_iter": 1000, "solver": "lbfgs", "random_state": 42}
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
