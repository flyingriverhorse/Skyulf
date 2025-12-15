"""Classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .sklearn_wrapper import SklearnApplier, SklearnCalculator


# --- Logistic Regression ---
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


class LogisticRegressionApplier(SklearnApplier):
    """Logistic Regression Applier."""

    pass


# --- Random Forest Classifier ---
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


class RandomForestClassifierApplier(SklearnApplier):
    """Random Forest Classifier Applier."""

    pass
