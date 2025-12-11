from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .sklearn_wrapper import SklearnCalculator, SklearnApplier

# --- Logistic Regression ---
class LogisticRegressionCalculator(SklearnCalculator):
    def __init__(self):
        super().__init__(
            model_class=LogisticRegression,
            default_params={
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42
            },
            problem_type="classification"
        )

class LogisticRegressionApplier(SklearnApplier):
    pass


# --- Random Forest Classifier ---
class RandomForestClassifierCalculator(SklearnCalculator):
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
            problem_type="classification"
        )

class RandomForestClassifierApplier(SklearnApplier):
    pass
