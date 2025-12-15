"""Hyperparameter definitions for V2 models."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HyperparameterField:
    """Describe a single tunable hyperparameter."""

    name: str
    label: str
    type: str  # "number", "select", "boolean"
    default: Any
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = (
        None  # For 'select' type: [{"label": "L1", "value": "l1"}]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- Logistic Regression ---
LOGISTIC_REGRESSION_PARAMS = [
    HyperparameterField(
        name="C",
        label="Inverse Regularization Strength (C)",
        type="number",
        default=1.0,
        min=0.0001,
        max=100.0,
        description="Smaller values specify stronger regularization.",
    ),
    HyperparameterField(
        name="penalty",
        label="Penalty",
        type="select",
        default="l2",
        options=[
            {"label": "L1", "value": "l1"},
            {"label": "L2", "value": "l2"},
            {"label": "ElasticNet", "value": "elasticnet"},
            {"label": "None", "value": None},
        ],
        description="Norm used in the penalization.",
    ),
    HyperparameterField(
        name="solver",
        label="Solver",
        type="select",
        default="lbfgs",
        options=[
            {"label": "LBFGS", "value": "lbfgs"},
            {"label": "Liblinear", "value": "liblinear"},
            {"label": "Newton-CG", "value": "newton-cg"},
            {"label": "SAG", "value": "sag"},
            {"label": "SAGA", "value": "saga"},
        ],
        description="Algorithm to use in the optimization problem.",
    ),
    HyperparameterField(
        name="max_iter",
        label="Max Iterations",
        type="number",
        default=100,
        min=10,
        max=10000,
        step=10,
        description="Maximum number of iterations taken for the solvers to converge.",
    ),
    HyperparameterField(
        name="l1_ratio",
        label="L1 Ratio",
        type="number",
        default=None,
        min=0.0,
        max=1.0,
        step=0.1,
        description="The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'.",
    ),
]

# --- Random Forest (Classifier & Regressor) ---
RANDOM_FOREST_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="The number of trees in the forest.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description="The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.",
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="The minimum number of samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="The minimum number of samples required to be at a leaf node.",
    ),
    HyperparameterField(
        name="bootstrap",
        label="Bootstrap",
        type="select",
        default=True,
        options=[
            {"label": "True", "value": True},
            {"label": "False", "value": False},
        ],
        description="Whether bootstrap samples are used when building trees.",
    ),
]

# Add criterion for Classifier only
RANDOM_FOREST_CLASSIFIER_PARAMS = RANDOM_FOREST_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="gini",
        options=[
            {"label": "Gini", "value": "gini"},
            {"label": "Entropy", "value": "entropy"},
            {"label": "Log Loss", "value": "log_loss"},
        ],
        description="The function to measure the quality of a split.",
    )
]

# --- Ridge Regression ---
RIDGE_REGRESSION_PARAMS = [
    HyperparameterField(
        name="alpha",
        label="Alpha",
        type="number",
        default=1.0,
        min=0.0,
        max=100.0,
        description="Regularization strength; must be a positive float.",
    ),
    HyperparameterField(
        name="solver",
        label="Solver",
        type="select",
        default="auto",
        options=[
            {"label": "Auto", "value": "auto"},
            {"label": "SVD", "value": "svd"},
            {"label": "Cholesky", "value": "cholesky"},
            {"label": "LSQR", "value": "lsqr"},
            {"label": "Sparse CG", "value": "sparse_cg"},
            {"label": "SAG", "value": "sag"},
            {"label": "SAGA", "value": "saga"},
        ],
        description="Solver to use in the computational routines.",
    ),
    HyperparameterField(
        name="fit_intercept",
        label="Fit Intercept",
        type="select",
        default=True,
        options=[
            {"label": "True", "value": True},
            {"label": "False", "value": False},
        ],
        description="Whether to calculate the intercept for this model.",
    ),
]

MODEL_HYPERPARAMETERS = {
    "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
    "random_forest_classifier": RANDOM_FOREST_CLASSIFIER_PARAMS,
    "random_forest_regressor": RANDOM_FOREST_PARAMS,
    "ridge_regression": RIDGE_REGRESSION_PARAMS,
}


def get_hyperparameters(model_key: str) -> List[Dict[str, Any]]:
    params = MODEL_HYPERPARAMETERS.get(model_key, [])
    return [p.to_dict() for p in params]


# --- Default Search Spaces ---
# These are used to populate the UI for Hyperparameter Tuning

DEFAULT_SEARCH_SPACES = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["saga"],
        "max_iter": [100, 200, 500, 1000],
        "l1_ratio": [0.1, 0.5, 0.7, 0.9],  # Only used for elasticnet
    },
    "random_forest_classifier": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy", "log_loss"],
        "bootstrap": [True, False],
    },
    "random_forest_regressor": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "bootstrap": [True, False],
    },
    "ridge_regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "fit_intercept": [True, False],
    },
}


def get_default_search_space(model_key: str) -> Dict[str, Any]:
    return DEFAULT_SEARCH_SPACES.get(model_key, {})
