"""Hyperparameter definitions for V2 models."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

@dataclass
class HyperparameterField:
    """Describe a single tunable hyperparameter."""
    name: str
    label: str
    type: str # "number", "select", "boolean"
    default: Any
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = None # For 'select' type: [{"label": "L1", "value": "l1"}]
    
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
        description="Smaller values specify stronger regularization."
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
        description="Algorithm to use in the optimization problem."
    ),
    HyperparameterField(
        name="max_iter",
        label="Max Iterations",
        type="number",
        default=100,
        min=10,
        max=10000,
        step=10,
        description="Maximum number of iterations taken for the solvers to converge."
    )
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
        description="The number of trees in the forest."
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description="The maximum depth of the tree. If None, nodes are expanded until all leaves are pure."
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="The minimum number of samples required to split an internal node."
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
        description="Regularization strength; must be a positive float."
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
        ],
        description="Solver to use in the computational routines."
    )
]

MODEL_HYPERPARAMETERS = {
    "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
    "random_forest_classifier": RANDOM_FOREST_PARAMS,
    "random_forest_regressor": RANDOM_FOREST_PARAMS,
    "ridge_regression": RIDGE_REGRESSION_PARAMS
}

def get_hyperparameters(model_key: str) -> List[Dict[str, Any]]:
    params = MODEL_HYPERPARAMETERS.get(model_key, [])
    return [p.to_dict() for p in params]
