"""Hyperparameter definitions for Ridge Regression models."""

from .base import HyperparameterField

RIDGE_REGRESSION_PARAMS = [
    HyperparameterField(
        name="alpha",
        label="Alpha (Regularization)",
        type="number",
        default=1.0,
        description="Regularization strength (higher = more regularization)",
        min=0.001,
        max=100.0,
        step=0.1,
    ),
    HyperparameterField(
        name="solver",
        label="Solver",
        type="select",
        default="auto",
        description="Algorithm for optimization",
        options=[
            {"value": "auto", "label": "Auto"},
            {"value": "svd", "label": "SVD"},
            {"value": "cholesky", "label": "Cholesky"},
            {"value": "lsqr", "label": "LSQR"},
            {"value": "sparse_cg", "label": "Sparse CG"},
            {"value": "sag", "label": "SAG"},
            {"value": "saga", "label": "SAGA"},
        ],
    ),
    HyperparameterField(
        name="max_iter",
        label="Max Iterations",
        type="number",
        default=None,
        description="Maximum iterations for solver (empty = auto)",
        min=100,
        max=10000,
        step=100,
        nullable=True,
    ),
    HyperparameterField(
        name="tol",
        label="Tolerance",
        type="number",
        default=0.001,
        description="Precision for stopping criteria",
        min=0.0001,
        max=0.1,
        step=0.0001,
    ),
]
