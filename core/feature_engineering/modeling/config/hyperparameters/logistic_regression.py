"""Hyperparameter definitions for Logistic Regression models."""

from .base import HyperparameterField

LOGISTIC_REGRESSION_PARAMS = [
    HyperparameterField(
        name="max_iter",
        label="Max Iterations",
        type="number",
        default=1000,
        description="Maximum number of iterations for solver convergence",
        min=100,
        max=10000,
        step=100,
    ),
    HyperparameterField(
        name="C",
        label="Regularization Strength (C)",
        type="number",
        default=1.0,
        description="Inverse of regularization strength (smaller = stronger)",
        min=0.001,
        max=100.0,
        step=0.1,
    ),
    HyperparameterField(
        name="solver",
        label="Solver",
        type="select",
        default="lbfgs",
        description=(
            "Algorithm for optimization; standardize features when using SAG or SAGA to help convergence"
        ),
        options=[
            {"value": "lbfgs", "label": "LBFGS"},
            {"value": "liblinear", "label": "Liblinear"},
            {"value": "newton-cg", "label": "Newton-CG"},
            {"value": "sag", "label": "SAG"},
            {"value": "saga", "label": "SAGA"},
        ],
    ),
    HyperparameterField(
        name="penalty",
        label="Penalty",
        type="select",
        default="l2",
        description="Regularization penalty",
        options=[
            {"value": "l1", "label": "L1 (Lasso)"},
            {"value": "l2", "label": "L2 (Ridge)"},
            {"value": "elasticnet", "label": "Elastic Net"},
            {"value": "none", "label": "None"},
        ],
    ),
    HyperparameterField(
        name="multi_class",
        label="Multi-class Strategy",
        type="select",
        default="auto",
        description=(
            "Strategy for multi-class classification; leave at the estimator default (auto) unless a specific scheme is required"
        ),
        options=[
            {"value": "auto", "label": "Estimator default (auto)"},
            {"value": "ovr", "label": "One-vs-Rest"},
            {"value": "multinomial", "label": "Multinomial"},
        ],
        nullable=True,
    ),
]
