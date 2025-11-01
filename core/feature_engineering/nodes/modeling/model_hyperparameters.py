"""Hyperparameter configurations for machine learning models."""

from typing import Dict, List, Any, Optional


class HyperparameterField:
    """Configuration for a single hyperparameter field."""
    
    def __init__(
        self,
        name: str,
        label: str,
        type: str,  # "number", "select", "boolean", "text"
        default: Any,
        description: str = "",
        min: Optional[float] = None,
        max: Optional[float] = None,
        step: Optional[float] = None,
        options: Optional[List[Dict[str, Any]]] = None,
        nullable: bool = False,
    ):
        self.name = name
        self.label = label
        self.type = type
        self.default = default
        self.description = description
        self.min = min
        self.max = max
        self.step = step
        self.options = options or []
        self.nullable = nullable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "description": self.description,
        }
        if self.min is not None:
            result["min"] = self.min
        if self.max is not None:
            result["max"] = self.max
        if self.step is not None:
            result["step"] = self.step
        if self.options:
            result["options"] = self.options
        if self.nullable:
            result["nullable"] = self.nullable
        return result


# Logistic Regression Hyperparameters
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
        description="Algorithm for optimization",
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
        description="Strategy for multi-class classification",
        options=[
            {"value": "auto", "label": "Auto"},
            {"value": "ovr", "label": "One-vs-Rest"},
            {"value": "multinomial", "label": "Multinomial"},
        ],
    ),
]

# Random Forest Classifier Hyperparameters
RANDOM_FOREST_CLASSIFIER_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        description="Number of trees in the forest",
        min=10,
        max=1000,
        step=10,
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        description="Maximum depth of trees (empty = no limit)",
        min=1,
        max=100,
        step=1,
        nullable=True,
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        description="Minimum samples required to split a node",
        min=2,
        max=20,
        step=1,
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        description="Minimum samples required at a leaf node",
        min=1,
        max=20,
        step=1,
    ),
    HyperparameterField(
        name="max_features",
        label="Max Features",
        type="select",
        default="sqrt",
        description="Number of features to consider for best split",
        options=[
            {"value": "sqrt", "label": "Square Root"},
            {"value": "log2", "label": "Log2"},
            {"value": "None", "label": "All Features"},
        ],
    ),
    HyperparameterField(
        name="random_state",
        label="Random State",
        type="number",
        default=42,
        description="Seed for reproducibility",
        min=0,
        max=9999,
        step=1,
    ),
]

# Random Forest Regressor Hyperparameters
RANDOM_FOREST_REGRESSOR_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        description="Number of trees in the forest",
        min=10,
        max=1000,
        step=10,
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        description="Maximum depth of trees (empty = no limit)",
        min=1,
        max=100,
        step=1,
        nullable=True,
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        description="Minimum samples required to split a node",
        min=2,
        max=20,
        step=1,
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        description="Minimum samples required at a leaf node",
        min=1,
        max=20,
        step=1,
    ),
    HyperparameterField(
        name="max_features",
        label="Max Features",
        type="select",
        default="sqrt",
        description="Number of features to consider for best split",
        options=[
            {"value": "sqrt", "label": "Square Root"},
            {"value": "log2", "label": "Log2"},
            {"value": "None", "label": "All Features"},
        ],
    ),
    HyperparameterField(
        name="random_state",
        label="Random State",
        type="number",
        default=42,
        description="Seed for reproducibility",
        min=0,
        max=9999,
        step=1,
    ),
]

# Ridge Regression Hyperparameters
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


# Registry mapping model types to their hyperparameter configurations
MODEL_HYPERPARAMETERS: Dict[str, List[HyperparameterField]] = {
    "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
    "random_forest_classifier": RANDOM_FOREST_CLASSIFIER_PARAMS,
    "random_forest_regressor": RANDOM_FOREST_REGRESSOR_PARAMS,
    "ridge_regression": RIDGE_REGRESSION_PARAMS,
}


def get_hyperparameters_for_model(model_type: str) -> List[Dict[str, Any]]:
    """Get hyperparameter configuration for a specific model type."""
    fields = MODEL_HYPERPARAMETERS.get(model_type, [])
    return [field.to_dict() for field in fields]


def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    """Get default hyperparameter values for a specific model type."""
    fields = MODEL_HYPERPARAMETERS.get(model_type, [])
    return {field.name: field.default for field in fields}
