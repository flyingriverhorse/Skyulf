"""Linear models: LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet."""

from ._field import HyperparameterField

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
        description=(
            "The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. "
            "Only used if penalty='elasticnet'."
        ),
    ),
]

# --- Linear Regression ---
LINEAR_REGRESSION_PARAMS = [
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

# --- Lasso Regression ---
LASSO_REGRESSION_PARAMS = [
    HyperparameterField(
        name="alpha",
        label="Alpha",
        type="number",
        default=1.0,
        min=0.0001,
        max=100.0,
        description="Constant that multiplies the L1 term.",
    ),
    HyperparameterField(
        name="selection",
        label="Selection",
        type="select",
        default="cyclic",
        options=[{"label": "Cyclic", "value": "cyclic"}, {"label": "Random", "value": "random"}],
        description="If set to 'random', a random coefficient is updated every iteration.",
    ),
]

# --- ElasticNet Regression ---
ELASTICNET_REGRESSION_PARAMS = [
    HyperparameterField(
        name="alpha",
        label="Alpha",
        type="number",
        default=1.0,
        min=0.0001,
        max=100.0,
        description="Constant that multiplies the penalty terms.",
    ),
    HyperparameterField(
        name="l1_ratio",
        label="L1 Ratio",
        type="number",
        default=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        description="The ElasticNet mixing parameter (0 <= l1_ratio <= 1).",
    ),
    HyperparameterField(
        name="selection",
        label="Selection",
        type="select",
        default="cyclic",
        options=[{"label": "Cyclic", "value": "cyclic"}, {"label": "Random", "value": "random"}],
        description="If set to 'random', a random coefficient is updated every iteration.",
    ),
]
