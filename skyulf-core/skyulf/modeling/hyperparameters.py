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

# --- SVM (SVC & SVR) ---
SVM_PARAMS = [
    HyperparameterField(
        name="C",
        label="C (Regularization)",
        type="number",
        default=1.0,
        min=0.01,
        max=1000.0,
        description="Regularization parameter. The strength of the regularization is inversely proportional to C.",
    ),
    HyperparameterField(
        name="kernel",
        label="Kernel",
        type="select",
        default="rbf",
        options=[
            {"label": "Linear", "value": "linear"},
            {"label": "Poly", "value": "poly"},
            {"label": "RBF", "value": "rbf"},
            {"label": "Sigmoid", "value": "sigmoid"},
        ],
        description="Specifies the kernel type to be used in the algorithm.",
    ),
    HyperparameterField(
        name="gamma",
        label="Gamma",
        type="select",
        default="scale",
        options=[{"label": "Scale", "value": "scale"}, {"label": "Auto", "value": "auto"}],
        description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.",
    ),
]

# --- K-Neighbors ---
KNN_PARAMS = [
    HyperparameterField(
        name="n_neighbors",
        label="Number of Neighbors",
        type="number",
        default=5,
        min=1,
        max=50,
        step=1,
        description="Number of neighbors to use.",
    ),
    HyperparameterField(
        name="weights",
        label="Weights",
        type="select",
        default="uniform",
        options=[
            {"label": "Uniform", "value": "uniform"},
            {"label": "Distance", "value": "distance"},
        ],
        description="Weight function used in prediction.",
    ),
    HyperparameterField(
        name="algorithm",
        label="Algorithm",
        type="select",
        default="auto",
        options=[
            {"label": "Auto", "value": "auto"},
            {"label": "Ball Tree", "value": "ball_tree"},
            {"label": "KD Tree", "value": "kd_tree"},
            {"label": "Brute", "value": "brute"},
        ],
        description="Algorithm used to compute the nearest neighbors.",
    ),
]

# --- Decision Tree ---
DECISION_TREE_PARAMS = [
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description="The maximum depth of the tree.",
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
]
DECISION_TREE_CLASSIFIER_PARAMS = DECISION_TREE_PARAMS + [
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
DECISION_TREE_REGRESSOR_PARAMS = DECISION_TREE_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="squared_error",
        options=[
            {"label": "Squared Error", "value": "squared_error"},
            {"label": "Friedman MSE", "value": "friedman_mse"},
            {"label": "Absolute Error", "value": "absolute_error"},
            {"label": "Poisson", "value": "poisson"},
        ],
        description="The function to measure the quality of a split.",
    )
]

# --- Gradient Boosting (Sklearn) ---
GRADIENT_BOOSTING_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="The number of boosting stages to perform.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.1,
        min=0.001,
        max=1.0,
        description="Shrinks the contribution of each tree by learning_rate.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=3,
        min=1,
        max=20,
        description="Maximum depth of the individual regression estimators.",
    ),
    HyperparameterField(
        name="subsample",
        label="Subsample",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="The fraction of samples to be used for fitting the individual base learners.",
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="Minimum samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="Minimum samples required at a leaf node.",
    ),
]

# --- AdaBoost ---
ADABOOST_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Estimators",
        type="number",
        default=50,
        min=10,
        max=1000,
        step=10,
        description="The maximum number of estimators at which boosting is terminated.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=1.0,
        min=0.001,
        max=5.0,
        description="Weight applied to each classifier at each boosting iteration.",
    ),
]

# --- XGBoost ---
XGBOOST_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Estimators",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="Number of gradient boosted trees.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=6,
        min=1,
        max=20,
        description="Maximum tree depth for base learners.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.3,
        min=0.001,
        max=1.0,
        description="Boosting learning rate (eta).",
    ),
    HyperparameterField(
        name="subsample",
        label="Subsample",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Subsample ratio of the training instances.",
    ),
    HyperparameterField(
        name="colsample_bytree",
        label="Colsample By Tree",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Subsample ratio of columns when constructing each tree.",
    ),
    HyperparameterField(
        name="min_child_weight",
        label="Min Child Weight",
        type="number",
        default=1,
        min=0,
        max=50,
        step=1,
        description="Minimum sum of instance weights in a child. Higher values regularize against overfitting.",
    ),
    HyperparameterField(
        name="gamma",
        label="Gamma (min_split_loss)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="Minimum loss reduction required to make a split. Higher = more conservative tree growth.",
    ),
    HyperparameterField(
        name="reg_alpha",
        label="L1 Regularization (reg_alpha)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L1 regularization term on leaf weights.",
    ),
    HyperparameterField(
        name="reg_lambda",
        label="L2 Regularization (reg_lambda)",
        type="number",
        default=1.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L2 regularization term on leaf weights. Default sklearn XGBoost is 1.",
    ),
]

# --- Extra Trees (Classifier & Regressor) ---
EXTRA_TREES_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="Number of trees in the forest.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=100,
        description="Maximum depth of the tree. None = unlimited.",
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        min=2,
        max=20,
        description="Minimum samples required to split an internal node.",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        min=1,
        max=20,
        description="Minimum samples required at a leaf node.",
    ),
    HyperparameterField(
        name="bootstrap",
        label="Bootstrap",
        type="select",
        default=False,
        options=[
            {"label": "False (default)", "value": False},
            {"label": "True", "value": True},
        ],
        description="Whether bootstrap samples are used (False = use full dataset).",
    ),
]

EXTRA_TREES_CLASSIFIER_PARAMS = EXTRA_TREES_PARAMS + [
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
        description="Function to measure the quality of a split.",
    )
]

EXTRA_TREES_REGRESSOR_PARAMS = EXTRA_TREES_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="squared_error",
        options=[
            {"label": "Squared Error", "value": "squared_error"},
            {"label": "Absolute Error", "value": "absolute_error"},
            {"label": "Friedman MSE", "value": "friedman_mse"},
        ],
        description="Function to measure the quality of a split.",
    )
]

# --- HistGradientBoosting (Classifier & Regressor) ---
HIST_GRADIENT_BOOSTING_PARAMS = [
    HyperparameterField(
        name="max_iter",
        label="Max Iterations (Trees)",
        type="number",
        default=100,
        min=10,
        max=1000,
        step=10,
        description="Maximum number of iterations (boosting rounds).",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.1,
        min=0.001,
        max=1.0,
        description="Shrinks the contribution of each tree.",
    ),
    HyperparameterField(
        name="max_leaf_nodes",
        label="Max Leaf Nodes",
        type="number",
        default=31,
        min=2,
        max=255,
        description="Maximum number of leaves per tree.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        min=1,
        max=20,
        description="Maximum depth per tree. None = unlimited (max_leaf_nodes is the effective limit).",
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=20,
        min=1,
        max=200,
        description="Minimum samples at a leaf node.",
    ),
    HyperparameterField(
        name="l2_regularization",
        label="L2 Regularization",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L2 penalty applied to the leaves' values.",
    ),
    HyperparameterField(
        name="max_bins",
        label="Max Bins",
        type="number",
        default=255,
        min=10,
        max=255,
        step=5,
        description="Maximum number of bins for feature discretisation. Higher = more precise but slower.",
    ),
]

# --- LightGBM (Classifier & Regressor) ---
LGBM_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Estimators",
        type="number",
        default=100,
        min=10,
        max=2000,
        step=10,
        description="Number of boosting rounds.",
    ),
    HyperparameterField(
        name="num_leaves",
        label="Num Leaves",
        type="number",
        default=31,
        min=2,
        max=512,
        step=1,
        description="Maximum number of leaves per tree. Controls model complexity; increase for more accuracy.",
    ),
    HyperparameterField(
        name="learning_rate",
        label="Learning Rate",
        type="number",
        default=0.1,
        min=0.001,
        max=1.0,
        description="Shrinks the contribution of each tree.",
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=-1,
        min=-1,
        max=50,
        description="Maximum tree depth. -1 = unlimited (controlled by num_leaves).",
    ),
    HyperparameterField(
        name="min_child_samples",
        label="Min Child Samples",
        type="number",
        default=20,
        min=1,
        max=200,
        description="Minimum samples required in a leaf. Regularises against overfitting.",
    ),
    HyperparameterField(
        name="subsample",
        label="Subsample (Bagging Fraction)",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Fraction of training data sampled per iteration.",
    ),
    HyperparameterField(
        name="colsample_bytree",
        label="Colsample By Tree (Feature Fraction)",
        type="number",
        default=1.0,
        min=0.1,
        max=1.0,
        step=0.1,
        description="Fraction of features sampled per tree.",
    ),
    HyperparameterField(
        name="reg_alpha",
        label="L1 Regularization (reg_alpha)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L1 regularization term on leaf weights.",
    ),
    HyperparameterField(
        name="reg_lambda",
        label="L2 Regularization (reg_lambda)",
        type="number",
        default=0.0,
        min=0.0,
        max=10.0,
        step=0.1,
        description="L2 regularization term on leaf weights.",
    ),
    HyperparameterField(
        name="boosting_type",
        label="Boosting Type",
        type="select",
        default="gbdt",
        options=[
            {"label": "GBDT (default)", "value": "gbdt"},
            {"label": "DART (dropout)", "value": "dart"},
            {"label": "GOSS (gradient-based sampling)", "value": "goss"},
        ],
        description="GBDT is standard gradient boosting. DART adds dropout. GOSS samples by gradient magnitude.",
    ),
]

# --- Gaussian Naive Bayes ---
GAUSSIAN_NB_PARAMS = [
    HyperparameterField(
        name="var_smoothing",
        label="Var Smoothing",
        type="number",
        default=1e-9,
        min=1e-12,
        max=1.0,
        description=(
            "Portion of the largest variance of all features that is added "
            "to variances for calculation stability."
        ),
    ),
]

MODEL_HYPERPARAMETERS = {
    "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
    "random_forest_classifier": RANDOM_FOREST_CLASSIFIER_PARAMS,
    "random_forest_regressor": RANDOM_FOREST_PARAMS,
    "ridge_regression": RIDGE_REGRESSION_PARAMS,
    "lasso_regression": LASSO_REGRESSION_PARAMS,
    "elasticnet_regression": ELASTICNET_REGRESSION_PARAMS,
    "linear_regression": LINEAR_REGRESSION_PARAMS,
    "svc": SVM_PARAMS,
    "svr": SVM_PARAMS,
    "k_neighbors_classifier": KNN_PARAMS,
    "k_neighbors_regressor": KNN_PARAMS,
    "decision_tree_classifier": DECISION_TREE_CLASSIFIER_PARAMS,
    "decision_tree_regressor": DECISION_TREE_REGRESSOR_PARAMS,
    "gradient_boosting_classifier": GRADIENT_BOOSTING_PARAMS,
    "gradient_boosting_regressor": GRADIENT_BOOSTING_PARAMS,
    "adaboost_classifier": ADABOOST_PARAMS,
    "adaboost_regressor": ADABOOST_PARAMS,
    "xgboost_classifier": XGBOOST_PARAMS,
    "xgboost_regressor": XGBOOST_PARAMS,
    "extra_trees_classifier": EXTRA_TREES_CLASSIFIER_PARAMS,
    "extra_trees_regressor": EXTRA_TREES_REGRESSOR_PARAMS,
    "hist_gradient_boosting_classifier": HIST_GRADIENT_BOOSTING_PARAMS,
    "hist_gradient_boosting_regressor": HIST_GRADIENT_BOOSTING_PARAMS,
    "lgbm_classifier": LGBM_PARAMS,
    "lgbm_regressor": LGBM_PARAMS,
    "gaussian_nb": GAUSSIAN_NB_PARAMS,
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
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "bootstrap": [True, False],
    },
    "ridge_regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "fit_intercept": [True, False],
    },
    "lasso_regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "elasticnet_regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.7, 0.9],
    },
    "linear_regression": {
        "fit_intercept": [True, False],
    },
    "svc": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
    },
    "svr": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
    },
    "k_neighbors_classifier": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "k_neighbors_regressor": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "decision_tree_classifier": {
        "max_depth": [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "decision_tree_regressor": {
        "max_depth": [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 5, 10, 20],
    },
    "gradient_boosting_regressor": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 5, 10, 20],
    },
    "adaboost_classifier": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.1, 1.0, 1.5],
    },
    "adaboost_regressor": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.1, 1.0, 1.5],
    },
    "xgboost_classifier": {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    },
    "xgboost_regressor": {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    },
    "gaussian_nb": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "extra_trees_classifier": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy", "log_loss"],
        "bootstrap": [False, True],
    },
    "extra_trees_regressor": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "absolute_error", "friedman_mse"],
        "bootstrap": [False, True],
    },
    "hist_gradient_boosting_classifier": {
        "max_iter": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63, 127],
        "max_depth": [None, 3, 5, 10],
        "min_samples_leaf": [10, 20, 50, 100],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0],
    },
    "hist_gradient_boosting_regressor": {
        "max_iter": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63, 127],
        "max_depth": [None, 3, 5, 10],
        "min_samples_leaf": [10, 20, 50, 100],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0],
    },
    "lgbm_classifier": {
        "n_estimators": [100, 200, 500, 1000],
        "num_leaves": [15, 31, 63, 127, 255],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 5, 10, 20],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.0, 0.01, 0.1, 1.0],
        "boosting_type": ["gbdt", "dart", "goss"],
    },
    "lgbm_regressor": {
        "n_estimators": [100, 200, 500, 1000],
        "num_leaves": [15, 31, 63, 127, 255],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 5, 10, 20],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.0, 0.01, 0.1, 1.0],
        "boosting_type": ["gbdt", "dart", "goss"],
    },
}


# ---------------------------------------------------------------------------
# Grid-safe search spaces (grid / halving_grid)
#
# These are intentionally trimmed to keep the cartesian product manageable.
# The full DEFAULT_SEARCH_SPACES above are designed for random/optuna/halving_random
# where only a subset of combinations is ever evaluated.
# ---------------------------------------------------------------------------
GRID_SEARCH_SPACES: Dict[str, Any] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
        "max_iter": [200, 500],
    },
    "random_forest_classifier": {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "criterion": ["gini", "entropy"],
    },
    "random_forest_regressor": {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "criterion": ["squared_error", "friedman_mse"],
    },
    "ridge_regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "saga"],
        "fit_intercept": [True, False],
    },
    "lasso_regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "elasticnet_regression": {
        "alpha": [0.01, 0.1, 1.0],
        "l1_ratio": [0.3, 0.5, 0.7],
    },
    "linear_regression": {
        "fit_intercept": [True, False],
    },
    "svc": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "svr": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "k_neighbors_classifier": {
        "n_neighbors": [3, 5, 9],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
    },
    "k_neighbors_regressor": {
        "n_neighbors": [3, 5, 9],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
    },
    "decision_tree_classifier": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "decision_tree_regressor": {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["squared_error", "friedman_mse"],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "min_samples_leaf": [1, 5],
    },
    "gradient_boosting_regressor": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "min_samples_leaf": [1, 5],
    },
    "adaboost_classifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    },
    "adaboost_regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    },
    "xgboost_classifier": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.1, 1.0, 5.0],
    },
    "xgboost_regressor": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.1, 1.0, 5.0],
    },
    "gaussian_nb": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "extra_trees_classifier": {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"],
    },
    "extra_trees_regressor": {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5],
        "criterion": ["squared_error", "friedman_mse"],
    },
    "hist_gradient_boosting_classifier": {
        "max_iter": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [20, 50],
    },
    "hist_gradient_boosting_regressor": {
        "max_iter": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [20, 50],
    },
    "lgbm_classifier": {
        "n_estimators": [100, 200, 500],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [-1, 5, 10],
        "subsample": [0.8, 1.0],
    },
    "lgbm_regressor": {
        "n_estimators": [100, 200, 500],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [-1, 5, 10],
        "subsample": [0.8, 1.0],
    },
}

_GRID_STRATEGIES = {"grid", "halving_grid"}


def get_default_search_space(model_key: str, strategy: str = "random") -> Dict[str, Any]:
    """Return the default search space for *model_key*.

    For grid-based strategies (``grid`` / ``halving_grid``) the trimmed
    ``GRID_SEARCH_SPACES`` dict is used so that the cartesian product stays
    manageable.  All other strategies (``random``, ``halving_random``,
    ``optuna``) use the richer ``DEFAULT_SEARCH_SPACES``.
    """
    if strategy in _GRID_STRATEGIES:
        return GRID_SEARCH_SPACES.get(model_key, DEFAULT_SEARCH_SPACES.get(model_key, {}))
    return DEFAULT_SEARCH_SPACES.get(model_key, {})
