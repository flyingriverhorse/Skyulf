"""Central registry: maps model_key → param list, plus search-space dicts."""

from typing import Any, Dict, List

from ._bayes import GAUSSIAN_NB_PARAMS
from ._calibration import CALIBRATED_CLASSIFIER_PARAMS
from ._ensemble import (
    STACKING_CLASSIFIER_PARAMS,
    STACKING_REGRESSOR_PARAMS,
    VOTING_CLASSIFIER_PARAMS,
    VOTING_REGRESSOR_PARAMS,
)
from ._linear import (
    ELASTICNET_REGRESSION_PARAMS,
    LASSO_REGRESSION_PARAMS,
    LINEAR_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    RIDGE_REGRESSION_PARAMS,
)
from ._neighbors import KNN_PARAMS
from ._svm import SVM_PARAMS
from ._tree import (
    ADABOOST_PARAMS,
    DECISION_TREE_CLASSIFIER_PARAMS,
    DECISION_TREE_REGRESSOR_PARAMS,
    EXTRA_TREES_CLASSIFIER_PARAMS,
    EXTRA_TREES_REGRESSOR_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    HIST_GRADIENT_BOOSTING_PARAMS,
    LGBM_PARAMS,
    RANDOM_FOREST_CLASSIFIER_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)

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
    "calibrated_classifier": CALIBRATED_CLASSIFIER_PARAMS,
    "voting_classifier": VOTING_CLASSIFIER_PARAMS,
    "stacking_classifier": STACKING_CLASSIFIER_PARAMS,
    "voting_regressor": VOTING_REGRESSOR_PARAMS,
    "stacking_regressor": STACKING_REGRESSOR_PARAMS,
}


def get_hyperparameters(model_key: str) -> List[Dict[str, Any]]:
    params = MODEL_HYPERPARAMETERS.get(model_key, [])
    return [p.to_dict() for p in params]


# ---------------------------------------------------------------------------
# Default search spaces
# Used to populate the UI for Hyperparameter Tuning.
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_SPACES: Dict[str, Any] = {
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
    "calibrated_classifier": {
        "method": ["sigmoid", "isotonic"],
        "cv": [3, 5, 10],
    },
    "voting_classifier": {
        "voting": ["soft", "hard"],
    },
    "stacking_classifier": {
        "cv": [3, 5, 10],
    },
    "stacking_regressor": {
        "cv": [3, 5, 10],
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
# Trimmed to keep the cartesian product manageable. The full
# DEFAULT_SEARCH_SPACES above are designed for random / optuna /
# halving_random where only a subset of combinations is ever evaluated.
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
    "calibrated_classifier": {
        "method": ["sigmoid", "isotonic"],
        "cv": [3, 5],
    },
    "voting_classifier": {
        "voting": ["soft", "hard"],
    },
    "stacking_classifier": {
        "cv": [3, 5],
    },
    "stacking_regressor": {
        "cv": [3, 5],
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
    ``GRID_SEARCH_SPACES`` dict is used so the cartesian product stays
    manageable. All other strategies (``random``, ``halving_random``,
    ``optuna``) use the richer ``DEFAULT_SEARCH_SPACES``.
    """
    if strategy in _GRID_STRATEGIES:
        return GRID_SEARCH_SPACES.get(model_key, DEFAULT_SEARCH_SPACES.get(model_key, {}))
    return DEFAULT_SEARCH_SPACES.get(model_key, {})


# ---------------------------------------------------------------------------
# Ensemble nested search spaces
# Maps each selectable base-learner key (as used by ``skyulf.modeling.ensemble``)
# to its registry key so its parameter grid can be expanded into nested
# ``<name>__<param>`` keys understood by sklearn's Voting/Stacking estimators.
# ---------------------------------------------------------------------------
_BASE_KEY_TO_REGISTRY_CLF: Dict[str, str] = {
    "logistic_regression": "logistic_regression",
    "random_forest": "random_forest_classifier",
    "extra_trees": "extra_trees_classifier",
    "gradient_boosting": "gradient_boosting_classifier",
    "hist_gradient_boosting": "hist_gradient_boosting_classifier",
    "adaboost": "adaboost_classifier",
    "decision_tree": "decision_tree_classifier",
    "gaussian_nb": "gaussian_nb",
    "svc": "svc",
    "knn": "k_neighbors_classifier",
    "xgboost": "xgboost_classifier",
    "lightgbm": "lgbm_classifier",
}

_BASE_KEY_TO_REGISTRY_REG: Dict[str, str] = {
    "linear_regression": "linear_regression",
    "ridge": "ridge_regression",
    "lasso": "lasso_regression",
    "elasticnet": "elasticnet_regression",
    "random_forest": "random_forest_regressor",
    "extra_trees": "extra_trees_regressor",
    "gradient_boosting": "gradient_boosting_regressor",
    "hist_gradient_boosting": "hist_gradient_boosting_regressor",
    "adaboost": "adaboost_regressor",
    "decision_tree": "decision_tree_regressor",
    "svr": "svr",
    "knn": "k_neighbors_regressor",
    "xgboost": "xgboost_regressor",
    "lightgbm": "lgbm_regressor",
}


def build_ensemble_search_space(
    ensemble_key: str,
    base_estimators: List[str],
    final_estimator: str = "",
    strategy: str = "random",
    problem_type: str = "classification",
) -> Dict[str, Any]:
    """Expand an ensemble's base learners into a nested tuning search space.

    Combines the ensemble's own meta-params (``voting`` / ``cv``) with each
    chosen base learner's parameter grid, prefixed as ``<name>__<param>`` so
    sklearn's Voting/Stacking estimators route them to the right sub-model.
    The stacking ``final_estimator`` is prefixed as ``final_estimator__<param>``.
    Unknown keys are skipped.
    """
    space: Dict[str, Any] = dict(get_default_search_space(ensemble_key, strategy))
    mapping = (
        _BASE_KEY_TO_REGISTRY_CLF if problem_type == "classification" else _BASE_KEY_TO_REGISTRY_REG
    )
    for key in base_estimators or []:
        reg_key = mapping.get(key)
        if not reg_key:
            continue
        for param, values in get_default_search_space(reg_key, strategy).items():
            space[f"{key}__{param}"] = values
    if final_estimator:
        reg_key = mapping.get(final_estimator)
        if reg_key:
            for param, values in get_default_search_space(reg_key, strategy).items():
                space[f"final_estimator__{param}"] = values
    return space
