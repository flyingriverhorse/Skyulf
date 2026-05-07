"""Hyperparameter definitions for V2 models.

Public API (preserved for backward compatibility with the previous
single-file ``hyperparameters.py``):

* :class:`HyperparameterField` — dataclass describing one tunable param.
* :data:`MODEL_HYPERPARAMETERS` — ``{model_key: [HyperparameterField, ...]}``.
* :data:`DEFAULT_SEARCH_SPACES` — random/optuna/halving_random spaces.
* :data:`GRID_SEARCH_SPACES` — trimmed grid/halving_grid spaces.
* :func:`get_hyperparameters(model_key)` — returns dict-form of the params.
* :func:`get_default_search_space(model_key, strategy)` — picks the right
  search-space dict based on the tuning strategy.

Per-family ``*_PARAMS`` constants are also re-exported so existing imports
(e.g. ``from skyulf.modeling.hyperparameters import LOGISTIC_REGRESSION_PARAMS``)
keep working.
"""

from ._bayes import GAUSSIAN_NB_PARAMS
from ._field import HyperparameterField
from ._linear import (
    ELASTICNET_REGRESSION_PARAMS,
    LASSO_REGRESSION_PARAMS,
    LINEAR_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    RIDGE_REGRESSION_PARAMS,
)
from ._neighbors import KNN_PARAMS
from ._registry import (
    DEFAULT_SEARCH_SPACES,
    GRID_SEARCH_SPACES,
    MODEL_HYPERPARAMETERS,
    get_default_search_space,
    get_hyperparameters,
)
from ._svm import SVM_PARAMS
from ._tree import (
    ADABOOST_PARAMS,
    DECISION_TREE_CLASSIFIER_PARAMS,
    DECISION_TREE_PARAMS,
    DECISION_TREE_REGRESSOR_PARAMS,
    EXTRA_TREES_CLASSIFIER_PARAMS,
    EXTRA_TREES_PARAMS,
    EXTRA_TREES_REGRESSOR_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    HIST_GRADIENT_BOOSTING_PARAMS,
    LGBM_PARAMS,
    RANDOM_FOREST_CLASSIFIER_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)

__all__ = [
    # Core
    "HyperparameterField",
    "MODEL_HYPERPARAMETERS",
    "DEFAULT_SEARCH_SPACES",
    "GRID_SEARCH_SPACES",
    "get_hyperparameters",
    "get_default_search_space",
    # Linear
    "LOGISTIC_REGRESSION_PARAMS",
    "LINEAR_REGRESSION_PARAMS",
    "RIDGE_REGRESSION_PARAMS",
    "LASSO_REGRESSION_PARAMS",
    "ELASTICNET_REGRESSION_PARAMS",
    # Tree-based
    "RANDOM_FOREST_PARAMS",
    "RANDOM_FOREST_CLASSIFIER_PARAMS",
    "DECISION_TREE_PARAMS",
    "DECISION_TREE_CLASSIFIER_PARAMS",
    "DECISION_TREE_REGRESSOR_PARAMS",
    "GRADIENT_BOOSTING_PARAMS",
    "ADABOOST_PARAMS",
    "XGBOOST_PARAMS",
    "EXTRA_TREES_PARAMS",
    "EXTRA_TREES_CLASSIFIER_PARAMS",
    "EXTRA_TREES_REGRESSOR_PARAMS",
    "HIST_GRADIENT_BOOSTING_PARAMS",
    "LGBM_PARAMS",
    # Other families
    "SVM_PARAMS",
    "KNN_PARAMS",
    "GAUSSIAN_NB_PARAMS",
]
