"""Contract tests for `random_state` exposure in the hyperparameter registry.

Every model whose estimator carries randomness must expose a seed control to
the UI (basic-mode fixed params), but the seed must never be offered as a
search-space candidate — tuning a seed is never meaningful. The seeding
default comes from the single owner `DEFAULT_RANDOM_STATE` (finding F-21).
"""

import pytest

from skyulf.modeling.hyperparameters import (
    MODEL_HYPERPARAMETERS,
    get_default_search_space,
    get_hyperparameters,
)
from skyulf.types import DEFAULT_RANDOM_STATE

SEEDED_MODELS = [
    "random_forest_classifier",
    "random_forest_regressor",
    "decision_tree_classifier",
    "decision_tree_regressor",
    "extra_trees_classifier",
    "extra_trees_regressor",
    "gradient_boosting_classifier",
    "gradient_boosting_regressor",
    "adaboost_classifier",
    "adaboost_regressor",
    "hist_gradient_boosting_classifier",
    "hist_gradient_boosting_regressor",
    "xgboost_classifier",
    "xgboost_regressor",
    "lgbm_classifier",
    "lgbm_regressor",
    "logistic_regression",
    "sgd_classifier",
    "ridge_regression",
    "lasso_regression",
    "elasticnet_regression",
    "calibrated_classifier",
    "kmeans",
    "minibatch_kmeans",
    "gaussian_mixture",
]

# Estimators without a `random_state` parameter must not expose a seed control.
UNSEEDED_MODELS = [
    "linear_regression",
    "svc",
    "svr",
    "k_neighbors_classifier",
    "k_neighbors_regressor",
    "gaussian_nb",
    "multinomial_nb",
    "bernoulli_nb",
    "birch",
    "voting_classifier",
    "stacking_classifier",
    "voting_regressor",
    "stacking_regressor",
]


def _seed_field(model_key: str) -> dict | None:
    fields = [p for p in get_hyperparameters(model_key) if p["name"] == "random_state"]
    return fields[0] if fields else None


@pytest.mark.parametrize("key", SEEDED_MODELS)
def test_seeded_models_expose_non_tunable_seed_field(key: str) -> None:
    field = _seed_field(key)
    assert field is not None, f"{key} should expose a random_state control"
    assert field["default"] == DEFAULT_RANDOM_STATE
    assert field["tunable"] is False
    assert field["type"] == "number"


@pytest.mark.parametrize("key", UNSEEDED_MODELS)
def test_unseeded_models_expose_no_seed_field(key: str) -> None:
    assert _seed_field(key) is None, f"{key} has no randomness; no seed control expected"


@pytest.mark.parametrize("strategy", ["random", "grid", "halving_grid", "optuna"])
def test_seed_never_lands_in_default_search_spaces(strategy: str) -> None:
    """A seed is never a sensible tuning target — no default search space may
    propose one, regardless of strategy."""
    for key in MODEL_HYPERPARAMETERS:
        space = get_default_search_space(key, strategy=strategy)
        assert "random_state" not in space, f"{key} ({strategy}) proposes a seed"


def test_seed_field_payload_shape() -> None:
    """The API payload the frontend renders must carry the tunable flag."""
    field = _seed_field("random_forest_classifier")
    assert field is not None
    assert set(field) >= {"name", "label", "type", "default", "min", "max", "step", "tunable"}
