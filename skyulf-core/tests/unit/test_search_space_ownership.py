"""Default and nested ensemble search spaces belong to each requesting caller."""

from copy import deepcopy

import pytest

from skyulf.modeling.hyperparameters import (
    _registry,
    build_ensemble_search_space,
    get_default_search_space,
)

STRATEGIES = ["grid", "halving_grid", "random", "halving_random", "optuna"]


@pytest.fixture(autouse=True)
def isolate_registry_for_mutation_checks(monkeypatch):
    """Failing ownership checks must not contaminate unrelated model-space tests."""
    monkeypatch.setattr(
        _registry, "DEFAULT_SEARCH_SPACES", deepcopy(_registry.DEFAULT_SEARCH_SPACES)
    )
    monkeypatch.setattr(_registry, "GRID_SEARCH_SPACES", deepcopy(_registry.GRID_SEARCH_SPACES))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("model_key", ["random_forest_classifier", "voting_regressor", "unknown"])
def test_editing_one_search_space_does_not_change_later_requests(model_key, strategy):
    """Customizing candidates locally must not change another run's defaults."""
    space = get_default_search_space(model_key, strategy)
    expected = deepcopy(space)
    if "n_estimators" in space:
        space["n_estimators"].append(999)
    space["caller_only"] = [1]

    assert get_default_search_space(model_key, strategy) == expected


@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_grid_fallback_search_space_is_independent(monkeypatch, strategy):
    """The default-space fallback must be detached just like an explicit grid entry."""
    monkeypatch.delitem(_registry.GRID_SEARCH_SPACES, "random_forest_classifier")
    space = get_default_search_space("random_forest_classifier", strategy)
    space["n_estimators"].clear()

    assert get_default_search_space("random_forest_classifier", strategy)["n_estimators"] == [
        50,
        100,
        200,
        500,
    ]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    ("ensemble_key", "problem_type"),
    [
        ("voting_classifier", "classification"),
        ("voting_regressor", "regression"),
        ("stacking_classifier", "classification"),
        ("stacking_regressor", "regression"),
    ],
)
def test_ensemble_candidates_do_not_mutate_registry_or_next_ensemble(
    ensemble_key, problem_type, strategy
):
    """Both meta-parameter and base-learner lists must be owned by one ensemble request."""
    space = build_ensemble_search_space(
        ensemble_key, ["random_forest"], strategy=strategy, problem_type=problem_type
    )
    expected = deepcopy(space)
    for candidates in space.values():
        candidates.clear()

    assert (
        build_ensemble_search_space(
            ensemble_key, ["random_forest"], strategy=strategy, problem_type=problem_type
        )
        == expected
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("problem_type", ["classification", "regression"])
@pytest.mark.parametrize("calibrate", [False, True])
def test_same_base_and_final_learner_have_independent_candidates(strategy, problem_type, calibrate):
    """Changing a base learner's candidates must not silently retune the stacking head."""
    ensemble_key = (
        "stacking_classifier" if problem_type == "classification" else "stacking_regressor"
    )
    space = build_ensemble_search_space(
        ensemble_key,
        ["random_forest"],
        final_estimator="random_forest",
        strategy=strategy,
        problem_type=problem_type,
        calibrate_base_models=calibrate,
    )
    infix = "estimator__" if calibrate and problem_type == "classification" else ""
    expected_final = list(space["final_estimator__n_estimators"])
    space[f"random_forest__{infix}n_estimators"].clear()

    assert space["final_estimator__n_estimators"] == expected_final
