"""Pin Canvas advanced search preparation without silently fitting default models."""

import pytest

from backend.ml_pipeline._execution.engine._node_runners import NodeRunnersMixin
from backend.ml_pipeline._execution.schemas import NodeConfig
from skyulf.modeling.classification import RandomForestClassifierCalculator
from skyulf.modeling.ensemble import VotingClassifierCalculator


def training_node(**tuning_overrides):
    """Reproduce Basic 400/7 switched to Advanced with fifty requested trials."""
    return NodeConfig(
        node_id="model",
        step_type="training",
        params={
            "run_mode": "tuned",
            "algorithm": "random_forest_classifier",
            "target_column": "target",
            "tuning_config": {
                "strategy": "random",
                "n_trials": 50,
                "random_state": 0,
                "search_space": {},
                **tuning_overrides,
            },
        },
    )


def test_plain_advanced_empty_space_requires_explicit_search():
    """An empty advanced search must fail before fifty trials become one default fit."""
    with pytest.raises(ValueError, match="Configure at least one search parameter"):
        NodeRunnersMixin()._prepare_tuning_config(
            training_node(), RandomForestClassifierCalculator()
        )


def test_advanced_explicit_space_and_zero_seed_are_preserved():
    """Advanced search choices and seed zero must survive backend preparation."""
    prepared = NodeRunnersMixin()._prepare_tuning_config(
        training_node(search_space={"n_estimators": [400], "max_depth": [7]}),
        RandomForestClassifierCalculator(),
    )
    assert prepared["search_space"] == {"n_estimators": [400], "max_depth": [7]}
    assert prepared["random_state"] == 0


def test_ensemble_automatic_space_remains_supported():
    """Ensembles already resolve empty incoming spaces into real nested search choices."""
    prepared = NodeRunnersMixin()._prepare_tuning_config(
        training_node(
            base_estimators=["random_forest", "logistic_regression"],
            base_estimator_params={"random_forest": {"n_estimators": 400, "max_depth": 7}},
            tune_base_models=True,
        ),
        VotingClassifierCalculator(),
    )
    assert any(key.startswith("random_forest__") for key in prepared["search_space"])
    assert prepared["base_estimator_params"]["random_forest"]["n_estimators"] == 400
    assert prepared["random_state"] == 0
