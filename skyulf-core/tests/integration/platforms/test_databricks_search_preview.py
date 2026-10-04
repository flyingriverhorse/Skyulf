"""Readable offline previews for bounded Core search and explanations."""

from copy import deepcopy

from skyulf.integrations.databricks.workflow_config import preview_workflow_config


def _search(workflow_config, **changes):
    """Select a registered ridge search without changing the source fixture."""
    model = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "ridge_regression", "params": {}},
        "strategy": "grid",
        "metric": "rmse",
        "n_trials": 10,
        "max_candidates": 1000,
        "search_space": {},
        "random_state": 17,
    }
    model.update(changes)
    workflow_config["pipeline"]["modeling"] = model
    return workflow_config


def test_preview_describes_auto_search_and_separate_holdout_objective(workflow_config):
    """Operators must see what is selected without mistaking promotion for tuning."""
    config = _search(workflow_config, strategy="grid")
    config.update(cv_enabled=True, cv_folds=3, cv_type="k_fold", cv_random_state=29)
    original = deepcopy(config)

    report = preview_workflow_config(config, action="train")

    assert "ridge_regression" in report
    assert "Core default search space" in report
    assert "grid" in report and "20 candidates" in report
    assert "max_candidates=1000" in report
    assert "rmse" in report and "heldout_rmse" in report
    assert "training partition only" in report
    assert "cv_random_state=29" in report and "random_state=17" in report
    assert config == original


def test_preview_describes_explicit_space_and_disabled_cv(workflow_config):
    """An override and single training-only split must be visible before running."""
    config = _search(
        workflow_config, strategy="random", search_space={"alpha": [0.1, 1.0]}, n_trials=7
    )
    report = preview_workflow_config(config, action="train")
    assert "explicit search space" in report
    assert "random" in report and "n_trials=7" in report
    assert "single training-only" in report
    assert "Final holdout" in report


def test_preview_describes_independent_nested_searches(workflow_config):
    """Nested CV must describe independent selection and the multiplied search budget."""
    config = _search(workflow_config)
    config.update(cv_enabled=True, cv_type="nested_cv", cv_folds=4, cv_inner_folds=2)
    report = preview_workflow_config(config, action="train")
    assert "nested_cv" in report
    assert "independent 2-fold inner search" in report
    assert "4 outer folds" in report and "separate final training search" in report
    assert "budgets apply to each search" in report


def test_preview_describes_optuna_soft_timeout_and_shap_bounds(workflow_config):
    """Optional limits should not promise to interrupt an in-flight fit."""
    config = _search(
        workflow_config,
        strategy="optuna",
        timeout=30,
        strategy_params={"sampler": "tpe", "pruner": "median"},
    )
    config["pipeline"]["explainability"] = {
        "method": "shap",
        "max_samples": 20,
        "max_features": 10,
        "max_display_samples": 5,
    }
    report = preview_workflow_config(config, action="train")
    assert "Optuna" in report and "30" in report and "in-flight fit" in report
    assert "SHAP" in report and "max_samples=20" in report
    assert "max_features=10" in report and "max_display_samples=5" in report
