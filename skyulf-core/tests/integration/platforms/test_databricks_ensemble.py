"""Ensemble selections must reach Core without silent defaults or changed fixed values."""

from copy import deepcopy

import pytest

from skyulf.integrations.databricks.training.tuning.cv import CVSpec
from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline


def _prepare(params, model="voting_regressor", **modeling):
    """Exercise the same admission boundary used by training and preview."""
    return prepare_search_pipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": model, "params": params},
                "strategy": "random",
                "n_trials": 2,
                "metric": "accuracy" if model.endswith("classifier") else "rmse",
                **modeling,
            },
        },
        CVSpec(enabled=True, folds=2),
        target_column="target",
        event_column=None,
    )["modeling"]


def test_named_weights_follow_selected_member_order():
    """Frontend-style named weights must not silently become equal weights."""
    params = {
        "base_estimators": ["ridge", "linear_regression"],
        "weights": {"linear_regression": 3, "ridge": 1},
    }
    before = deepcopy(params)
    result = _prepare(params)
    assert result["base_model"]["params"]["weights"] == [1, 3]
    assert params == before


@pytest.mark.parametrize(
    "model,members,axis",
    [
        ("voting_classifier", ["logistic_regression"], "logistic_regression__C"),
        ("stacking_classifier", ["logistic_regression"], "logistic_regression__C"),
        ("voting_regressor", ["ridge"], "ridge__alpha"),
        ("stacking_regressor", ["ridge"], "ridge__alpha"),
    ],
)
def test_selected_ensemble_members_are_tuned_by_default(model, members, axis):
    """Omitting the component-tuning flag must still search the chosen learners."""
    params = {"base_estimators": members}
    result = _prepare(params, model)
    assert len(result["search_space"][axis]) > 1
    assert result["base_model"]["params"]["tune_base_models"] is True
    assert "tune_base_models" not in params


def test_explicit_component_tuning_opt_out_is_preserved():
    """A deliberate fixed-component recipe must keep its base learners fixed."""
    result = _prepare(
        {"base_estimators": ["ridge"], "tune_base_models": False}, "stacking_regressor"
    )
    assert result["base_model"]["params"]["tune_base_models"] is False
    assert "ridge__alpha" not in result["search_space"]


@pytest.mark.parametrize("calibrated", [False, True])
def test_auto_space_preserves_fixed_base_model_parameters(calibrated):
    """Automatic axes must respect values configured inside base_estimator_params."""
    result = _prepare(
        {
            "base_estimators": ["random_forest", "logistic_regression"],
            "tune_base_models": True,
            "calibrate_base_models": calibrated,
            "base_estimator_params": {"random_forest": {"n_estimators": 3}},
        },
        "voting_classifier",
    )
    prefix = "random_forest__estimator__" if calibrated else "random_forest__"
    assert result["search_space"][prefix + "n_estimators"] == [3]


@pytest.mark.parametrize(
    "params,model,match",
    [
        ({"base_estimators": ["not_a_model"]}, "voting_regressor", "base_estimators"),
        ({"base_estimators": []}, "voting_regressor", "base_estimators"),
        ({"base_estimators": ["ridge", "ridge"]}, "voting_regressor", "base_estimators"),
        ({"base_estimators": ["ridge"], "weights": [0]}, "voting_regressor", "weights"),
        ({"base_estimators": ["ridge"], "weights": [float("nan")]}, "voting_regressor", "weights"),
        ({"base_estimators": ["ridge"], "weights": {"lasso": 1}}, "voting_regressor", "weights"),
        (
            {"base_estimators": ["ridge"], "base_estimator_params": {"lasso": {"alpha": 1}}},
            "voting_regressor",
            "base_estimator_params",
        ),
        (
            {"base_estimators": ["ridge"], "base_estimator_params": {"ridge": {"alphaa": 1}}},
            "voting_regressor",
            "alphaa",
        ),
        ({"calibrate_base_models": True}, "voting_regressor", "classification"),
        ({"calibrate_base_models": "yes"}, "voting_classifier", "boolean"),
        (
            {"calibrate_base_models": True, "calibration_cv": 1},
            "voting_classifier",
            "calibration_cv",
        ),
        ({"weights": [1, 1, 1]}, "stacking_regressor", "weights"),
        ({"final_estimator": "unknown"}, "stacking_regressor", "final_estimator"),
        ({"final_estimator_params": {"unknown": 1}}, "stacking_regressor", "unknown"),
        ({"cv": "prefit"}, "stacking_regressor", "cv"),
    ],
)
def test_invalid_ensemble_settings_fail_before_fit(params, model, match):
    """Misconfigured ensembles must fail instead of silently selecting another recipe."""
    with pytest.raises(ValueError, match=match):
        _prepare(params, model)


def test_explicit_nested_search_cannot_override_fixed_base_parameter():
    """Conflicting manual axes must not change a user-declared fixed parameter."""
    with pytest.raises(ValueError, match="conflicts"):
        _prepare(
            {"base_estimators": ["ridge"], "base_estimator_params": {"ridge": {"alpha": 1}}},
            search_space={"ridge__alpha": [0.1, 1]},
        )


def test_ordinary_ensemble_uses_the_same_admission_and_named_weights():
    """Library callers using fixed models must not bypass ensemble recipe checks."""
    config = {
        "modeling": {
            "type": "voting_regressor",
            "params": {
                "base_estimators": ["ridge", "linear_regression"],
                "weights": {"ridge": 2},
            },
        }
    }
    result = prepare_search_pipeline(config, CVSpec(), target_column="target", event_column=None)
    assert result["modeling"]["params"]["weights"] == [2, 1]
    config["modeling"]["params"]["base_estimators"] = ["typo"]
    with pytest.raises(ValueError, match="base_estimators"):
        CVSpec().validate_pipeline(config, target_column="target")
    invalid = {"modeling": {"type": "voting_regressor", "params": {"bogus": 1}}}
    with pytest.raises(ValueError, match="Unknown ensemble parameter"):
        CVSpec().validate_pipeline(invalid, target_column="target")


def test_disabled_calibration_still_rejects_invalid_dormant_settings():
    """A stored calibration typo must fail before later enabling the feature."""
    with pytest.raises(ValueError, match="calibration_method"):
        _prepare(
            {"calibrate_base_models": False, "calibration_method": "typo"}, "voting_classifier"
        )


def test_fixed_ensemble_cannot_silently_ignore_a_tuning_request():
    """Base-model search requires the existing tuner wrapper even for SDK callers."""
    config = {"modeling": {"type": "voting_regressor", "params": {"tune_base_models": True}}}
    with pytest.raises(ValueError, match="hyperparameter_tuner"):
        prepare_search_pipeline(config, CVSpec(), target_column="target", event_column=None)
