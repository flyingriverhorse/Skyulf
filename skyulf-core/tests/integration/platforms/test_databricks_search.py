"""Bound local search settings before a Databricks training source is read."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.tuning.cv import CVSpec
from skyulf.integrations.databricks.training.tuning.search import (
    base_model_config,
    prepare_search_pipeline,
)


def _pipeline(**changes):
    """Build a small registered regression search with explicit trial limits."""
    model = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "ridge_regression", "params": {"fit_intercept": True}},
        "strategy": "grid",
        "metric": "rmse",
        "search_space": {"alpha": [0.1, 1.0]},
        "n_trials": 2,
        "random_state": 17,
    }
    model.update(changes)
    return {"preprocessing": [], "modeling": model}


def test_base_model_resolves_wrapper_without_mutating_input():
    """Task checks must see the selected model, while replay retains the wrapper."""
    pipeline = _pipeline()
    original = deepcopy(pipeline)

    assert base_model_config(pipeline) == {
        "type": "ridge_regression",
        "params": {"fit_intercept": True},
    }
    assert pipeline == original


def test_search_preparation_replays_shared_cv_and_fixed_parameters():
    """A submitted search must use shared folds and keep fixed model settings."""
    pipeline = _pipeline()
    original = deepcopy(pipeline)
    cv = CVSpec(enabled=True, folds=3, method="k_fold", shuffle=True, random_state=29)

    prepared = prepare_search_pipeline(pipeline, cv, target_column="target", event_column=None)

    assert prepared["modeling"]["search_space"] == {
        "alpha": [0.1, 1.0],
        "fit_intercept": [True],
    }
    assert prepared["modeling"]["cv_enabled"] is True
    assert prepared["modeling"]["cv_folds"] == 3
    assert prepared["modeling"]["cv_random_state"] == 29
    assert prepared["modeling"]["random_state"] == 17
    assert prepared["modeling"]["n_jobs"] == 1
    assert pipeline == original


def test_ordinary_model_is_deep_copied_without_search_settings():
    """Default training must retain the prior Core configuration exactly."""
    pipeline = {"modeling": {"type": "ridge_regression", "params": {"alpha": 1.0}}}

    prepared = prepare_search_pipeline(
        pipeline, CVSpec(), target_column="target", event_column=None
    )

    assert prepared == pipeline
    assert prepared is not pipeline
    assert prepared["modeling"] is not pipeline["modeling"]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"strategy": "halving_grid", "strategy_params": {"max_resources": -1}}, "max_resources"),
        ({"metric": "heldout_rmse"}, "metric"),
        ({"metric": "f1"}, "Regression"),
        ({"search_space": {"missing_parameter": [1]}}, "missing_parameter"),
        ({"search_space": {"alpha": [float("nan")]}}, "finite"),
        ({"search_space": {"alpha": []}}, "nonempty"),
        ({"search_space": {"alpha": [1, 2, 3]}, "max_candidates": 2}, "max_candidates"),
        ({"n_trials": 0}, "n_trials"),
        ({"n_trials": 1001}, "n_trials"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"timeout": 30}, "timeout"),
        ({"strategy_params": {"factor": 2}}, "strategy_params"),
        ({"tune_threshold": True}, "tune_threshold"),
        ({"n_jobs": 2}, "n_jobs"),
        ({"random_state": True}, "random_state"),
    ],
)
def test_invalid_search_is_rejected_before_training(changes, error):
    """Invalid budgets and silent Core fallbacks must fail during preflight."""
    with pytest.raises(ValueError, match=error):
        prepare_search_pipeline(
            _pipeline(**changes),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_conflicting_fixed_and_searched_parameter_is_rejected():
    """A fixed base parameter cannot be overridden by candidate values."""
    with pytest.raises(ValueError, match="fit_intercept"):
        prepare_search_pipeline(
            _pipeline(search_space={"alpha": [1.0], "fit_intercept": [False]}),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_matching_singleton_fixed_parameter_is_accepted():
    """An explicitly repeated fixed parameter remains a single candidate axis."""
    prepared = prepare_search_pipeline(
        _pipeline(search_space={"alpha": [1.0], "fit_intercept": [True]}),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["search_space"]["fit_intercept"] == [True]


def test_disabled_cv_is_kept_for_training_only_shuffle_split():
    """Search without enabled folds must use Core's training partition split."""
    prepared = prepare_search_pipeline(
        _pipeline(),
        CVSpec(enabled=False, random_state=31),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["cv_enabled"] is False
    assert prepared["modeling"]["cv_random_state"] == 31


def test_conflicting_cv_override_is_rejected():
    """Wrapper folds cannot silently disagree with the workflow's folds."""
    with pytest.raises(ValueError, match="cv_folds"):
        prepare_search_pipeline(
            _pipeline(cv_folds=2),
            CVSpec(enabled=True, folds=3),
            target_column="target",
            event_column=None,
        )


def test_estimator_parallelism_cannot_be_searched():
    """An estimator's own n_jobs axis must not multiply trial and fold workers."""
    with pytest.raises(ValueError, match="n_jobs"):
        prepare_search_pipeline(
            _pipeline(
                base_model={"type": "random_forest_regressor", "params": {}},
                search_space={"n_jobs": [1, 2]},
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_ensemble_fixed_parallelism_cannot_multiply_fold_workers():
    """Structural ensemble n_jobs must also stay single-worker during search."""
    with pytest.raises(ValueError, match="n_jobs"):
        prepare_search_pipeline(
            _pipeline(
                base_model={"type": "voting_regressor", "params": {"n_jobs": 2}},
                search_space={"weights": [None]},
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_nested_ensemble_parallelism_is_rejected():
    """Selected base learners cannot start workers inside each search fold."""
    with pytest.raises(ValueError, match="n_jobs"):
        prepare_search_pipeline(
            _pipeline(
                base_model={
                    "type": "voting_regressor",
                    "params": {
                        "base_estimators": ["random_forest", "ridge"],
                        "base_estimator_params": {"random_forest": {"n_jobs": 2}},
                    },
                },
                search_space={"weights": [None]},
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_search_validates_fold_preprocessing_before_fit():
    """A missing transformer cannot fail only after a search begins."""
    pipeline = _pipeline()
    pipeline["preprocessing"] = [{"name": "broken", "transformer": "missing_transformer"}]
    with pytest.raises(ValueError, match="missing_transformer"):
        prepare_search_pipeline(
            pipeline,
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
def test_halving_uses_core_supported_resource_controls(strategy):
    """Existing successive-halving choices must remain usable with bounded axes."""
    prepared = prepare_search_pipeline(
        _pipeline(
            strategy=strategy,
            strategy_params={"factor": 2, "resource": "n_samples", "min_resources": 4},
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["strategy"] == strategy
    assert prepared["modeling"]["strategy_params"]["factor"] == 2


def test_halving_estimator_resource_requires_explicit_maximum():
    """Tree budget must have an explicit maximum before Core fits a candidate."""
    with pytest.raises(ValueError, match="max_resources"):
        prepare_search_pipeline(
            _pipeline(
                strategy="halving_random",
                base_model={"type": "random_forest_regressor", "params": {}},
                search_space={"max_depth": [2, 3]},
                strategy_params={"resource": "n_estimators", "min_resources": 10},
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_halving_estimator_resource_is_preserved_with_bound():
    """A bounded tree count can be forwarded to Core halving search."""
    prepared = prepare_search_pipeline(
        _pipeline(
            strategy="halving_random",
            base_model={"type": "random_forest_regressor", "params": {}},
            search_space={"max_depth": [2, 3]},
            strategy_params={"resource": "n_estimators", "min_resources": 10, "max_resources": 40},
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["strategy_params"]["max_resources"] == 40


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
def test_halving_automatic_space_leaves_resource_to_scheduler(strategy):
    """Automatic axes must not conflict with the selected halving tree budget."""
    recipe = _pipeline(
        strategy=strategy,
        base_model={"type": "random_forest_regressor", "params": {}},
        search_space={},
        max_candidates=10000,
        strategy_params={"resource": "n_estimators", "min_resources": 4, "max_resources": 8},
    )
    prepared = prepare_search_pipeline(
        recipe, CVSpec(enabled=True), target_column="target", event_column=None
    )
    assert "n_estimators" not in prepared["modeling"]["search_space"]
    assert "max_depth" in prepared["modeling"]["search_space"]
    assert prepared["modeling"]["strategy_params"]["max_resources"] == 8
    assert recipe["modeling"]["search_space"] == {}


@pytest.mark.parametrize("space", [{}, {"n_estimators": [4, 8]}])
def test_halving_explicit_resource_conflicts_remain_rejected(space):
    """An explicit resource axis or fixed value cannot override the scheduler."""
    params = {"n_estimators": 4} if not space else {}
    with pytest.raises(ValueError, match="cannot also be a fixed or searched parameter"):
        prepare_search_pipeline(
            _pipeline(
                strategy="halving_random",
                base_model={"type": "random_forest_regressor", "params": params},
                search_space=space,
                strategy_params={
                    "resource": "n_estimators",
                    "min_resources": 4,
                    "max_resources": 8,
                },
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_ensemble_structural_base_selection_is_not_a_candidate_axis():
    """A selected ensemble's learner list must be resolved before tuning."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={
                "type": "voting_regressor",
                "params": {"base_estimators": ["ridge", "lasso"]},
            },
            search_space={"weights": [None]},
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["base_model"]["params"]["base_estimators"] == ["ridge", "lasso"]
    assert "base_estimators" not in prepared["modeling"]["search_space"]


def test_ensemble_empty_space_uses_core_model_defaults():
    """An ensemble's auto-built axes must be budgeted rather than skipped."""
    with pytest.raises(ValueError, match="max_candidates"):
        prepare_search_pipeline(
            _pipeline(
                base_model={"type": "voting_classifier", "params": {}},
                metric="accuracy",
                search_space={},
                max_candidates=1,
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_ensemble_base_model_tuning_control_builds_nested_axes():
    """Core's tune_base_models option must expand selected learners, not reach the estimator."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={
                "type": "voting_regressor",
                "params": {
                    "base_estimators": ["ridge", "lasso"],
                    "tune_base_models": True,
                },
            },
            search_space={},
            max_candidates=10_000,
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    space = prepared["modeling"]["search_space"]
    assert any(name.startswith("ridge__") for name in space)
    assert "tune_base_models" not in space
    assert (
        prepare_search_pipeline(
            prepared, CVSpec(enabled=True), target_column="target", event_column=None
        )
        == prepared
    )


def test_unknown_wrapper_option_is_rejected_before_fit():
    """Misspelled CV settings cannot be silently filtered out by Core."""
    with pytest.raises(ValueError, match="cv_foldz"):
        prepare_search_pipeline(
            _pipeline(cv_foldz=3),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_halving_resource_cannot_also_be_a_fixed_base_parameter():
    """A fixed tree count cannot compete with successive-halving's resource budget."""
    with pytest.raises(ValueError, match="n_estimators"):
        prepare_search_pipeline(
            _pipeline(
                strategy="halving_random",
                base_model={"type": "random_forest_regressor", "params": {"n_estimators": 20}},
                search_space={"max_depth": [2, 3]},
                strategy_params={
                    "resource": "n_estimators",
                    "min_resources": 10,
                    "max_resources": 40,
                },
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_default_estimator_parallelism_is_forced_to_one():
    """Core's n_jobs=-1 default must not multiply search folds and workers."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={"type": "random_forest_regressor", "params": {"n_estimators": 3}},
            search_space={"max_depth": [2]},
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["search_space"]["n_jobs"] == [1]


def test_ensemble_learner_parallelism_is_forced_to_one():
    """Nested random forests cannot escape the local single-worker budget."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={
                "type": "voting_regressor",
                "params": {
                    "base_estimators": ["random_forest", "ridge"],
                    "base_estimator_params": {"random_forest": {"n_estimators": 3}},
                },
            },
            search_space={"weights": [None]},
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["search_space"]["random_forest__n_jobs"] == [1]


def test_default_voting_ensemble_uses_core_selected_learners():
    """An ensemble with no explicit members must resolve Core's own defaults."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={"type": "voting_regressor", "params": {}}, search_space={"weights": [None]}
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["base_model"]["params"]["base_estimators"] == [
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    ]
    assert prepared["modeling"]["search_space"]["random_forest__n_jobs"] == [1]


def test_fitted_default_ensemble_uses_one_worker(tmp_path):
    """Single-worker settings must survive candidate selection and final refit."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={
                "type": "voting_regressor",
                "params": {
                    "base_estimator_params": {
                        "random_forest": {"n_estimators": 3},
                        "gradient_boosting": {"n_estimators": 3},
                    },
                },
            },
            search_space={"weights": [None]},
        ),
        CVSpec(enabled=True, folds=2),
        target_column="target",
        event_column=None,
    )
    frame = pd.DataFrame({"x": np.arange(24, dtype=float), "target": np.arange(24) * 2.0})
    artifact = fit_workflow(
        prepared,
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / "ensemble",
        max_rows=30,
        max_bytes=100_000,
    )
    assert artifact.pipeline.model_estimator is not None
    estimator = artifact.pipeline.model_estimator._unwrap_tuned_model()
    assert estimator.n_jobs == 1
    assert dict(estimator.estimators)["random_forest"].n_jobs == 1


def test_fitted_forest_uses_one_worker(tmp_path):
    """The direct forest's Core default -1 must not survive final refit."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={"type": "random_forest_regressor", "params": {"n_estimators": 3}},
            search_space={"max_depth": [2]},
        ),
        CVSpec(enabled=True, folds=2),
        target_column="target",
        event_column=None,
    )
    frame = pd.DataFrame({"x": np.arange(24, dtype=float), "target": np.arange(24) * 2.0})
    artifact = fit_workflow(
        prepared,
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / "forest",
        max_rows=30,
        max_bytes=100_000,
    )
    assert artifact.pipeline.model_estimator is not None
    assert artifact.pipeline.model_estimator._unwrap_tuned_model().n_jobs == 1


def test_temporal_search_uses_workflow_event_column():
    """A conflicting requested sort column cannot invalidate chronological CV."""
    cv = CVSpec(enabled=True, method="time_series_split", shuffle=False)
    with pytest.raises(ValueError, match="cv_time_column"):
        prepare_search_pipeline(
            _pipeline(cv_time_column="x"),
            cv,
            target_column="target",
            event_column="event_time",
        )
    prepared = prepare_search_pipeline(
        _pipeline(), cv, target_column="target", event_column="event_time"
    )
    assert prepared["modeling"]["cv_time_column"] == "event_time"


def test_non_temporal_search_rejects_unused_time_column():
    """A time column setting cannot be accepted when the splitter ignores it."""
    with pytest.raises(ValueError, match="cv_time_column"):
        prepare_search_pipeline(
            _pipeline(cv_time_column="event_time"),
            CVSpec(enabled=True),
            target_column="target",
            event_column="event_time",
        )


def test_omitted_search_defaults_are_recorded_for_replay():
    """Effective artifacts and lifecycle logging need concrete Core defaults."""
    pipeline = {
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression", "params": {}},
            "metric": "rmse",
            "search_space": {"alpha": [1.0]},
        }
    }
    cv = CVSpec(enabled=True)
    prepared = prepare_search_pipeline(pipeline, cv, target_column="target", event_column=None)
    assert prepared["modeling"]["strategy"] == "random"
    assert prepared["modeling"]["n_trials"] == 10
    assert prepared["modeling"]["random_state"] == 42
    assert prepared["modeling"]["max_candidates"] == 1000
    assert (
        prepare_search_pipeline(prepared, cv, target_column="target", event_column=None) == prepared
    )


def test_plain_model_empty_space_uses_core_registry_grid():
    """Advanced search starts with existing per-model Core defaults."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={"type": "ridge_regression", "params": {}}, search_space={}, n_trials=20
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    space = prepared["modeling"]["search_space"]
    assert space["alpha"] == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert space["fit_intercept"] == [True, False]


def test_fixed_model_setting_overrides_automatic_axis():
    """Selected base settings stay fixed when defaults are generated automatically."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={"type": "ridge_regression", "params": {"alpha": 0.5}}, search_space={}
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["search_space"]["alpha"] == [0.5]


def test_fixed_ensemble_structure_is_not_automatically_searched():
    """A chosen ensemble voting mode must survive automatic defaults."""
    prepared = prepare_search_pipeline(
        _pipeline(
            base_model={
                "type": "voting_classifier",
                "params": {"voting": "hard", "base_estimators": ["logistic_regression"]},
            },
            metric="accuracy",
            search_space={},
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert "voting" not in prepared["modeling"]["search_space"]


def test_explicit_search_cannot_override_fixed_ensemble_structure():
    """A searched structural setting must not silently replace the selected ensemble."""
    with pytest.raises(ValueError, match="voting"):
        prepare_search_pipeline(
            _pipeline(
                base_model={"type": "voting_classifier", "params": {"voting": "hard"}},
                metric="accuracy",
                search_space={"voting": ["soft"]},
            ),
            CVSpec(enabled=True),
            target_column="target",
            event_column=None,
        )


def test_halving_digit_string_resource_limit_is_normalized():
    """Template integer text must become a stable numeric Core setting."""
    prepared = prepare_search_pipeline(
        _pipeline(strategy="halving_random", strategy_params={"min_resources": "10"}),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["strategy_params"]["min_resources"] == 10


def test_optuna_settings_use_core_fields():
    """Optional sampler, pruner and timeout reach Core without being flattened."""
    pytest.importorskip("optuna_integration")
    prepared = prepare_search_pipeline(
        _pipeline(
            strategy="optuna", timeout=30, strategy_params={"sampler": "random", "pruner": "none"}
        ),
        CVSpec(enabled=True),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["timeout"] == 30
    assert prepared["modeling"]["strategy_params"] == {"sampler": "random", "pruner": "none"}


def test_nested_cv_uses_shared_outer_setting_for_core_inner_folds():
    """Nested diagnostics and search must share one explicit CV configuration."""
    prepared = prepare_search_pipeline(
        _pipeline(),
        CVSpec(enabled=True, folds=4, method="nested_cv"),
        target_column="target",
        event_column=None,
    )
    assert prepared["modeling"]["cv_type"] == "nested_cv"
    assert prepared["modeling"]["cv_folds"] == 4
