"""Nested selection must preserve temporal and entity boundaries at both levels."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.cv_policy import policy_splitter, prepare_policy_data
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.pipeline import SkyulfPipeline
from skyulf.registry import NodeRegistry


def policy_rows():
    """Give each entity repeated observations and retain an independent event clock."""
    rng = np.random.default_rng(14)
    X = pd.DataFrame({"x": rng.normal(size=96), "z": rng.normal(size=96)})
    y = pd.Series(2 * X.x + rng.normal(size=96), name="target")
    X["event"] = pd.date_range("2024-01-01", periods=96, tz="UTC")
    X["entity"] = np.repeat(np.arange(24), 4)
    return X, y


@pytest.mark.parametrize("policy", ["time_series_split", "group_k_fold"])
def test_nested_policy_matches_independent_searches(policy):
    """Every outer winner and score must match independently constructed sklearn loops."""
    X, y = policy_rows()
    temporal = policy == "time_series_split"
    X = X.drop(columns="entity" if temporal else "event")
    config = TuningConfig(
        strategy="grid",
        metric="mse",
        search_space={"alpha": [0.1, 10.0]},
        cv_type="nested_cv",
        cv_nested_type=policy,
        cv_folds=3,
        cv_inner_folds=2,
        cv_shuffle=False,
        cv_time_column="event" if temporal else None,
        cv_group_column=None if temporal else "entity",
        cv_gap=1 if temporal else 0,
        cv_test_size=8 if temporal else None,
    )
    tuner = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")())
    model, result = tuner.fit(X, y, config)
    features = X[["x", "z"]]
    splitter = TimeSeriesSplit(3, gap=1, test_size=8) if temporal else GroupKFold(3)
    groups = None if temporal else X.entity
    for fold, (train, test) in zip(
        result.nested_cv["folds"], splitter.split(X, y, groups), strict=True
    ):
        inner = TimeSeriesSplit(2, gap=1, test_size=8) if temporal else GroupKFold(2)
        fit_args = {} if groups is None else {"groups": groups.iloc[train]}
        search = GridSearchCV(
            Ridge(), {"alpha": [0.1, 10.0]}, cv=inner, scoring="neg_mean_squared_error"
        ).fit(features.iloc[train], y.iloc[train], **fit_args)
        assert fold["best_params"]["alpha"] == search.best_params_["alpha"]
        assert fold["outer_score"] == pytest.approx(search.score(features.iloc[test], y.iloc[test]))
        assert fold["split"]["train_rows"] == len(train)
    assert model.n_features_in_ == 2
    assert result.nested_cv["split_policy"]["method"] == policy


def test_nested_time_sorts_without_moving_labels_or_using_clock_as_feature():
    """Shuffling source order must leave chronological selection and predictions unchanged."""
    X, y = policy_rows()
    X = X.drop(columns="entity")
    config = TuningConfig(
        strategy="grid",
        metric="mse",
        search_space={"alpha": [0.1, 10.0]},
        cv_type="nested_cv",
        cv_nested_type="time_series_split",
        cv_shuffle=False,
        cv_time_column="event",
        cv_folds=3,
        cv_inner_folds=2,
    )
    tuner = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")())
    ordered, a = tuner.fit(X, y, config)
    order = np.random.default_rng(12).permutation(len(X))
    shuffled, b = tuner.fit(X.iloc[order], y.iloc[order], config)
    assert a.nested_cv == b.nested_cv
    assert np.allclose(ordered.predict(X[["x", "z"]]), shuffled.predict(X[["x", "z"]]))


@pytest.mark.parametrize("strategy", ["grid", "random", "optuna", "halving_grid", "halving_random"])
@pytest.mark.parametrize("policy", ["time_series_split", "group_k_fold", "stratified_group_k_fold"])
def test_all_strategies_respect_policy_partitions(strategy, policy):
    """Searcher-backed strategies must receive the same isolated partitions as grid search."""
    X, y = policy_rows()
    temporal = policy == "time_series_split"
    classification = policy == "stratified_group_k_fold"
    X = X.drop(columns="entity" if temporal else "event")
    if classification:
        y = pd.Series(np.tile(["no", "yes"], 48), name="target")
    config = TuningConfig(
        strategy=strategy,
        metric="accuracy" if classification else "mse",
        search_space={"max_depth": [1, 3]} if classification else {"alpha": [0.1, 10.0]},
        cv_type="nested_cv",
        cv_nested_type=policy,
        cv_folds=2,
        cv_inner_folds=2,
        cv_shuffle=False,
        cv_time_column="event" if temporal else None,
        cv_group_column=None if temporal else "entity",
        n_trials=2,
        strategy_params={"min_resources": "exhaust", "factor": 2, "pruner": "none"},
    )
    tuner = TuningCalculator(
        NodeRegistry.get_calculator(
            "decision_tree_classifier" if classification else "ridge_regression"
        )()
    )
    _, result = tuner.fit(X, y, config)
    assert len(result.nested_cv["folds"]) == 2
    assert all(np.isfinite(f["outer_score"]) for f in result.nested_cv["folds"])
    assert result.nested_cv["split_policy"]["method"] == policy


@pytest.mark.parametrize("bad", ["missing", "ties", "shuffle"])
def test_invalid_temporal_metadata_fails_before_search(bad):
    """Invalid event boundaries cannot be accepted as a successful temporal evaluation."""
    X, y = policy_rows()
    X = X.drop(columns="entity")
    if bad == "missing":
        X.loc[1, "event"] = pd.NaT
    if bad == "ties":
        X["event"] = X.event.iloc[0]
    config = TuningConfig(
        cv_type="nested_cv",
        cv_nested_type="time_series_split",
        cv_time_column="event",
        cv_shuffle=bad == "shuffle",
        cv_folds=2,
        cv_inner_folds=2,
    )
    with pytest.raises(ValueError, match="timestamp|time|shuffle"):
        TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(X, y, config)


def test_rolling_time_windows_match_independent_partitions():
    """Gap and rolling limits must apply to inner, outer and final search partitions."""
    X, y = policy_rows()
    X = X.drop(columns="entity")
    config = TuningConfig(
        strategy="grid",
        metric="mse",
        search_space={"alpha": [1.0]},
        cv_type="nested_cv",
        cv_nested_type="time_series_split",
        cv_time_column="event",
        cv_shuffle=False,
        cv_folds=3,
        cv_inner_folds=2,
        cv_gap=2,
        cv_test_size=6,
        cv_max_train_size=30,
    )
    _, result = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
        X, y, config
    )
    for fold, (train, test) in zip(
        result.nested_cv["folds"],
        TimeSeriesSplit(3, gap=2, test_size=6, max_train_size=30).split(X),
        strict=True,
    ):
        assert fold["split"]["train_rows"] == len(train) == 30
        assert fold["split"]["test_rows"] == len(test) == 6
        assert fold["split"]["train_end"] == X.event.iloc[train[-1]].value
        for inner, (a, b) in zip(
            fold["inner_splits"],
            TimeSeriesSplit(2, gap=2, test_size=6, max_train_size=30).split(train),
            strict=True,
        ):
            assert inner["train_end"] == X.event.iloc[train[a[-1]]].value
            assert inner["test_start"] == X.event.iloc[train[b[0]]].value
    assert all(f["train_rows"] <= 30 for f in result.nested_cv["final_splits"])


@pytest.mark.parametrize("bad", ["missing", "few_groups", "class_coverage"])
def test_invalid_group_membership_rejects_before_model_fit(bad, monkeypatch):
    """Missing entities and unusable folds cannot become apparently successful searches."""
    X, y = policy_rows()
    X = X.drop(columns="event")
    if bad == "missing":
        X["entity"] = X.entity.astype(float)
        X.loc[0, "entity"] = np.nan
    elif bad == "few_groups":
        X["entity"] = 1
    else:
        y = pd.Series((X.entity == 0).astype(int))
    calculator = NodeRegistry.get_calculator(
        "decision_tree_classifier" if bad == "class_coverage" else "ridge_regression"
    )()

    def forbidden_fit(*args, **kwargs):
        """Ensure invalid memberships fail before any estimator trains."""
        pytest.fail("Invalid fold reached estimator fitting")

    monkeypatch.setattr(calculator.model_class, "fit", forbidden_fit)
    config = TuningConfig(
        cv_type="nested_cv",
        cv_nested_type="group_k_fold",
        cv_group_column="entity",
        cv_folds=3,
        cv_inner_folds=2,
        cv_shuffle=False,
    )
    with pytest.raises(ValueError, match="missing|group|class"):
        TuningCalculator(calculator).fit(X, y, config)


def test_shuffled_groups_replay_without_splitting_entities():
    """The seed must reproduce whole-group assignment on every supported sklearn version."""
    X, y = policy_rows()
    X = X.drop(columns="event")
    config = TuningConfig(cv_type="group_k_fold", cv_group_column="entity", cv_folds=3)
    _, labels, metadata = prepare_policy_data(X, y, config, "regression")
    a = policy_splitter(config, "regression", labels, metadata)
    b = policy_splitter(config, "regression", labels, metadata)
    for (train, test), (again_train, again_test) in zip(a.split(), b.split(), strict=True):
        assert np.array_equal(train, again_train)
        assert np.array_equal(test, again_test)
        assert set(X.entity.iloc[train]).isdisjoint(X.entity.iloc[test])


@pytest.mark.parametrize("policy", ["group_k_fold", "time_series_split"])
@pytest.mark.parametrize("partition", ["test", "validation"])
def test_pipeline_rejects_invalid_reserved_policy_partition_before_fit(
    policy, partition, monkeypatch
):
    """Reserved SDK partitions must satisfy entity/time boundaries before any model fits."""
    X, y = policy_rows()
    temporal = policy == "time_series_split"
    data = X.drop(columns="entity" if temporal else "event").assign(target=y)
    modeling = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "ridge_regression"},
        "strategy": "grid",
        "metric": "mse",
        "search_space": {"alpha": [1.0]},
        "cv_type": "nested_cv",
        "cv_nested_type": policy,
        "cv_folds": 2,
        "cv_inner_folds": 2,
        "cv_shuffle": False,
        "cv_time_column": "event" if temporal else None,
        "cv_group_column": None if temporal else "entity",
    }
    heldout = data.iloc[:8].copy()
    dataset = SplitDataset(
        train=data,
        test=heldout if partition == "test" else data.iloc[:0],
        validation=heldout if partition == "validation" else None,
    )

    def reject_fit(*args, **kwargs):
        """Fail if invalid split metadata reaches estimator fitting."""
        pytest.fail("Invalid reserved partition reached model fitting")

    monkeypatch.setattr(Ridge, "fit", reject_fit)
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": modeling})
    with pytest.raises(ValueError, match="Reserved.*holdout"):
        pipeline.fit(dataset, "target")
