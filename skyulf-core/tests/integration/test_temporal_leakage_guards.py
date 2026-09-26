"""Temporal feature guards must reject concrete future/target leakage paths."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline, validate_leakage_safety
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.time_series.lag import LagFeaturesApplier, LagFeaturesCalculator
from skyulf.preprocessing.time_series.rolling import (
    RollingAggregateApplier,
    RollingAggregateCalculator,
)

PAIRS = [
    (LagFeaturesCalculator, LagFeaturesApplier),
    (RollingAggregateCalculator, RollingAggregateApplier),
]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("calculator,applier", PAIRS)
def test_missing_declared_sort_column_cannot_silently_use_input_order(engine, calculator, applier):
    """A typo must not turn a chronological lag/window into a future-dependent feature."""
    frame = pd.DataFrame({"time": [3, 1, 2], "value": [30.0, 10.0, 20.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    state = calculator().fit(frame, {"columns": ["value"], "sort_by": "time_typo"})
    with pytest.raises(ValueError, match="sort_by.*time_typo"):
        applier().apply(frame, state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", [-1, 0, 1.5, True])
def test_direct_lag_artifact_cannot_read_future_or_current_rows(engine, lag):
    """The public applier must enforce the positive-lag contract as well as fit."""
    frame = pd.DataFrame({"value": [10.0, 20.0, 30.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError, match="positive.*lag|lags.*positive"):
        LagFeaturesApplier().apply(frame, {"columns": ["value"], "lags": [lag]})


@pytest.mark.parametrize("placement", ["before", "after", "external", "no_split"])
def test_known_current_target_rolling_is_rejected_at_every_split_placement(placement):
    """A split cannot protect a feature that contains its own row's answer."""
    rolling = {
        "name": "target_roll",
        "transformer": "RollingAggregate",
        "params": {"columns": ["target"], "window": 1},
    }
    split = {
        "name": "split",
        "transformer": "TrainTestSplitter",
        "params": {"target_column": "target", "test_size": 0.25},
    }
    steps = [rolling, split] if placement == "before" else [split, rolling]
    if placement in {"external", "no_split"}:
        steps = [rolling]
    config = {"preprocessing": steps, "modeling": {"type": "linear_regression"}}
    with pytest.raises(ValueError, match="RollingAggregate.*target.*current row"):
        validate_leakage_safety(
            config, target_column="target", already_split=placement == "external"
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("external", [False, True])
def test_real_pipeline_stops_target_rolling_before_model_fit(engine, external):
    """The real fit entrypoint must enforce the guard, including an external SplitDataset."""
    frame = pd.DataFrame({"x": range(12), "target": [(i * 13) % 17 for i in range(12)]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    data = SplitDataset(train=frame[:8], test=frame[8:]) if external else frame
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "roll",
                    "transformer": "RollingAggregate",
                    "params": {"columns": ["target"], "window": 1},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    with pytest.raises(ValueError, match="RollingAggregate.*target.*current row"):
        pipeline.fit(data, target_column="target")
    assert not pipeline.is_fitted()


def test_direct_calculator_rejects_known_target_but_retains_feature_rolling():
    """Direct FE users with target context need the same protection as full pipelines."""
    frame = pd.DataFrame({"target": [1.0, 2.0], "observed": [3.0, 4.0]})
    with pytest.raises(ValueError, match="RollingAggregate.*target.*current row"):
        RollingAggregateCalculator().fit(frame, {"columns": ["target"], "target_column": "target"})
    state = RollingAggregateCalculator().fit(
        frame, {"columns": ["observed"], "target_column": "target"}
    )
    assert state["columns"] == ["observed"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("calculator,applier", PAIRS)
def test_future_rows_and_other_entities_do_not_change_prior_features(engine, calculator, applier):
    """For ordered observed features, history must be causal within each configured group."""
    original = pd.DataFrame(
        {"g": ["a", "b", "a", "b"], "t": [1, 1, 2, 2], "v": [10.0, 100.0, 20.0, 200.0]}
    )
    extended = pd.concat(
        [pd.DataFrame({"g": ["a", "b"], "t": [3, 3], "v": [9999.0, -9999.0]}), original],
        ignore_index=True,
    )
    if engine == "polars":
        original, extended = pl.from_pandas(original), pl.from_pandas(extended)
    state = calculator().fit(
        original, {"columns": ["v"], "group_by": ["g"], "sort_by": "t", "window": 2}
    )
    first = applier().apply(original, state)
    second = applier().apply(extended, state)
    if engine == "polars":
        first, second = first.to_pandas(), second.to_pandas()
    feature = "v_lag_1" if calculator is LagFeaturesCalculator else "v_roll_mean_2"
    np.testing.assert_allclose(first[feature], second.loc[second.t <= 2, feature], equal_nan=True)
    expected = (
        [np.nan, np.nan, 10.0, 100.0]
        if calculator is LagFeaturesCalculator
        else [10.0, 100.0, 15.0, 150.0]
    )
    np.testing.assert_allclose(first[feature], expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("calculator,applier", PAIRS)
def test_new_batch_does_not_inherit_training_history(engine, calculator, applier):
    """Saved configuration is not a history buffer for live or separately scored batches."""
    training = pd.DataFrame({"t": [1, 2], "v": [10.0, 20.0]})
    incoming = pd.DataFrame({"t": [3, 4], "v": [30.0, 40.0]})
    if engine == "polars":
        training, incoming = pl.from_pandas(training), pl.from_pandas(incoming)
    state = calculator().fit(training, {"columns": ["v"], "sort_by": "t", "window": 2})
    result = applier().apply(incoming, state)
    if engine == "polars":
        result = result.to_pandas()
    feature = "v_lag_1" if calculator is LagFeaturesCalculator else "v_roll_mean_2"
    expected = [np.nan, 30.0] if calculator is LagFeaturesCalculator else [30.0, 35.0]
    np.testing.assert_allclose(result[feature], expected, equal_nan=True)


@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
def test_target_rolling_is_invalid_even_when_ordering_warnings_are_disabled(mode):
    """An explicit leakage-warning override cannot make the current answer a valid feature."""
    config = {
        "preprocessing": [
            {"name": "split", "transformer": "TrainTestSplitter", "params": {"target_column": "y"}},
            {"name": "roll", "transformer": "RollingAggregate", "params": {"columns": ["y"]}},
        ]
    }
    with pytest.raises(ValueError, match="RollingAggregate.*target.*current row"):
        validate_leakage_safety(config, on_leakage=mode)
