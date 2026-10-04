"""Opt-in temporal artifacts must bridge batches without mutating fitted state."""

import json
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.time_series.history import TemporalHistorySession


def _frame(engine, times, values, groups=None):
    """Build equivalent native frames with stable observation identifiers."""
    frame = pd.DataFrame({"t": times, "v": values, "g": groups or ["a"] * len(times)})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _engineer(kind):
    """Configure one temporal step using the public recipe contract."""
    return FeatureEngineer(
        [
            {
                "name": "history",
                "transformer": kind,
                "params": {
                    "columns": ["v"],
                    "sort_by": "t",
                    "group_by": ["g"],
                    "history_mode": "carry",
                    "lags": [1, 3],
                    "window": 2,
                    "aggregations": ["mean"],
                },
            }
        ]
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "kind, column, expected",
    [
        ("LagFeatures", "v_lag_1", 2.5),
        ("RollingAggregate", "v_roll_mean_2", 3.25),
    ],
)
def test_integer_incoming_values_do_not_truncate_float_history(engine, kind, column, expected):
    """Batch dtype inference must not round the saved temporal context."""
    engineer = _engineer(kind)
    engineer.fit_transform(_frame(engine, [1, 2], [1.5, 2.5]))

    result = engineer.transform(_frame(engine, [3], [4]), preserve_rows=True)

    assert result[column].to_list() == [expected]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_integer_incoming_values_accept_missing_history(engine):
    """Missing historical values must survive a later integer-valued batch."""
    engineer = _engineer("LagFeatures")
    engineer.fit_transform(_frame(engine, [1, 2], [1.5, None]))

    result = engineer.transform(_frame(engine, [3], [4]), preserve_rows=True)

    assert pd.isna(result["v_lag_1"].to_list()[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_null_only_history_keeps_lag_values_numeric(engine):
    """A null-only saved tail must not coerce the next batch's numeric lags to text."""
    engineer = _engineer("LagFeatures")
    engineer.steps_config[0]["params"]["lags"] = [1]
    engineer.fit_transform(_frame(engine, [1, 2], [1.0, np.nan]))

    result = engineer.transform(_frame(engine, [4, 5], [5, 6]), preserve_rows=True)

    values = result["v_lag_1"].to_list()
    assert pd.isna(values[0])
    assert values[1] == 5
    if engine == "polars":
        assert result.schema["v_lag_1"].is_numeric()
    else:
        assert pd.api.types.is_numeric_dtype(result["v_lag_1"].dtype)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "kind, column, expected",
    [
        ("LagFeatures", "v_lag_1", [30.0, 40.0]),
        ("RollingAggregate", "v_roll_mean_2", [35.0, 45.0]),
    ],
)
def test_saved_tail_continues_holdout_without_mutating_artifact(engine, kind, column, expected):
    """Separate holdout rows must use training history while the seed stays immutable."""
    engineer = _engineer(kind)
    train = _frame(engine, [1, 2, 3], [10.0, 20.0, 30.0])
    test = _frame(engine, [4, 5], [40.0, 50.0])
    output, _ = engineer.fit_transform(SplitDataset(train=train, test=test))
    np.testing.assert_allclose(output.test[column].to_list(), expected)
    before = deepcopy(engineer.fitted_steps[0]["artifact"])
    scored = engineer.transform(test, preserve_rows=True)
    np.testing.assert_allclose(scored[column].to_list(), expected)
    assert engineer.fitted_steps[0]["artifact"] == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_training_does_not_read_its_own_tail(engine):
    """Fitted history must never turn the first training lag into a future value."""
    engineer = _engineer("LagFeatures")
    output, _ = engineer.fit_transform(_frame(engine, [1, 2, 3], [10.0, 20.0, 30.0]))
    np.testing.assert_allclose(
        np.asarray(output["v_lag_1"].to_list(), dtype=float), [np.nan, 10.0, 20.0], equal_nan=True
    )
    assert "history_seed" in engineer.fitted_steps[0]["artifact"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_history_preserves_requested_order_and_isolates_entities(engine):
    """History is keyed by entity and must not reorder predictions or their labels."""
    engineer = _engineer("LagFeatures")
    engineer.fit_transform(
        _frame(engine, [1, 1, 2, 2], [10.0, 100.0, 20.0, 200.0], ["a", "b", "a", "b"])
    )
    incoming = _frame(engine, [4, 3, 3, 3], [40.0, 300.0, 30.0, 900.0], ["a", "b", "a", "new"])
    result = engineer.transform(incoming, preserve_rows=True)
    np.testing.assert_allclose(
        np.asarray(result["v_lag_1"].to_list(), dtype=float),
        [30.0, 200.0, 20.0, np.nan],
        equal_nan=True,
    )
    assert result["g"].to_list() == ["a", "b", "a", "new"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "kind,column", [("LagFeatures", "v_lag_3"), ("RollingAggregate", "v_roll_mean_2")]
)
def test_chunks_and_reload_match_one_batch(engine, kind, column):
    """Persisted continuation must match a single causal batch, even across model reloads."""
    engineer = _engineer(kind)
    engineer.fit_transform(_frame(engine, [1, 2, 3], [10.0, 20.0, 30.0]))
    expected = engineer.transform(_frame(engine, [4, 5, 6, 7], [40.0, 50.0, 60.0, 70.0]))
    values = []
    state = None
    for times, observations in [([4, 5], [40.0, 50.0]), ([6, 7], [60.0, 70.0])]:
        engineer = pickle.loads(pickle.dumps(engineer))
        with TemporalHistorySession("model-v1", state) as session:
            output = engineer.transform(_frame(engine, times, observations), preserve_rows=True)
            repeated = engineer.transform(_frame(engine, times, observations), preserve_rows=True)
        assert output[column].to_list() == repeated[column].to_list()
        state = json.loads(json.dumps(session.state))
        values.extend(output[column].to_list())
    np.testing.assert_allclose(values, expected[column].to_list())
    assert len(next(iter(state["steps"].values()))) <= 3


def test_failed_prediction_discards_proposal_and_retry_is_deterministic():
    """Model failure after preprocessing must leave committed continuation reusable."""
    engineer = _engineer("LagFeatures")
    engineer.fit_transform(_frame("pandas", [1, 2, 3], [10.0, 20.0, 30.0]))
    failed = TemporalHistorySession("v1")
    with pytest.raises(RuntimeError, match="model failed"), failed:
        engineer.transform(_frame("pandas", [4], [40.0]))
        raise RuntimeError("model failed")
    assert failed.state is None
    proposals = []
    for _ in range(2):
        with TemporalHistorySession("v1") as session:
            engineer.transform(_frame("pandas", [4], [40.0]))
        proposals.append(session.state)
    assert proposals[0] == proposals[1]


@pytest.mark.parametrize("times", [[3], [2], [4, 4], [None]])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_late_ambiguous_and_missing_times_are_rejected(engine, times):
    """A continuation must never reorder committed time or guess ambiguous boundaries."""
    engineer = _engineer("LagFeatures")
    engineer.fit_transform(_frame(engine, [1, 2, 3], [10.0, 20.0, 30.0]))
    with pytest.raises(ValueError, match="Temporal history"):
        engineer.transform(_frame(engine, times, [40.0] * len(times)))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_timestamp_precision_survives_json_continuation(engine):
    """Submicrosecond event times must remain distinct in both serialized engine paths."""
    times = list(pd.date_range("2026-01-01", periods=5, freq="ns", tz="UTC"))
    engineer = _engineer("LagFeatures")
    engineer.fit_transform(_frame(engine, times[:3], [10.0, 20.0, 30.0]))
    with TemporalHistorySession("v1") as first:
        engineer.transform(_frame(engine, times[3:4], [40.0]))
    with TemporalHistorySession("v1", json.loads(json.dumps(first.state))):
        result = engineer.transform(_frame(engine, times[4:], [50.0]))
    assert result["v_lag_1"].to_list() == [40.0]


def test_wrong_model_and_oversized_context_are_rejected():
    """State cannot silently cross model versions or discard entities to fit its budget."""
    with pytest.raises(ValueError, match="different model"):
        TemporalHistorySession("v2", {"version": 1, "model_id": "v1", "steps": {}})
    engineer = _engineer("LagFeatures")
    engineer.steps_config[0]["params"]["history_max_rows"] = 2
    with pytest.raises(ValueError, match="history_max_rows"):
        engineer.fit_transform(_frame("pandas", [1, 2, 3], [10.0, 20.0, 30.0]))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_gap_rows_never_enter_fold_history(engine):
    """Excluded gap observations cannot become context for the heldout fold."""
    from sklearn.model_selection import TimeSeriesSplit

    frame = _frame(engine, list(range(12)), list(np.arange(12.0)))
    for train, heldout in TimeSeriesSplit(n_splits=2, gap=2, test_size=2).split(np.arange(12)):
        engineer = _engineer("LagFeatures")
        training = frame[train.tolist()] if engine == "polars" else frame.iloc[train]
        testing = frame[heldout.tolist()] if engine == "polars" else frame.iloc[heldout]
        engineer.fit_transform(training)
        result = engineer.transform(testing)
        assert result["v_lag_1"].to_list() == [float(train[-1]), float(heldout[0])]


@pytest.mark.parametrize("method", ["time_series_split", "nested_cv"])
@pytest.mark.parametrize("strategy", ["grid", "random", "optuna", "halving_grid", "halving_random"])
def test_temporal_cv_refits_history_inside_each_fold(method, strategy):
    """Ordinary and nested tuning must retain time for lag creation and exclude it from models."""
    from skyulf.modeling._tuning.engine import TuningCalculator
    from skyulf.modeling._tuning.schemas import TuningConfig
    from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
    from skyulf.registry import NodeRegistry

    frame = _frame("pandas", list(pd.date_range("2026-01-01", periods=40)), np.arange(40.0))
    frame = frame.drop(columns="g")
    steps = deepcopy(_engineer("RollingAggregate").steps_config)
    steps[0]["params"]["group_by"] = None
    adapter = FeatureEngineerFoldAdapter(steps, "target")
    config = TuningConfig(
        strategy=strategy,
        metric="mse",
        search_space={"alpha": [1.0]},
        cv_type=method,
        cv_nested_type="time_series_split",
        cv_folds=2,
        cv_inner_folds=2,
        cv_time_column="t",
        cv_shuffle=False,
        n_trials=1,
        strategy_params={"factor": 2, "min_resources": "exhaust", "pruner": "none"},
    )
    result = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).tune(
        frame,
        pd.Series(np.arange(40.0), name="target"),
        config,
        preprocessing=adapter,
        preprocessing_frames=(frame, pd.Series(np.arange(40.0), name="target")),
    )
    assert np.isfinite(result.best_score)
    if strategy == "grid":
        model, fitted = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
            frame,
            pd.Series(np.arange(40.0), name="target"),
            config,
            preprocessing=adapter,
        )
        assert model.n_features_in_ == 2
        assert np.isfinite(fitted.best_score)


def test_pre_split_history_and_target_history_are_rejected():
    """A saved tail must not include heldout rows or future target labels."""
    from skyulf.leakage import step_learns_from_data, validate_temporal_target

    assert step_learns_from_data("LagFeatures", {"history_mode": "carry"})
    with pytest.raises(ValueError, match="prediction target"):
        validate_temporal_target(
            "LagFeatures", {"history_mode": "carry", "columns": ["target"]}, target_column="target"
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_chained_history_does_not_expand_scaler_training_population(engine):
    """Each temporal step stores its own inputs while subsequent fits see requested rows only."""
    steps = deepcopy(_engineer("RollingAggregate").steps_config)
    second = deepcopy(steps[0])
    second["name"] = "second"
    second["params"]["columns"] = ["v_roll_mean_2"]
    steps += [
        second,
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["v"]}},
    ]
    engineer = FeatureEngineer(steps)
    train = _frame(engine, [1, 2, 3], [10.0, 20.0, 30.0])
    output, _ = engineer.fit_transform(SplitDataset(train=train, test=_frame(engine, [4], [40.0])))
    assert len(output.train) == 3
    assert len(output.test) == 1
    assert output.test["v_roll_mean_2_roll_mean_2"].to_list() == [30.0]
    np.testing.assert_allclose(output.test["v"].to_list(), [20.0 / np.std([10.0, 20.0, 30.0])])


def test_history_continuation_rejects_missing_steps():
    """An incomplete continuation cannot silently reset part of a fitted feature chain."""
    engineer = _engineer("RollingAggregate")
    engineer.fit_transform(_frame("pandas", [1, 2, 3], [10.0, 20.0, 30.0]))
    with (
        pytest.raises(ValueError, match="missing a fitted step"),
        TemporalHistorySession("v1", {"version": 1, "model_id": "v1", "steps": {}}),
    ):
        engineer.transform(_frame("pandas", [4], [40.0]))


def test_tuning_without_cv_uses_explicit_later_validation_history():
    """Disabling CV must still allow tuning against an explicitly later validation batch."""
    from skyulf.modeling._tuning.engine import TuningCalculator
    from skyulf.modeling._tuning.schemas import TuningConfig
    from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
    from skyulf.registry import NodeRegistry

    steps = deepcopy(_engineer("RollingAggregate").steps_config)
    steps += [
        {
            "name": "drop_keys",
            "transformer": "DropMissingColumns",
            "params": {"columns": ["t", "g"], "missing_threshold": None},
        }
    ]
    adapter = FeatureEngineerFoldAdapter(steps, "target")
    train = _frame("pandas", list(range(20)), list(np.arange(20.0)))
    heldout = _frame("pandas", list(range(20, 25)), list(np.arange(20.0, 25.0)))
    y = pd.Series(np.arange(20.0), name="target")
    valid_y = pd.Series(np.arange(20.0, 25.0), name="target")
    result = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).tune(
        train,
        y,
        TuningConfig(
            strategy="grid", metric="mse", search_space={"alpha": [1.0]}, cv_enabled=False
        ),
        preprocessing=adapter,
        preprocessing_frames=(train, y),
        validation_frames=(heldout, valid_y),
        validation_data=(heldout, valid_y),
    )
    assert np.isfinite(result.best_score)
