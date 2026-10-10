"""Training split markers stay separate from direct feature/target extraction."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.data.dataset import SplitDataset
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry


def _frame(engine):
    """Keep native schemas, null features and duplicate pandas row labels visible."""
    frame = pd.DataFrame({"x": [0.0, None, 2.0, 3.0, 4.0, 5.0], "target": [0, 1] * 3})
    frame.index = [9, 2, 9, -1, 8, 0]
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_feature_target_metadata_uses_genuine_saved_state_without_execution(engine, monkeypatch):
    """The native pair operation has row context without executing the training boundary."""
    node = "feature_target_split"
    config = {"target_column": np.str_("target")}
    state = NodeRegistry.get_calculator(node)().fit(_frame(engine), config)
    applier: Any = NodeRegistry.get_applier(node)
    before = artifact_digest(state)

    def forbidden(*args, **kwargs):
        """Metadata inspection must not invoke native splitting or refit."""
        raise AssertionError("Unexpected split execution")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, config, state, engine=engine) == (
        ExecutionCapability(engine, "apply", "local", "preserve", "row")
    )
    assert applier.validate_inference_state(state) is state
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("config", [{}, {"target_column": None}, {"target_column": ""}])
def test_unusable_feature_target_fit_state_is_not_certified(config, engine):
    """Fit can retain an absent target; inspection must preserve the native apply rejection."""
    node = "feature_target_split"
    frame = _frame(engine)
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    with pytest.raises(ValueError, match="Target column"):
        NodeRegistry.get_applier(node)().apply(frame, state)
    with pytest.raises(ValueError):
        get_inference_capability(node, config, state, engine=engine)


def _assert_equal(actual, expected):
    """Compare native schema, values and pandas labels without tolerance or conversion."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    elif isinstance(actual, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    elif isinstance(actual, pl.DataFrame):
        assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_series_equal(actual, expected, check_exact=True)


def _slice(frame, positions):
    """Select positions without resetting duplicate pandas row labels."""
    return (
        frame.iloc[positions] if isinstance(frame, (pd.DataFrame, pd.Series)) else frame[positions]
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_native_feature_target_pairs_keep_rows_schema_and_values_without_refit(engine, monkeypatch):
    """Direct extraction must preserve both X and y across null, chunk and empty requests."""
    node = "feature_target_split"
    frame = _frame(engine)
    target = [0.0, None, 0.0, 1.0, 0.0, 1.0]
    if engine == "polars":
        frame = frame.with_columns(pl.Series("target", target))
    else:
        frame["target"] = target
    config = {"target_column": "target"}
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    applier = NodeRegistry.get_applier(node)()
    before, original = artifact_digest(state), deepcopy(frame)

    def forbidden(*args, **kwargs):
        """A saved name never needs another fit."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    full_x, full_y = applier.apply(frame, state)
    assert list(full_x.columns) == ["x"]
    _assert_equal(full_y, frame["target"])
    for positions in ([0], [1, 2, 3], [4, 5], [5, 4, 3, 2, 1, 0], []):
        actual_x, actual_y = applier.apply(_slice(frame, positions), state)
        _assert_equal(actual_x, _slice(full_x, positions))
        _assert_equal(actual_y, _slice(full_y, positions))
    pair = (full_x, full_y)
    assert applier.apply(pair, state) is pair
    with pytest.raises(ValueError, match="not found"):
        applier.apply(full_x, state)
    _assert_equal(frame, original)
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_feature_target_split_dataset_preserves_existing_pairs_weights_and_coverage(engine):
    """Separating targets must not repartition train/test/validation or detach train weights."""
    frame = _frame(engine)
    node = "feature_target_split"
    state = NodeRegistry.get_calculator(node)().fit(frame, {"target_column": "target"})
    applier = NodeRegistry.get_applier(node)()
    ready_pair = applier.apply(frame.head(2), state)
    weights = np.array([1.0, 2.0])
    coverage = {"train": {"rows_in": 6, "rows_out": 6}}
    dataset = SplitDataset(
        train=ready_pair,
        test=frame.head(3),
        validation=(frame.tail(1), None),
        train_sample_weight=weights,
        evaluation_coverage=coverage,
    )
    result = applier.apply(dataset, state)
    assert result.train is ready_pair
    assert result.train_sample_weight is weights
    assert result.evaluation_coverage == coverage and result.evaluation_coverage is not coverage
    for actual, source in ((result.test, frame.head(3)), (result.validation, frame.tail(1))):
        expected = applier.apply(source, state)
        _assert_equal(actual[0], expected[0])
        _assert_equal(actual[1], expected[1])
    assert dataset.test is not result.test


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_target_only_frame_retains_native_zero_feature_limitation(engine):
    """Polars zero-width features lose height even though the extracted target retains rows."""
    frame = pd.DataFrame({"target": [0, 1]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"target_column": "target"}
    state = NodeRegistry.get_calculator("feature_target_split")().fit(frame, config)
    features, target = NodeRegistry.get_applier("feature_target_split")().apply(frame, state)
    assert features.shape == ((0, 0) if engine == "polars" else (2, 0))
    assert target.to_list() == [0, 1]
    # Context describes native extraction, not successful model input shape for every frame.
    capability = get_inference_capability("feature_target_split", config, state, engine=engine)
    assert capability is not None and capability.context == "row"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target", [0, False, np.str_("target"), " "])
def test_feature_target_fit_keeps_its_existing_name_conversion(target, engine):
    """Configured falsey numbers become legitimate string names before saved-state inspection."""
    name = str(target)
    frame = pd.DataFrame({"x": [1, 2], name: [3, 4]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    state = NodeRegistry.get_calculator("feature_target_split")().fit(
        frame, {"target_column": target}
    )
    applier: Any = NodeRegistry.get_applier("feature_target_split")()
    assert state == {"type": "feature_target_split", "target_column": name}
    assert applier.validate_inference_state(state) is state
    _, output = applier.apply(frame, state)
    _assert_equal(output, frame[name])


@pytest.mark.parametrize(
    "state",
    [
        None,
        {},
        {"type": "wrong", "target_column": "y"},
        {"type": "feature_target_split", "target_column": ["y"]},
        {"type": "feature_target_split", "target_column": "y", "extra": True},
    ],
)
def test_feature_target_rejects_malformed_saved_state_without_mutation(state):
    """Malformed recipes cannot acquire a row declaration by being treated as no-ops."""
    before = deepcopy(state)
    applier: Any = NodeRegistry.get_applier("feature_target_split")
    with pytest.raises(ValueError):
        applier.validate_inference_state(state)
    assert state == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_direct_feature_target_pair_is_outside_frame_probe_shape(engine):
    """A synthetic active record cannot stand in for a genuinely omitted pipeline marker."""
    node = "feature_target_split"
    config = {"target_column": "target"}
    frame = _frame(engine)
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    record = {
        "name": "target",
        "type": node,
        "params": config,
        "artifact": state,
        "applier": NodeRegistry.get_applier(node)(),
    }
    detail, output = _probe_step(
        record,
        {"name": "target", "transformer": node, "params": config},
        frame,
        engine,
        (1, 3),
        (32, 1_000_000),
        active=True,
        project_sha=None,
    )
    assert detail["state_validation"] == "node_owned"
    assert detail["context"] == "row" and detail["status"] == "failed"
    assert detail["checks"] == [
        {"name": "full", "status": "failed", "reason": "invalid_output_frame_or_budget"}
    ]
    _assert_equal(output, frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["TrainTestSplitter", "Split"])
def test_native_train_test_split_remains_partitioned_and_undeclared(node, engine):
    """Training partitions require dataset context and cannot promise a frame row effect."""
    frame = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    calculator = NodeRegistry.get_calculator(node)()
    applier: Any = NodeRegistry.get_applier(node)()
    state = calculator.fit(frame, {})
    assert state == {"type": "split"}
    before, original = artifact_digest(state), deepcopy(frame)
    result = applier.apply(frame, state)
    repeat = applier.apply(frame, state)
    assert isinstance(result, SplitDataset)
    assert len(result.train) == 16 and len(result.test) == 4 and result.validation is None
    _assert_equal(result.train, repeat.train)
    _assert_equal(result.test, repeat.test)
    assert get_inference_capability(node, {}, state, engine=engine) is None
    assert "inference_capability" not in vars(type(applier))
    for count in (0, 1):
        with pytest.raises(ValueError):
            applier.apply(frame.head(count), state)
    _assert_equal(frame, original)
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_native_split_keeps_saved_options_and_weighted_xy_alignment(engine):
    """Native optional validation and stratification stay training behavior, with shared row gathers."""
    frame = pd.DataFrame({"x": range(24), "target": [0, 1] * 12})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {
        "test_size": np.float64(0.25),
        "validation_size": 0.25,
        "random_state": np.int64(7),
        "shuffle": np.bool_(True),
        "stratify": True,
        "target_column": "target",
        "_route": "ignored",
    }
    state = NodeRegistry.get_calculator("TrainTestSplitter")().fit(frame, config)
    assert "_route" not in state and state["random_state"] is config["random_state"]
    before = artifact_digest(state)
    result = NodeRegistry.get_applier("TrainTestSplitter")().apply(
        frame, state, sample_weight=np.arange(24) + 1.0
    )
    assert len(result.train[0]) == 12 and len(result.test[0]) == len(result.validation[0]) == 6
    for features, target in (result.train, result.test, result.validation):
        assert target.to_list() == [value % 2 for value in features["x"].to_list()]
    assert result.train_sample_weight.tolist() == [
        value + 1.0 for value in result.train[0]["x"].to_list()
    ]
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["TrainTestSplitter", "Split", "feature_target_split"])
def test_actual_engineer_omits_split_artifacts_and_never_splits_prediction(
    node, engine, monkeypatch
):
    """Training recipes persist as topology while actual fitted inference history has no split record."""
    params = (
        {"target_column": "target", "test_size": 0.25}
        if node != "feature_target_split"
        else {"target_column": "target"}
    )
    steps = [{"name": "boundary", "transformer": node, "params": params}]
    engineer = FeatureEngineer(steps)
    snapshots = []
    transformed, _ = engineer.fit_transform(_frame(engine), on_split=snapshots.append)
    assert isinstance(transformed, tuple if node == "feature_target_split" else SplitDataset)
    assert len(snapshots) == (0 if node == "feature_target_split" else 1)
    assert engineer.steps_config == steps and engineer.fitted_steps == []
    before = artifact_digest(engineer.steps_config)

    def forbidden(*args, **kwargs):
        """Inference must not execute an unrecorded training recipe."""
        raise AssertionError("Training boundary executed during prediction")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier(node), "apply", forbidden)
    sample = _frame(engine).select("x") if engine == "polars" else _frame(engine)[["x"]]
    assert engineer.transform(sample, preserve_rows=True) is sample
    assert engineer.transform(sample.head(1), preserve_rows=True).shape == (1, 1)
    assert engineer.transform(sample.head(0), preserve_rows=True).shape == (0, 1)
    assert len(snapshots) == (0 if node == "feature_target_split" else 1)
    assert engineer.fitted_steps == [] and artifact_digest(engineer.steps_config) == before


@pytest.mark.parametrize("node", ["TrainTestSplitter", "Split", "feature_target_split"])
def test_split_metadata_does_not_admit_spark_or_partition_execution(node):
    """Local native behavior must not extend worker capability declarations."""
    config = {"target_column": "target"}
    state = NodeRegistry.get_calculator(node)().fit(_frame("pandas"), config)
    assert get_inference_capability(node, config, state, engine="spark") is None
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", "pandas", config=config, execution_kind="python_batch")
