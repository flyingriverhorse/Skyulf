"""Scalar output columns must follow the input height, including zero-column requests."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.inference._probe_frames import reverse_frame
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.preprocessing.feature_generation.interaction import (
    FeatureInteractionApplier,
    FeatureInteractionCalculator,
)
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("height", [0, 1, 4])
@pytest.mark.parametrize(
    "value,dtype",
    [(7, pl.Int32), (2.5, pl.Float64), ("fallback", pl.String), (True, pl.Boolean)],
)
def test_simple_restores_saved_columns_at_exact_input_height(lazy, height, value, dtype):
    """Restoration must neither create a row nor collapse existing rows without an input column."""
    state = SimpleImputerCalculator().fit(
        pl.DataFrame({"x": [1, 2]}),
        {"columns": ["x"], "strategy": "constant", "fill_value": value},
    )
    state_bytes = pickle.dumps(state)
    state = pickle.loads(state_bytes)
    frame = pl.DataFrame(np.empty((height, 0)))
    target = np.arange(height)
    result, output_target = SimpleImputerApplier().apply(
        (frame.lazy() if lazy else frame, target), state
    )
    if lazy:
        assert isinstance(result, pl.LazyFrame)
        result = result.collect()
    expected = pl.DataFrame({"x": pl.Series([value] * height, dtype=dtype)})
    assert_frame_equal(result, expected, check_exact=True)
    assert output_target is target
    assert frame.shape == (height, 0)
    assert pickle.dumps(state) == state_bytes


@pytest.mark.parametrize(
    "value,dtype", [(np.int64(5), pl.Int64), (np.float32(2.5), pl.Float32), (2**64 - 1, pl.UInt64)]
)
def test_simple_restoration_keeps_native_scalar_width_with_and_without_anchor(value, dtype):
    """Row-count anchoring must retain the original scalar's native width and exact large values."""
    state = SimpleImputerCalculator().fit(
        pl.DataFrame({"x": [1, 2]}),
        {"columns": ["x"], "strategy": "constant", "fill_value": value},
    )
    for frame in (pl.DataFrame(np.empty((3, 0))), pl.DataFrame({"keep": [8, 3, 1]})):
        full = SimpleImputerApplier().apply(frame, state)
        assert full["x"].dtype == dtype
        assert full["x"].to_list() == [value] * 3
        for positions in ([0], [1, 2], [2, 1, 0], []):
            assert_frame_equal(
                SimpleImputerApplier().apply(frame[positions], state), full[positions]
            )
    assert state["fill_values"]["x"] == value


def test_simple_all_null_mean_retains_null_restoration_schema():
    """A saved missing mean must keep its existing Polars Null column without creating rows."""
    state = SimpleImputerCalculator().fit(
        pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Float64)}),
        {"columns": ["x"], "strategy": "mean"},
    )
    for height in (0, 3):
        result = SimpleImputerApplier().apply(pl.DataFrame(np.empty((height, 0))), state)
        assert_frame_equal(result, pl.DataFrame({"x": pl.Series([None] * height, dtype=pl.Null)}))
    assert state["fill_values"]["x"] is None


@pytest.mark.parametrize("node", ["SimpleImputer", "FeatureInteraction"])
def test_scalar_column_plan_never_collects_lazy_input(node, monkeypatch):
    """Zero-row handling must remain a native lazy expression without executing a request early."""
    config = (
        {"columns": ["x"], "strategy": "constant", "fill_value": 7}
        if node == "SimpleImputer"
        else {"columns": [], "include_bias": True}
    )
    state = NodeRegistry.get_calculator(node)().fit(pl.DataFrame({"x": [1, 2]}), config)
    frame = pl.DataFrame().lazy()

    def forbidden(*args, **kwargs):
        """Early collection would turn a lazy input contract into eager execution."""
        raise AssertionError("Apply must not collect lazy inputs")

    with monkeypatch.context() as patch:
        patch.setattr(pl.LazyFrame, "collect", forbidden)
        result = NodeRegistry.get_applier(node)().apply(frame, state)
        assert isinstance(result, pl.LazyFrame)
    assert result.collect().shape == (0, 1)


@pytest.mark.parametrize("node", ["SimpleImputer", "FeatureInteraction"])
@pytest.mark.parametrize("height", [0, 1, 4])
def test_zero_column_saved_diagnostic_passes_exact_checks(node, height, monkeypatch):
    """A genuine saved artifact must pass row, schema, mutation and empty checks after restoration."""
    config = (
        {"columns": ["x"], "strategy": "constant", "fill_value": 7}
        if node == "SimpleImputer"
        else {"columns": [], "include_bias": True}
    )
    state = NodeRegistry.get_calculator(node)().fit(pl.DataFrame({"x": [1, 2]}), config)
    record = {
        "name": "shape",
        "type": node,
        "params": config,
        "artifact": state,
        "applier": NodeRegistry.get_applier(node)(),
    }

    def forbidden(*args, **kwargs):
        """Saved diagnostics must never refit the artifact on the incoming request."""
        raise AssertionError("Unexpected inference-time fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    detail, output = _probe_step(
        record,
        {"name": "shape", "transformer": node, "params": config},
        pl.DataFrame(np.empty((height, 0))),
        "polars",
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    assert detail["context"] == "row"
    assert detail["status"] == "passed", detail
    assert all(check["status"] == "passed" for check in detail["checks"])
    assert output.shape == (height, 1)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("height", [0, 1, 4])
@pytest.mark.parametrize("with_columns", [False, True])
def test_probe_reversal_preserves_empty_and_ordinary_native_frames(engine, height, with_columns):
    """Diagnostic reordering must not erase row count when no feature can anchor the permutation."""
    if engine == "pandas":
        frame = pd.DataFrame(index=[8, 3, 3, 1][:height])
        if with_columns:
            frame["value"] = range(height)
        original = frame.copy(deep=True)
        result = reverse_frame(frame)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, frame.iloc[::-1])
        pd.testing.assert_frame_equal(frame, original)
    else:
        frame = (
            pl.DataFrame({"value": pl.Series(range(height), dtype=pl.Int64)})
            if with_columns
            else pl.DataFrame(np.empty((height, 0)))
        )
        original = frame.clone()
        result = reverse_frame(frame)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == frame.shape
        if with_columns:
            assert result["value"].to_list() == list(reversed(range(height)))
        assert_frame_equal(frame, original)
    assert result.shape == (height, int(with_columns))


def test_interaction_keeps_existing_bias_values_and_dtype():
    """Row-count expressions must not replace a pre-existing bias column with generated ones."""
    frame = pl.DataFrame({"interaction_bias": pl.Series([7, 8], dtype=pl.Int16)})
    state = FeatureInteractionCalculator().fit(frame, {"columns": [], "include_bias": True})
    for chunk in (frame, frame.head(1), frame.head(0)):
        assert_frame_equal(FeatureInteractionApplier().apply(chunk, state), chunk)
    assert frame["interaction_bias"].to_list() == [7, 8]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("height", [0, 1, 4])
@pytest.mark.parametrize("columns", [[], ["a", "b"]])
def test_interaction_bias_preserves_height_without_present_features(lazy, height, columns):
    """Bias-only inference must preserve row count when configured product columns are absent."""
    state = FeatureInteractionCalculator().fit(
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
        {"columns": columns, "include_bias": True},
    )
    state_bytes = pickle.dumps(state)
    state = pickle.loads(state_bytes)
    frame = pl.DataFrame(np.empty((height, 0)))
    target = np.arange(height)
    result, output_target = FeatureInteractionApplier().apply(
        (frame.lazy() if lazy else frame, target), state
    )
    if lazy:
        assert isinstance(result, pl.LazyFrame)
        result = result.collect()
    assert_frame_equal(
        result, pl.DataFrame({"interaction_bias": pl.Series([1.0] * height, dtype=pl.Float64)})
    )
    assert output_target is target
    assert frame.shape == (height, 0)
    assert pickle.dumps(state) == state_bytes
