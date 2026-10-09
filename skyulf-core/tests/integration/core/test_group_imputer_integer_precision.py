"""Group imputation must retain exact learned integer values and typed group identities."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator


def _native(frame, engine):
    """Keep explicit nullable integer storage when entering either native engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _equal(actual, expected):
    """Pin exact values, row labels and nullable dtypes without a numeric tolerance."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,value",
    [("Int64", 2**53 + 1), ("Int64", 2**63 - 1), ("Int64", -(2**63)), ("UInt64", 2**64 - 1)],
)
def test_group_mode_keeps_exact_fitted_values_and_request_fills(engine, dtype, value, monkeypatch):
    """Nulls, fallback groups and request partitioning must never round saved integer modes."""
    training = _native(
        pd.DataFrame(
            {"g": ["a", "a", "a", "b"], "x": pd.Series([value, None, value, 1], dtype=dtype)}
        ),
        engine,
    )
    original = training.copy(deep=True) if engine == "pandas" else training.clone()
    state = GroupImputerCalculator().fit(
        training, {"columns": ["x"], "group_by": "g", "strategy": "mode"}
    )
    assert state["fill_values"]["x"] == value
    assert state["group_values"]["x"] == [["a", value], ["b", 1]]
    assert type(state["fill_values"]["x"]) is int
    _equal(training, original)
    state = pickle.loads(pickle.dumps(state))
    saved = pickle.dumps(state)
    frame = pd.DataFrame(
        {
            "g": ["a", "new", None, "b", "new"],
            "x": pd.Series([None, None, None, None, value], dtype=dtype),
        }
    )
    frame.index = [8, 3, 3, 7, 1]
    expected = frame.copy()
    expected["x"] = pd.array([value, value, value, 1, value], dtype=dtype)
    frame, expected = _native(frame, engine), _native(expected, engine)
    original = frame.copy(deep=True) if engine == "pandas" else frame.clone()

    def forbidden(*args, **kwargs):
        """Saved modes must replay without recalculating training statistics."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(GroupImputerCalculator, "fit", forbidden)
    applier = GroupImputerApplier()
    for positions in ([0, 1, 2, 3, 4], [0], [1], [2], [3], [4], [4, 3, 2, 1, 0], []):
        request = frame.iloc[positions] if engine == "pandas" else frame[positions]
        wanted = expected.iloc[positions] if engine == "pandas" else expected[positions]
        _equal(applier.apply(request, state), wanted)
    _equal(frame, original)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
@pytest.mark.parametrize("dtype,value", [("Int64", 2**53 + 1), ("UInt64", 2**64 - 1)])
def test_training_integer_group_keys_stay_exact_beside_nulls(engine, strategy, dtype, value):
    """Fitted group keys cannot move to a neighboring integer before statistics are stored."""
    frame = _native(
        pd.DataFrame({"g": pd.Series([value, None, 1], dtype=dtype), "x": [11.0, 22.0, 33.0]}),
        engine,
    )
    state = GroupImputerCalculator().fit(
        frame, {"columns": ["x"], "group_by": "g", "strategy": strategy}
    )
    assert state["group_values"]["x"] == [[1, 33.0], [value, 11.0]]
    sample = _native(
        pd.DataFrame({"g": pd.Series([value, value - 1, None], dtype=dtype), "x": [np.nan] * 3}),
        engine,
    )
    result = GroupImputerApplier().apply(sample, state)
    assert result["x"].to_list() == [11.0, state["fill_values"]["x"], state["fill_values"]["x"]]


@pytest.mark.parametrize(
    "dtype", ["Int64", "UInt64", "int64[pyarrow]", "uint64[pyarrow]", "object"]
)
def test_pandas_saved_integer_lookup_does_not_promote_missing_groups(dtype):
    """Existing exact artifacts must fill fallback rows without coercing their learned integers."""
    value = 2**53 + 1
    state = {
        "columns": ["x"],
        "group_by": "g",
        "strategy": "most_frequent",
        "group_values": {"x": [["a", value]]},
        "fill_values": {"x": 1},
    }
    frame = pd.DataFrame({"g": ["a", "unseen", None], "x": pd.Series([None] * 3, dtype=dtype)})
    expected = frame.copy()
    expected["x"] = pd.Series([value, 1, 1], dtype=dtype)
    _equal(GroupImputerApplier().apply(frame, state), expected)


@pytest.mark.parametrize("strategy", ["mean", "most_frequent"])
def test_pandas_mixed_numeric_group_keys_do_not_round_mapping_index(strategy):
    """A floating group key must not make another exact integer key match its rounded neighbor."""
    value = 2**53 + 1
    state = {
        "columns": ["x"],
        "group_by": "g",
        "strategy": strategy,
        "group_values": {"x": [[value, 11], [1.5, 22]]},
        "fill_values": {"x": 3},
    }
    frame = pd.DataFrame({"g": pd.Series([value, value - 1, 1.5], dtype=object), "x": [np.nan] * 3})
    result = GroupImputerApplier().apply(frame, state)
    assert result["x"].to_list() == [11.0, 3.0, 22.0]


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_group_float_outputs_keep_their_native_dtype(dtype, strategy):
    """Integer precision repairs must not turn floating imputation outputs into object columns."""
    frame = pd.DataFrame({"g": ["a", "new", "a"], "x": pd.Series([None, None, 1.25], dtype=dtype)})
    state = {
        "columns": ["x"],
        "group_by": "g",
        "strategy": strategy,
        "group_values": {"x": [["a", 3.0]]},
        "fill_values": {"x": 2.0},
    }
    expected = frame.copy()
    expected["x"] = pd.Series([3.0, 2.0, 1.25], dtype=dtype)
    _equal(GroupImputerApplier().apply(frame, state), expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype,value", [("Int64", 2**53 + 1), ("UInt64", 2**64 - 1)])
def test_group_global_integer_mode_without_observed_group_keys(engine, dtype, value):
    """An empty learned group map must not round the exact global fallback."""
    training = _native(
        pd.DataFrame({"g": [None] * 3, "x": pd.Series([value, None, value], dtype=dtype)}),
        engine,
    )
    state = GroupImputerCalculator().fit(
        training, {"columns": ["x"], "group_by": "g", "strategy": "mode"}
    )
    assert state["group_values"]["x"] == []
    frame = _native(
        pd.DataFrame({"g": [None, "new"], "x": pd.Series([None, None], dtype=dtype)}), engine
    )
    assert GroupImputerApplier().apply(frame, state)["x"].to_list() == [value, value]


@pytest.mark.parametrize("dtype", ["Int8", "Int64", "UInt64"])
def test_pandas_invalid_float_mode_retains_native_cast_error(dtype):
    """Exact integer mapping must retain rejection of an out-of-range saved float mode."""
    frame = pd.DataFrame({"g": ["a", "new", None], "x": pd.Series([None, 2, None], dtype=dtype)})
    state = {
        "columns": ["x"],
        "group_by": "g",
        "strategy": "most_frequent",
        "group_values": {"x": [["a", 1e30]]},
        "fill_values": {"x": None},
    }
    with pytest.raises(TypeError, match="cannot safely cast"):
        GroupImputerApplier().apply(frame, state)
