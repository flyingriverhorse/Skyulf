"""Zero-column wrapper operations must preserve sample cardinality."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.engines.registry import EngineRegistry
from skyulf.engines.sklearn_bridge import SklearnBridge


@pytest.fixture(params=["pandas", "polars"])
def wrapped_frame(request):
    """Use three identifiable samples to distinguish no features from no samples."""
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    engine = EngineRegistry.get(request.param)
    return engine.wrap(engine.from_pandas(frame))


@pytest.mark.parametrize("operation", ["select", "drop", "chained_drop"])
def test_empty_column_projection_preserves_rows_through_bridges(wrapped_frame, operation):
    """Removing every feature must keep all samples through wrapping and conversion."""
    if operation == "select":
        empty = wrapped_frame.select([])
    elif operation == "drop":
        empty = wrapped_frame.drop(["a", "b"])
    else:
        empty = wrapped_frame.drop(["a"]).drop(["b"])

    assert empty.shape == (3, 0)
    assert len(empty) == 3
    assert empty.columns == []
    assert empty.to_native().shape == (3, 0)
    assert empty.to_pandas().shape == (3, 0)
    assert empty.to_arrow().num_rows == 3
    assert empty.to_arrow().num_columns == 0

    empty = EngineRegistry.wrap(empty.select([]).drop([]).copy().to_native())
    restored = pickle.loads(pickle.dumps(empty))
    X, y = SklearnBridge.to_sklearn((restored, [0, 1, 0]))
    assert X.shape == (3, 0)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(y, [0, 1, 0])
    assert wrapped_frame.shape == (3, 2)


@pytest.mark.parametrize("values", [7, [7, 8, 9]], ids=["scalar", "sequence"])
def test_add_column_after_empty_selection_keeps_sample_count(wrapped_frame, values):
    """Restoring a feature must broadcast or assign to the existing samples."""
    empty = wrapped_frame.select([])
    result = empty.with_column("c", values)
    expected = [7, 7, 7] if isinstance(values, int) else [7, 8, 9]
    assert result.shape == (3, 1)
    assert result.to_pandas()["c"].tolist() == expected
    assert empty.shape == (3, 0)


def test_add_wrong_length_column_after_empty_selection_rejects(wrapped_frame):
    """An empty feature set must not let a new column silently replace sample count."""
    empty = wrapped_frame.select([])
    with pytest.raises((ValueError, pl.exceptions.ShapeError)):
        empty.with_column("c", [7, 8])
    assert empty.shape == (3, 0)


@pytest.mark.parametrize("engine_name", ["pandas", "polars"])
def test_zero_sample_projection_stays_empty(engine_name):
    """Retaining height must not invent samples for genuinely empty input."""
    engine = EngineRegistry.get(engine_name)
    empty = engine.wrap(engine.from_pandas(pd.DataFrame({"a": []}))).select([])
    assert isinstance(empty, SkyulfPandasWrapper | SkyulfPolarsWrapper)
    X, _ = SklearnBridge.to_sklearn(empty.drop([]).copy())
    assert X.shape == (0, 0)
    assert empty.to_pandas().shape == (0, 0)


@pytest.mark.parametrize(
    ("key", "rows"),
    [
        ((slice(None), []), 3),
        ((slice(1, 3), []), 2),
        (([2, 0], []), 2),
        (([], []), 0),
        ((1, []), 1),
        ((slice(None), slice(0, 0)), 3),
        ([False], 3),
        (pl.Series([False]), 3),
        (np.array([False]), 3),
        (pl.Series([], dtype=pl.String), 3),
    ],
)
def test_polars_item_empty_projection_respects_selected_rows(key, rows):
    """Column indexing must preserve the requested row subset without hidden columns."""
    wrapped = EngineRegistry.wrap(pl.DataFrame({"a": [1, 2, 3]}))
    native = wrapped[key]
    assert isinstance(native, pl.DataFrame)
    assert native.shape == (rows, 0)
    assert native.columns == []


def test_polars_boolean_tuple_selects_columns_without_losing_rows():
    """A boolean tuple is a native column mask, not a scalar row index."""
    wrapped = EngineRegistry.wrap(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert wrapped[False, False].shape == (3, 0)


@pytest.mark.parametrize(
    ("key", "rows"),
    [(slice(1, 3), 2), ([2, 0], 2), ([], 0), (1, 1), (slice(None, None, -1), 3)],
)
def test_polars_item_rows_after_empty_projection_respects_cardinality(key, rows):
    """Row indexing on a zero-column wrapper must retain the selected sample count."""
    wrapped = EngineRegistry.wrap(pl.DataFrame({"a": [1, 2, 3]})).select([])
    native = wrapped[key]
    assert native.shape == (rows, 0)


@pytest.mark.parametrize("key", [(3, []), (-4, []), ([3], []), (slice(None, None, 0), [])])
def test_polars_empty_projection_validates_row_selector(key):
    """An empty feature selection must not hide invalid row positions or slices."""
    wrapped = EngineRegistry.wrap(pl.DataFrame({"a": [1, 2, 3]})).select([])
    with pytest.raises((IndexError, ValueError, pl.exceptions.OutOfBoundsError)):
        wrapped[key]
    assert wrapped.shape == (3, 0)


def test_polars_filter_after_empty_selection_preserves_matching_rows():
    """Native filtering must still reduce sample count after all features are removed."""
    wrapped = EngineRegistry.wrap(pl.DataFrame({"a": [1, 2, 3]})).select([])
    filtered = EngineRegistry.wrap(wrapped.to_native().filter(pl.Series([True, False, True])))
    assert filtered.select([]).shape == (2, 0)
    assert filtered.to_arrow().num_rows == 2
    assert wrapped.shape == (3, 0)


def test_polars_empty_projection_still_validates_missing_columns():
    """Height retention must not bypass native validation of requested columns."""
    wrapped = EngineRegistry.wrap(pl.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        wrapped.drop(["a", "missing"])
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        wrapped.select(["missing"])
    assert wrapped.shape == (3, 1)
