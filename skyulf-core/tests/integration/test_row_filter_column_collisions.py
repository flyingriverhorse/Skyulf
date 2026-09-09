"""Row filtering must preserve user columns that resemble internal row indices."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.preprocessing.drop_and_missing.deduplicate import (
    DeduplicateApplier,
    DeduplicateCalculator,
)
from skyulf.preprocessing.drop_and_missing.drop_rows import (
    DropMissingRowsApplier,
    DropMissingRowsCalculator,
)


def _frame(engine: str, data: dict[str, list]) -> Any:
    """Build native or wrapped inputs without changing user column names."""
    if engine == "pandas":
        return pd.DataFrame(data, index=[7] * len(next(iter(data.values()))))
    frame = pl.DataFrame(data)
    return SkyulfPolarsWrapper(frame) if engine == "wrapped-polars" else frame


def _assert_selected(actual: Any, source: Any, positions: list[int]) -> None:
    """Check selected values, all column names, and the caller's frame type."""
    assert type(actual) is type(source)
    if isinstance(actual, SkyulfPolarsWrapper):
        actual, source = actual.to_native(), source.to_native()
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, source.iloc[positions])
    else:
        assert actual.equals(source.gather(positions))


@pytest.mark.parametrize("engine", ["pandas", "polars", "wrapped-polars"])
@pytest.mark.parametrize("with_y", [False, True])
@pytest.mark.parametrize(
    "config,positions",
    [
        pytest.param({}, [0], id="any"),
        pytest.param({"how": "all"}, [0, 1, 2], id="all"),
        pytest.param({"subset": ["__idx__"]}, [0, 2], id="subset"),
        pytest.param({"threshold": 2}, [0], id="absolute-threshold"),
        pytest.param({"missing_threshold": 50}, [0, 1, 2], id="percentage-threshold"),
    ],
)
def test_drop_missing_preserves_index_named_feature(
    engine: str, with_y: bool, config: dict[str, Any], positions: list[int]
) -> None:
    """A real __idx__ feature participates in missingness and survives filtering."""
    X = _frame(engine, {"__idx__": [1.0, None, 3.0, np.nan], "value": [10.0, 20.0, None, None]})
    y = [100, 200, 300, 400]
    artifact = DropMissingRowsCalculator().fit(X, config)
    result = DropMissingRowsApplier().apply((X, y) if with_y else X, artifact)
    X_out = result[0] if with_y else result
    _assert_selected(X_out, X, positions)
    if with_y:
        assert result[1] == [y[position] for position in positions]


@pytest.mark.parametrize("engine", ["pandas", "polars", "wrapped-polars"])
@pytest.mark.parametrize("feature_name", ["feature", "__idx__"])
@pytest.mark.parametrize(
    "calculator,applier,values,config,positions",
    [
        pytest.param(
            DropMissingRowsCalculator,
            DropMissingRowsApplier,
            [1.0, None, 3.0, None],
            {},
            [0, 2],
            id="drop-missing",
        ),
        pytest.param(
            DropMissingRowsCalculator,
            DropMissingRowsApplier,
            [None, None, None, None],
            {},
            [],
            id="drop-all-missing",
        ),
        pytest.param(
            DeduplicateCalculator,
            DeduplicateApplier,
            [1.0, 1.0, 2.0, 3.0],
            {"keep": "first"},
            [0, 2, 3],
            id="keep-first",
        ),
        pytest.param(
            DeduplicateCalculator,
            DeduplicateApplier,
            [1.0, 1.0, 2.0, 3.0],
            {"keep": "last"},
            [1, 2, 3],
            id="keep-last",
        ),
        pytest.param(
            DeduplicateCalculator,
            DeduplicateApplier,
            [1.0, 1.0, 2.0, 3.0],
            {"keep": "none"},
            [2, 3],
            id="keep-none",
        ),
        pytest.param(
            DeduplicateCalculator,
            DeduplicateApplier,
            [1.0, 1.0, 1.0, 1.0],
            {"keep": "none"},
            [],
            id="drop-all-duplicates",
        ),
    ],
)
def test_row_filter_preserves_index_named_multioutput_target(
    engine: str,
    feature_name: str,
    calculator: Any,
    applier: Any,
    values: list,
    config: dict[str, Any],
    positions: list[int],
) -> None:
    """Multi-output targets keep every user column aligned with surviving features."""
    X = _frame(engine, {feature_name: values})
    target_engine = "polars" if engine == "wrapped-polars" else engine
    y = _frame(target_engine, {"__idx__": [11, 22, 33, 44], "other": [101, 202, 303, 404]})
    artifact = calculator().fit(X, config)
    X_out, y_out = applier().apply((X, y), artifact)
    _assert_selected(X_out, X, positions)
    _assert_selected(y_out, y, positions)


@pytest.mark.parametrize("engine", ["pandas", "polars", "wrapped-polars"])
def test_deduplicate_subset_ignores_index_named_nonkey_columns(engine: str) -> None:
    """Tracking positions must not turn nonkey user columns into deduplication keys."""
    X = _frame(engine, {"__idx__": [11, 22, 33, 44], "__idx___": [5, 6, 7, 8], "key": [1, 1, 2, 2]})
    y = np.array([[10, 100], [20, 200], [30, 300], [40, 400]])
    artifact = DeduplicateCalculator().fit(X, {"subset": ["key"], "keep": "last"})
    X_out, y_out = DeduplicateApplier().apply((X, y), artifact)
    _assert_selected(X_out, X, [1, 3])
    np.testing.assert_array_equal(y_out, [[20, 200], [40, 400]])
