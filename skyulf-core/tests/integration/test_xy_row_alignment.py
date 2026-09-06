"""X/y row alignment across preprocessing nodes that reorder or drop rows.

Any node that changes X's row set must hand ``y`` the *same* row positions, or
the pair desynchronises: downstream training then consumes the two positionally
and silently learns from wrong labels, with no error and no warning. Regression
coverage for OC-163 (``sort_by`` permuted X but not y), OC-165 (pandas lag
``drop_na`` never filtered y at all) and OC-166 (the outlier helpers returned an
engine-neutral y untouched while X lost rows).

Every case runs on both engines across all four target shapes the dispatcher
accepts, because the original defects were shape- and engine-specific: the
pandas lag path ignored y for every shape, the polars lag path crashed on
numpy/list y, and the outlier helpers no-oped on numpy/list y but worked for
polars Series y — so a single-shape test would have passed either way.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing import (
    LagFeaturesApplier,
    LagFeaturesCalculator,
    RollingAggregateApplier,
    RollingAggregateCalculator,
)
from skyulf.preprocessing._helpers import select_rows_by_position
from skyulf.preprocessing.outliers.elliptic import (
    EllipticEnvelopeApplier,
    EllipticEnvelopeCalculator,
)
from skyulf.preprocessing.outliers.iqr import IQRApplier, IQRCalculator
from skyulf.preprocessing.outliers.manual_bounds import (
    ManualBoundsApplier,
    ManualBoundsCalculator,
)
from skyulf.preprocessing.outliers.zscore import ZScoreApplier, ZScoreCalculator
from skyulf.preprocessing.time_series._common import (
    sort_with_positions_pandas,
    sort_with_positions_polars,
)

# (engine, y-shape) product. ``list`` and ``numpy`` are the engine-neutral shapes
# ``_check_xy_engine_parity`` documents as always accepted; ``series``/``frame``
# are the engine-matched ones.
_ENGINE_Y_SHAPES = [
    pytest.param(engine, shape, id=f"{engine}-{shape}")
    for engine in ("pandas", "polars")
    for shape in ("list", "numpy", "series", "frame")
]


def _frame(engine: str, data: dict[str, list]) -> Any:
    """Build an X frame of the requested engine."""
    return pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)


def _make_y(engine: str, shape: str, values: list) -> Any:
    """Build a target of the requested shape for the given engine."""
    if shape == "list":
        return list(values)
    if shape == "numpy":
        return np.asarray(values)
    if shape == "series":
        return pl.Series(values) if engine == "polars" else pd.Series(values)
    return _frame(engine, {"target": values})


def _as_list(y: Any) -> list:
    """Flatten any supported target shape to a plain list for comparison."""
    if isinstance(y, pl.DataFrame):
        return y.to_series().to_list()
    if isinstance(y, pl.Series):
        return y.to_list()
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0].tolist()
    if isinstance(y, np.ndarray):
        return y.reshape(-1).tolist()
    if isinstance(y, pd.Series):
        return y.tolist()
    return list(y)


def _col(frame: Any, name: str) -> list:
    """Read one column of a pandas or polars frame as a list."""
    return frame[name].to_list() if isinstance(frame, pl.DataFrame) else frame[name].tolist()


# ---------------------------------------------------------------------------
# OC-163 — sort_by must permute y identically to X
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_lag_sort_by_permutes_y_with_x(engine: str, y_shape: str) -> None:
    """``LagFeatures`` sorting X by time must carry y through the same permutation.

    ``y`` mirrors the time column scaled by 100, so a correct permutation is
    unambiguous: rows sorted to time ``[1, 2, 3]`` must carry ``[100, 200, 300]``.
    Pre-fix every shape returned the original ``[300, 100, 200]``.
    """
    X = _frame(engine, {"time": [3, 1, 2], "value": [30, 10, 20]})
    y = _make_y(engine, y_shape, [300, 100, 200])
    art = LagFeaturesCalculator().fit(X, {"columns": ["value"], "lags": [1], "sort_by": "time"})
    X_out, y_out = LagFeaturesApplier().apply((X, y), art)
    assert _col(X_out, "time") == [1, 2, 3]
    assert _as_list(y_out) == [100, 200, 300]


@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_rolling_sort_by_permutes_y_with_x(engine: str, y_shape: str) -> None:
    """``RollingAggregate`` must permute y with X, exactly like the lag node."""
    X = _frame(engine, {"time": [3, 1, 2], "value": [30, 10, 20]})
    y = _make_y(engine, y_shape, [300, 100, 200])
    art = RollingAggregateCalculator().fit(
        X, {"columns": ["value"], "window": 2, "sort_by": "time", "aggregations": ["mean"]}
    )
    X_out, y_out = RollingAggregateApplier().apply((X, y), art)
    assert _col(X_out, "time") == [1, 2, 3]
    assert _as_list(y_out) == [100, 200, 300]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_sort_without_y_still_sorts_x(engine: str) -> None:
    """A frame-only input (no target) must keep sorting X exactly as before."""
    X = _frame(engine, {"time": [3, 1, 2], "value": [30, 10, 20]})
    art = LagFeaturesCalculator().fit(X, {"columns": ["value"], "lags": [1], "sort_by": "time"})
    X_out = LagFeaturesApplier().apply(X, art)
    assert _col(X_out, "time") == [1, 2, 3]


@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_missing_sort_by_leaves_y_untouched(engine: str, y_shape: str) -> None:
    """Without ``sort_by`` no row is reordered, so y must come back unchanged."""
    X = _frame(engine, {"time": [3, 1, 2], "value": [30, 10, 20]})
    y = _make_y(engine, y_shape, [300, 100, 200])
    art = LagFeaturesCalculator().fit(X, {"columns": ["value"], "lags": [1]})
    _, y_out = LagFeaturesApplier().apply((X, y), art)
    assert _as_list(y_out) == [300, 100, 200]


# ---------------------------------------------------------------------------
# OC-165 — lag drop_na must filter y, on both engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_lag_drop_na_filters_y(engine: str, y_shape: str) -> None:
    """``drop_na`` removes the first (null-lag) row from X, so y must lose it too.

    Pre-fix the pandas path returned all three targets beside two feature rows;
    the polars path raised ``AttributeError`` for numpy and list targets because
    it called ``.filter`` on them directly.
    """
    X = _frame(engine, {"value": [10, 20, 30]})
    y = _make_y(engine, y_shape, [100, 200, 300])
    art = LagFeaturesCalculator().fit(X, {"columns": ["value"], "lags": [1], "drop_na": True})
    X_out, y_out = LagFeaturesApplier().apply((X, y), art)
    assert _col(X_out, "value") == [20, 30]
    assert _as_list(y_out) == [200, 300]


@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_lag_sort_by_and_drop_na_compose(engine: str, y_shape: str) -> None:
    """Sorting then dropping null lags must apply both row changes to y in order.

    The two operations compose: sort ``[3, 1, 2]`` to ``[1, 2, 3]``, then drop the
    first row whose lag is null, leaving times ``[2, 3]`` and targets ``[200, 300]``.
    """
    X = _frame(engine, {"time": [3, 1, 2], "value": [30, 10, 20]})
    y = _make_y(engine, y_shape, [300, 100, 200])
    art = LagFeaturesCalculator().fit(
        X, {"columns": ["value"], "lags": [1], "sort_by": "time", "drop_na": True}
    )
    X_out, y_out = LagFeaturesApplier().apply((X, y), art)
    assert _col(X_out, "time") == [2, 3]
    assert _as_list(y_out) == [200, 300]


def test_lag_drop_na_survives_duplicate_pandas_index() -> None:
    """Row selection must stay positional: a duplicated index must not expand y.

    This is the OC-12 failure mode — label-based selection returns every row
    matching a duplicated label — reaching the lag node instead of the drop node.
    """
    X = pd.DataFrame({"value": [10, 20, 30]}, index=pd.Index([0, 0, 1]))
    y = pd.Series([100, 200, 300], index=pd.Index([0, 0, 1]))
    art = LagFeaturesCalculator().fit(X, {"columns": ["value"], "lags": [1], "drop_na": True})
    X_out, y_out = LagFeaturesApplier().apply((X, y), art)
    assert len(X_out) == 2
    assert y_out.tolist() == [200, 300]


# ---------------------------------------------------------------------------
# OC-166 — outlier nodes must filter every target shape
# ---------------------------------------------------------------------------

_OUTLIER_NODES = [
    pytest.param(IQRCalculator, IQRApplier, {"columns": ["x"]}, id="IQR"),
    pytest.param(ZScoreCalculator, ZScoreApplier, {"columns": ["x"], "threshold": 1}, id="ZScore"),
    pytest.param(
        ManualBoundsCalculator,
        ManualBoundsApplier,
        {"bounds": {"x": {"lower": 0, "upper": 10}}},
        id="ManualBounds",
    ),
]

_OUTLIERS = [1.0, 2.0, 3.0, 4.0, 100.0]
_OUTLIER_Y = [10, 20, 30, 40, 1000]


@pytest.mark.parametrize("calc,applier,config", _OUTLIER_NODES)
@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_outlier_nodes_filter_y_in_sync(
    calc: Any, applier: Any, config: dict, engine: str, y_shape: str
) -> None:
    """Each outlier node must drop y's outlier row on both engines, every shape.

    Pre-fix the polars helper returned numpy and list targets with all five rows
    beside a four-row X; the pandas helper raised ``TypeError`` on a list target.
    """
    X = _frame(engine, {"x": list(_OUTLIERS)})
    y = _make_y(engine, y_shape, list(_OUTLIER_Y))
    art = calc().fit(X, config)
    X_out, y_out = applier().apply((X, y), art)
    assert len(X_out) == 4
    assert _as_list(y_out) == [10, 20, 30, 40]


@pytest.mark.parametrize("engine,y_shape", _ENGINE_Y_SHAPES)
def test_elliptic_envelope_filters_y_in_sync(engine: str, y_shape: str) -> None:
    """``EllipticEnvelope`` must filter y too — it had its own inline pass-through.

    Its polars branch used ``y.filter(mask) if hasattr(y, "filter") else y``, a
    fourth copy of the OC-166 defect written inline rather than in the shared
    helper, so numpy and list targets kept all eight rows beside a seven-row X.
    """
    data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0]
    X = _frame(engine, {"x": data})
    y = _make_y(engine, y_shape, list(range(8)))
    art = EllipticEnvelopeCalculator().fit(X, {"columns": ["x"], "contamination": 0.125})
    X_out, y_out = EllipticEnvelopeApplier().apply((X, y), art)
    assert len(X_out) == len(_as_list(y_out))
    assert _as_list(y_out) == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# The shared helper, and the sort positions that feed it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param([100, 200, 300], [300, 100, 200], id="list"),
        pytest.param(np.array([100, 200, 300]), [300, 100, 200], id="numpy"),
        pytest.param(pd.Series([100, 200, 300]), [300, 100, 200], id="pandas-series"),
        pytest.param(pd.DataFrame({"t": [100, 200, 300]}), [300, 100, 200], id="pandas-frame"),
        pytest.param(pl.Series([100, 200, 300]), [300, 100, 200], id="polars-series"),
        pytest.param(pl.DataFrame({"t": [100, 200, 300]}), [300, 100, 200], id="polars-frame"),
    ],
)
def test_select_rows_by_position_handles_every_target_shape(y: Any, expected: list) -> None:
    """One position array must reorder all six frame/array target shapes alike."""
    assert _as_list(select_rows_by_position(y, np.array([2, 0, 1]))) == expected


def test_select_rows_by_position_preserves_type_and_2d_rows() -> None:
    """Type is preserved, and a 2-D multi-output target is selected row-wise."""
    assert isinstance(select_rows_by_position([1, 2, 3], np.array([2])), list)
    out = select_rows_by_position(np.array([[1, 10], [2, 20], [3, 30]]), np.array([2, 0]))
    assert out.tolist() == [[3, 30], [1, 10]]


def test_select_rows_by_position_passes_through_none() -> None:
    """``None`` for either argument means no row changed, so y is returned as-is."""
    sentinel = object()
    assert select_rows_by_position(None, np.array([0])) is None
    assert select_rows_by_position(sentinel, None) is sentinel


def test_select_rows_by_position_rejects_unknown_shape() -> None:
    """An unselectable target must raise, not silently desync from X."""
    with pytest.raises(TypeError, match="Cannot select rows from y"):
        select_rows_by_position({"a": 1}, np.array([0]))


def test_pandas_sort_positions_accept_polars_positions_too() -> None:
    """A polars position Series must index non-polars targets (cross-engine positions)."""
    order = pl.Series([2, 0, 1], dtype=pl.UInt32)
    assert _as_list(select_rows_by_position(np.array([100, 200, 300]), order)) == [300, 100, 200]
    assert select_rows_by_position([100, 200, 300], order) == [300, 100, 200]


@pytest.mark.parametrize(
    "sort_by,expected_positions",
    [
        pytest.param("t", [1, 3, 2, 0, 4], id="ties-are-stable"),
        pytest.param("n", [2, 4, 0, 1, 3], id="nulls-last"),
        pytest.param(None, None, id="no-sort"),
        pytest.param("absent", None, id="unknown-column"),
    ],
)
def test_pandas_sort_positions_match_sort_values(
    sort_by: str | None, expected_positions: list | None
) -> None:
    """The returned positions must reproduce ``sort_values`` exactly, ties and nulls included.

    Two independent sort implementations could drift; deriving the positions from
    pandas' own ``sort_values`` is what makes X and y agree by construction.
    """
    df = pd.DataFrame(
        {"t": [3, 1, 2, 1, 3], "n": [3.0, np.nan, 1.0, np.nan, 2.0], "v": list("abcde")}
    )
    out, positions = sort_with_positions_pandas(df, sort_by)
    if expected_positions is None:
        assert positions is None
        assert out is df
        return
    assert positions.tolist() == expected_positions
    expected = df.sort_values(sort_by, kind="mergesort")
    assert out.index.tolist() == expected.index.tolist()
    assert out.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_pandas_sort_positions_survive_duplicate_index() -> None:
    """Positions must be positional, so a duplicated index cannot expand or misorder."""
    df = pd.DataFrame({"t": [3, 1, 2], "v": list("abc")}, index=pd.Index([0, 0, 1]))
    out, positions = sort_with_positions_pandas(df, "t")
    assert positions.tolist() == [1, 2, 0]
    assert out["v"].tolist() == ["b", "c", "a"]
    assert out.index.tolist() == [0, 1, 0]


def test_pandas_sort_positions_handle_nullable_and_string_keys() -> None:
    """Nullable Int64 and string keys must both yield positions, not raise."""
    nullable = pd.DataFrame({"t": pd.Series([3, None, 1], dtype="Int64"), "v": list("abc")})
    _, positions = sort_with_positions_pandas(nullable, "t")
    assert positions.tolist() == [2, 0, 1]
    strings = pd.DataFrame({"t": ["c", "a", "b"], "v": [1, 2, 3]})
    _, str_positions = sort_with_positions_pandas(strings, "t")
    assert str_positions.tolist() == [1, 2, 0]


@pytest.mark.parametrize(
    "sort_by,expected_order",
    [
        pytest.param("t", [1, 3, 2, 0, 4], id="ties-are-stable"),
        pytest.param("n", [2, 4, 0, 1, 3], id="nulls-last"),
        pytest.param(None, None, id="no-sort"),
        pytest.param("absent", None, id="unknown-column"),
    ],
)
def test_polars_sort_positions_match_dataframe_sort(
    sort_by: str | None, expected_order: list | None
) -> None:
    """``X.gather(order)`` must equal the ``X.sort(...)`` it replaces, on every key type."""
    X = pl.DataFrame({"t": [3, 1, 2, 1, 3], "n": [3.0, None, 1.0, None, 2.0], "v": list("abcde")})
    out, order = sort_with_positions_polars(X, sort_by)
    if expected_order is None:
        assert order is None
        assert out is X
        return
    assert order.to_list() == expected_order
    assert out.equals(X.sort(sort_by, nulls_last=True, maintain_order=True))


def test_polars_sort_positions_handle_date_keys() -> None:
    """A date sort key must order correctly, matching the pandas path's semantics."""
    X = pl.DataFrame(
        {
            "t": pd.to_datetime(["2021-03-01", "2021-01-01", "2021-02-01"]),
            "v": list("abc"),
        }
    )
    out, order = sort_with_positions_polars(X, "t")
    assert order.to_list() == [1, 2, 0]
    assert out["v"].to_list() == ["b", "c", "a"]
