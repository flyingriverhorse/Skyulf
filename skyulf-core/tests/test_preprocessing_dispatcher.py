"""Tests for skyulf.preprocessing.dispatcher (apply_dual_engine / fit_dual_engine)."""

import logging

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.dispatcher import apply_dual_engine, fit_dual_engine


def _pandas_apply(X, y, params):
    """Pandas apply_func stub: adds 1 to every value in column 'a'."""
    X = X.copy()
    X["a"] = X["a"] + 1
    return X, y


def _polars_apply(X, y, params):
    """Polars apply_func stub: adds 1 to every value in column 'a'."""
    X = X.with_columns((pl.col("a") + 1).alias("a"))
    return X, y


def _pandas_fit(X, y, params):
    """Pandas fit_func stub: returns the mean of column 'a'."""
    return {"mean_a": float(X["a"].mean())}


def _polars_fit(X, y, params):
    """Polars fit_func stub: returns the mean of column 'a'."""
    return {"mean_a": float(X["a"].mean())}


def test_apply_dual_engine_dispatches_to_pandas_path():
    """A pandas DataFrame input should route through pandas_func."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = apply_dual_engine(df, {}, _polars_apply, _pandas_apply)
    assert list(result["a"]) == [2, 3, 4]


def test_apply_dual_engine_dispatches_to_polars_path():
    """A polars DataFrame input should route through polars_func."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = apply_dual_engine(df, {}, _polars_apply, _pandas_apply)
    assert result["a"].to_list() == [2, 3, 4]


def test_apply_dual_engine_converts_wrapper_to_pandas_before_pandas_func():
    """A SkyulfDataFrame wrapper exposing to_pandas() should be converted before pandas_func runs."""
    from skyulf.engines.pandas_engine import SkyulfPandasWrapper

    df = pd.DataFrame({"a": [1, 2, 3]})
    wrapper = SkyulfPandasWrapper(df)

    captured = {}

    def _pandas_apply_capture(X, y, params):
        captured["type"] = type(X)
        return X, y

    apply_dual_engine(wrapper, {}, _polars_apply, _pandas_apply_capture)
    assert captured["type"] is pd.DataFrame


def test_apply_dual_engine_propagates_pandas_exception():
    """An exception raised inside pandas_func should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="boom"):
        apply_dual_engine(df, {}, _polars_apply, _raising)


def test_apply_dual_engine_pandas_exception_is_logged(caplog):
    """A pandas_func failure should be logged (with traceback) before propagating.

    Regression test: the dispatcher previously had a dead
    ``try/except: raise e`` with the actual logging call commented out,
    meaning engine-dispatch failures went completely unlogged at this
    central chokepoint used by ~50 preprocessing nodes.
    """

    def _raising(X, y, params):
        raise ValueError("boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with caplog.at_level(logging.ERROR, logger="skyulf.preprocessing.dispatcher"):
        with pytest.raises(ValueError, match="boom"):
            apply_dual_engine(df, {}, _polars_apply, _raising)
    assert any("Pandas engine apply failed" in rec.message for rec in caplog.records)


def test_apply_dual_engine_propagates_polars_exception():
    """An exception raised inside polars_func should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("polars boom")

    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="polars boom"):
        apply_dual_engine(df, {}, _raising, _pandas_apply)


def test_apply_dual_engine_handles_tuple_input():
    """(X, y) tuple input should be unpacked, processed, and repacked as a tuple."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    result = apply_dual_engine((X, y), {}, _polars_apply, _pandas_apply)
    assert isinstance(result, tuple)
    assert list(result[0]["a"]) == [2, 3, 4]
    assert list(result[1]) == [0, 1, 0]


def test_fit_dual_engine_dispatches_to_pandas_path():
    """A pandas DataFrame input should route through the pandas fit function."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = fit_dual_engine(df, {}, _polars_fit, _pandas_fit)
    assert result == {"mean_a": 2.0}


def test_fit_dual_engine_dispatches_to_polars_path():
    """A polars DataFrame input should route through the polars fit function."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = fit_dual_engine(df, {}, _polars_fit, _pandas_fit)
    assert result == {"mean_a": 2.0}


def test_fit_dual_engine_converts_wrapper_to_pandas_before_pandas_func():
    """A wrapper exposing to_pandas() should be converted before the pandas fit func runs."""
    from skyulf.engines.pandas_engine import SkyulfPandasWrapper

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    wrapper = SkyulfPandasWrapper(df)

    captured = {}

    def _pandas_fit_capture(X, y, params):
        captured["type"] = type(X)
        return {}

    fit_dual_engine(wrapper, {}, _polars_fit, _pandas_fit_capture)
    assert captured["type"] is pd.DataFrame


def test_fit_dual_engine_propagates_pandas_exception():
    """An exception raised inside the pandas fit function should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("fit boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="fit boom"):
        fit_dual_engine(df, {}, _polars_fit, _raising)


def test_fit_dual_engine_propagates_polars_exception():
    """An exception raised inside the polars fit function should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("polars fit boom")

    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="polars fit boom"):
        fit_dual_engine(df, {}, _raising, _pandas_fit)


def test_fit_dual_engine_result_is_plain_dict():
    """fit_dual_engine should coerce the func's Mapping return into a plain dict."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    result = fit_dual_engine(df, {}, _polars_fit, _pandas_fit)
    assert type(result) is dict
