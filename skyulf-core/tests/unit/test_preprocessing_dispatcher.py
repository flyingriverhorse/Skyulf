"""Tests for skyulf.preprocessing.dispatcher helpers."""

import logging

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.dispatcher import (
    apply_dual_engine,
    fit_dual_engine,
    fit_transform_train_dual_engine,
)


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


def _pandas_fit_transform_train(X, y, params):
    """Pandas train-transform stub: returns params plus a special train view."""
    X = X.copy()
    X["a"] = X["a"] + 100
    return {"mean_a": 2.0}, X, y


def _polars_fit_transform_train(X, y, params):
    """Polars train-transform stub: returns params plus a special train view."""
    X = X.with_columns((pl.col("a") + 100).alias("a"))
    return {"mean_a": 2.0}, X, y


def test_apply_dual_engine_dispatches_to_pandas_path():
    """A pandas DataFrame input should route through the pandas implementation."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = apply_dual_engine(df, {}, {"polars": _polars_apply, "pandas": _pandas_apply})
    assert list(result["a"]) == [2, 3, 4]


def test_apply_dual_engine_dispatches_to_polars_path():
    """A polars DataFrame input should route through the polars implementation."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = apply_dual_engine(df, {}, {"polars": _polars_apply, "pandas": _pandas_apply})
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

    apply_dual_engine(wrapper, {}, {"polars": _polars_apply, "pandas": _pandas_apply_capture})
    assert captured["type"] is pd.DataFrame


def test_apply_dual_engine_propagates_pandas_exception():
    """An exception raised inside the pandas implementation should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="boom"):
        apply_dual_engine(df, {}, {"polars": _polars_apply, "pandas": _raising})


def test_apply_dual_engine_pandas_value_error_is_logged_without_traceback(caplog):
    """A routine input error should be logged without a traceback before propagating."""

    def _raising(X, y, params):
        raise ValueError("boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with (
        caplog.at_level(logging.DEBUG, logger="skyulf.preprocessing.dispatcher"),
        pytest.raises(ValueError, match="boom"),
    ):
        apply_dual_engine(df, {}, {"polars": _polars_apply, "pandas": _raising})
    records = [rec for rec in caplog.records if "Pandas engine apply failed" in rec.message]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].exc_info is None


def test_apply_dual_engine_pandas_unexpected_exception_is_logged_with_traceback(caplog):
    """An unexpected failure should retain exception-level traceback logging."""

    def _raising(X, y, params):
        raise RuntimeError("boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with (
        caplog.at_level(logging.DEBUG, logger="skyulf.preprocessing.dispatcher"),
        pytest.raises(RuntimeError, match="boom"),
    ):
        apply_dual_engine(df, {}, {"polars": _polars_apply, "pandas": _raising})
    records = [rec for rec in caplog.records if "Pandas engine apply failed" in rec.message]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None


def test_apply_dual_engine_propagates_polars_exception():
    """An exception raised inside the polars implementation should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("polars boom")

    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="polars boom"):
        apply_dual_engine(df, {}, {"polars": _raising, "pandas": _pandas_apply})


def test_apply_dual_engine_handles_tuple_input():
    """(X, y) tuple input should be unpacked, processed, and repacked as a tuple."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    result = apply_dual_engine((X, y), {}, {"polars": _polars_apply, "pandas": _pandas_apply})
    assert isinstance(result, tuple)
    assert list(result[0]["a"]) == [2, 3, 4]
    assert list(result[1]) == [0, 1, 0]


def test_fit_dual_engine_dispatches_to_pandas_path():
    """A pandas DataFrame input should route through the pandas fit function."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = fit_dual_engine(df, {}, {"polars": _polars_fit, "pandas": _pandas_fit})
    assert result == {"mean_a": 2.0}


def test_fit_dual_engine_dispatches_to_polars_path():
    """A polars DataFrame input should route through the polars fit function."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = fit_dual_engine(df, {}, {"polars": _polars_fit, "pandas": _pandas_fit})
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

    fit_dual_engine(wrapper, {}, {"polars": _polars_fit, "pandas": _pandas_fit_capture})
    assert captured["type"] is pd.DataFrame


def test_fit_dual_engine_propagates_pandas_exception():
    """An exception raised inside the pandas fit function should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("fit boom")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="fit boom"):
        fit_dual_engine(df, {}, {"polars": _polars_fit, "pandas": _raising})


def test_fit_dual_engine_propagates_polars_exception():
    """An exception raised inside the polars fit function should propagate unmodified."""

    def _raising(X, y, params):
        raise ValueError("polars fit boom")

    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="polars fit boom"):
        fit_dual_engine(df, {}, {"polars": _raising, "pandas": _pandas_fit})


def test_fit_dual_engine_result_is_plain_dict():
    """fit_dual_engine should coerce the func's Mapping return into a plain dict."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    result = fit_dual_engine(df, {}, {"polars": _polars_fit, "pandas": _pandas_fit})
    assert type(result) is dict


def test_fit_transform_train_dual_engine_dispatches_to_pandas_path():
    """A pandas DataFrame input should route through the pandas train-transform helper."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    params, result = fit_transform_train_dual_engine(
        df,
        {},
        {"polars": _polars_fit_transform_train, "pandas": _pandas_fit_transform_train},
    )
    assert params == {"mean_a": 2.0}
    assert list(result["a"]) == [101, 102, 103]


def test_fit_transform_train_dual_engine_dispatches_to_polars_path():
    """A polars DataFrame input should route through the polars train-transform helper."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    params, result = fit_transform_train_dual_engine(
        df,
        {},
        {"polars": _polars_fit_transform_train, "pandas": _pandas_fit_transform_train},
    )
    assert params == {"mean_a": 2.0}
    assert result["a"].to_list() == [101, 102, 103]


# ---------------------------------------------------------------------------
# F-09: unmapped engines must fail loudly, never collect to pandas silently
# ---------------------------------------------------------------------------


class _StubSparkEngine:
    """Stand-in for a registered third engine (no EngineRegistry mutation)."""

    name = "spark"


def test_apply_dual_engine_raises_for_unmapped_engine(monkeypatch):
    """An engine with no registered implementation must raise NotImplementedError."""
    monkeypatch.setattr(
        "skyulf.preprocessing.dispatcher.get_engine", lambda data=None: _StubSparkEngine
    )
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="No 'spark' implementation registered"):
        apply_dual_engine(df, {}, {"polars": _polars_apply, "pandas": _pandas_apply})


def test_apply_dual_engine_unmapped_engine_error_lists_available_keys(monkeypatch):
    """The error must name the available implementations to make typos findable."""
    monkeypatch.setattr(
        "skyulf.preprocessing.dispatcher.get_engine", lambda data=None: _StubSparkEngine
    )
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError) as exc_info:
        apply_dual_engine(df, {}, {"polars": _polars_apply})
    message = str(exc_info.value)
    assert "available: polars" in message


def test_apply_dual_engine_raises_before_any_pandas_conversion():
    """The loud failure must precede any to_pandas() collect of the input."""

    class _SpyFrame:
        def __init__(self):
            self.to_pandas_calls = 0

        def to_pandas(self):
            self.to_pandas_calls += 1
            return pd.DataFrame({"a": [1, 2, 3]})

    spy = _SpyFrame()
    # Unknown type resolves to the default (pandas) engine; a mapping without
    # a "pandas" key must raise instead of silently converting.
    with pytest.raises(NotImplementedError, match="No 'pandas' implementation registered"):
        apply_dual_engine(spy, {}, {"polars": _polars_apply})
    assert spy.to_pandas_calls == 0


def test_apply_dual_engine_raises_for_engine_without_prep_path(monkeypatch):
    """A mapped engine with no input-preparation branch must fail loudly too."""
    monkeypatch.setattr(
        "skyulf.preprocessing.dispatcher.get_engine", lambda data=None: _StubSparkEngine
    )

    called = []

    def _spark_apply(X, y, params):
        called.append(X)
        return X, y

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="input-preparation path"):
        apply_dual_engine(df, {}, {"spark": _spark_apply})
    assert called == []


def test_fit_dual_engine_raises_for_unmapped_engine(monkeypatch):
    """The fit entry point must fail loudly for unmapped engines as well."""
    monkeypatch.setattr(
        "skyulf.preprocessing.dispatcher.get_engine", lambda data=None: _StubSparkEngine
    )
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(NotImplementedError, match="No 'spark' implementation registered"):
        fit_dual_engine(df, {}, {"polars": _polars_fit, "pandas": _pandas_fit})


def test_fit_dual_engine_accepts_same_function_for_both_engines():
    """One engine-agnostic function registered under both keys must work (woe pattern)."""

    def _shared_fit(X, y, params):
        return {"n_rows": len(X)}

    impls = {"polars": _shared_fit, "pandas": _shared_fit}
    assert fit_dual_engine(pd.DataFrame({"a": [1, 2]}), {}, impls) == {"n_rows": 2}
    assert fit_dual_engine(pl.DataFrame({"a": [1, 2]}), {}, impls) == {"n_rows": 2}


# ---------------------------------------------------------------------------
# F-27: mixed-engine (X, y) pairs must fail with a clear TypeError
# ---------------------------------------------------------------------------


def test_apply_dual_engine_rejects_mixed_engine_tuple():
    """pandas X with a polars y must raise TypeError, not a downstream AttributeError."""
    X = pd.DataFrame({"a": [1, 2]})
    y = pl.Series("t", [0, 1])
    with pytest.raises(TypeError, match="[Mm]ixed engines"):
        apply_dual_engine((X, y), {}, {"polars": _polars_apply, "pandas": _pandas_apply})


def test_apply_dual_engine_rejects_polars_x_with_pandas_y():
    """polars X with a pandas y must raise TypeError too."""
    X = pl.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    with pytest.raises(TypeError, match="[Mm]ixed engines"):
        apply_dual_engine((X, y), {}, {"polars": _polars_apply, "pandas": _pandas_apply})


def test_fit_dual_engine_rejects_mixed_engine_tuple():
    """The fit entry point must reject mixed engines as well."""
    X = pl.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([0, 1])
    with pytest.raises(TypeError, match="[Mm]ixed engines"):
        fit_dual_engine((X, y), {}, {"polars": _polars_fit, "pandas": _pandas_fit})


def test_fit_transform_train_dual_engine_rejects_mixed_engine_tuple():
    """The fit_transform_train entry point must reject mixed engines as well."""
    X = pd.DataFrame({"a": [1, 2]})
    y = pl.Series("t", [0, 1])
    with pytest.raises(TypeError, match="[Mm]ixed engines"):
        fit_transform_train_dual_engine(
            (X, y),
            {},
            {"polars": _polars_fit_transform_train, "pandas": _pandas_fit_transform_train},
        )


def test_apply_dual_engine_allows_engine_neutral_y():
    """A plain list y carries no engine and must stay accepted on either path."""
    impls = {"polars": _polars_apply, "pandas": _pandas_apply}
    X_pd = pd.DataFrame({"a": [1, 2]})
    result = apply_dual_engine((X_pd, [0, 1]), {}, impls)
    assert list(result[0]["a"]) == [2, 3]

    X_pl = pl.DataFrame({"a": [1, 2]})
    result_pl = apply_dual_engine((X_pl, [0, 1]), {}, impls)
    X_out = result_pl[0]
    assert X_out["a"].to_list() == [2, 3]
