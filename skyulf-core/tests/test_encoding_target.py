"""Tests for the TargetEncoder Calculator/Applier (skyulf.preprocessing.encoding.target)."""

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import LabelEncoder, TargetEncoder

from skyulf.preprocessing.encoding.target import (
    TargetEncoderApplier,
    TargetEncoderCalculator,
)

settings.register_profile(
    "encoding_target",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("encoding_target")


def test_binary_target_encoding_matches_raw_sklearn() -> None:
    """The node's binary-target encoding matches a manually-fitted sklearn TargetEncoder."""
    X = pd.DataFrame({"city": ["a", "b", "a", "b", "a", "b"]})
    y = pd.Series([1, 0, 1, 0, 0, 1], name="target")
    config: dict[str, Any] = {"columns": ["city"], "smooth": "auto", "target_type": "binary"}

    calc = TargetEncoderCalculator()
    applier = TargetEncoderApplier()
    params = calc.fit((X, y), config)
    X_out, y_out = applier.apply((X, y), dict(params))

    raw_encoder = TargetEncoder(smooth="auto", target_type="binary")
    raw_encoder.fit(X[["city"]].to_numpy(), y.to_numpy())
    expected = raw_encoder.transform(X[["city"]].to_numpy())

    np.testing.assert_allclose(X_out["city"].to_numpy(), expected[:, 0])
    assert list(y_out) == list(y)


def test_no_target_returns_empty_params_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Fitting without a y and without a resolvable target_column is a documented no-op."""
    X = pd.DataFrame({"city": ["a", "b", "a"]})
    calc = TargetEncoderCalculator()

    with caplog.at_level(logging.WARNING):
        params = calc.fit(X, {"columns": ["city"]})

    assert params == {}
    assert any("requires a target variable" in rec.message for rec in caplog.records)


def test_user_picked_no_columns_short_circuits() -> None:
    """columns=[] means the user explicitly disabled encoding; fit must return {}."""
    X = pd.DataFrame({"city": ["a", "b"]})
    y = pd.Series([1, 0], name="target")
    calc = TargetEncoderCalculator()

    params = calc.fit((X, y), {"columns": []})
    assert params == {}


def test_target_column_pulled_from_x_when_y_missing() -> None:
    """When y is None but target_column names a column in X, that column becomes y."""
    df = pd.DataFrame({"city": ["a", "b", "a", "b"], "target": [1, 0, 1, 0]})
    config: dict[str, Any] = {"columns": ["city"], "target_column": "target"}
    calc = TargetEncoderCalculator()

    params = calc.fit(df, config)
    assert params["columns"] == ["city"]
    assert params["encoder_object"] is not None


def test_multiclass_target_creates_per_class_columns() -> None:
    """Multiclass targets fan out into `<col>_cls{i}` columns and drop the original."""
    X = pd.DataFrame({"city": ["a", "b", "c", "a", "b", "c"] * 3})
    y = pd.Series([0, 1, 2, 1, 2, 0] * 3, name="target")
    config: dict[str, Any] = {"columns": ["city"], "target_type": "multiclass"}

    calc = TargetEncoderCalculator()
    applier = TargetEncoderApplier()
    params = calc.fit((X, y), config)
    X_out, y_out = applier.apply((X, y), dict(params))

    assert "city" not in X_out.columns
    assert {"city_cls0", "city_cls1", "city_cls2"}.issubset(set(X_out.columns))
    assert list(y_out) == list(y)


def test_string_multiclass_target_is_label_encoded_internally() -> None:
    """String multiclass targets are label-encoded before sklearn's TargetEncoder.fit."""
    X = pd.DataFrame({"city": ["a", "b", "c", "a", "b", "c"] * 3})
    y = pd.Series(["red", "green", "blue", "green", "blue", "red"] * 3, name="target")
    config: dict[str, Any] = {"columns": ["city"], "target_type": "multiclass"}

    calc = TargetEncoderCalculator()
    params = calc.fit((X, y), config)

    le = LabelEncoder()
    y_int = le.fit_transform(y.to_numpy())
    raw_encoder = TargetEncoder(smooth="auto", target_type="multiclass")
    raw_encoder.fit(X[["city"]].to_numpy(), y_int)

    fitted = params["encoder_object"]
    np.testing.assert_allclose(
        fitted.transform(X[["city"]].to_numpy()),
        raw_encoder.transform(X[["city"]].to_numpy()),
    )


def test_unseen_category_at_apply_falls_back_to_global_mean() -> None:
    """An unseen category at apply time is imputed with sklearn's global target mean."""
    X = pd.DataFrame({"city": ["a", "b", "a", "b", "a", "b"]})
    y = pd.Series([1, 0, 1, 0, 1, 0], name="target")
    config: dict[str, Any] = {"columns": ["city"], "target_type": "binary"}

    calc = TargetEncoderCalculator()
    applier = TargetEncoderApplier()
    params = calc.fit((X, y), config)

    test_X = pd.DataFrame({"city": ["unseen"]})
    X_out = applier.apply(test_X, dict(params))

    encoder = params["encoder_object"]
    np.testing.assert_allclose(X_out["city"].to_numpy(), encoder.target_mean_)


def test_constant_column_single_category() -> None:
    """A constant categorical column still fits (single category -> single stat)."""
    X = pd.DataFrame({"city": ["same"] * 6})
    y = pd.Series([1, 0, 1, 0, 1, 0], name="target")
    config: dict[str, Any] = {"columns": ["city"], "target_type": "binary"}

    calc = TargetEncoderCalculator()
    applier = TargetEncoderApplier()
    params = calc.fit((X, y), config)
    X_out, _ = applier.apply((X, y), dict(params))

    # Every row shares the same (only) category, so encoded values are identical.
    assert X_out["city"].nunique() == 1


def test_unresolved_columns_short_circuit_apply() -> None:
    """Applying params whose columns no longer exist in X is a no-op."""
    X = pd.DataFrame({"other": [1, 2, 3]})
    y = pd.Series([1, 0, 1], name="target")
    train_X = pd.DataFrame({"city": ["a", "b", "a"]})
    calc = TargetEncoderCalculator()
    applier = TargetEncoderApplier()
    params = calc.fit((train_X, y), {"columns": ["city"], "target_type": "binary"})

    X_out, y_out = applier.apply((X, y), dict(params))
    pd.testing.assert_frame_equal(X_out, X)
    assert list(y_out) == list(y)


# ---------------------------------------------------------------------------
# Polars parity
# ---------------------------------------------------------------------------


@st.composite
def _categorical_binary_frame(
    draw: st.DrawFn, *, min_rows: int = 20, max_rows: int = 60
) -> pd.DataFrame:
    """Generate a frame with one categorical feature and a binary target."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cats = draw(st.lists(st.sampled_from(["x", "y", "z"]), min_size=n, max_size=n))
    target = draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
    assume(len(set(target)) == 2 and len(set(cats)) > 1)
    return pd.DataFrame({"city": cats, "target": target})


@given(df=_categorical_binary_frame())
def test_target_encoder_fit_engine_parity(df: pd.DataFrame) -> None:
    """pandas and polars fits must produce numerically identical encodings."""
    config: dict[str, Any] = {"columns": ["city"], "target_type": "binary"}

    pd_params = dict(TargetEncoderCalculator().fit((df[["city"]], df["target"]), dict(config)))
    pl_params = dict(
        TargetEncoderCalculator().fit(
            (pl.from_pandas(df[["city"]]), pl.Series("target", df["target"])), dict(config)
        )
    )

    assert pd_params["columns"] == pl_params["columns"]
    sample = df[["city"]].to_numpy()
    np.testing.assert_allclose(
        cast(TargetEncoder, pd_params["encoder_object"]).transform(sample),
        cast(TargetEncoder, pl_params["encoder_object"]).transform(sample),
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Polars apply path (binary + multiclass + error handling)
# ---------------------------------------------------------------------------


def test_polars_apply_binary_matches_pandas() -> None:
    """The polars apply path replaces columns in-place identically to pandas."""
    X_pd = pd.DataFrame({"city": ["a", "b", "a", "b", "a", "b"]})
    y_pd = pd.Series([1, 0, 1, 0, 0, 1], name="target")
    config: dict[str, Any] = {"columns": ["city"], "target_type": "binary"}

    params = dict(TargetEncoderCalculator().fit((X_pd, y_pd), config))
    out_pd, _ = TargetEncoderApplier().apply((X_pd, y_pd), dict(params))

    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.Series("target", y_pd)
    out_pl, y_out_pl = TargetEncoderApplier().apply((X_pl, y_pl), dict(params))

    np.testing.assert_allclose(out_pd["city"].to_numpy(), out_pl["city"].to_numpy())
    assert list(y_out_pl) == list(y_pd)


def test_polars_apply_multiclass_creates_per_class_columns_and_drops_original() -> None:
    """Polars apply path fans multiclass targets into `<col>_cls{i}` columns."""
    X_pd = pd.DataFrame({"city": ["a", "b", "c", "a", "b", "c"] * 3})
    y_pd = pd.Series([0, 1, 2, 1, 2, 0] * 3, name="target")
    config: dict[str, Any] = {"columns": ["city"], "target_type": "multiclass"}

    params = dict(TargetEncoderCalculator().fit((X_pd, y_pd), config))
    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.Series("target", y_pd)
    out_pl, y_out_pl = TargetEncoderApplier().apply((X_pl, y_pl), dict(params))

    assert "city" not in out_pl.columns
    assert {"city_cls0", "city_cls1", "city_cls2"}.issubset(set(out_pl.columns))
    assert list(y_out_pl) == list(y_pd)


def test_polars_apply_no_valid_columns_is_noop() -> None:
    """When configured columns aren't present in X, the polars apply path returns X, y unchanged."""
    train_X = pd.DataFrame({"city": ["a", "b", "a"]})
    y = pd.Series([1, 0, 1], name="target")
    params = dict(
        TargetEncoderCalculator().fit((train_X, y), {"columns": ["city"], "target_type": "binary"})
    )

    X_pl = pl.DataFrame({"other": [1, 2, 3]})
    y_pl = pl.Series("target", [1, 0, 1])
    out_pl, y_out_pl = TargetEncoderApplier().apply((X_pl, y_pl), dict(params))

    assert out_pl.equals(X_pl)
    assert list(y_out_pl) == list(y_pl)


def test_polars_apply_exception_is_caught_and_returns_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transform-time exception in the polars apply path is caught, logged, and X/y returned as-is."""
    X_pl = pl.DataFrame({"city": ["a", "b", "a"]})
    y_pl = pl.Series("target", [1, 0, 1])

    class _BrokenEncoder:
        def transform(self, _x: Any) -> Any:
            raise ValueError("boom")

    params = {"columns": ["city"], "encoder_object": _BrokenEncoder()}
    with caplog.at_level(logging.ERROR):
        out_pl, y_out_pl = TargetEncoderApplier().apply((X_pl, y_pl), dict(params))

    assert out_pl.equals(X_pl)
    assert list(y_out_pl) == list(y_pl)
    assert any("Target Encoding failed" in rec.message for rec in caplog.records)


def test_pandas_apply_exception_is_caught_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transform-time exception in the pandas apply path is caught and logged."""
    X_pd = pd.DataFrame({"city": ["a", "b", "a"]})
    y_pd = pd.Series([1, 0, 1], name="target")

    class _BrokenEncoder:
        def transform(self, _x: Any) -> Any:
            raise ValueError("boom")

    params = {"columns": ["city"], "encoder_object": _BrokenEncoder()}
    with caplog.at_level(logging.ERROR):
        out_pd, y_out_pd = TargetEncoderApplier().apply((X_pd, y_pd), dict(params))

    pd.testing.assert_frame_equal(out_pd, X_pd)
    assert any("Target Encoding failed" in rec.message for rec in caplog.records)


def test_polars_fit_extracts_y_from_target_column() -> None:
    """Polars fit path pulls y out of X via target_column when y is missing."""
    df = pl.DataFrame({"city": ["a", "b", "a", "b"], "target": [1, 0, 1, 0]})
    config: dict[str, Any] = {"columns": ["city"], "target_column": "target"}

    params = TargetEncoderCalculator().fit(df, config)
    assert params["columns"] == ["city"]
    assert params["encoder_object"] is not None


def test_polars_fit_no_target_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Polars fit without a resolvable target logs a warning and returns {}."""
    df = pl.DataFrame({"city": ["a", "b", "a"]})
    with caplog.at_level(logging.WARNING):
        params = TargetEncoderCalculator().fit(df, {"columns": ["city"]})

    assert params == {}
    assert any("requires a target variable" in rec.message for rec in caplog.records)


def test_polars_fit_no_resolvable_columns_returns_empty() -> None:
    """Polars fit falls back to auto-detection; a purely-numeric frame yields no columns."""
    df = pl.DataFrame({"amount": [1, 2, 3]})
    y = pl.Series("target", [1, 0, 1])
    params = TargetEncoderCalculator().fit((df, y), {})
    assert params == {}


def test_pandas_fit_no_resolvable_columns_returns_empty() -> None:
    """Pandas fit falls back to auto-detection; a purely-numeric frame yields no columns."""
    df = pd.DataFrame({"amount": [1, 2, 3]})
    y = pd.Series([1, 0, 1], name="target")
    params = TargetEncoderCalculator().fit((df, y), {})
    assert params == {}


def test_unknown_label_type_error_is_translated_to_actionable_message() -> None:
    """sklearn's 'unknown label type' ValueError is translated into an actionable message.

    A multi-output (2-D) target forces sklearn's `type_of_target` to infer
    'multiclass-multioutput', which is unsupported and raises the exact
    ValueError that ``_fit_target_encoder`` catches and re-raises with a
    clearer, actionable message.
    """
    X = pd.DataFrame({"city": ["a", "b", "a", "b"]})
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    config: dict[str, Any] = {"columns": ["city"], "target_type": "auto"}

    with pytest.raises(ValueError, match="TargetEncoder failed"):
        TargetEncoderCalculator().fit((X, y), config)


def test_maybe_extract_y_polars_returns_y_when_target_col_missing_from_x() -> None:
    """_maybe_extract_y_polars falls through to `y` unchanged if target_col isn't in X."""
    from skyulf.preprocessing.encoding.target import _maybe_extract_y_polars

    X = pl.DataFrame({"city": ["a", "b"]})
    assert _maybe_extract_y_polars(X, None, "missing_col") is None


def test_y_to_numpy_uses_to_pandas_fallback() -> None:
    """_y_to_numpy converts via `.to_pandas().to_numpy()` for objects lacking `.to_numpy`."""
    from skyulf.preprocessing.encoding.target import _y_to_numpy

    class _ToPandasOnly:
        def to_pandas(self) -> pd.Series:
            return pd.Series([1, 2, 3])

    result = _y_to_numpy(_ToPandasOnly())
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_fit_target_encoder_reraises_unrelated_value_error() -> None:
    """A ValueError unrelated to label-type/multiclass issues is re-raised as-is."""
    from skyulf.preprocessing.encoding.target import _fit_target_encoder

    X = pd.DataFrame({"city": ["a", "b"]})
    y = pd.Series([1, 0], name="target")
    config: dict[str, Any] = {"smooth": "not-a-valid-smooth-value", "target_type": "binary"}

    with pytest.raises(ValueError) as exc_info:
        _fit_target_encoder(X, y, config)
    assert "TargetEncoder failed" not in str(exc_info.value)
