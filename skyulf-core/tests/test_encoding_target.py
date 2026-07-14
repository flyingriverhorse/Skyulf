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
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.core.schema import SkyulfSchema
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

_no_target_cases = TestCaseLoader(
    "preprocessing/encoding_target", group="no_target_returns_empty"
).load_with_ids()
_no_resolvable_columns_cases = TestCaseLoader(
    "preprocessing/encoding_target", group="no_resolvable_columns"
).load_with_ids()


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


class TestNoTargetReturnsEmptyAndWarns:
    """Fitting without a resolvable target logs a warning and returns {}.
    Scenarios (pandas/polars) loaded from
    ``tests/test_cases/preprocessing/encoding_target.json`` (group ``no_target_returns_empty``).
    """

    @pytest.mark.parametrize(_no_target_cases[0], _no_target_cases[1], ids=_no_target_cases[2])
    def test_no_target_returns_empty_params_and_warns(
        self, engine: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        X = (
            pl.DataFrame({"city": ["a", "b", "a"]})
            if engine == "polars"
            else pd.DataFrame({"city": ["a", "b", "a"]})
        )
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


def test_polars_apply_exception_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transform-time exception in the polars apply path propagates (raised), with
    the dispatcher logging the failure."""
    X_pl = pl.DataFrame({"city": ["a", "b", "a"]})
    y_pl = pl.Series("target", [1, 0, 1])

    class _BrokenEncoder:
        def transform(self, _x: Any) -> Any:
            raise ValueError("boom")

    params = {"columns": ["city"], "encoder_object": _BrokenEncoder()}
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="boom"):
        TargetEncoderApplier().apply((X_pl, y_pl), dict(params))

    assert any("engine apply failed" in rec.message for rec in caplog.records)


def test_pandas_apply_exception_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transform-time exception in the pandas apply path propagates (raised), with
    the dispatcher logging the failure."""
    X_pd = pd.DataFrame({"city": ["a", "b", "a"]})
    y_pd = pd.Series([1, 0, 1], name="target")

    class _BrokenEncoder:
        def transform(self, _x: Any) -> Any:
            raise ValueError("boom")

    params = {"columns": ["city"], "encoder_object": _BrokenEncoder()}
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="boom"):
        TargetEncoderApplier().apply((X_pd, y_pd), dict(params))

    assert any("engine apply failed" in rec.message for rec in caplog.records)


def test_polars_fit_extracts_y_from_target_column() -> None:
    """Polars fit path pulls y out of X via target_column when y is missing."""
    df = pl.DataFrame({"city": ["a", "b", "a", "b"], "target": [1, 0, 1, 0]})
    config: dict[str, Any] = {"columns": ["city"], "target_column": "target"}

    params = TargetEncoderCalculator().fit(df, config)
    assert params["columns"] == ["city"]
    assert params["encoder_object"] is not None


class TestFitNoResolvableColumnsReturnsEmpty:
    """A purely-numeric frame yields no encodable columns, so fit() returns {}.
    Scenarios (pandas/polars) loaded from
    ``tests/test_cases/preprocessing/encoding_target.json`` (group ``no_resolvable_columns``).
    """

    @pytest.mark.parametrize(
        _no_resolvable_columns_cases[0],
        _no_resolvable_columns_cases[1],
        ids=_no_resolvable_columns_cases[2],
    )
    def test_fit_no_resolvable_columns_returns_empty(self, engine: str) -> None:
        if engine == "polars":
            df = pl.DataFrame({"amount": [1, 2, 3]})
            y = pl.Series("target", [1, 0, 1])
        else:
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


def test_infer_output_schema_preserves_columns_for_binary_target() -> None:
    """Binary/regression target encoding replaces values in place — schema unchanged."""
    input_schema = SkyulfSchema.from_columns(["city", "target"])
    output_schema = TargetEncoderCalculator().infer_output_schema(
        input_schema, {"target_type": "binary", "columns": ["city"]}
    )
    assert output_schema is input_schema


def test_infer_output_schema_returns_none_for_multiclass_target() -> None:
    """Multiclass target encoding fans out into ``{col}_cls{i}`` columns and drops
    the originals (see ``_target_apply_polars``/``_target_apply_pandas``), so the
    output columns are data-dependent (unknown number of classes) and can't be
    confidently predicted from config alone. Regression test for a bug where this
    unconditionally returned ``input_schema`` unchanged, misleading downstream
    schema consumers about the real output columns.
    """
    input_schema = SkyulfSchema.from_columns(["city", "target"])
    output_schema = TargetEncoderCalculator().infer_output_schema(
        input_schema, {"target_type": "multiclass", "columns": ["city"]}
    )
    assert output_schema is None


def test_infer_output_schema_returns_none_for_default_auto_target_type() -> None:
    """The default ``target_type="auto"`` resolves to multiclass at fit time
    whenever y has more than two classes, so it must also be treated as
    "unknown" — not confidently binary/regression. Regression test for a gap
    where only the explicit "multiclass" string was handled, missing the
    common default config that omits ``target_type`` entirely.
    """
    input_schema = SkyulfSchema.from_columns(["city", "target"])
    output_schema = TargetEncoderCalculator().infer_output_schema(
        input_schema, {"columns": ["city"]}
    )
    assert output_schema is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.
    ``plan_type`` (no NaN, 3 categories) + binary ``churned`` target exercises the
    TargetEncoder on production-like data: each plan group gets its own smoothed
    target statistic, and the result is a numeric column of the same length.
    """

    def test_plan_type_target_encoding_with_binary_churn_target(self) -> None:
        """TargetEncoder on ``plan_type`` with binary ``churned`` produces a numeric column.

        Each distinct plan_type value maps to a different smoothed mean of the
        binary churn target — verifies real-data fit→apply cycle without leakage.
        """
        df = load_sample_dataset("customers")
        X = df[["plan_type"]].copy()
        y = df["churned"]
        config: dict[str, Any] = {"columns": ["plan_type"], "target_type": "binary"}

        params = TargetEncoderCalculator().fit((X, y), config)
        X_out, y_out = TargetEncoderApplier().apply((X, y), dict(params))

        assert len(X_out) == len(df)
        assert pd.api.types.is_float_dtype(X_out["plan_type"])
        # Binary target encoding produces values in (0, 1) representing the smoothed churn rate.
        assert X_out["plan_type"].between(0.0, 1.0).all()
        assert list(y_out) == list(y)
