"""Coverage-gap tests for transformations: power (Box-Cox/Yeo-Johnson), simple (log/sqrt/etc).

Verifies real transform math against ``sklearn``/``numpy`` reference values,
plus edge cases: non-positive data with box-cox, missing columns, negative
inputs for log/sqrt, and pandas/polars parity for simple transformations.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import PowerTransformer as SkPowerTransformer
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
    _extract_scaler_params,
    _filter_power_columns,
)
from skyulf.preprocessing.transformations.simple import (
    SimpleTransformationApplier,
    SimpleTransformationCalculator,
)

_empty_artifact_cases = TestCaseLoader("preprocessing/power_transformer_empty_artifact").load()
_simple_method_cases = TestCaseLoader("preprocessing/simple_transformation_methods").load()

# ---------------------------------------------------------------------------
# PowerTransformer
# ---------------------------------------------------------------------------


def test_power_transformer_yeo_johnson_matches_sklearn_reference() -> None:
    """Fit+apply reproduces sklearn's own PowerTransformer output exactly."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "yeo-johnson", "columns": ["a"]})
    out = PowerTransformerApplier().apply(df, art)

    reference = SkPowerTransformer(method="yeo-johnson", standardize=True)
    expected = reference.fit_transform(df[["a"]])
    np.testing.assert_allclose(out["a"].to_numpy(), expected[:, 0], rtol=1e-6, atol=1e-6)


def test_power_transformer_box_cox_excludes_non_positive_columns() -> None:
    """Box-Cox drops columns containing zero/negative values from ``columns``."""
    df = pd.DataFrame({"pos": [1.0, 2.0, 3.0], "has_zero": [0.0, 1.0, 2.0]})
    art = PowerTransformerCalculator().fit(
        df, {"method": "box-cox", "columns": ["pos", "has_zero"]}
    )
    assert art["columns"] == ["pos"]


@pytest.mark.parametrize(*_empty_artifact_cases)
def test_power_transformer_returns_empty_artifact(
    df_data: dict[str, list], fit_config: dict
) -> None:
    """Scenarios where fit() must yield an empty artifact — no valid/positive/numeric columns.

    Loaded from ``tests/test_cases/preprocessing/power_transformer_empty_artifact.json``.
    """
    df = pd.DataFrame(df_data)
    art = PowerTransformerCalculator().fit(df, fit_config)
    assert art == {}


def test_power_transformer_apply_missing_lambdas_is_noop() -> None:
    """Applying an artifact without fitted ``lambdas`` leaves the frame unchanged."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = PowerTransformerApplier().apply(df, {"columns": ["a"]})
    pd.testing.assert_frame_equal(out, df)


def test_power_transformer_polars_apply_matches_pandas() -> None:
    """Polars apply path reproduces the same transformed values as pandas."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "yeo-johnson", "columns": ["a"]})
    pandas_out = PowerTransformerApplier().apply(df, art)
    polars_out = PowerTransformerApplier().apply(pl.from_pandas(df), art)
    np.testing.assert_allclose(
        pandas_out["a"].to_numpy(), np.array(polars_out["a"].to_list()), rtol=1e-6, atol=1e-6
    )


def test_power_transformer_standardize_false_skips_scaler() -> None:
    """``standardize=False`` skips the internal StandardScaler reconstruction."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    art = PowerTransformerCalculator().fit(
        df, {"method": "yeo-johnson", "standardize": False, "columns": ["a"]}
    )
    assert art["scaler_params"] == {}
    out = PowerTransformerApplier().apply(df, art)
    assert out["a"].notna().all()


def test_power_transformer_infer_output_schema_is_identity() -> None:
    """Power transforms are applied in place, so the schema is unchanged."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    out = PowerTransformerCalculator().infer_output_schema(schema, {"columns": ["a"]})
    assert out is schema


def test_filter_power_columns_empty_input_returns_empty() -> None:
    """_filter_power_columns must short-circuit on an empty `cols` list directly."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert _filter_power_columns(df, [], "yeo-johnson") == []


def test_extract_scaler_params_returns_empty_when_no_scaler_present() -> None:
    """_extract_scaler_params must return {} when the transformer lacks a `_scaler` (line 84)."""
    transformer = SkPowerTransformer(method="yeo-johnson", standardize=True)
    # Freshly constructed (unfitted) transformer has no `_scaler` attribute yet.
    assert _extract_scaler_params(transformer, standardize=True) == {}


def test_power_transformer_apply_polars_missing_lambdas_is_noop() -> None:
    """Polars apply with no fitted `lambdas` must return the frame unchanged (line 102)."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    pl_df = pl.from_pandas(df)
    out = PowerTransformerApplier().apply(pl_df, {"columns": ["a"]})
    assert out.equals(pl_df)


def test_power_transformer_apply_polars_no_valid_columns_is_noop() -> None:
    """Polars apply where none of the fitted columns exist must no-op (line 105)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "yeo-johnson", "columns": ["a"]})
    other_df = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    pl_df = pl.from_pandas(other_df)
    out = PowerTransformerApplier().apply(pl_df, art)
    assert out.equals(pl_df)


def test_power_transformer_apply_pandas_no_valid_columns_is_noop() -> None:
    """Pandas apply where none of the fitted columns exist must no-op (line 123)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "yeo-johnson", "columns": ["a"]})
    other_df = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    out = PowerTransformerApplier().apply(other_df, art)
    pd.testing.assert_frame_equal(out, other_df)


def test_power_transformer_apply_polars_swallows_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transform failure on the polars path must be logged, not raised (lines 112-114)."""
    import skyulf.preprocessing.transformations.power as power_mod

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "yeo-johnson", "columns": ["a"]})

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated transform failure")

    monkeypatch.setattr(power_mod, "_power_transform_array", _boom)
    pl_df = pl.from_pandas(df)
    out = PowerTransformerApplier().apply(pl_df, art)
    # Exception swallowed; original frame returned unchanged.
    assert out.equals(pl_df)


def test_power_transformer_apply_pandas_swallows_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transform failure on the pandas path must be logged, not raised (lines 130-131)."""
    import skyulf.preprocessing.transformations.power as power_mod

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "yeo-johnson", "columns": ["a"]})

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated transform failure")

    monkeypatch.setattr(power_mod, "_power_transform_array", _boom)
    out = PowerTransformerApplier().apply(df, art)
    # Exception swallowed; original values returned unchanged.
    pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# SimpleTransformation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_simple_method_cases)
def test_simple_transformation_method(
    values: list[float], method: str, extra_params: dict, expected: list[float | None]
) -> None:
    """Verifies per-method transform math and edge-case handling (NaN masking, clipping, unknown methods).

    Loaded from ``tests/test_cases/preprocessing/simple_transformation_methods.json``.
    """
    df = pd.DataFrame({"a": values})
    config = {"column": "a", "method": method, **extra_params}
    art = SimpleTransformationCalculator().fit(df, {"transformations": [config]})
    out = SimpleTransformationApplier().apply(df, art)
    result = out["a"].tolist()
    assert len(result) == len(expected)
    for actual, exp in zip(result, expected):
        if exp is None:
            assert np.isnan(actual)
        else:
            np.testing.assert_allclose(actual, exp, rtol=1e-6, atol=1e-9)


def test_simple_transformation_missing_column_skipped() -> None:
    """A transformation targeting a nonexistent column is skipped gracefully."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "nonexistent", "method": "log"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    assert list(out.columns) == ["a"]


def test_simple_transformation_no_transformations_returns_unchanged() -> None:
    """An empty ``transformations`` list is a full no-op."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    art = SimpleTransformationCalculator().fit(df, {"transformations": []})
    out = SimpleTransformationApplier().apply(df, art)
    pd.testing.assert_frame_equal(out, df)


def test_simple_transformation_polars_no_transformations_returns_unchanged() -> None:
    """An empty ``transformations`` list must no-op on the polars apply path (line 26)."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    pl_df = pl.from_pandas(df)
    art = SimpleTransformationCalculator().fit(df, {"transformations": []})
    out = SimpleTransformationApplier().apply(pl_df, art)
    assert out.equals(pl_df)


def test_simple_transformation_polars_missing_column_skipped() -> None:
    """A transformation targeting a nonexistent column must be skipped (line 33)."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    pl_df = pl.from_pandas(df)
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "nonexistent", "method": "log"}]}
    )
    out = SimpleTransformationApplier().apply(pl_df, art)
    assert list(out.columns) == ["a"]


def test_simple_transformation_polars_unknown_method_skipped() -> None:
    """An unrecognised method must be skipped on the polars apply path (line 36)."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    pl_df = pl.from_pandas(df)
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "bogus_method"}]}
    )
    out = SimpleTransformationApplier().apply(pl_df, art)
    assert out["a"].to_list() == [1.0, 2.0]


def test_simple_transformation_polars_parity_log_and_square() -> None:
    """Polars and pandas apply paths agree on log1p + square transformations."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    art = SimpleTransformationCalculator().fit(
        df,
        {
            "transformations": [
                {"column": "a", "method": "log"},
                {"column": "b", "method": "square"},
            ]
        },
    )
    pandas_out = SimpleTransformationApplier().apply(df, art)
    polars_out = SimpleTransformationApplier().apply(pl.from_pandas(df), art)
    np.testing.assert_allclose(pandas_out["a"].to_numpy(), np.array(polars_out["a"].to_list()))
    np.testing.assert_allclose(pandas_out["b"].to_numpy(), np.array(polars_out["b"].to_list()))


def test_simple_transformation_infer_output_schema_is_identity() -> None:
    """Simple transformations replace values in place; schema is unchanged."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    out = SimpleTransformationCalculator().infer_output_schema(schema, {})
    assert out is schema


# ---------------------------------------------------------------------------
# Real-shaped dataset integration check
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.

    Verifies that SimpleTransformationCalculator handles the real-world
    ``income`` column (which has NaN values): NaN rows must be preserved
    through a log transform, and non-missing rows must produce finite output.
    """

    def test_simple_log_on_income_preserves_nan(self) -> None:
        df = load_sample_dataset("customers")
        art = SimpleTransformationCalculator().fit(
            df, {"transformations": [{"column": "income", "method": "log"}]}
        )
        out = SimpleTransformationApplier().apply(df, art)
        # Rows where income was NaN must remain NaN after log.
        assert out.loc[df["income"].isna(), "income"].isna().all()
        # Non-missing rows must yield finite log1p values.
        non_missing = ~df["income"].isna()
        assert out.loc[non_missing, "income"].notna().all()
