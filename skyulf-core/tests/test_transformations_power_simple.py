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

from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
)
from skyulf.preprocessing.transformations.simple import (
    SimpleTransformationApplier,
    SimpleTransformationCalculator,
)

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


def test_power_transformer_no_valid_columns_returns_empty_artifact() -> None:
    """If every candidate column fails the box-cox positivity check, artifact is empty."""
    df = pd.DataFrame({"has_zero": [0.0, -1.0, 2.0]})
    art = PowerTransformerCalculator().fit(df, {"method": "box-cox", "columns": ["has_zero"]})
    assert art == {}


def test_power_transformer_no_columns_configured_returns_empty() -> None:
    """``user_picked_no_columns`` short-circuits fit to an empty artifact."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    art = PowerTransformerCalculator().fit(df, {"columns": []})
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


# ---------------------------------------------------------------------------
# SimpleTransformation
# ---------------------------------------------------------------------------


def test_simple_transformation_log_negative_values_become_nan() -> None:
    """Negative inputs to "log" are masked to NaN before ``log1p``."""
    df = pd.DataFrame({"a": [-1.0, 0.0, 3.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "log"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    assert np.isnan(out["a"].iloc[0])
    assert out["a"].iloc[1] == 0.0
    np.testing.assert_allclose(out["a"].iloc[2], np.log1p(3.0))


def test_simple_transformation_sqrt_negative_values_become_nan() -> None:
    """Negative inputs to "sqrt" are masked to NaN."""
    df = pd.DataFrame({"a": [-4.0, 4.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "sqrt"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    assert np.isnan(out["a"].iloc[0])
    assert out["a"].iloc[1] == 2.0


def test_simple_transformation_square() -> None:
    """ "square" squares every value, matching ``numpy.square``."""
    df = pd.DataFrame({"a": [-3.0, 0.0, 4.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "square"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    assert out["a"].tolist() == [9.0, 0.0, 16.0]


def test_simple_transformation_reciprocal_zero_becomes_nan() -> None:
    """Dividing by zero in "reciprocal" produces NaN, not inf/exception."""
    df = pd.DataFrame({"a": [0.0, 2.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "reciprocal"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    assert np.isnan(out["a"].iloc[0])
    assert out["a"].iloc[1] == 0.5


def test_simple_transformation_exp_clips_large_values() -> None:
    """ "exp" clips inputs above ``clip_threshold`` before exponentiating."""
    df = pd.DataFrame({"a": [1000.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "exp", "clip_threshold": 10}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    np.testing.assert_allclose(out["a"].iloc[0], np.exp(10))


def test_simple_transformation_cube_root_handles_negative_values() -> None:
    """ "cube_root" handles negative inputs correctly (unlike sqrt/log)."""
    df = pd.DataFrame({"a": [-8.0, 8.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "cube_root"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    np.testing.assert_allclose(out["a"].tolist(), [-2.0, 2.0])


def test_simple_transformation_unknown_method_skipped() -> None:
    """An unrecognised method is skipped, leaving the source column untouched."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    art = SimpleTransformationCalculator().fit(
        df, {"transformations": [{"column": "a", "method": "bogus_method"}]}
    )
    out = SimpleTransformationApplier().apply(df, art)
    assert out["a"].tolist() == [1.0, 2.0]


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
