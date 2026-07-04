"""PyArrow-backed pandas dtype coverage for representative preprocessing nodes.

Backlog item 2.4 — opt-in test parametrisation for ``pandas[pyarrow]`` dtypes.
Builds small ``pandas.ArrowDtype``-backed DataFrames (via
``pd.Series(..., dtype=pd.ArrowDtype(...))`` and ``convert_dtypes(dtype_backend="pyarrow")``)
and runs them through the same fit/apply path as an equivalent numpy-backed
DataFrame, asserting the results agree. Where a node genuinely mishandles a
pyarrow dtype today, the case is marked ``xfail`` with a clear reason instead
of being hidden or skipped, so the gap stays visible in CI.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from skyulf.preprocessing.encoding.one_hot import (  # noqa: E402
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
)
from skyulf.preprocessing.imputation.simple import (  # noqa: E402
    SimpleImputerApplier,
    SimpleImputerCalculator,
)
from skyulf.preprocessing.scaling.standard import (  # noqa: E402
    StandardScalerApplier,
    StandardScalerCalculator,
)


def _fit_apply(calculator: Any, applier: Any, df: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Run ``fit`` then ``apply`` on ``df`` and return the transformed DataFrame."""
    params = calculator.fit(df, config)
    return applier.apply(df, params)


# ---------------------------------------------------------------------------
# StandardScaler — numeric pyarrow dtype
# ---------------------------------------------------------------------------


def test_standard_scaler_pyarrow_matches_numpy_backend() -> None:
    """StandardScaler produces numerically equal output for numpy vs pyarrow-backed columns."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    df_numpy = pd.DataFrame({"a": values})
    df_arrow = pd.DataFrame({"a": pd.Series(values, dtype=pd.ArrowDtype(pa.float64()))})

    config = {"columns": ["a"]}
    out_numpy = _fit_apply(StandardScalerCalculator(), StandardScalerApplier(), df_numpy, config)
    out_arrow = _fit_apply(StandardScalerCalculator(), StandardScalerApplier(), df_arrow, config)

    np.testing.assert_allclose(
        out_numpy["a"].to_numpy(dtype=float),
        out_arrow["a"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_standard_scaler_pyarrow_convert_dtypes_backend() -> None:
    """StandardScaler works on a frame built via ``convert_dtypes(dtype_backend="pyarrow")``."""
    df_numpy = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]})
    df_arrow = df_numpy.convert_dtypes(dtype_backend="pyarrow")
    assert isinstance(df_arrow["a"].dtype, pd.ArrowDtype)

    config = {"columns": ["a"]}
    out_numpy = _fit_apply(StandardScalerCalculator(), StandardScalerApplier(), df_numpy, config)
    out_arrow = _fit_apply(StandardScalerCalculator(), StandardScalerApplier(), df_arrow, config)

    np.testing.assert_allclose(
        out_numpy["a"].to_numpy(dtype=float),
        out_arrow["a"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# SimpleImputer — float works, int silently truncates the fill value
# ---------------------------------------------------------------------------


def test_simple_imputer_mean_pyarrow_float_matches_numpy_backend() -> None:
    """Mean-strategy SimpleImputer fills NaNs identically for numpy vs pyarrow float columns."""
    values = [1.0, 2.0, 3.0, None, 5.0]
    df_numpy = pd.DataFrame({"a": values})
    df_arrow = pd.DataFrame({"a": pd.Series(values, dtype=pd.ArrowDtype(pa.float64()))})

    config = {"columns": ["a"], "strategy": "mean"}
    out_numpy = _fit_apply(SimpleImputerCalculator(), SimpleImputerApplier(), df_numpy, config)
    out_arrow = _fit_apply(SimpleImputerCalculator(), SimpleImputerApplier(), df_arrow, config)

    np.testing.assert_allclose(
        out_numpy["a"].to_numpy(dtype=float),
        out_arrow["a"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.xfail(
    reason=(
        "SimpleImputer's pandas apply path uses Series.fillna(mean_value) directly; "
        "on an int64[pyarrow] column this silently truncates the (float) mean fill "
        "value to an int instead of raising or upcasting, unlike pandas' own nullable "
        "Int64 dtype which raises TypeError for the same input. Genuine pyarrow dtype "
        "gap in skyulf.preprocessing.imputation.simple, not a test bug."
    ),
    strict=True,
)
def test_simple_imputer_mean_pyarrow_int_matches_numpy_backend() -> None:
    """Mean-strategy SimpleImputer on an int64[pyarrow] column should not silently truncate."""
    values = [1, 2, 3, None, 5]
    df_numpy = pd.DataFrame({"a": values}, dtype="float64")
    df_arrow = pd.DataFrame({"a": pd.Series(values, dtype=pd.ArrowDtype(pa.int64()))})

    config = {"columns": ["a"], "strategy": "mean"}
    out_numpy = _fit_apply(SimpleImputerCalculator(), SimpleImputerApplier(), df_numpy, config)
    out_arrow = _fit_apply(SimpleImputerCalculator(), SimpleImputerApplier(), df_arrow, config)

    np.testing.assert_allclose(
        out_numpy["a"].to_numpy(dtype=float),
        out_arrow["a"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# OneHotEncoder — string pyarrow dtype
# ---------------------------------------------------------------------------


def test_one_hot_encoder_pyarrow_string_matches_numpy_backend() -> None:
    """OneHotEncoder produces the same encoded columns for numpy vs pyarrow string columns."""
    values = ["x", "y", "x", "y", "x"]
    df_numpy = pd.DataFrame({"cat": values})
    df_arrow = pd.DataFrame({"cat": pd.Series(values, dtype=pd.ArrowDtype(pa.string()))})

    config = {"columns": ["cat"]}
    out_numpy = _fit_apply(OneHotEncoderCalculator(), OneHotEncoderApplier(), df_numpy, config)
    out_arrow = _fit_apply(OneHotEncoderCalculator(), OneHotEncoderApplier(), df_arrow, config)

    encoded_cols = [c for c in out_numpy.columns if c.startswith("cat_")]
    assert encoded_cols, "Expected one-hot encoded columns to be produced"
    for col in encoded_cols:
        np.testing.assert_array_equal(
            out_numpy[col].to_numpy(dtype=int),
            out_arrow[col].to_numpy(dtype=int),
        )
