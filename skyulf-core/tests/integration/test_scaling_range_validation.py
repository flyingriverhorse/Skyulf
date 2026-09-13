"""Scaling ranges fail clearly at the public fit boundary on both engines."""

from array import array
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.scaling.minmax import MinMaxScalerApplier, MinMaxScalerCalculator
from skyulf.preprocessing.scaling.robust import RobustScalerApplier, RobustScalerCalculator


@pytest.mark.parametrize("frame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
@pytest.mark.parametrize(
    ("calculator", "key"),
    [(MinMaxScalerCalculator, "feature_range"), (RobustScalerCalculator, "quantile_range")],
)
@pytest.mark.parametrize(
    "bounds",
    [
        None,
        [],
        [0],
        [0, 1, 2],
        [None, 1],
        [0, None],
        [float("nan"), 1],
        [0, float("inf")],
        [float("-inf"), 1],
        ["0", 1],
        [False, 1],
        np.array(0),
        np.array([[0], [1]]),
    ],
)
def test_invalid_range_fails_with_named_value_error(
    frame: Any, calculator: Any, key: str, bounds: Any
) -> None:
    """Malformed UI/API bounds must not crash comparisons or poison fitted statistics."""
    with pytest.raises(ValueError, match=key):
        calculator().fit(frame({"x": [0.0, 2.0, 4.0]}), {"columns": ["x"], key: bounds})


@pytest.mark.parametrize("frame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
@pytest.mark.parametrize(
    ("calculator", "key", "bounds"),
    [
        (MinMaxScalerCalculator, "feature_range", [1, 1]),
        (MinMaxScalerCalculator, "feature_range", [2, 1]),
        (RobustScalerCalculator, "quantile_range", [-1, 75]),
        (RobustScalerCalculator, "quantile_range", [25, 101]),
        (RobustScalerCalculator, "quantile_range", [80, 20]),
    ],
)
def test_invalid_range_order_and_percentiles(
    frame: Any, calculator: Any, key: str, bounds: Any
) -> None:
    """Invalid orders and percentiles must identify the field to correct."""
    with pytest.raises(ValueError, match=key):
        calculator().fit(frame({"x": [0.0, 2.0, 4.0]}), {"columns": ["x"], key: bounds})


@pytest.mark.parametrize("frame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
@pytest.mark.parametrize(
    ("calculator", "applier", "key", "bounds", "expected"),
    [
        (MinMaxScalerCalculator, MinMaxScalerApplier, "feature_range", [-1, 1], [-1, 0, 1]),
        (RobustScalerCalculator, RobustScalerApplier, "quantile_range", [0, 100], [-0.5, 0, 0.5]),
        (RobustScalerCalculator, RobustScalerApplier, "quantile_range", [50, 50], [-2, 0, 2]),
    ],
)
def test_valid_json_ranges_keep_scaled_values(
    frame: Any, calculator: Any, applier: Any, key: str, bounds: Any, expected: list[float]
) -> None:
    """JSON list coercion and previously supported quantile endpoints keep their meaning."""
    source = frame({"x": [0.0, 2.0, 4.0]})
    params = calculator().fit(source, {"columns": ["x"], key: bounds})
    result = applier().apply(source, params)
    np.testing.assert_allclose(result["x"].to_numpy(), expected)


def test_numpy_range_remains_supported() -> None:
    """Programmatic callers can retain their existing one-dimensional numeric arrays."""
    source = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
    params = MinMaxScalerCalculator().fit(
        source, {"columns": ["x"], "feature_range": np.array([-1.0, 1.0])}
    )
    np.testing.assert_allclose(MinMaxScalerApplier().apply(source, params)["x"], [-1, 0, 1])


@pytest.mark.parametrize(
    "bounds",
    [
        range(2),
        pd.Series([0, 1]),
        pd.Index([0, 1]),
        array("d", [0, 1]),
        [Decimal("0"), Decimal("1")],
    ],
)
def test_programmatic_minmax_range_compatibility(bounds: Any) -> None:
    """Previously accepted numeric iterables and Decimal endpoints must retain scaled values."""
    source = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
    params = MinMaxScalerCalculator().fit(source, {"columns": ["x"], "feature_range": bounds})
    np.testing.assert_allclose(MinMaxScalerApplier().apply(source, params)["x"], [0, 0.5, 1])


@pytest.mark.parametrize(
    "bounds", [range(0, 101, 100), pd.Series([0, 100]), pd.Index([0, 100]), array("d", [0, 100])]
)
def test_programmatic_robust_range_compatibility(bounds: Any) -> None:
    """Non-JSON callers can continue providing supported iterable quantile bounds."""
    source = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
    params = RobustScalerCalculator().fit(source, {"columns": ["x"], "quantile_range": bounds})
    np.testing.assert_allclose(RobustScalerApplier().apply(source, params)["x"], [-0.5, 0, 0.5])
