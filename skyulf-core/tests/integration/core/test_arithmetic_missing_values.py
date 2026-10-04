"""Arithmetic features must fill native nulls and NaNs consistently."""

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("fillna", [None, 0, 10])
@pytest.mark.parametrize("method", ["add", "subtract", "multiply", "divide"])
def test_arithmetic_fills_missing_primary_and_secondary_operands(
    engine: str, fillna: int | None, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing values in either operand must use the configured replacement on replay."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {
        "a": [1.0, float("nan"), None, 6.0, 6.0],
        "b": [2.0, 2.0, 2.0, float("nan"), None],
    }
    # Direct construction preserves the distinction between Polars NaN and null.
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    operation = {
        "operation_type": "arithmetic",
        "method": method,
        "input_columns": ["a"],
        "secondary_columns": ["b"],
        "constants": [2],
        "output_column": "result",
    }
    if fillna is not None:
        operation["fillna"] = fillna
    params = FeatureGenerationCalculator().fit(frame, {"operations": [operation]})

    result = FeatureGenerationApplier().apply(frame, params)

    replacement = fillna or 0
    expected = {
        "add": [5, replacement + 4, replacement + 4, 8 + replacement, 8 + replacement],
        "subtract": [-3, replacement - 4, replacement - 4, 4 - replacement, 4 - replacement],
        "multiply": [4, replacement * 4, replacement * 4, 12 * replacement, 12 * replacement],
        "divide": [
            0.25,
            replacement / 4,
            replacement / 4,
            3 / max(replacement, 1e-9),
            3 / max(replacement, 1e-9),
        ],
    }
    assert result[["a", "b"]].equals(frame)
    assert result["result"].to_list() == pytest.approx(expected[method])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("values,expected", [([1, 3], [3, 5]), ([None, None], [12, 12])])
def test_arithmetic_fill_preserves_integer_and_null_only_inputs(
    engine: str, values: list, expected: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN normalization must also support integer and untyped null-only columns."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"a": values}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    config = {
        "operations": [
            {
                "operation_type": "arithmetic",
                "method": "add",
                "input_columns": ["a"],
                "constants": [2],
                "fillna": 10,
                "output_column": "result",
            }
        ]
    }
    params = FeatureGenerationCalculator().fit(frame, config)

    result = FeatureGenerationApplier().apply(frame, params)

    assert result["result"].to_list() == pytest.approx(expected)
