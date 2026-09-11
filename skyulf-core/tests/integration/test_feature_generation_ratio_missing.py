"""Native engine inputs must agree on missing values in feature ratios."""

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("epsilon", [1e-9, 1e-3])
def test_ratio_treats_native_nan_and_null_as_zero(
    engine: str, epsilon: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN in one operand must not poison a ratio or hide its other contributions."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    nan = float("nan")
    inf = float("inf")
    rows = [
        (1.0, 0.0, nan, 0.0),
        (nan, 0.0, 2.0, 0.0),
        (1.0, nan, 2.0, 0.0),
        (1.0, 2.0, nan, 4.0),
        (nan, 2.0, 4.0, 0.0),
        (None, nan, None, nan),
        (nan, None, -epsilon / 2, None),
        (1.0, nan, -epsilon / 2, nan),
        (nan, -2.0, nan, -epsilon / 2),
        (inf, 0.0, 2.0, 0.0),
        (1.0, 0.0, inf, 0.0),
    ]
    columns = ["n1", "n2", "d1", "d2"]
    # Construct each engine directly: pl.from_pandas would erase native NaNs.
    frame = (
        pd.DataFrame(rows, columns=columns)
        if engine == "pandas"
        else pl.DataFrame(rows, schema=columns, orient="row")
    )
    config = {
        "epsilon": epsilon,
        "operations": [
            {
                "operation_type": "ratio",
                "input_columns": ["n1", "n2"],
                "secondary_columns": ["d1", "d2"],
                "output_column": "ratio",
            }
        ],
    }

    params = FeatureGenerationCalculator().fit(frame, config)
    result = FeatureGenerationApplier().apply(frame, params)

    expected = [1 / epsilon, 0.0, 0.5, 0.75, 0.5, 0.0, 0.0, -1 / epsilon, 2 / epsilon, inf, 0.0]
    assert result["ratio"].to_list() == pytest.approx(expected)
    assert result[columns].equals(frame)
    assert params["epsilon"] == epsilon


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "numerators,denominators,expected",
    [([2, 6], [4, 3], [0.5, 2.0]), ([None, None], [None, None], [0.0, 0.0])],
)
def test_ratio_preserves_integer_and_untyped_null_inputs(
    engine: str,
    numerators: list,
    denominators: list,
    expected: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalizing NaNs must keep integer and null-only inputs usable on both engines."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"n": numerators, "d": denominators}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    config = {
        "operations": [
            {
                "operation_type": "ratio",
                "input_columns": ["n"],
                "secondary_columns": ["d"],
                "output_column": "ratio",
            }
        ]
    }

    params = FeatureGenerationCalculator().fit(frame, config)
    result = FeatureGenerationApplier().apply(frame, params)

    assert result["ratio"].to_list() == pytest.approx(expected)
