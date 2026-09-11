"""Regression coverage for ratio denominator clamping across existing engines."""

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("epsilon,clamped_ratio", [(None, 1e9), (1e-3, 1000.0)])
def test_ratio_preserves_denominator_sign(engine, epsilon, clamped_ratio, monkeypatch) -> None:
    """Clamping a negative denominator must not invert a generated feature's sign."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    threshold = 1e-9 if epsilon is None else epsilon
    frame = pd.DataFrame(
        [
            (1.0, -2.0, 0.0, -0.5),
            (1.0, -1e-15, 0.0, -clamped_ratio),
            (1.0, 1e-15, 0.0, clamped_ratio),
            (1.0, -threshold, 0.0, -clamped_ratio),
            (1.0, threshold, 0.0, clamped_ratio),
            (1.0, 0.0, 0.0, clamped_ratio),
            (1.0, -0.0, 0.0, clamped_ratio),
            (1.0, None, 0.0, clamped_ratio),
            (1.0, float("nan"), 0.0, clamped_ratio),
            (-1.0, -1e-15, 0.0, clamped_ratio),
            (None, -1e-15, 0.0, 0.0),
            (float("nan"), -1e-15, 0.0, 0.0),
            (1.0, -2 * threshold, 1.5 * threshold, -clamped_ratio),
            (1.0, 2 * threshold, -1.5 * threshold, clamped_ratio),
        ],
        columns=["numerator", "denominator", "adjustment", "expected"],
    )
    expected = frame.pop("expected").to_list()
    if engine == "polars":
        # Use Polars nulls for pandas missing values; native NaN handling is separate.
        frame = pl.from_pandas(frame)
    config = {
        "operations": [
            {
                "operation_type": "ratio",
                "input_columns": ["numerator"],
                "secondary_columns": ["denominator", "adjustment"],
                "output_column": "ratio",
            }
        ]
    }
    if epsilon is not None:
        config["epsilon"] = epsilon

    params = FeatureGenerationCalculator().fit(frame, config)
    result = FeatureGenerationApplier().apply(frame, params)

    assert result["ratio"].to_list() == pytest.approx(expected)
