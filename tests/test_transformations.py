import numpy as np
import pandas as pd
import pytest
from skyulf.preprocessing.transformations import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
    SimpleTransformationApplier,
    SimpleTransformationCalculator,
)


def test_power_transformer():
    # Skewed data
    df = pd.DataFrame({"A": np.random.exponential(size=100)})

    calc = PowerTransformerCalculator()
    params = calc.fit(df, {"method": "yeo-johnson", "columns": ["A"]})

    applier = PowerTransformerApplier()
    res = applier.apply(df, params)

    # Should be more gaussian (mean approx 0, std approx 1)
    assert abs(res["A"].mean()) < 0.5
    assert abs(res["A"].std() - 1) < 0.5


def test_log_transformer():
    df = pd.DataFrame({"A": [0, 1, 10, 100]})

    calc = SimpleTransformationCalculator()
    params = calc.fit(df, {"transformations": [{"column": "A", "method": "log"}]})

    applier = SimpleTransformationApplier()
    res = applier.apply(df, params)

    assert res["A"].iloc[0] == 0  # log1p(0) = 0
    assert res["A"].iloc[3] > 4  # log1p(100) approx 4.6
