"""Constant imputation supports nullable text without changing numeric defaults."""

import numpy as np
import pandas as pd
import pytest

from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_constant_imputer_fills_nullable_text(dtype):
    """Rare grouping's nullable strings must compose with explicit missing-category filling."""
    frame = pd.DataFrame({"cat": pd.Series(["a", pd.NA, None], dtype=dtype)})
    before = frame.copy(deep=True)
    state = SimpleImputerCalculator().fit(
        frame, {"columns": ["cat"], "strategy": "constant", "fill_value": "Missing"}
    )
    result = SimpleImputerApplier().apply(frame, state)
    assert result["cat"].tolist() == ["a", "Missing", "Missing"]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "float64"])
def test_constant_numeric_default_remains_zero(dtype):
    """Normalizing text must not change sklearn's numeric default to a string fill."""
    frame = pd.DataFrame({"value": pd.Series([1, None, 3], dtype=dtype)})
    state = SimpleImputerCalculator().fit(frame, {"columns": ["value"], "strategy": "constant"})
    np.testing.assert_allclose(
        SimpleImputerApplier().apply(frame, state)["value"].to_numpy(dtype=float), [1, 0, 3]
    )
