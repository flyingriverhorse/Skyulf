"""Regress automatic selector task inference for pandas extension targets."""

import numpy as np
import pandas as pd
import pytest

from skyulf.preprocessing.feature_selection import (
    ModelBasedSelectionCalculator,
    UnivariateSelectionCalculator,
)
from skyulf.preprocessing.feature_selection._common import (
    _infer_problem_type,
    _prepare_sklearn_y,
    _resolve_problem_type,
)


@pytest.fixture
def multiclass_features() -> pd.DataFrame:
    """Keep each of eleven classes repeated so a numeric cutoff cannot mask dtype errors."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "signal": np.repeat(np.arange(11), 4) + rng.normal(0, 0.1, 44),
            "noise": rng.normal(size=44),
        }
    )


@pytest.mark.parametrize(
    "calculator", [UnivariateSelectionCalculator, ModelBasedSelectionCalculator]
)
@pytest.mark.parametrize("dtype", ["object", "string[python]", "string[pyarrow]", "category"])
@pytest.mark.parametrize("embedded_target", [False, True])
def test_string_target_dtype_preserves_selector_result(
    multiclass_features: pd.DataFrame,
    calculator: type,
    dtype: str,
    embedded_target: bool,
) -> None:
    """Equivalent class labels must select the same features across pandas storage dtypes."""
    labels = np.repeat([f"class-{i}" for i in range(11)], 4)
    expected_target = pd.Series(labels, dtype=object, name="target")
    target = pd.Series(labels, dtype=dtype, name="target")
    config = {"k": 1, "max_features": 1, "target_column": "target"}
    expected = calculator().fit((multiclass_features, expected_target), config)
    data = (
        multiclass_features.assign(target=target)
        if embedded_target
        else (multiclass_features, target)
    )

    actual = calculator().fit(data, config)

    assert actual == expected
    assert actual["selected_columns"] == ["signal"]


@pytest.mark.parametrize(
    "calculator", [UnivariateSelectionCalculator, ModelBasedSelectionCalculator]
)
@pytest.mark.parametrize("offset", [0, 0.5])
def test_numeric_categories_preserve_numeric_task_semantics(
    multiclass_features: pd.DataFrame, calculator: type, offset: float
) -> None:
    """Numeric category storage must retain the existing numeric target heuristic."""
    values = np.repeat(np.arange(11), 4) + offset
    target = pd.Series(values, dtype="category", name="target")
    expected_target = pd.Series(values, name="target")
    config = {"k": 1, "max_features": 1}
    expected = calculator().fit((multiclass_features, expected_target), config)

    actual = calculator().fit((multiclass_features, target), config)

    assert _infer_problem_type(target) == "regression"
    assert actual == expected


@pytest.mark.parametrize("dtype", ["int64", "Int64", "float64", "Float64", "category"])
@pytest.mark.parametrize("unique_count, expected", [(10, "classification"), (11, "regression")])
def test_numeric_target_cutoff_is_preserved(dtype: str, unique_count: int, expected: str) -> None:
    """Adding extension label support must preserve the documented numeric cutoff."""
    target = pd.Series(np.repeat(np.arange(unique_count), 4), dtype=dtype)

    assert _infer_problem_type(target) == expected


@pytest.mark.parametrize("declared", ["classification", "regression"])
@pytest.mark.parametrize("dtype", ["string", "category", "float64"])
def test_explicit_problem_type_takes_precedence(declared: str, dtype: str) -> None:
    """A caller's task choice must keep precedence over dtype inference."""
    target = pd.Series(np.arange(11), dtype=dtype)

    assert _resolve_problem_type(declared, target) == declared


def test_explicit_regression_keeps_numeric_category_values() -> None:
    """Regression overrides must use numeric labels as values rather than category codes."""
    values = np.array([1.5, 4.5, 2.5, 1.5])
    target = pd.Series(values, dtype="category")

    prepared = _prepare_sklearn_y(target, "regression")

    np.testing.assert_array_equal(prepared, values)
