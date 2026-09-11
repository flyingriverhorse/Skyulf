"""Unsupported feature operations must fail before returning a successful no-op."""

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation import (
    FEATURE_MATH_ALLOWED_TYPES,
    FeatureGenerationApplier,
)
from skyulf.registry import NodeRegistry


def test_polynomial_is_not_advertised_as_a_feature_math_operation() -> None:
    """Consumers of the public allow-list must not offer an unimplemented operation."""
    assert "polynomial" not in FEATURE_MATH_ALLOWED_TYPES


@pytest.mark.parametrize("node_id", ["FeatureGeneration", "FeatureMath", "FeatureGenerationNode"])
@pytest.mark.parametrize("operation_type", ["polynomial", "not_an_operation"])
def test_fit_rejects_unsupported_operation(node_id, operation_type) -> None:
    """All registry aliases must reject unsupported requests before producing an artifact."""
    frame = pd.DataFrame({"x": [2.0, 3.0]})
    calculator = NodeRegistry.get_calculator(node_id)()
    with pytest.raises(ValueError, match=operation_type):
        calculator.fit(
            frame,
            {"operations": [{"operation_type": operation_type, "input_columns": ["x"]}]},
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("operation_type", ["polynomial", "not_an_operation"])
def test_replay_rejects_unsupported_operation(engine, operation_type) -> None:
    """Old artifacts cannot bypass validation and silently omit a requested feature."""
    frame = pd.DataFrame({"x": [2.0, 3.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    params = {
        "operations": [
            {"method": "add", "input_columns": ["x"], "constants": [1.0]},
            {"operation_type": operation_type, "input_columns": ["x"]},
        ]
    }
    with pytest.raises(ValueError, match=operation_type):
        FeatureGenerationApplier().apply(frame, params)


def test_omitted_operation_type_still_defaults_to_arithmetic() -> None:
    """Existing row-local configs without an explicit operation type remain valid."""
    frame = pd.DataFrame({"x": [2.0, 3.0]})
    params = NodeRegistry.get_calculator("FeatureMath")().fit(
        frame,
        {"operations": [{"method": "add", "input_columns": ["x"], "constants": [1.0]}]},
    )
    result = FeatureGenerationApplier().apply(frame, params)
    assert result["arithmetic_0"].tolist() == [3.0, 4.0]
