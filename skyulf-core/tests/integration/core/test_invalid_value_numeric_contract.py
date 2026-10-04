"""OC-140 regressions for numeric rules on explicitly selected nonnumeric data."""

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.preprocessing.cleaning.invalid_value import (
    InvalidValueReplacementApplier,
    InvalidValueReplacementCalculator,
)


@pytest.fixture(params=["pandas", "wrapped_pandas", "polars", "wrapped_polars"])
def engine(request: pytest.FixtureRequest) -> str:
    """Exercise native inputs and both public engine wrappers without engine conversion."""
    return request.param


def _frame(data: dict[str, Any], engine: str) -> Any:
    """Build equivalent frames while retaining the requested public input shape."""
    frame = pd.DataFrame(data)
    if engine == "pandas":
        return frame
    if engine == "wrapped_pandas":
        return SkyulfPandasWrapper(frame)
    polars_frame = pl.from_pandas(frame)
    return SkyulfPolarsWrapper(polars_frame) if engine == "wrapped_polars" else polars_frame


def _assert_equal(actual: Any, expected: Any) -> None:
    """Compare both values and dtypes, including whether a wrapper was retained."""
    assert type(actual) is type(expected)
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    elif isinstance(actual, SkyulfPandasWrapper):
        pd.testing.assert_frame_equal(actual.to_native(), expected.to_native())
    elif isinstance(actual, SkyulfPolarsWrapper):
        assert_frame_equal(actual.to_native(), expected.to_native())
    else:
        assert_frame_equal(actual, expected)


def _expected_output(frame: Any) -> Any:
    """Preserve the dispatcher's existing native-pandas and wrapped-Polars outputs."""
    return frame.to_native() if isinstance(frame, SkyulfPandasWrapper) else frame


@pytest.mark.parametrize("operation", ["fit", "apply"])
@pytest.mark.parametrize(
    "config",
    [
        {"rule": "negative"},
        {"rule": "negative_to_nan"},
        {"rule": "zero"},
        {"rule": "custom_range", "min_value": 0},
        {"rule": "custom_range", "max_value": 100},
        {"replace_inf": True},
        {"replace_neg_inf": True},
    ],
)
def test_numeric_operations_reject_text_without_modifying_source(
    engine: str, operation: str, config: dict[str, Any]
) -> None:
    """Selected text must fail clearly before either engine can erase or rewrite data."""
    data = {"numeric": [-3.0, 0.0, 2.0], "text": ["hello", "-3", "2"]}
    frame = _frame(data, engine)
    original = _frame(data, engine)
    params = {"columns": ["numeric", "text"], **config}
    node = (
        InvalidValueReplacementCalculator()
        if operation == "fit"
        else InvalidValueReplacementApplier()
    )

    with pytest.raises(ValueError) as exc:
        getattr(node, operation)(frame, params)

    assert str(exc.value) == (
        "InvalidValueReplacement requires numeric columns; non-numeric columns: ['text']. "
        "Convert these columns to a numeric type before applying numeric rules."
    )
    _assert_equal(frame, original)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["5", "-3", "7"], dtype="string"),
        pd.Series(["low", "medium", "high"], dtype="category"),
        pd.Series([True, False, True], dtype="boolean"),
        pd.Series(pd.date_range("2026-01-01", periods=3)),
        pd.Series(pd.to_timedelta([-1, 0, 1], unit="D")),
    ],
    ids=["numeric_text", "category", "boolean", "datetime", "duration"],
)
def test_nonnumeric_dtypes_rejected_for_canvas_mode(engine: str, values: pd.Series) -> None:
    """The Canvas numeric-only contract also applies to direct Core configuration."""
    frame = _frame({"value": values}, engine)

    with pytest.raises(ValueError, match="non-numeric columns: .*value"):
        InvalidValueReplacementCalculator().fit(
            frame, {"columns": ["value"], "mode": "negative_to_nan"}
        )


def test_apply_revalidates_numeric_artifact_after_schema_drift(engine: str) -> None:
    """An artifact fitted on numbers must reject text encountered during inference."""
    training = _frame({"value": [5.0, -3.0, 7.0]}, engine)
    inference = _frame({"value": ["5", "oops", "7"]}, engine)
    artifact = InvalidValueReplacementCalculator().fit(
        training, {"columns": ["value"], "mode": "percentage_bounds"}
    )

    with pytest.raises(ValueError, match="non-numeric columns: .*value"):
        InvalidValueReplacementApplier().apply(inference, artifact)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"rule": None, "replace_inf": False, "replace_neg_inf": False},
        {"rule": "unknown"},
        {"rule": "custom_range"},
    ],
)
def test_inactive_operations_preserve_text_and_target(engine: str, config: dict[str, Any]) -> None:
    """Configurations without an effective numeric operation remain true no-ops."""
    frame = _frame({"text": ["hello", "-3", "2"]}, engine)
    target = np.array([1, 0, 1])
    artifact = InvalidValueReplacementCalculator().fit(
        (frame, target), {"columns": ["text"], **config}
    )

    result, result_target = InvalidValueReplacementApplier().apply((frame, target), artifact)

    _assert_equal(result, _expected_output(frame))
    assert result_target is target


def test_auto_selection_preserves_text_and_target(engine: str) -> None:
    """Numeric validation must not reject untouched text or alter the target selection."""
    data = {
        "number": [-3.0, 0.0, 2.0],
        "text": ["a", "b", "c"],
        "duration": pd.to_timedelta([-1, 0, 1], unit="D"),
        "target": [-1, 1, 0],
    }
    frame = _frame(data, engine)
    original = _frame(data, engine)
    target = np.array([0, 1, 0])
    artifact = InvalidValueReplacementCalculator().fit(
        (frame, target), {"mode": "negative_to_nan", "target_column": "target", "value": 0.0}
    )

    result, result_target = InvalidValueReplacementApplier().apply((frame, target), artifact)

    assert artifact["columns"] == ["number"]
    _assert_equal(result, _expected_output(_frame({**data, "number": [0.0, 0.0, 2.0]}, engine)))
    _assert_equal(frame, original)
    assert result_target is target


def test_empty_selection_with_active_rule_preserves_text(engine: str) -> None:
    """An explicit empty selection must not become implicit validation of all columns."""
    frame = _frame({"text": ["a", "b", "c"]}, engine)
    artifact = InvalidValueReplacementCalculator().fit(
        frame, {"columns": [], "mode": "negative_to_nan"}
    )

    result = InvalidValueReplacementApplier().apply(frame, artifact)

    assert artifact == {}
    _assert_equal(result, _expected_output(frame))


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([Decimal("-3"), Decimal("2"), None]),
        pd.Series([-3, 2, None], dtype="Int64"),
        pd.Series([-3.0, 2.0, None], dtype="Float64"),
    ],
    ids=["decimal", "nullable_integer", "nullable_float"],
)
def test_numeric_extension_values_remain_supported(engine: str, values: pd.Series) -> None:
    """Numeric validation must retain Decimal and nullable numeric replacement support."""
    frame = _frame({"value": values}, engine)
    original = _frame({"value": values}, engine)
    artifact = InvalidValueReplacementCalculator().fit(
        frame, {"mode": "negative_to_nan", "value": 0.0}
    )

    result = InvalidValueReplacementApplier().apply(frame, artifact)

    assert artifact["columns"] == ["value"]
    assert type(result) is type(_expected_output(frame))
    native = result.to_native() if isinstance(result, SkyulfPolarsWrapper) else result
    result_values = native["value"].to_list()
    assert result_values[:2] == [0.0, 2.0]
    assert pd.isna(result_values[2])
    _assert_equal(frame, original)
