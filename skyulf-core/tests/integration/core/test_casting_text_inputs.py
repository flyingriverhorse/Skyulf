"""Regressions for datetime and categorical boolean casting across engines."""

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.casting import CastingApplier, CastingCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_type", ["datetime", "date", "datetime64[ns]"])
@pytest.mark.parametrize("coerce_on_error", [True, False])
def test_casting_datetime_preserves_valid_text(
    engine: str, target_type: str, coerce_on_error: bool
) -> None:
    """Date-only text must survive casting alongside timestamps and missing rows."""
    values = ["2024-01-01", "2024-06-15", "2024-06-15T12:34:56", None]
    frame: Any = pd.DataFrame({"x": values, "keep": [1, 2, 3, 4]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    original = frame.clone() if engine == "polars" else frame.copy()
    artifact = CastingCalculator().fit(
        frame, {"column_types": {"x": target_type}, "coerce_on_error": coerce_on_error}
    )

    result = CastingApplier().apply(frame, artifact)

    actual = result["x"].to_list()
    assert actual[:3] == [
        datetime(2024, 1, 1),
        datetime(2024, 6, 15),
        datetime(2024, 6, 15, 12, 34, 56),
    ]
    assert pd.isna(actual[3])
    assert result["keep"].to_list() == [1, 2, 3, 4]
    assert frame.equals(original)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["2024-01-01", "invalid", None], [datetime(2024, 1, 1), None, None]),
        (["invalid", None], [None, None]),
        ([None, None], [None, None]),
        ([], []),
    ],
    ids=["mixed-invalid", "all-invalid", "all-null", "empty"],
)
def test_casting_datetime_default_coercion_preserves_missingness(
    engine: str, values: list[str | None], expected: list[datetime | None]
) -> None:
    """Bad date tokens must become missing without erasing valid dates or raising."""
    frame: Any = pd.DataFrame({"x": pd.Series(values, dtype="string")})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    artifact = CastingCalculator().fit(frame, {"columns": ["x"], "target_type": "datetime"})

    result = CastingApplier().apply(frame, artifact)

    actual = [None if pd.isna(value) else value for value in result["x"].to_list()]
    assert actual == expected
    if engine == "polars":
        assert result.schema["x"] == pl.Datetime("us")
    else:
        assert pd.api.types.is_datetime64_any_dtype(result["x"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("values", [["2024-01-01", "invalid"], ["invalid", None]])
def test_casting_datetime_strict_rejects_invalid_text(
    engine: str, values: list[str | None]
) -> None:
    """Strict datetime casting must reject invalid tokens instead of nulling them."""
    frame: Any = pd.DataFrame({"x": values})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    artifact = CastingCalculator().fit(
        frame, {"columns": ["x"], "target_type": "datetime", "coerce_on_error": False}
    )

    with pytest.raises((ValueError, pl.exceptions.InvalidOperationError)):
        CastingApplier().apply(frame, artifact)


@pytest.mark.parametrize("source_dtype", [pl.String, pl.Categorical, pl.Enum])
@pytest.mark.parametrize("coerce_on_error", [True, False])
def test_casting_categorical_booleans_match_string_aliases(
    source_dtype: Any, coerce_on_error: bool
) -> None:
    """Categorical labels must use the text alias table while preserving nulls."""
    values = ["true", "false", " YES ", "off", "1", "0", None]
    if source_dtype == pl.Enum:
        source_dtype = pl.Enum([value for value in values if value is not None])
    frame = pl.DataFrame({"x": pl.Series(values, dtype=source_dtype)})
    artifact = CastingCalculator().fit(
        frame, {"column_types": {"x": "bool"}, "coerce_on_error": coerce_on_error}
    )

    result = CastingApplier().apply(frame, artifact)

    assert result["x"].dtype == pl.Boolean
    assert result["x"].to_list() == [True, False, True, False, True, False, None]


@pytest.mark.parametrize("engine", ["pandas", "polars-categorical", "polars-enum"])
def test_casting_categorical_boolean_invalid_tokens_coerce(engine: str) -> None:
    """Invalid categorical flags must become missing just like pandas categories."""
    values = ["true", "false", "maybe", None]
    frame: Any = pd.DataFrame({"x": pd.Categorical(values)})
    if engine != "pandas":
        dtype = (
            pl.Categorical
            if engine == "polars-categorical"
            else pl.Enum(["true", "false", "maybe"])
        )
        frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    artifact = CastingCalculator().fit(frame, {"columns": ["x"], "target_type": "boolean"})

    result = CastingApplier().apply(frame, artifact)

    actual = [None if pd.isna(value) else value for value in result["x"].to_list()]
    assert actual == [True, False, None, None]


@pytest.mark.parametrize("source_dtype", [pl.String, pl.Categorical, pl.Enum(["true", "maybe"])])
def test_casting_categorical_boolean_invalid_tokens_raise(source_dtype: Any) -> None:
    """Strict categorical casts must surface the established invalid-token error."""
    frame = pl.DataFrame({"x": pl.Series(["true", "maybe", None], dtype=source_dtype)})
    artifact = CastingCalculator().fit(
        frame, {"columns": ["x"], "target_type": "boolean", "coerce_on_error": False}
    )

    with pytest.raises(ValueError, match="not recognized as true/false"):
        CastingApplier().apply(frame, artifact)
