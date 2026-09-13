"""Pin observable casting and text-dispatch behavior after helper extraction."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing.casting import CastingApplier, CastingCalculator
from skyulf.preprocessing.vectorization.count_vectorizer import (
    CountVectorizerApplier,
    CountVectorizerCalculator,
)


def _frame_for_engine(frame: pd.DataFrame, engine: str) -> Any:
    """Exercise native frames and the supported public Polars wrapper."""
    if engine == "pandas":
        return frame
    native = pl.from_pandas(frame)
    return EngineRegistry.wrap(native) if engine == "polars_wrapped" else native


def _as_pandas(frame: Any) -> pd.DataFrame:
    """Expose frame values without depending on wrapper internals."""
    return frame if isinstance(frame, pd.DataFrame) else frame.to_pandas()


@pytest.mark.parametrize("engine", ["pandas", "polars", "polars_wrapped"])
def test_strict_integral_float_cast_preserves_width_nulls_and_rows(engine: str) -> None:
    """Strict casts must accept integral floats without widening or losing missing rows."""
    source = pd.DataFrame(
        {"quantity": [12.0, None, -4.0], "keep": ["first", "missing", "last"]},
        index=pd.Index([31, 9, 52], name="row"),
    )
    frame = _frame_for_engine(source, engine)
    original = _as_pandas(frame).copy(deep=True)
    labels = pd.Series([1, 0, 1], index=source.index, name="target")
    artifact = CastingCalculator().fit(
        (frame, labels),
        {
            "columns": ["quantity", "absent"],
            "target_type": "int16",
            "coerce_on_error": False,
        },
    )

    result, result_labels = CastingApplier().apply((frame, labels), artifact)
    values = _as_pandas(result)

    assert artifact["type_map"] == {"quantity": "int16"}
    assert artifact["categories"] == {}
    assert result_labels is labels
    assert type(result) is type(frame)
    assert values.columns.tolist() == ["quantity", "keep"]
    assert values["quantity"].dropna().tolist() == [12, -4]
    assert values["quantity"].isna().tolist() == [False, True, False]
    if engine == "pandas":
        assert result["quantity"].dtype == pd.Int16Dtype()
        pd.testing.assert_index_equal(result.index, source.index)
    else:
        native = result.to_native() if engine == "polars_wrapped" else result
        assert native.schema == {"quantity": pl.Int16, "keep": pl.String}
    pd.testing.assert_series_equal(values["keep"], _as_pandas(frame)["keep"])
    pd.testing.assert_frame_equal(_as_pandas(frame), original)


@pytest.mark.parametrize("engine", ["polars", "polars_wrapped"])
@pytest.mark.parametrize("fractional", [0.25, 100000.001])
def test_strict_polars_integer_cast_rejects_fractional_values(
    engine: str, fractional: float
) -> None:
    """Strict casting must reject fractions even when large values look nearly integral."""
    frame = _frame_for_engine(
        pd.DataFrame({"quantity": [1.0, fractional, None], "keep": [7, 8, 9]}), engine
    )
    original = _as_pandas(frame).copy(deep=True)
    artifact = CastingCalculator().fit(
        frame,
        {"column_types": {"quantity": "int"}, "coerce_on_error": False},
    )

    with pytest.raises(ValueError, match="quantity.*fractional values"):
        CastingApplier().apply(frame, artifact)

    pd.testing.assert_frame_equal(_as_pandas(frame), original)


@pytest.mark.parametrize("engine", ["pandas", "polars", "polars_wrapped"])
@pytest.mark.parametrize("drop_original", [False, True])
def test_nonstring_vectorization_restores_engine_and_preserves_text_semantics(
    engine: str, drop_original: bool
) -> None:
    """Boolean text must keep pandas spelling and restore its input engine after fallback."""
    source = pd.DataFrame(
        {"flag": [True, False, None], "keep": pd.array([7, 8, 9], dtype="int16")},
        index=pd.Index([31, 9, 52], name="row"),
    )
    frame = _frame_for_engine(source, engine)
    original = _as_pandas(frame).copy(deep=True)
    labels = pd.Series([1, 0, 1], index=source.index, name="target")
    artifact = CountVectorizerCalculator().fit(
        (frame, labels),
        {"columns": ["flag"], "lowercase": False, "drop_original": drop_original},
    )

    result, result_labels = CountVectorizerApplier().apply((frame, labels), artifact)
    values = _as_pandas(result)

    assert result_labels is labels
    assert type(result) is type(frame)
    expected_columns = ["keep", "flag__count__False", "flag__count__True"]
    if not drop_original:
        expected_columns.insert(0, "flag")
        pd.testing.assert_series_equal(values["flag"], original["flag"])
    assert values.columns.tolist() == expected_columns
    assert values[["flag__count__False", "flag__count__True"]].values.tolist() == [
        [0, 1],
        [1, 0],
        [0, 0],
    ]
    pd.testing.assert_series_equal(values["keep"], original["keep"])
    if engine == "pandas":
        pd.testing.assert_index_equal(result.index, source.index)
    else:
        native = result.to_native() if engine == "polars_wrapped" else result
        assert native.schema["keep"] == pl.Int16
        if not drop_original:
            assert native.schema["flag"] == pl.Boolean
    pd.testing.assert_frame_equal(_as_pandas(frame), original)
