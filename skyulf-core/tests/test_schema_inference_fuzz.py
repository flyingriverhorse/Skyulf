"""Fuzz tests for schema inference (item 6.3).

Generates weird-but-valid frames (all-null columns, mixed dtypes, single-row,
single-column, empty) and asserts that schema inference never crashes:

* ``SkyulfSchema.from_dataframe`` always returns a consistent schema.
* Every registered Calculator's ``infer_output_schema`` returns ``None`` or a
  ``SkyulfSchema`` without raising, for a generic column-based config.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from skyulf.preprocessing._schema import SkyulfSchema
from skyulf.preprocessing.base import BaseCalculator
from skyulf.registry import NodeRegistry

_COLUMN_NAMES = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=6)


@st.composite
def _weird_frame(draw: st.DrawFn) -> pd.DataFrame:
    """Build a small frame mixing null columns, dtypes, and degenerate shapes."""
    n_cols = draw(st.integers(min_value=1, max_value=4))
    n_rows = draw(st.integers(min_value=0, max_value=5))
    names = draw(st.lists(_COLUMN_NAMES, min_size=n_cols, max_size=n_cols, unique=True))
    data: Dict[str, List[Any]] = {}
    for name in names:
        kind = draw(st.sampled_from(["int", "float", "str", "all_null", "bool"]))
        if kind == "int":
            data[name] = [draw(st.integers(-100, 100)) for _ in range(n_rows)]
        elif kind == "float":
            data[name] = [
                draw(st.floats(allow_nan=True, allow_infinity=False, width=32))
                for _ in range(n_rows)
            ]
        elif kind == "str":
            data[name] = [draw(st.text(max_size=4)) for _ in range(n_rows)]
        elif kind == "bool":
            data[name] = [draw(st.booleans()) for _ in range(n_rows)]
        else:  # all_null
            data[name] = [None] * n_rows
    return pd.DataFrame(data)


@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
@given(df=_weird_frame())
def test_from_dataframe_never_crashes(df: pd.DataFrame) -> None:
    schema = SkyulfSchema.from_dataframe(df)
    assert isinstance(schema, SkyulfSchema)
    # Columns are preserved exactly and dtypes only describe known columns.
    assert list(schema.columns) == list(df.columns)
    assert set(schema.dtypes).issubset(set(df.columns))
    assert len(schema) == len(df.columns)


def _all_calculators() -> List[type]:
    calculators: List[type] = []
    for node_id in NodeRegistry.get_all_metadata():
        try:
            calc = NodeRegistry.get_calculator(node_id)
        except (ValueError, KeyError):
            continue
        if isinstance(calc, type) and issubclass(calc, BaseCalculator):
            calculators.append(calc)
    return calculators


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}),  # normal
        pd.DataFrame({"a": [1]}),  # single row, single column
        pd.DataFrame({"a": [None, None], "b": [1, 2]}),  # all-null column
        pd.DataFrame({"a": pd.Series([], dtype="float64")}),  # empty
        pd.DataFrame({"a": [1, "x", 3.0]}),  # mixed/object dtype
        pd.DataFrame({"a": [np.nan, np.inf, -np.inf]}),  # non-finite
    ],
)
def test_infer_output_schema_never_crashes(df: pd.DataFrame) -> None:
    schema = SkyulfSchema.from_dataframe(df)
    config = {"columns": list(df.columns)}
    for calc_cls in _all_calculators():
        try:
            instance = calc_cls()
        except TypeError:
            continue  # Calculators that need constructor args (e.g. tuners).
        result = instance.infer_output_schema(schema, config)
        assert result is None or isinstance(result, SkyulfSchema)
