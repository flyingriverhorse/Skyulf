"""Group imputer previews must select the same fill columns as fitted execution."""

import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
@pytest.mark.parametrize("columns", [None, [], ["value"], ["target"], ["missing"]])
def test_group_schema_matches_fitted_column_selection(engine, strategy, columns):
    """Automatic selection excludes keys and targets; explicit selection keeps its own meaning."""
    frame = pd.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "value": pd.Series([1, None, 5, 7], dtype="Int64"),
            "other": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
            "text": ["a", "a", "b", "b"],
        }
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"group_by": "group", "strategy": strategy, "target_column": "target"}
    if columns is not None:
        config["columns"] = columns
    calculator = GroupImputerCalculator()
    state = calculator.fit(frame, config)
    actual = SkyulfSchema.from_dataframe(GroupImputerApplier().apply(frame, state))
    predicted = calculator.infer_output_schema(SkyulfSchema.from_dataframe(frame), config)

    if not columns and strategy in ("mean", "median"):
        assert predicted is None
        return
    assert predicted is not None
    assert predicted.columns == actual.columns
    assert {col: dtype.lower() for col, dtype in predicted.dtypes.items()} == {
        col: dtype.lower() for col, dtype in actual.dtypes.items()
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("columns", [None, []])
def test_auto_group_schema_does_not_guess_binary_or_constant_selection(engine, columns):
    """Schema alone cannot tell which numeric columns automatic fitting will exclude."""
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "x": [1.0, None, 3.0, None],
            "flag": [0, 1, 0, 1],
            "constant": [7, 7, 7, 7],
        }
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"group_by": "group", "strategy": "mean"}
    if columns is not None:
        config["columns"] = columns
    calculator = GroupImputerCalculator()
    state = calculator.fit(frame, config)
    output = GroupImputerApplier().apply(frame, state)
    actual = SkyulfSchema.from_dataframe(output)
    assert actual.dtypes["flag"].lower() == "int64"
    assert actual.dtypes["constant"].lower() == "int64"
    assert calculator.infer_output_schema(SkyulfSchema.from_dataframe(frame), config) is None


@pytest.mark.parametrize("node", ["SimpleImputer", "KNNImputer", "IterativeImputer"])
def test_other_imputers_keep_empty_selection_as_noop(node):
    """Group-specific automatic selection must not widen an explicit empty selection elsewhere."""
    frame = pd.DataFrame({"value": pd.Series([1, None, 3], dtype="Int64")})
    calculator = NodeRegistry.get_calculator(node)()
    state = calculator.fit(frame, {"columns": []})
    schema = SkyulfSchema.from_dataframe(frame)
    predicted = calculator.infer_output_schema(schema, {"columns": []})
    output = NodeRegistry.get_applier(node)().apply(frame, state)

    assert predicted == schema
    pd.testing.assert_frame_equal(output, frame)
