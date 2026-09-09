"""Compatibility controls for row-local calendar parsing and neutral target filtering."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node_type", ["DropMissingRows", "Deduplicate"])
@pytest.mark.parametrize("target_kind", ["numpy", "list"])
@pytest.mark.parametrize("empty", [False, True], ids=["survivors", "empty"])
def test_neutral_target_filtering_preserves_container_shape_and_positions(
    engine, node_type, target_kind, empty
):
    """Filtering must preserve multi-output labels, empty array shape, and positional alignment."""
    values = [1.0, 1.0, None, 3.0]
    params = {"subset": ["x"]}
    if empty:
        values = [None] * 4 if node_type == "DropMissingRows" else [1.0] * 4
        if node_type == "Deduplicate":
            params["keep"] = "none"
    pandas_frame = pd.DataFrame({"x": values}, index=[8, 8, 2, 2])
    frame = pandas_frame if engine == "pandas" else pl.from_pandas(pandas_frame)
    original = np.asarray([[10, 100], [11, 110], [12, 120], [13, 130]], dtype=np.int16)
    target = original if target_kind == "numpy" else original.tolist()
    calculator = NodeRegistry.get_calculator(node_type)()
    artifact = calculator.fit((frame, target), params)
    result, result_target = NodeRegistry.get_applier(node_type)().apply((frame, target), artifact)
    kept = [] if empty else ([0, 1, 3] if node_type == "DropMissingRows" else [0, 2, 3])
    assert isinstance(result_target, type(target))
    assert len(result) == len(kept)
    if target_kind == "numpy":
        assert result_target.dtype == original.dtype
        assert result_target.shape == (len(kept), 2)
        np.testing.assert_array_equal(result_target, original[kept])
    else:
        assert result_target == original[kept].tolist()
    if engine == "pandas":
        assert result.index.tolist() == pandas_frame.index[kept].tolist()


_CALENDAR_CASES = {
    "mixed_strings": (
        ["04/05/2024", "2024-01-02", None, "bad"],
        [4, 1, None, None],
        [14, 1, None, None],
    ),
    "native_date": ([date(2024, 1, 2), None], [1, None], [1, None]),
    "native_datetime": ([datetime(2024, 1, 2, 23, 15), None], [1, None], [1, None]),
    "all_null": ([None, None], [None, None], [None, None]),
    "empty": ([], [], []),
}


def _calendar_config(node_type, features):
    """Choose each public node's existing calendar configuration shape."""
    if node_type == "DateFeatures":
        return {"columns": ["date"], "features": features}
    return {
        "operations": [
            {
                "operation_type": "datetime_extract",
                "input_columns": ["date"],
                "datetime_features": features,
            }
        ]
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node_type", ["DateFeatures", "FeatureGeneration", "FeatureGenerationNode", "FeatureMath"]
)
@pytest.mark.parametrize("case", list(_CALENDAR_CASES))
def test_calendar_parsing_keeps_native_null_and_empty_input_contracts(engine, node_type, case):
    """All calendar paths must retain requested features for valid, missing, and empty inputs."""
    values, expected_months, expected_weeks = _CALENDAR_CASES[case]
    frame = pd.DataFrame({"date": values}) if engine == "pandas" else pl.DataFrame({"date": values})
    week = "weekofyear" if node_type == "DateFeatures" else "week"
    config = _calendar_config(node_type, [week, "month"])
    artifact = NodeRegistry.get_calculator(node_type)().fit(frame, config)
    result = NodeRegistry.get_applier(node_type)().apply(frame, artifact)
    assert isinstance(result, type(frame))
    assert "date_month" in result.columns and f"date_{week}" in result.columns
    months = [None if pd.isna(value) else int(value) for value in result["date_month"].to_list()]
    weeks = [None if pd.isna(value) else int(value) for value in result[f"date_{week}"].to_list()]
    assert months == expected_months
    assert weeks == expected_weeks


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node_type", ["DateFeatures", "FeatureGeneration", "FeatureGenerationNode", "FeatureMath"]
)
def test_calendar_timezone_strings_preserve_existing_timezone_interpretation(engine, node_type):
    """Row-local parsing must retain each path's existing timezone treatment for valid strings."""
    values = ["2024-01-02T23:00:00-05:00", None, "bad"]
    frame = pd.DataFrame({"date": values}) if engine == "pandas" else pl.DataFrame({"date": values})
    config = _calendar_config(node_type, ["day", "hour"])
    artifact = NodeRegistry.get_calculator(node_type)().fit(frame, config)
    result = NodeRegistry.get_applier(node_type)().apply(frame, artifact)
    day, hour = (2, 23) if node_type == "DateFeatures" and engine == "pandas" else (3, 4)
    assert result["date_day"][0] == day
    assert result["date_hour"][0] == hour
    assert pd.isna(result["date_day"][1]) and pd.isna(result["date_day"][2])
