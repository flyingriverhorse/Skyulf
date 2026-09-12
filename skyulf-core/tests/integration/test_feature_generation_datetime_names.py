"""Datetime outputs must honor configured names without losing existing columns."""

from copy import deepcopy
from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


@pytest.fixture(params=["pandas", "polars", "pandas-wrapper", "polars-wrapper"])
def frame_factory(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Exercise both native and public wrapper dispatch without ambient engine overrides."""
    engine = request.param
    monkeypatch.setenv("SKYULF_ENGINE", engine.split("-")[0])

    def make_frame(data: dict[str, Any]) -> Any:
        """Construct the selected engine directly from the same source values."""
        if engine == "pandas":
            return pd.DataFrame(data)
        if engine == "polars":
            return pl.DataFrame(data)
        if engine == "pandas-wrapper":
            return SkyulfPandasWrapper(pd.DataFrame(data))
        return SkyulfPolarsWrapper(pl.DataFrame(data))

    return make_frame


def _native(frame: Any) -> Any:
    """Expose native columns for value assertions on every public input form."""
    return frame.to_native() if hasattr(frame, "to_native") else frame


def _date_op(**overrides: Any) -> dict[str, Any]:
    """Mirror the Canvas operation payload while allowing naming edge cases."""
    return {
        "operation_type": "datetime_extract",
        "method": "year",
        "input_columns": ["dt"],
        "datetime_features": ["year"],
        "output_column": "",
        **overrides,
    }


@pytest.mark.parametrize(
    "naming,features,expected",
    [
        ({"output_column": "custom_year"}, ["year"], ["custom_year"]),
        ({"output_column": "calendar"}, ["year", "month"], ["calendar_year", "calendar_month"]),
        ({}, ["year", "month"], ["dt_year", "dt_month"]),
        ({"output_prefix": "calendar"}, ["year"], ["calendar_dt_year"]),
        (
            {"output_prefix": "ignored", "output_column": "custom_year"},
            ["year"],
            ["custom_year"],
        ),
    ],
)
def test_datetime_output_naming(frame_factory, naming, features, expected) -> None:
    """A UI output name must reach generated columns and retain unnamed defaults."""
    frame = frame_factory({"dt": ["2024-01-15", "2025-07-04"]})
    config = {"operations": [_date_op(datetime_features=features, **naming)]}
    params = FeatureGenerationCalculator().fit(frame, config)
    result = _native(FeatureGenerationApplier().apply(frame, params))

    assert list(result.columns) == ["dt", *expected]
    assert result[expected[0]].to_list() == [2024, 2025]
    if len(features) > 1:
        assert result[expected[1]].to_list() == [1, 7]
    assert list(frame.columns) == ["dt"]


@pytest.mark.parametrize("custom_name", [None, "dt_year"])
@pytest.mark.parametrize("allow_overwrite", [False, True])
def test_datetime_input_and_generated_collisions(
    frame_factory, custom_name, allow_overwrite
) -> None:
    """Existing and earlier generated columns survive unless overwrite is explicit."""
    frame = frame_factory({"dt": ["2024-01-15"], "dt_year": [99], "dt_year_1": [98]})
    op = _date_op(output_column=custom_name)
    config = {"allow_overwrite": allow_overwrite, "operations": [op, deepcopy(op)]}
    params = FeatureGenerationCalculator().fit(frame, config)
    result = _native(FeatureGenerationApplier().apply(frame, params))

    if allow_overwrite:
        assert list(result.columns) == ["dt", "dt_year", "dt_year_1"]
        assert result["dt_year"].to_list() == [2024]
    else:
        assert list(result.columns) == ["dt", "dt_year", "dt_year_1", "dt_year_2", "dt_year_3"]
        assert result["dt_year"].to_list() == [99]
        assert result["dt_year_2"].to_list() == result["dt_year_3"].to_list() == [2024]
    assert result["dt_year_1"].to_list() == [98]
    assert _native(frame)["dt_year"].to_list() == [99]


def test_multiple_sources_keep_distinct_names_at_inference(frame_factory) -> None:
    """Missing inference sources cannot change the names of remaining configured outputs."""
    training = frame_factory({"dt": ["2024-01-15"], "end": ["2026-02-03"]})
    config = {"operations": [_date_op(input_columns=["dt", "end"], output_column="calendar")]}
    params = FeatureGenerationCalculator().fit(training, config)
    applier = FeatureGenerationApplier()
    fitted = _native(applier.apply(training, params))
    inferred = _native(applier.apply(frame_factory({"dt": ["2025-01-15"]}), params))

    assert list(fitted.columns) == ["dt", "end", "calendar_dt_year", "calendar_end_year"]
    assert fitted["calendar_end_year"].to_list() == [2026]
    assert list(inferred.columns) == ["dt", "calendar_dt_year"]
    assert inferred["calendar_dt_year"].to_list() == [2025]


@pytest.mark.parametrize("allow_overwrite", [False, True])
def test_datetime_repeated_features_reserve_each_output_name(
    frame_factory, allow_overwrite
) -> None:
    """Batched Polars expressions must reserve aliases before they are materialized."""
    frame = frame_factory({"dt": ["2024-01-15"]})
    params = FeatureGenerationCalculator().fit(
        frame,
        {
            "allow_overwrite": allow_overwrite,
            "operations": [_date_op(datetime_features=["year", "year"])],
        },
    )
    result = _native(FeatureGenerationApplier().apply(frame, params))

    expected_columns = ["dt", "dt_year"] if allow_overwrite else ["dt", "dt_year", "dt_year_1"]
    assert list(result.columns) == expected_columns
    assert all(result[column].to_list() == [2024] for column in expected_columns[1:])


def test_datetime_names_replay_without_mutating_artifact_or_config(frame_factory) -> None:
    """Replay stays deterministic and old plain artifacts need no new naming fields."""
    frame = frame_factory({"dt": ["2024-01-15"], "named": [99]})
    config = {"operations": [_date_op(output_column="named")]}
    saved_config = deepcopy(config)
    params = FeatureGenerationCalculator().fit(frame, config)
    saved_params = deepcopy(params)
    applier = FeatureGenerationApplier()
    first = _native(applier.apply(frame, params))
    replay = _native(applier.apply(frame, params))
    repeated = _native(applier.apply(first, params))
    legacy = _native(applier.apply(frame, saved_config))

    assert first.equals(replay) and first.equals(legacy)
    assert list(first.columns) == ["dt", "named", "named_1"]
    assert list(repeated.columns) == ["dt", "named", "named_1", "named_2"]
    assert repeated["named_2"].to_list() == [2024]
    assert params == saved_params and config == saved_config


def test_datetime_named_output_feeds_later_operation(frame_factory) -> None:
    """Downstream operations must read the configured datetime name in the same node."""
    frame = frame_factory({"dt": ["2024-01-15", "2025-01-15"]})
    config = {
        "operations": [
            _date_op(output_column="calendar_year"),
            {
                "operation_type": "arithmetic",
                "method": "subtract",
                "input_columns": ["calendar_year"],
                "constants": [2000],
                "output_column": "elapsed",
            },
        ]
    }
    calculator = FeatureGenerationCalculator()
    predicted = calculator.infer_output_schema(SkyulfSchema.from_dataframe(frame), config)
    params = calculator.fit(frame, config)
    result = _native(FeatureGenerationApplier().apply(frame, params))
    actual_schema = SkyulfSchema.from_dataframe(result)

    assert predicted is None  # This node intentionally falls back to runtime introspection.
    assert actual_schema.columns == ("dt", "calendar_year", "elapsed")
    assert result["elapsed"].to_list() == [24, 25]


@pytest.mark.parametrize("allow_overwrite", [False, True])
def test_datetime_output_can_target_its_source(frame_factory, allow_overwrite) -> None:
    """An explicit source name respects overwrite without mutating the caller's frame."""
    frame = frame_factory({"dt": ["2024-01-15"]})
    params = FeatureGenerationCalculator().fit(
        frame,
        {
            "allow_overwrite": allow_overwrite,
            "operations": [_date_op(output_column="dt")],
        },
    )
    result = _native(FeatureGenerationApplier().apply(frame, params))

    assert result["dt" if allow_overwrite else "dt_1"].to_list() == [2024]
    assert list(result.columns) == (["dt"] if allow_overwrite else ["dt", "dt_1"])
    assert _native(frame)["dt"].to_list() == ["2024-01-15"]


def test_datetime_named_output_supplies_fitted_group_aggregation(frame_factory) -> None:
    """Fit-time intermediate columns must match the names used during inference replay."""
    frame = frame_factory(
        {"dt": ["2024-01-15", "2024-02-15", "2025-03-15"], "value": [2.0, 4.0, 8.0]}
    )
    config = {
        "operations": [
            _date_op(output_column="calendar_year"),
            {
                "operation_type": "group_agg",
                "method": "mean",
                "input_columns": ["calendar_year"],
                "secondary_columns": ["value"],
                "output_column": "annual_mean",
            },
        ]
    }
    params = FeatureGenerationCalculator().fit(frame, config)
    inference = frame_factory({"dt": ["2024-12-31", "2025-12-31"], "value": [99.0, 99.0]})
    result = _native(FeatureGenerationApplier().apply(inference, params))

    assert result["calendar_year"].to_list() == [2024, 2025]
    assert result["annual_mean"].to_list() == [3.0, 8.0]
