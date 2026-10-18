"""New FeatureMath fits validate configuration and keep calendar dtypes stable across batches."""

from copy import deepcopy

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.preprocessing.feature_generation.generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)


def _frame(engine):
    """Use a valid date and an invalid neighbor to expose pandas batch-dependent calendar types."""
    frame = pd.DataFrame({"date": ["2024-01-07T12:13:14Z", "bad"], "x": [2.0, 3.0]})
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "operation",
    [
        {"operation_type": "arithmetic", "method": "typo", "input_columns": ["x"]},
        {
            "operation_type": "datetime_extract",
            "input_columns": ["date"],
            "datetime_features": ["year", "typo"],
        },
        {
            "operation_type": "group_agg",
            "input_columns": ["date"],
            "secondary_columns": ["x"],
            "method": "typo",
        },
    ],
)
def test_unknown_feature_math_choices_fail_before_fitting(engine, operation):
    """Typos must explain valid choices instead of silently omitting requested columns."""
    with pytest.raises(ValueError, match="typo"):
        FeatureGenerationCalculator().fit(_frame(engine), {"operations": [operation]})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_feature_math_operation_errors_propagate_but_legacy_replay_keeps_skip(engine):
    """New validated artifacts must not hide execution failures while old saved artifacts stay readable."""
    config = {
        "operations": [
            {
                "operation_type": "arithmetic",
                "method": "add",
                "input_columns": ["x"],
                "round_digits": "bad",
                "output_column": "result",
            }
        ]
    }
    frame = _frame(engine)
    state = FeatureGenerationCalculator().fit(frame, config)
    with pytest.raises((TypeError, ValueError)):
        FeatureGenerationApplier().apply(frame, state)
    legacy = deepcopy(state)
    legacy.pop("operation_contract_version")
    assert list(FeatureGenerationApplier().apply(frame, legacy).columns) == ["date", "x"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_datetime_features_have_nullable_int64_for_full_singleton_and_empty(engine):
    """A bad date in another batch must not change a numeric calendar feature's persisted dtype."""
    features = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "weekday",
        "is_weekend",
        "hour",
        "minute",
        "second",
    ]
    frame = _frame(engine)
    state = FeatureGenerationCalculator().fit(
        frame,
        {
            "operations": [
                {
                    "operation_type": "datetime_extract",
                    "input_columns": ["date"],
                    "datetime_features": features,
                }
            ]
        },
    )
    result = FeatureGenerationApplier().apply(frame, state)
    for name in features:
        assert str(result[f"date_{name}"].dtype) == "Int64"
    for count in (0, 1):
        got = FeatureGenerationApplier().apply(frame.head(count), state)
        expected = result.head(count)
        if engine == "polars":
            assert_polars_equal(got, expected)
        else:
            pd.testing.assert_frame_equal(got, expected)
    assert state["operation_contract_version"] == 1


def test_legacy_calendar_artifact_keeps_historical_dtype():
    """The new numeric output contract must not silently rewrite a saved legacy date operation."""
    state = {
        "type": "feature_generation",
        "operations": [
            {
                "operation_type": "datetime_extract",
                "input_columns": ["date"],
                "datetime_features": ["year"],
            }
        ],
        "epsilon": 1e-9,
        "allow_overwrite": False,
    }
    result = FeatureGenerationApplier().apply(_frame("pandas").head(1), state)
    assert str(result["date_year"].dtype) == "int32"


@pytest.mark.parametrize("version", [True, 2, "1"])
def test_feature_math_unknown_contract_versions_are_not_admitted(version):
    """Strict inference validation must distinguish supported integer versions from truthy values."""
    state = dict(FeatureGenerationCalculator().fit(_frame("pandas"), {"operations": []}))
    state["operation_contract_version"] = version
    with pytest.raises(ValueError, match="version"):
        FeatureGenerationApplier.validate_inference_state(state)
