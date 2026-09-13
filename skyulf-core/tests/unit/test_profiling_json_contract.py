"""Profile JSON projections must agree with the existing finite-or-null JSON encoder."""

import json
import math

import pytest

from skyulf.profiling.schemas import (
    BoxPlotStats,
    ClusterStats,
    CorrelationMatrix,
    DatasetProfile,
    DateStats,
    GeoPoint,
    HistogramBin,
    NormalityTestResult,
    NumericStats,
    PCAPoint,
    RuleNode,
    TextStats,
    TimeSeriesPoint,
)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "model,fields",
    [
        (DateStats, {"duration_days": "invalid"}),
        (TextStats, {"avg_length": "invalid", "sentiment_distribution": {"neutral": "invalid"}}),
        (HistogramBin, {"start": "invalid", "end": 2.0, "count": 4}),
        (
            NormalityTestResult,
            {"test_name": "adf", "statistic": "invalid", "p_value": 0.1, "is_normal": False},
        ),
        (CorrelationMatrix, {"columns": ["x"], "values": [["invalid"]]}),
        (PCAPoint, {"x": "invalid", "y": 1.0}),
        (GeoPoint, {"lat": "invalid", "lon": 1.0}),
        (TimeSeriesPoint, {"date": "2026-01-01", "values": {"value": "invalid"}}),
        (BoxPlotStats, {"min": "invalid", "q1": 1.0, "median": 2.0, "q3": 3.0, "max": 4.0}),
        (
            ClusterStats,
            {"cluster_id": 0, "size": 2, "percentage": 50.0, "center": {"x": "invalid"}},
        ),
        (
            RuleNode,
            {"id": 0, "impurity": "invalid", "samples": 2, "value": ["invalid"], "is_leaf": True},
        ),
    ],
)
def test_typed_profile_models_have_finite_json_projection(model, fields, invalid):
    """Scalar, vector, and matrix fields must not emit SQL JSON NaN/Infinity tokens."""
    payload = json.loads(json.dumps(fields).replace('"invalid"', json.dumps(invalid)))
    profile = model.model_validate(payload)
    dumped = profile.model_dump(mode="json")
    assert dumped == json.loads(profile.model_dump_json())
    assert json.loads(json.dumps(dumped, allow_nan=False)) == dumped


def test_profile_json_normalization_preserves_internal_values_and_dump_options():
    """Null JSON projections must preserve mathematical values, field names, and caller exclusions."""
    profile = DatasetProfile(
        row_count=2,
        column_count=1,
        duplicate_rows=0,
        missing_cells_percentage=0.0,
        memory_usage_mb=1.0,
        columns={},
        vif={"x": float("inf")},
        target_correlations={"x": float("nan")},
        sample_data=[
            {"label": "inf", "values": [float("nan"), {"low": float("-inf"), "finite": 3.5}]}
        ],
    )
    payload = profile.model_dump(mode="json", exclude={"generated_at"}, exclude_none=True)
    assert payload["vif"] == {"x": None}
    assert payload["target_correlations"] == {"x": None}
    assert payload["sample_data"] == [
        {"label": "inf", "values": [None, {"low": None, "finite": 3.5}]}
    ]
    assert "generated_at" not in payload and "geospatial" not in payload
    assert profile.vif is not None and math.isinf(profile.vif["x"])
    assert profile.target_correlations is not None and math.isnan(profile.target_correlations["x"])
    assert "vif" in DatasetProfile.model_json_schema(mode="serialization")["properties"]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_numeric_stats_coercion_cannot_reintroduce_nonfinite_json():
    """Pydantic's string-to-float coercion must not bypass the JSON boundary sanitizer."""
    stats = NumericStats.model_validate(
        {"mean": "Infinity", "normality_test": {"value": float("nan")}}
    )
    assert stats.model_dump(mode="json")["mean"] is None
    assert stats.model_dump(mode="json")["normality_test"] == {"value": None}
