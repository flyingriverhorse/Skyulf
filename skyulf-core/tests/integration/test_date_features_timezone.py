"""Calendar features need one fitted timezone contract across engines and replay."""

import json
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.time_series.date_features import (
    DateFeaturesApplier,
    DateFeaturesCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fitted_date_features_use_utc_across_mixed_offsets(engine):
    """DST offsets and midnight boundaries must describe the same calendar instant on both engines."""
    values = [
        "2024-03-31T00:30:00+02:00",
        "2024-03-31T03:30:00+03:00",
        "2024-12-31T23:30:00-02:00",
        "invalid",
        None,
    ]
    expected = {
        "year": [2024, 2024, 2025, None, None],
        "month": [3, 3, 1, None, None],
        "day": [30, 31, 1, None, None],
        "dayofweek": [5, 6, 2, None, None],
        "dayofyear": [90, 91, 1, None, None],
        "quarter": [1, 1, 1, None, None],
        "weekofyear": [13, 13, 1, None, None],
        "hour": [22, 0, 1, None, None],
        "minute": [30, 30, 30, None, None],
        "is_weekend": [1, 1, 0, None, None],
        "is_month_start": [0, 0, 1, None, None],
        "is_month_end": [0, 1, 0, None, None],
    }
    data = pd.DataFrame({"date": values}) if engine == "pandas" else pl.DataFrame({"date": values})
    before = data.copy() if isinstance(data, pd.DataFrame) else data.to_pandas()
    artifact = DateFeaturesCalculator().fit(data, {"columns": ["date"], "features": list(expected)})
    saved = json.loads(json.dumps(artifact))
    result = DateFeaturesApplier().apply(data, saved)

    for feature, want in expected.items():
        got = result[f"date_{feature}"].to_list()
        assert [None if pd.isna(value) else value for value in got] == want
    pd.testing.assert_frame_equal(
        data if isinstance(data, pd.DataFrame) else data.to_pandas(), before
    )
    assert result["date"].to_list() == values


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("timezone", [None, "Europe/Paris"])
def test_native_date_features_match_string_instants(engine, timezone):
    """Native timezone dtypes must follow the same UTC rule as CSV strings without mutating inputs."""
    timestamps = pd.date_range("2024-03-30T23:30:00", periods=3, freq="h", tz=timezone)
    data = pd.DataFrame({"date": timestamps})
    if engine == "polars":
        data = pl.from_pandas(data)
    artifact = DateFeaturesCalculator().fit(
        data, {"columns": ["date"], "features": ["day", "hour"]}
    )
    result = DateFeaturesApplier().apply(data, artifact)

    assert result["date_day"].to_list() == ([30, 31, 31] if timezone is None else [30, 30, 31])
    assert result["date_hour"].to_list() == ([23, 0, 1] if timezone is None else [22, 23, 0])
    assert result["date"].dtype == data["date"].dtype


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_legacy_artifact_keeps_its_saved_calendar_interpretation(engine):
    """Loading an old trained model must not silently change its feature values."""
    data = {"date": ["2024-01-01T00:30:00+02:00"]}
    data = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    artifact = {"type": "date_features", "columns": ["date"], "features": ["year", "day", "hour"]}
    result = DateFeaturesApplier().apply(data, artifact)

    assert result["date_year"].to_list() == ([2024] if engine == "pandas" else [2023])
    assert result["date_day"].to_list() == ([1] if engine == "pandas" else [31])
    assert result["date_hour"].to_list() == ([0] if engine == "pandas" else [22])


def test_date_pipeline_replays_equivalent_instants_after_serialization():
    """A model trained on offsets must predict identically for equivalent UTC input after reload."""
    data = pd.DataFrame(
        {
            "date": [f"2024-01-01T0{hour}:30:00+02:00" for hour in range(2, 6)],
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "calendar",
                    "transformer": "DateFeatures",
                    "params": {"columns": ["date"], "features": ["hour"], "drop_original": True},
                }
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(data, target_column="target")
    reloaded = pickle.loads(pickle.dumps(pipeline))
    equivalent = pd.DataFrame({"date": [f"2024-01-01T0{hour}:30:00Z" for hour in range(4)]})

    np.testing.assert_array_equal(reloaded.predict(equivalent), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        reloaded.predict(data.drop(columns="target")), [0.0, 1.0, 2.0, 3.0]
    )
