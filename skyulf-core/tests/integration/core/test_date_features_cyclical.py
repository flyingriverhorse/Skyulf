"""Local calendar cycles must survive fitted replay without depending on neighboring rows."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf import SkyulfPipeline
from skyulf.core.schema import SkyulfSchema
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.preprocessing.time_series.date_features import (
    DateFeaturesApplier,
    DateFeaturesCalculator,
)

CYCLES = [
    f"{part}_{wave}"
    for part in ("hour", "dayofweek", "month", "dayofyear")
    for wave in ("sin", "cos")
]


def _native(frame, engine):
    """Retain typed dates and nontrivial pandas row labels in each native engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _slice(frame, positions):
    """Use positional subsets without normalizing pandas labels."""
    return frame[positions] if isinstance(frame, pl.DataFrame) else frame.iloc[positions]


def _equal(actual, expected):
    """Assert both generated dtypes and row identity across batch boundaries."""
    if isinstance(actual, pl.DataFrame):
        assert_polars_equal(actual, expected)
    else:
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_calendar_cycles_use_zero_based_positions_and_local_leap_year(engine):
    """Cycles must wrap at calendar boundaries with an explicit per-year day count."""
    frame = _native(
        pd.DataFrame(
            {
                "date": [
                    "2024-01-01T06:00:17Z",
                    "2024-07-02T18:00:59Z",
                    "2023-12-31T00:00:00Z",
                    None,
                    "invalid",
                ]
            },
            index=[9, 2, 9, -1, 3],
        ),
        engine,
    )
    state = DateFeaturesCalculator().fit(
        frame, {"columns": ["date"], "features": ["second", *CYCLES]}
    )
    result = DateFeaturesApplier().apply(frame, state)
    components = {
        "hour": ([6, 18, 0], [24, 24, 24]),
        "dayofweek": ([0, 1, 6], [7, 7, 7]),
        "month": ([0, 6, 11], [12, 12, 12]),
        "dayofyear": ([0, 183, 364], [366, 366, 365]),
    }
    assert result["date_second"].to_list()[:3] == [17, 59, 0]
    for part, (positions, periods) in components.items():
        angles = 2 * np.pi * np.array(positions) / np.array(periods)
        for wave, function in (("sin", np.sin), ("cos", np.cos)):
            values = result[f"date_{part}_{wave}"].to_numpy()
            np.testing.assert_allclose(values[:3], function(angles), atol=1e-14)
            assert pd.isna(values[3:]).all()
    assert state["timezone"] == "UTC"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("representation", ["strings", "naive", "aware", "epoch"])
def test_selected_timezone_converts_instants_before_calendar_extraction(engine, representation):
    """DST folds and gaps must follow UTC instants rather than ambiguous local localization."""
    values = [
        "2024-03-31T00:30:17Z",
        "2024-03-31T01:30:18Z",
        "2024-10-27T00:30:19Z",
        "2024-10-27T01:30:20Z",
    ]
    timestamps = pd.to_datetime(values, utc=True)
    column = {
        "strings": values,
        "naive": timestamps.tz_localize(None),
        "aware": timestamps.tz_convert("Asia/Tokyo"),
        "epoch": timestamps.as_unit("s").astype(np.int64),
    }[representation]
    frame = _native(pd.DataFrame({"date": column}), engine)
    config = {
        "columns": ["date"],
        "features": ["hour", "second", "hour_sin", "hour_cos"],
        "timezone": "Europe/Berlin",
    }
    if representation == "epoch":
        config["epoch_unit"] = "s"
    state = DateFeaturesCalculator().fit(frame, config)
    result = DateFeaturesApplier().apply(frame, state)
    assert result["date_hour"].to_list() == [1, 3, 2, 2]
    assert result["date_second"].to_list() == [17, 18, 19, 20]
    np.testing.assert_allclose(
        result["date_hour_cos"].to_numpy(), np.cos(2 * np.pi * np.array([1, 3, 2, 2]) / 24)
    )
    assert state["timezone"] == "Europe/Berlin"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_local_calendar_is_row_local_and_has_stable_empty_schema(engine):
    """Saved IANA settings must retain nulls, names and dtypes for singleton, reordered and empty requests."""
    frame = _native(
        pd.DataFrame(
            {
                "date": ["2023-12-31T23:30:00Z", "2024-02-29T22:30:00Z", None, "invalid"],
                "keep": [4, 3, 2, 1],
            },
            index=[8, 2, 8, -1],
        ),
        engine,
    )
    before = deepcopy(frame)
    config = {
        "columns": ["date"],
        "features": ["year", "dayofyear", "second", *CYCLES],
        "timezone": "Europe/Istanbul",
        "drop_original": True,
    }
    state = json.loads(json.dumps(DateFeaturesCalculator().fit(frame, config)))
    saved = deepcopy(state)
    applier = DateFeaturesApplier()
    capability = applier.inference_capability(state, engine=engine)
    assert capability is not None and capability.context == "row"
    full = applier.apply(frame, state)
    assert full["date_year"].to_list()[:2] == [2024, 2024]
    assert full["date_dayofyear"].to_list()[:2] == [1, 61]
    for positions in ([0], [1, 2], [3], [3, 2, 1, 0], []):
        _equal(applier.apply(_slice(frame, positions), state), _slice(full, positions))
    preview = DateFeaturesCalculator().infer_output_schema(
        SkyulfSchema.from_dataframe(frame), config
    )
    assert preview is not None and preview.column_list() == list(full.columns)
    assert preview.dtypes["date_hour_sin"].lower() == "float64"
    assert preview.dtypes["date_second"] == "Int64"
    _equal(frame, before)
    assert state == saved


@pytest.mark.parametrize("timezone", ["", "No/Such_Zone", "+03:00", None, 3, ["UTC"]])
def test_new_fit_and_schema_reject_invalid_timezone_even_without_columns(timezone):
    """Invalid timezone options must fail during configuration rather than silently produce UTC."""
    config = {"columns": [], "timezone": timezone}
    calculator = DateFeaturesCalculator()
    with pytest.raises(ValueError, match="timezone.*IANA"):
        calculator.fit(pd.DataFrame({"date": ["2024-01-01"]}), config)
    with pytest.raises(ValueError, match="timezone.*IANA"):
        calculator.infer_output_schema(
            SkyulfSchema.from_dataframe(pd.DataFrame({"date": ["2024-01-01"]})), config
        )


@pytest.mark.parametrize("timezone", ["", "No/Such_Zone", "+03:00", 3])
def test_saved_timezone_must_be_valid_before_row_admission(timezone):
    """A malformed saved timezone cannot claim deterministic supported inference."""
    state = {
        "type": "date_features",
        "columns": ["date"],
        "features": ["hour"],
        "timezone": timezone,
    }
    with pytest.raises(ValueError, match="timezone"):
        DateFeaturesApplier.inference_capability(state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_real_pipeline_save_load_keeps_local_cycles_and_cross_engine_predictions(engine, tmp_path):
    """A fitted model must retain its calendar basis through disk persistence and cross-engine replay."""
    frame = pd.DataFrame(
        {
            "date": [f"2024-01-01T{hour:02d}:00:00Z" for hour in range(0, 24, 3)],
            "target": np.arange(8.0),
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "calendar",
                    "transformer": "DateFeatures",
                    "params": {
                        "columns": ["date"],
                        "features": ["second", *CYCLES],
                        "timezone": "America/New_York",
                        "drop_original": True,
                    },
                }
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    )
    pipeline.fit(_native(frame, engine), target_column="target")
    save_pipeline(pipeline, tmp_path / "calendar")
    restored = load_pipeline(tmp_path / "calendar")
    query = frame.drop(columns="target")
    for apply_engine in ("pandas", "polars"):
        result = predict_pipeline(_native(query, apply_engine), restored)
        np.testing.assert_array_equal(result["prediction"], frame["target"])
    assert (tmp_path / "calendar").exists()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_month_end_does_not_shift_clock_into_dst_gap(engine):
    """Calendar flags must not require constructing a nonexistent same-time timestamp on the next day."""
    frame = _native(
        pd.DataFrame(
            {
                "date": [
                    "2024-03-30T01:30:00Z",
                    "2024-03-31T01:30:00Z",
                    "2024-10-26T00:30:00Z",
                    "2024-10-31T01:30:00Z",
                ]
            }
        ),
        engine,
    )
    state = DateFeaturesCalculator().fit(
        frame,
        {
            "columns": ["date"],
            "features": ["is_month_start", "is_month_end"],
            "timezone": "Europe/Berlin",
        },
    )
    result = DateFeaturesApplier().apply(frame, state)
    assert result["date_is_month_end"].to_list() == [0, 1, 0, 1]
    assert result["date_is_month_start"].to_list() == [0, 0, 0, 0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_all_null_dates_keep_cycle_types_and_explicit_local_state(engine):
    """Null-only columns must produce typed null features instead of losing generated empty schemas."""
    frame = _native(pd.DataFrame({"date": pd.Series([None, None], dtype="string")}), engine)
    state = DateFeaturesCalculator().fit(
        frame, {"columns": ["date"], "features": ["second", *CYCLES], "timezone": "Asia/Kolkata"}
    )
    result = DateFeaturesApplier().apply(frame, state)
    for feature in ["second", *CYCLES]:
        assert pd.isna(result[f"date_{feature}"].to_numpy()).all()
    _equal(DateFeaturesApplier().apply(_slice(frame, []), state), _slice(result, []))
    assert state["timezone"] == "Asia/Kolkata"
