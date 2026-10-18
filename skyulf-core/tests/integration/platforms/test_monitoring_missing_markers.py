"""Per-column missing markers make sentinel readings count as missing in monitoring."""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
    parse_drift_thresholds,
)
from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    build_monitoring_report,
)
from skyulf.profiling.drift import DriftCalculator

AS_OF = datetime(2026, 10, 17, tzinfo=UTC)
MARKERS: Any = {"missing_markers": {"co_gt": [-200]}}


def _frames(marker: float | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reference and current sensor rows whose sentinel is either -200 or a real null."""
    reference = pd.DataFrame(
        {
            "record_id": range(40),
            "co_gt": [marker if i % 10 == 0 else 1.0 + i / 10 for i in range(40)],
            "humidity": [40.0 + i for i in range(40)],
        }
    )
    current = pd.DataFrame(
        {
            "record_id": range(100, 130),
            "co_gt": [marker if i % 3 == 0 else 2.0 + i / 10 for i in range(30)],
            "humidity": [50.0 + i for i in range(30)],
        }
    )
    predictions = pd.DataFrame({"record_id": current["record_id"], "prediction": 1.0})
    return reference, current, predictions


def _report(frames, thresholds, engine=pd.DataFrame):
    """Build the local report with an engine choice for the three input frames."""
    reference, current, predictions = (
        frame if engine is pd.DataFrame else pl.from_pandas(frame) for frame in frames
    )
    return build_monitoring_report(
        reference,
        current,
        predictions,
        None,
        feature_columns=("co_gt", "humidity"),
        record_key_columns=("record_id",),
        target_column="no2_gt",
        result_available_at_column="result_available_at",
        as_of=AS_OF,
        task="regression",
        thresholds=thresholds,
    )


@pytest.mark.parametrize("engine", [pd.DataFrame, pl.DataFrame])
def test_markers_measure_exactly_like_real_nulls(engine):
    """A -200 reading must not count as a measurement in missingness or drift.

    Markers become nulls, not NaN, so the comparison uses Polars nulls for both engines.
    """
    marked = _report(_frames(-200.0), MARKERS, engine)
    nulled = _report(_frames(None), None, pl.DataFrame)
    assert marked["metrics"] == nulled["metrics"]
    missing = next(
        item
        for item in marked["metrics"]
        if item["column_name"] == "co_gt" and item["metric_name"] == "missing_fraction"
    )
    assert missing["value"] == pytest.approx(10 / 30)
    assert missing["threshold"] == pytest.approx(4 / 40)


def test_without_markers_the_sentinel_stays_a_measurement():
    """The default must keep existing reports unchanged, -200 included."""
    report = _report(_frames(-200.0), {}, pd.DataFrame)
    assert report == _report(_frames(-200.0), None, pd.DataFrame)
    missing = next(
        item
        for item in report["metrics"]
        if item["column_name"] == "co_gt" and item["metric_name"] == "missing_fraction"
    )
    assert missing["value"] == 0.0


def test_bundle_parses_markers_and_core_drift_still_rejects_them():
    """Markers are a monitoring option; Core drift must not accept and silently ignore them."""
    assert parse_drift_thresholds('{"psi": 0.3, "missing_markers": {"co_gt": [-200]}}') == {
        "psi": 0.3,
        "missing_markers": {"co_gt": [-200]},
    }
    frame = pl.DataFrame({"co_gt": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Unknown monitoring drift thresholds"):
        DriftCalculator(frame, frame).calculate_drift(MARKERS)


@pytest.mark.parametrize(
    "markers",
    [
        [],
        {},
        {"co_gt": -200},
        {"co_gt": []},
        {"co_gt": [True]},
        {"co_gt": ["-200"]},
        {"co_gt": [float("nan")]},
        {"": [-200]},
        {"co_gt": list(range(101))},
        {"co_gt": [10**1000]},
        {"co_gt": [2**64]},
    ],
)
def test_invalid_markers_fail_before_data_access(markers):
    """A malformed marker policy must stop enrollment rather than be partly applied."""
    with pytest.raises(ValueError, match="missing_markers"):
        parse_drift_thresholds(f'{{"missing_markers": {json.dumps(markers)}}}')


@pytest.mark.parametrize(
    ("markers", "message"),
    [
        ({"no2_gt": [-200]}, "not monitored features"),
        ({"day_part": [-200]}, "floating-point"),
    ],
)
def test_markers_must_name_numeric_monitored_features(markers, message):
    """A typo or a text column must fail the report instead of measuring the raw value."""
    reference, current, predictions = _frames(-200.0)
    reference["day_part"] = "night"
    current["day_part"] = "night"
    with pytest.raises(ValueError, match=message):
        build_monitoring_report(
            reference,
            current,
            predictions,
            None,
            feature_columns=("co_gt", "humidity", "day_part"),
            record_key_columns=("record_id",),
            target_column="no2_gt",
            result_available_at_column="result_available_at",
            as_of=AS_OF,
            task="regression",
            thresholds={"missing_markers": markers},
        )


BIG = 2**53


def _single_column_report(values, dtype, markers):
    """Report one Polars feature column whose reference and current rows are identical."""
    frame = pl.DataFrame({"record_id": range(len(values)), "x": pl.Series(values, dtype=dtype)})
    predictions = pl.DataFrame({"record_id": range(len(values)), "prediction": 1.0})
    return build_monitoring_report(
        frame,
        frame,
        predictions,
        None,
        feature_columns=("x",),
        record_key_columns=("record_id",),
        target_column="y",
        result_available_at_column="result_available_at",
        as_of=AS_OF,
        task="regression",
        thresholds={"missing_markers": {"x": markers}},
    )


def _missing_fraction(report):
    """Return the current missing fraction of the single monitored column."""
    return next(
        item["value"] for item in report["metrics"] if item["metric_name"] == "missing_fraction"
    )


@pytest.mark.parametrize("engine", [pd.DataFrame, pl.DataFrame])
def test_large_integer_marker_never_matches_its_float_neighbour(engine):
    """2**53 + 1 must null only itself, although both values round to the same float."""
    values = [BIG, BIG + 1, BIG + 1, 7]
    if engine is pd.DataFrame:
        frame = pl.from_pandas(pd.DataFrame({"x": pd.Series(values, dtype="int64")}))
        values, dtype = frame["x"].to_list(), frame["x"].dtype
    else:
        dtype = pl.Int64
    assert _missing_fraction(_single_column_report(values, dtype, [BIG + 1])) == 0.5


@pytest.mark.parametrize(
    ("values", "dtype", "markers", "message"),
    [
        ([1, 2], pl.Int64, [1.5], "whole numbers within Int64"),
        ([1, 2], pl.UInt8, [-200], "whole numbers within UInt8"),
        ([1.0, 2.0], pl.Float64, [BIG + 1], "no exact Float64 value"),
        ([1.0, 2.0], pl.Float32, [2**24 + 1], "no exact Float32 value"),
        ([1.0, 2.0], pl.Float32, [0.1], "no exact Float32 value"),
        ([1.0, 2.0], pl.Float32, [1e39], "no exact Float32 value"),
        (["1.5", "2.5"], pl.Decimal(10, 2), [-200], "integer or floating-point"),
    ],
)
def test_markers_without_an_exact_column_value_fail(values, dtype, markers, message):
    """A marker that cannot equal a stored value exactly must fail, not match approximately."""
    if dtype == pl.Decimal(10, 2):
        values = pl.Series(values).cast(dtype).to_list()
    with pytest.raises(ValueError, match=message):
        _single_column_report(values, dtype, markers)


def test_whole_float_marker_applies_to_an_integer_column():
    """-200.0 written as a float in JSON still marks the integer reading -200."""
    report = _single_column_report([-200, 5, 6, 7], pl.Int32, [-200.0])
    assert _missing_fraction(report) == 0.25


def test_exact_float32_marker_nulls_only_itself():
    """-200 and 2**24 are exact in Float32 and must not touch the neighbouring 2**24 + 2."""
    values = [-200.0, 2.0**24, 2.0**24 + 2, 5.0]
    report = _single_column_report(values, pl.Float32, [-200, 2**24])
    assert _missing_fraction(report) == 0.5
