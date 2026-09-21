"""Pin the numeric histogram's right-closed bins to real analyze-filter results."""

import json
from pathlib import Path

import polars as pl
import pytest

from backend.eda.router import FilterRequest
from backend.eda.tasks import _run_eda_analyzer


@pytest.fixture
def histogram_fixture():
    """Share the real producer fixture with browser tests of histogram drill-down."""
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "frontend/ml-canvas/e2e/fixtures/eda-histogram-boundaries.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _frame(fixture):
    """Include excluded rows at edges to catch loss of the existing cohort filter."""
    return pl.DataFrame(
        {
            "value": fixture["values"] + fixture["excluded_values"],
            "cohort": ["keep"] * len(fixture["values"])
            + ["exclude"] * len(fixture["excluded_values"]),
        }
    )


def test_profile_histogram_fixture_matches_the_actual_producer(histogram_fixture):
    """Every edge and its neighbors must keep the browser fixture aligned with Core binning."""
    fixture = histogram_fixture
    profile = _run_eda_analyzer(_frame(fixture), {"filters": fixture["existing_filters"]})
    column = profile.columns["value"]

    assert profile.row_count == fixture["profile"]["row_count"] == 62
    assert column.dtype == "Numeric"
    assert column.missing_count == 1
    assert column.missing_percentage == pytest.approx(
        fixture["profile"]["columns"]["value"]["missing_percentage"]
    )
    assert column.histogram is not None
    assert [bucket.count for bucket in column.histogram] == fixture["expected_bin_counts"]
    assert [bucket.count for bucket in column.histogram] == [4] + [3] * 19
    assert [bucket.model_dump() for bucket in column.histogram] == fixture["profile"]["columns"][
        "value"
    ]["histogram"]


@pytest.mark.parametrize("case_index", [0, 1, 2], ids=["first", "middle", "last"])
def test_histogram_range_filters_select_exact_bin_rows(histogram_fixture, case_index):
    """Drill-down must retain the cohort filter, exclude nulls, and match the displayed bin count."""
    fixture = histogram_fixture
    case = fixture["cases"][case_index]
    filters = [FilterRequest.model_validate(item).model_dump() for item in case["filters"]]
    profile = _run_eda_analyzer(_frame(fixture), {"filters": filters})

    assert profile.active_filters is not None
    assert [item.model_dump() for item in profile.active_filters] == filters
    assert profile.row_count == fixture["expected_bin_counts"][case["bin_index"]]
    assert profile.sample_data is not None
    assert [row["value"] for row in profile.sample_data] == case["expected_values"]
    assert all(row["cohort"] == "keep" for row in profile.sample_data)
