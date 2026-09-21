"""Anchor null-statistic browser regressions to actual serialized analyzer output."""

import json
import math
from pathlib import Path

import polars as pl

from skyulf.profiling.analyzer import EDAAnalyzer


def _browser_fixture():
    """Load the two real producer projections consumed by browser regression tests."""
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "frontend/ml-canvas/e2e/fixtures/eda-null-producer.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_single_row_statistics_serialize_as_unavailable():
    """A single observation has no sample variance or deviation, rather than measured zero."""
    profile = EDAAnalyzer(pl.DataFrame({"value": [4.2]})).analyze()
    column = profile.model_dump(mode="json")["columns"]["value"]

    assert column["numeric_stats"]["mean"] == 4.2
    assert column["numeric_stats"]["std"] is None
    assert column["numeric_stats"]["variance"] is None
    assert column == _browser_fixture()["singleton_column"]


def test_real_outlier_overflow_serializes_unknown_deviation_without_losing_values():
    """A finite observation versus a tiny median can overflow its relative deviation to JSON null."""
    profile = EDAAnalyzer(pl.DataFrame({"tiny": [1e-308] * 99 + [10.0]})).analyze()
    assert profile.outliers is not None
    explanations = profile.outliers.top_outliers[0].explanation
    assert explanations is not None
    raw_explanation = explanations[0]
    assert math.isinf(raw_explanation["diff_pct"])
    payload = profile.model_dump(mode="json")
    json.dumps(payload, allow_nan=False)
    outliers = payload["outliers"]
    point = outliers["top_outliers"][0]

    assert point["index"] == 99
    assert point["explanation"] == [
        {"feature": "tiny", "value": 10.0, "median": 1e-308, "diff_pct": None}
    ]
    fixture_outliers = _browser_fixture()["outliers"]
    assert outliers["total_outliers"] == fixture_outliers["total_outliers"] == 1
    assert point["explanation"] == fixture_outliers["top_outliers"][0]["explanation"]
    assert point["values"] == fixture_outliers["top_outliers"][0]["values"]
