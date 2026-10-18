"""Repeated high-cardinality labels remain measurable without treating IDs as categories."""

import math

import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    _drift_evidence,
    _frame,
)
from skyulf.profiling.drift import DriftCalculator


def _report(reference, current):
    """Exercise the full feature-family correction, not only a grouping helper."""
    return DriftCalculator(
        pl.DataFrame({"category": pl.Series(reference, dtype=pl.String)}),
        pl.DataFrame({"category": pl.Series(current, dtype=pl.String)}),
    ).calculate_drift()


@pytest.mark.parametrize("shifted", [False, True])
def test_repeated_ninety_category_feature_is_measured(shifted):
    """ZIP-sized categorical domains must produce real evidence rather than unavailable rows."""
    reference = [f"zip{i:03}" for i in range(90)] * 20
    current = ["zip000"] * len(reference) if shifted else reference
    report = _report(reference, current)
    result = report.column_drifts["category"]
    assert result.drift_detected is shifted
    assert report.drifted_columns_count == int(shifted)
    assert result.evidence.reference_count == result.evidence.current_count == 1800
    assert result.evidence.status == ("supported" if shifted else "not_detected")
    assert result.evidence.test.endswith("_top50_other")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_category_rate_counts_unseen_labels_not_known_tail(engine):
    """Known tail categories belong to other without being misreported as newly observed."""
    reference = {"category": [f"zip{i:03}" for i in range(90)] * 20 + [None] * 30}
    current = {"category": ["zip089"] * 75 + ["new_zip"] * 25 + [None] * 100}
    frame = pd.DataFrame if engine == "pandas" else pl.DataFrame
    metrics, _, _, _ = _drift_evidence(
        _frame(frame(reference), "reference"),
        _frame(frame(current), "current"),
        ("category",),
        None,
    )
    novelty = next(item for item in metrics if item["metric_name"] == "new_category_rate")
    assert novelty["value"] == 0.25
    assert novelty["threshold"] is None
    assert novelty["has_issue"] is False


@pytest.mark.parametrize("count", [51, 90, 10000])
def test_uniqueness_boundary_uses_nonnull_reference_count(count):
    """Exactly half unique is admissible; nulls must not make ID-like data look repeated."""
    categories = [f"cat{i:05}" for i in range(count)]
    assert "category" in _report(categories * 2, categories * 2).column_drifts
    assert not _report(categories * 2, [None] * 20).column_drifts
    ids = categories + categories[:-1] + [None] * 200
    assert not _report(ids, ids).column_drifts


@pytest.mark.parametrize("side", ["reference", "current"])
def test_category_summary_cap_is_explicit(side):
    """Both sides stay bounded even when every category is repeated often enough."""
    small = [f"cat{i:05}" for i in range(90)] * 20
    large = [f"cat{i:05}" for i in range(10001)] * 2
    reference, current = (large, small) if side == "reference" else (small, large)
    assert not _report(reference, current).column_drifts


def test_known_tail_turnover_is_diagnostic_without_an_invented_policy():
    """Other-bin turnover exposes novelty but does not claim a statistically tested shift."""
    head = [f"cat{i:03}" for i in range(50)] * 30
    tail = [f"tail{i:03}" for i in range(40)] * 10
    current = head + [f"new{i:03}" for i in range(40)] * 10
    result = _report(head + tail, current).column_drifts["category"]
    metrics = {metric.metric: metric for metric in result.metrics}
    assert metrics["psi_categorical"].value == 0.0
    assert metrics["new_category_rate"].value == pytest.approx(400 / 1900)
    assert not metrics["new_category_rate"].has_drift
    assert not result.drift_detected


def test_top_categories_are_reference_based_deterministic_and_collision_free():
    """Reordering input and a literal other label cannot change the reference grouping."""
    reference = ["other"] * 1000 + [f"cat{i:03}" for i in range(89)] * 20
    current = [f"cat{i:03}" for i in range(89)] * 20 + ["new"] * 1000
    result = _report(reference, current).column_drifts["category"]
    assert result.drift_detected
    assert (
        result.model_dump_json()
        == _report(reference[::-1], current[::-1]).column_drifts["category"].model_dump_json()
    )


def test_top_fifty_cut_breaks_count_ties_by_label():
    """A label exactly at the tied cutoff stays explicit regardless of input order."""
    reference = [f"cat{i:03}" for i in range(90)] * 20
    current = ["cat049"] * 1800
    epsilon = 0.5 / 1800
    expected = (
        (1 - 1 / 90) * math.log(90)
        + 49 * (epsilon - 1 / 90) * math.log(epsilon * 90)
        + (epsilon - 40 / 90) * math.log(epsilon / (40 / 90))
    )
    result = _report(reference, current).column_drifts["category"]
    assert result.metrics[0].value == pytest.approx(expected)
    assert (
        result.model_dump_json()
        == _report(reference[::-1], current).column_drifts["category"].model_dump_json()
    )


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum([f"zip{i:03}" for i in range(90)])]
)
def test_observed_categorical_dtypes_share_grouped_evidence(dtype):
    """Categorical and enum storage must not change observed counts or grouping."""
    values = [f"zip{i:03}" for i in range(90)] * 20
    frame = pl.DataFrame({"category": pl.Series(values, dtype=dtype)})
    actual = DriftCalculator(frame, frame).calculate_drift()
    assert actual.model_dump_json() == _report(values, values).model_dump_json()


def test_low_cardinality_payload_is_unchanged():
    """The established small-category API stays byte-identical, including evidence and suggestions."""
    result = _report(["a", "b", "c"] * 10, ["a", "b", "c"] * 10)
    assert result.column_drifts["category"].model_dump_json() == (
        '{"column":"category","metrics":[{"metric":"psi_categorical","value":0.0,'
        '"has_drift":false,"threshold":0.2,"raw_value":null}],"drift_detected":false,'
        '"suggestions":["No statistical support for a distribution change; equivalence '
        'is not established."],"distribution":null,"evidence":{"status":"not_detected",'
        '"test":"chi_square","p_value":1.0,"adjusted_p_value":1.0,"significance_level":0.05,'
        '"reference_count":30,"current_count":30,"reason":"No statistical support for a '
        'distribution change; equivalence is not established."}}'
    )
