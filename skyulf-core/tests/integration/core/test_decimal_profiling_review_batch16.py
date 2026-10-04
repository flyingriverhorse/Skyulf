"""Decimal profiling exposes approximate metrics without losing the exact source values."""

from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.distributions import calculate_histogram


def test_decimal_numeric_profile_preserves_source_and_samples():
    """Supported decimal data must receive numeric statistics without rewriting its exact values."""
    values = [Decimal("1.25"), Decimal("2.50"), None, Decimal("5.75")]
    frame = pl.DataFrame({"amount": values})
    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()
    column = profile.columns["amount"]
    assert column.dtype == "Numeric"
    assert column.numeric_stats is not None
    assert column.numeric_stats.mean == pytest.approx(np.mean([1.25, 2.5, 5.75]))
    assert column.numeric_stats.std == pytest.approx(np.std([1.25, 2.5, 5.75], ddof=1))
    assert column.histogram is not None and sum(bucket.count for bucket in column.histogram) == 3
    assert column.missing_count == 1 and not column.is_constant
    assert profile.sample_data == frame.to_dicts()
    assert analyzer.df.equals(frame) and analyzer.df.schema["amount"].is_decimal()
    assert any(
        alert.column == "amount" and "approximat" in alert.message for alert in profile.alerts
    )
    assert '"1.25"' in profile.model_dump_json()


def test_decimal_joint_analytics_use_local_float_views():
    """Decimal features and targets must participate in numeric diagnostics on safe views."""
    rng = np.random.default_rng(28)
    x = np.round(rng.normal(size=40), 2)
    target = np.round(2 * x + rng.normal(size=40), 2)
    z = rng.normal(size=40)
    frame = pl.DataFrame(
        {
            "amount": [Decimal(str(value)) for value in x],
            "z": z,
            "target": [Decimal(str(value)) for value in target],
        }
    )
    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="Regression")
    assert profile.vif is not None and set(profile.vif) == {"amount", "z"}
    assert profile.target_correlations is not None
    assert profile.target_correlations["amount"] == pytest.approx(np.corrcoef(x, target)[0, 1])
    assert profile.outliers is not None and profile.outliers.analyzed_rows == 40
    assert profile.columns["amount"].normality_test is not None
    assert profile.sample_data == frame.to_dicts()


def test_decimal_classification_anova_uses_safe_float_view():
    """Decimal feature significance must match its safe numeric projection for class targets."""
    values = [Decimal("1.1"), Decimal("2.2"), Decimal("4.4"), Decimal("5.5")] * 8
    frame = pl.DataFrame({"amount": values, "target": ["a", "a", "b", "b"] * 8})
    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="Classification")
    oracle = EDAAnalyzer(frame.with_columns(pl.col("amount").cast(pl.Float64))).analyze(
        target_col="target", task_type="Classification"
    )
    assert profile.target_interactions and oracle.target_interactions
    assert profile.target_interactions[0].p_value == pytest.approx(
        oracle.target_interactions[0].p_value
    )
    assert profile.sample_data == frame.to_dicts()


def test_unsafe_decimal_target_cannot_produce_numeric_rule_tree():
    """A target whose variation vanishes in Float64 must not produce explanatory predictions."""
    frame = pl.DataFrame(
        {
            "feature": np.arange(20.0),
            "target": [Decimal("9007199254740993.01"), Decimal("9007199254740993.02")] * 10,
        }
    )
    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="Regression")
    assert profile.columns["target"].numeric_stats is None
    assert not profile.target_correlations
    assert profile.rule_tree is None
    assert profile.task_type == "Regression"


def test_decimal_precision_collapse_keeps_metrics_unavailable():
    """A float projection that erases real variation must not publish a false constant."""
    frame = pl.DataFrame(
        {
            "amount": [Decimal("9007199254740993.01"), Decimal("9007199254740993.02")] * 10,
            "safe": np.arange(20.0),
        }
    )
    profile = EDAAnalyzer(frame).analyze()
    column = profile.columns["amount"]
    assert column.dtype == "Numeric"
    assert column.numeric_stats is None and column.histogram is None
    assert column.is_constant is False
    assert profile.correlations is None and profile.vif is None
    assert any(
        alert.column == "amount" and "precision" in alert.message.lower()
        for alert in profile.alerts
    )
    assert profile.sample_data == frame.to_dicts()


@pytest.mark.parametrize(
    "values,constant", [([None, None], False), ([Decimal("1.25"), Decimal("1.25")], True)]
)
def test_decimal_missing_and_constant_controls(values, constant):
    """Numeric support must retain honest missing and repeated-value semantics."""
    frame = pl.DataFrame({"amount": pl.Series(values, dtype=pl.Decimal(38, 2))})
    column = EDAAnalyzer(frame).analyze().columns["amount"]
    assert column.dtype == "Numeric"
    assert column.is_constant is constant
    assert column.missing_count == values.count(None)


@pytest.mark.parametrize(
    "values,available",
    [
        ([Decimal("1.25"), Decimal("2.50")], True),
        ([Decimal("9007199254740993.01"), Decimal("9007199254740993.02")], False),
    ],
)
def test_decimal_histogram_public_helper_respects_projection_precision(values, available):
    """Standalone histogram calls must share the profiler's safe Decimal plotting boundary."""
    frame = pl.DataFrame({"amount": values})
    histogram = calculate_histogram(frame.lazy(), "amount")
    if available:
        assert histogram is not None and sum(bucket.count for bucket in histogram) == 2
    else:
        assert histogram is None
