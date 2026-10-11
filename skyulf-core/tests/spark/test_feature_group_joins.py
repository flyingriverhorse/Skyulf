"""Real Spark joins preserve observation grain and prevent future feature leakage."""

from datetime import datetime

import pytest

from skyulf.integrations.databricks.features.config import FeatureGroup, FeaturePlan
from skyulf.integrations.databricks.features.joins import join_feature_groups


def _plan(lookup="exact", allow_missing=False):
    """Use one domain so failures isolate observation and feature time semantics."""
    group = FeatureGroup(
        "activity", "raw", "features", "unused", ("amount",), lookup, allow_missing
    )
    return FeaturePlan("base", "merged", ("id",), "observed_at", (group,))


def _frame(spark, rows, value):
    """Keep timestamp types explicit even for empty or nullable fixtures."""
    return spark.createDataFrame(rows, f"id long, observed_at timestamp, {value} double")


def test_asof_uses_latest_past_feature_and_keeps_observations(spark):
    """Later feature values must never leak into an earlier labeled observation."""
    day = lambda n: datetime(2026, 1, n)
    base = _frame(spark, [(1, day(2), 0.0), (1, day(4), 1.0)], "target")
    features = _frame(spark, [(1, day(1), 2.0), (1, day(3), 4.0), (1, day(5), 99.0)], "amount")
    result = join_feature_groups(base, {"activity": features}, _plan("asof"))
    assert [(r.target, r.amount) for r in result.orderBy("observed_at").collect()] == [
        (0.0, 2.0),
        (1.0, 4.0),
    ]


def test_missing_match_is_explicit_and_nullable_values_still_match(spark):
    """A null feature value is distinct from having no feature record at all."""
    date = datetime(2026, 1, 2)
    base = _frame(spark, [(1, date, 0.0), (2, date, 1.0)], "target")
    features = _frame(spark, [(1, date, None)], "amount")
    with pytest.raises(ValueError, match="missing"):
        join_feature_groups(base, {"activity": features}, _plan())
    result = join_feature_groups(base, {"activity": features}, _plan(allow_missing=True))
    assert result.count() == 2
    assert result.where("amount IS NULL").count() == 2


@pytest.mark.parametrize("bad", ["duplicate", "null_key", "null_time", "collision"])
def test_bad_join_grain_or_names_fail_before_publish(spark, bad):
    """Multiplicity, null keys and ambiguous names must not silently corrupt training."""
    date = datetime(2026, 1, 2)
    base = _frame(spark, [(1, date, 0.0)], "target")
    rows = [(1, date, 2.0)]
    if bad == "duplicate":
        rows *= 2
    if bad == "null_key":
        rows = [(None, date, 2.0)]
    if bad == "null_time":
        rows = [(1, None, 2.0)]
    if bad == "collision":
        base = base.withColumnRenamed("target", "amount")
    with pytest.raises(ValueError):
        join_feature_groups(base, {"activity": _frame(spark, rows, "amount")}, _plan())


def test_time_dtypes_must_match_without_implicit_conversion(spark):
    """Day dates and instant timestamps must not silently join using session settings."""
    date = datetime(2026, 1, 2)
    base = _frame(spark, [(1, date, 0.0)], "target")
    features = _frame(spark, [(1, date, 2.0)], "amount").selectExpr(
        "id", "CAST(observed_at AS DATE) observed_at", "amount"
    )
    with pytest.raises(ValueError, match="types"):
        join_feature_groups(base, {"activity": features}, _plan())


def test_record_key_named_count_retains_duplicate_detection(spark):
    """A legal key called count must not collide with Spark's aggregation result name."""
    date = datetime(2026, 1, 2)
    base = _frame(spark, [(1, date, 0.0)], "target").withColumnRenamed("id", "count")
    features = _frame(spark, [(1, date, 2.0)], "amount").withColumnRenamed("id", "count")
    plan = FeaturePlan("base", "merged", ("count",), "observed_at", _plan().groups)
    result = join_feature_groups(base, {"activity": features}, plan)
    assert result.first().amount == 2.0
    with pytest.raises(ValueError, match="unique"):
        join_feature_groups(base, {"activity": features.unionByName(features)}, plan)


@pytest.mark.parametrize("name", ["__SKYULF_FEATURE_TIME", "unsafe.dot", "unsafe`tick"])
def test_unsafe_passthrough_names_fail_before_join_expression_resolution(spark, name):
    """Spark expression names and case-folded internal aliases cannot reach join assembly."""
    date = datetime(2026, 1, 2)
    base = _frame(spark, [(1, date, 0.0)], "target").withColumnRenamed("target", name)
    features = _frame(spark, [(1, date, 2.0)], "amount")
    with pytest.raises(ValueError, match="column|metadata"):
        join_feature_groups(base, {"activity": features}, _plan())
