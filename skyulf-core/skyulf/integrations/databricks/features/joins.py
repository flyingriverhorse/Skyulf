"""Check feature grain and perform distributed exact or backward temporal joins."""

from functools import reduce
from importlib import import_module
from operator import and_, or_
from typing import Any

from ..shared._contracts import column_name
from .config import FeatureGroup, FeaturePlan


def validate_feature_frame(frame: Any, keys: tuple[str, ...], timestamp: str) -> int:
    """Reject ambiguous identifiers, null keys, duplicate records and untyped dates."""
    F = import_module("pyspark.sql.functions")

    if frame.isStreaming:
        raise ValueError("Feature production requires a bounded Spark DataFrame.")
    _validate_columns(frame, (*keys, timestamp), timestamp)
    grain = (*keys, timestamp)
    if frame.where(reduce(or_, (F.col(name).isNull() for name in grain))).limit(1).count():
        raise ValueError("Feature key/timestamp contains null values.")
    counts = frame.groupBy(*grain).agg(F.count(F.lit(1)).alias("__skyulf_group_count"))
    if counts.where(F.col("__skyulf_group_count") > 1).limit(1).count():
        raise ValueError("Feature key/timestamp must be unique.")
    return frame.count()


def _validate_columns(frame: Any, grain: tuple[str, ...], timestamp: str) -> None:
    """Keep both control and passthrough names safe for Spark expression resolution."""
    names = frame.columns
    if len({name.lower() for name in names}) != len(names):
        raise ValueError("Feature frame has duplicate or case-ambiguous columns.")
    for name in names:
        column_name(name)
    if set(grain) - set(names):
        raise ValueError("Feature frame is missing declared key/timestamp columns.")
    if frame.schema[timestamp].dataType.typeName() not in {"date", "timestamp", "timestamp_ntz"}:
        raise ValueError("Feature timestamp must have a date or timestamp type.")


def _validate_join(base: Any, features: Any, plan: FeaturePlan, group: FeatureGroup) -> None:
    """Check both key transport and feature ownership before constructing a join."""
    validate_feature_frame(features, plan.keys, plan.timestamp)
    if set(group.columns) - set(features.columns):
        raise ValueError(f"Feature group {group.name} is missing configured feature columns.")
    if {n.lower() for n in group.columns}.intersection(n.lower() for n in base.columns):
        raise ValueError(f"Feature group {group.name} columns overlap the base table.")
    if any(base.schema[n].dataType != features.schema[n].dataType for n in plan.record_keys):
        raise ValueError(f"Feature group {group.name} key/timestamp types must match the base.")


def _join_group(base: Any, features: Any, plan: FeaturePlan, group: FeatureGroup) -> Any:
    """Match features without collecting records or using observation-dependent fits."""
    F = import_module("pyspark.sql.functions")
    Window = import_module("pyspark.sql.window").Window

    left = base.alias("observations")
    right = features.select(*plan.record_keys, *group.columns).alias("features")
    condition = reduce(and_, (left[key] == right[key] for key in plan.keys))
    time_condition = (
        left[plan.timestamp] >= right[plan.timestamp]
        if group.lookup == "asof"
        else left[plan.timestamp] == right[plan.timestamp]
    )
    joined = left.join(right, condition & time_condition, "left").select(
        *(left[name] for name in base.columns),
        *(right[name] for name in group.columns),
        right[plan.timestamp].alias("__skyulf_feature_time"),
    )
    if group.lookup == "asof":
        order = Window.partitionBy(*plan.record_keys).orderBy(
            F.col("__skyulf_feature_time").desc_nulls_last()
        )
        joined = (
            joined.withColumn("__skyulf_rank", F.row_number().over(order))
            .where(F.col("__skyulf_rank") == 1)
            .drop("__skyulf_rank")
        )
    if (
        not group.allow_missing
        and joined.where(F.col("__skyulf_feature_time").isNull()).limit(1).count()
    ):
        raise ValueError(f"Feature group {group.name} has missing observation matches.")
    return joined.drop("__skyulf_feature_time")


def join_feature_groups(base: Any, frames: dict[str, Any], plan: FeaturePlan) -> Any:
    """Preserve each unique observation and reject join losses or multiplication.

    Exact joins use entity plus timestamp. As-of joins select the latest feature
    timestamp no later than that observation. Input timestamps must already be
    typed consistently; this function never guesses a timezone or casts strings.
    Feature values may be null; missing records require explicit allow_missing.
    """
    if set(frames) != {group.name for group in plan.groups}:
        raise ValueError("Feature frames must match the configured groups exactly.")
    expected_rows = validate_feature_frame(base, plan.keys, plan.timestamp)
    result = base
    for group in plan.groups:
        _validate_join(result, frames[group.name], plan, group)
        result = _join_group(result, frames[group.name], plan, group)
        if result.count() != expected_rows:
            raise ValueError(f"Feature join changed the observation row count: {group.name}.")
    return result
