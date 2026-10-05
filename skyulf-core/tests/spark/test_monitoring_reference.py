"""Prepared reference populations retain typed nulls across Delta writes."""

from types import SimpleNamespace

import pandas as pd


def test_prepared_frame_preserves_all_null_source_type(spark):
    """An all-null training column must not become a NullType Delta column."""
    from skyulf.integrations.databricks.spark_monitoring_reference import _prepared_frame

    original = spark.createDataFrame([(1, None)], "id long, optional string")
    frame = _prepared_frame(
        spark,
        pd.DataFrame({"id": [1, 2], "optional": [None, None]}),
        original.schema,
        SimpleNamespace(event_column=None, result_available_at_column=None),
    )
    assert frame.schema["optional"].dataType.typeName() == "string"
    assert frame.where("optional IS NULL").count() == 2
