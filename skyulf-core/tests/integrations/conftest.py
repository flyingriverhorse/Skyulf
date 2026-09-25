"""Optional real Delta fixture shared by the batch integration tests."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def workflow_config():
    """Use a complete offline project so notebook tests exercise real configuration checks."""
    return {
        "config_version": 1,
        "task": "regression",
        "engine": "polars",
        "training_table": "workspace.test.source",
        "score_source_table": "workspace.test.source",
        "prediction_table": "workspace.test.predictions",
        "model_name": "workspace.test.model",
        "model_version": "1",
        "score_model_selection": "champion",
        "promotion_policy": "manual_approval",
        "score_handoff": "after_alias_change",
        "model_change_mode": "incremental_append",
        "record_key_columns": ["id", "record_id"],
        "input_columns": ["x"],
        "target_column": "target",
        "split_strategy": "temporal",
        "filter_unavailable_results": True,
        "result_cutoff": "2026-03-01T00:00:00+00:00",
        "event_column": "event_time",
        "result_available_at_column": "label_at",
        "training_version": 0,
        "start": "2026-01-01T00:00:00+00:00",
        "holdout_start": "2026-02-01T00:00:00+00:00",
        "cutoff": "2026-03-01T00:00:00+00:00",
        "monthly_lookback_months": 4,
        "max_rows": 1000,
        "max_input_mb": 1,
        "metric": "heldout_rmse",
        "quality_threshold": 5.0,
        "min_improvement": 0.0,
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }


@pytest.fixture(scope="session")
def delta_spark(tmp_path_factory):
    """Run actual Delta I/O; required lanes fail rather than silently skip."""
    if importlib.util.find_spec("delta") is None:
        if os.environ.get("SKYULF_REQUIRE_DELTA") == "1":
            pytest.fail("Install requirements-delta.txt and Java 17.")
        pytest.skip("Delta runtime is optional; install requirements-delta.txt.")
    spark_sql = importlib.import_module("pyspark.sql")
    root = tmp_path_factory.mktemp("delta")
    with pytest.MonkeyPatch.context() as env:
        env.setenv("PYSPARK_PYTHON", sys.executable)
        env.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)
        env.setenv("SPARK_LOCAL_IP", "127.0.0.1")
        builder = (
            spark_sql.SparkSession.builder.master("local[2]")
            .appName("skyulf-delta-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.databricks.delta.snapshotPartitions", "2")
            .config("spark.sql.warehouse.dir", (root / "warehouse").as_uri())
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
            )
        )
        jars_dir = os.environ.get("SKYULF_DELTA_JARS")
        if jars_dir:
            builder = builder.config(
                "spark.jars", ",".join(str(p) for p in Path(jars_dir).glob("*.jar"))
            )
        else:
            builder = importlib.import_module("delta").configure_spark_with_delta_pip(builder)
        session = builder.getOrCreate()
        try:
            yield session
        finally:
            session.stop()
