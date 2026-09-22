"""Optional real Delta fixture shared by the batch integration tests."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest


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
