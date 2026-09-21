"""Optional Spark fixtures; the dedicated lane fails when Spark is unavailable."""

import importlib
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    """Run a local JVM with two workers and the current interpreter for Python tasks."""
    if importlib.util.find_spec("pyspark") is None:
        message = "Spark tests require requirements-spark.txt and Java 17."
        if os.environ.get("SKYULF_REQUIRE_SPARK") == "1":
            pytest.fail(message)
        pytest.skip(message)

    # Keep optional runtime imports out of the base environment's type resolution.
    spark_sql = importlib.import_module("pyspark.sql")

    work_dir = tmp_path_factory.mktemp("spark")
    with pytest.MonkeyPatch.context() as env:
        env.setenv("PYSPARK_PYTHON", sys.executable)
        env.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)
        env.setenv("SPARK_LOCAL_IP", "127.0.0.1")
        session = (
            spark_sql.SparkSession.builder.master("local[2]")
            .appName("skyulf-spark-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "3")
            .config("spark.sql.warehouse.dir", (work_dir / "warehouse").as_uri())
            .config("spark.local.dir", str(work_dir / "local"))
            .getOrCreate()
        )
        try:
            yield session
        finally:
            session.stop()


@pytest.fixture(params=["pandas", "polars"])
def fitted_regression_pipeline(request):
    """Different feature scales expose swapped columns and accidental double preprocessing."""
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"x": rng.normal(100, 20, 40), "z": rng.normal(5, 2, 40)})
    frame["target"] = 2 * frame.x - 3 * frame.z + 7
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x", "z"]}},
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["x", "z"]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    data = pl.from_pandas(frame) if request.param == "polars" else frame
    pipeline.fit(SplitDataset(train=data, test=data.head(0)), target_column="target")
    return pipeline


@pytest.fixture
def raw_frame():
    """Held-out values and a nondefault index expose re-fitting and row misalignment."""
    return pd.DataFrame({"x": [150.0, 80.0], "z": [9.0, 3.0]}, index=[17, 4])
