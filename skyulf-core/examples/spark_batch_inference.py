"""Run a small Spark batch with a standalone classification bundle."""

import argparse
import os
import sys

import pandas as pd

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.data.dataset import SplitDataset
from skyulf.inference import build_bundle, predict_spark
from skyulf.pipeline import SkyulfPipeline


def build_demo_bundle():
    """Train a pandas model and freeze its raw-input inference contract."""
    training = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "label": ["no", "no", "no", "yes", "yes", "yes"],
        }
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="label")
    return build_bundle(pipeline, input_stage="raw", feature_order=("x",))


def run_example(mode: str) -> None:
    """Execute a distributed classification batch and print keyed predictions."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("skyulf-spark-batch-inference-example")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        incoming = spark.createDataFrame(
            [(101, -2.5), (102, -0.25), (103, 0.25), (104, 2.5)],
            "id long, x double",
        ).repartition(2)
        output = predict_spark(
            incoming,
            build_demo_bundle(),
            frame_spec=FrameSpec(row_keys=("id",)),
            options=ExecutionOptions("spark", python_batch_rows=2),
            mode=mode,
        )
        output.orderBy("id").show()
    finally:
        spark.stop()


def main() -> None:
    """Select the native or worker-local Python feature-engineering path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("native_features", "python_pipeline"), default="native_features"
    )
    args = parser.parse_args()
    run_example(args.mode)


if __name__ == "__main__":
    main()
