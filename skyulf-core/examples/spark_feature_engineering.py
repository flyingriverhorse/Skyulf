"""Fit, persist and restore native Spark FE; run with the optional Spark environment.

Use --state-path features.json to retain the portable bytes. The application owns
the Spark session and file I/O; the core API only exports/imports bytes. This demo
uses local[2] and a tiny fixture; Databricks callers supply their own session.
"""

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.preprocessing.pipeline import FeatureEngineer


def run_example(spark, state_path: Path | None = None) -> None:
    """Demonstrate training-only state and key-based parity after reload/repartition."""
    train = spark.createDataFrame(
        [(1, 1.0, 0, "unused"), (2, None, 1, "unused"), (3, 3.0, 0, "unused")],
        "id long, amount double, label long, extra string",
    ).select("id", "amount", "label")
    options = ExecutionOptions("spark", state_max_bytes=64 * 1024)
    engineer = FeatureEngineer(
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["amount"], "strategy": "mean"},
            },
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["amount"]}},
        ],
        frame_spec=FrameSpec(("id",), "label"),
        execution_options=options,
    )
    engineer.fit_transform(train)
    payload = engineer.export_state()
    if state_path is not None:
        state_path.write_bytes(payload)
        payload = state_path.read_bytes()
    restored = FeatureEngineer.from_state(
        payload, frame_spec=FrameSpec(("id",)), execution_options=options
    )
    batch = spark.createDataFrame([(10, 3.0), (20, None), (30, 1.0)], "id long, amount double")
    # Only this three-row demonstration result is collected; production can write a Spark table.
    rows = {row.id: row.amount for row in restored.transform(batch.repartition(2)).collect()}
    expected = {10: math.sqrt(1.5), 20: 0.0, 30: -math.sqrt(1.5)}
    assert all(
        math.isclose(rows[key], value, rel_tol=1e-10, abs_tol=1e-12)
        for key, value in expected.items()
    )
    assert restored.export_state() == payload
    print(f"Portable bytes: {len(payload)}; keyed features: {rows}")


def main() -> None:
    """Own the local session lifecycle and optional example artifact path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path)
    args = parser.parse_args()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark_sql = importlib.import_module("pyspark.sql")
    spark = (
        spark_sql.SparkSession.builder.master("local[2]")
        .appName("skyulf-feature-state-example")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        run_example(spark, args.state_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
