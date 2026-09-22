"""Validate registry-to-Spark parity before attempting the complete platform gate.

Run in a caller-selected Spark environment with a compatible installed Skyulf
wheel and MLflow. This probe creates a uniquely named test model and tracking
run, retains them for inspection, and never modifies an existing alias. It does
not deploy jobs, coordinate distributed publishers or write Delta tables.
"""

import argparse
import importlib
import json
import os
import platform
import re
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.data.dataset import SplitDataset
from skyulf.inference import build_bundle, predict_local, predict_spark
from skyulf.pipeline import SkyulfPipeline


def build_gold_bundle(engine: str):
    """Fit mean imputation and scaling with an independently known y=2*x oracle."""
    if engine not in ("pandas", "polars"):
        raise ValueError("training_engine must be pandas or polars.")
    training = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [0.0, 2.0, 4.0, 6.0]})
    if engine == "polars":
        training = pl.from_pandas(training)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x"], "strategy": "mean"},
                },
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    return bundle, pd.DataFrame({"x": [-1.0, np.nan, 2.5]})


def check_predictions(rows: list[tuple[int, float]]) -> None:
    """Compare keyed gold predictions, including the training-only imputed mean."""
    ordered = sorted(rows)
    if [row[0] for row in ordered] != [101, 102, 103]:
        raise AssertionError("Prediction keys differ from the three gold input keys.")
    np.testing.assert_allclose(
        [row[1] for row in ordered], [-2.0, 3.0, 5.0], rtol=1e-10, atol=1e-10
    )


def installation_evidence() -> dict[str, Any]:
    """Report the actual imported distribution without claiming a wheel was installed."""
    package = distribution("skyulf-core")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    import skyulf

    return {
        "module_path": str(Path(skyulf.__file__).resolve()),
        "editable": bool(direct.get("dir_info", {}).get("editable", False)),
        "version": package.version,
    }


def run_smoke(
    spark: Any,
    *,
    model_prefix: str,
    experiment_name: str,
    tracking_uri: str,
    registry_uri: str,
    training_engine: str = "pandas",
) -> dict[str, Any]:
    """Publish a test-owned bundle, download its pinned version and exercise workers."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}", model_prefix):
        raise ValueError("model_prefix must use simple registry identifiers.")
    if len(model_prefix.split(".")) not in (1, 3):
        raise ValueError("model_prefix must be model or catalog.schema.model.")
    if registry_uri.startswith("databricks-uc") and len(model_prefix.split(".")) != 3:
        raise ValueError("Unity Catalog requires catalog.schema.model_prefix.")
    from skyulf.integrations.mlflow.model import log_model
    from skyulf.integrations.mlflow.registry import (
        load_registered_bundle,
        register_model,
        resolve_model,
    )
    from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

    original, gold = build_gold_bundle(training_engine)
    check_predictions(
        list(zip([101, 102, 103], predict_local(gold, original).prediction, strict=True))
    )
    model_name = model_prefix + "_" + uuid4().hex
    print(f"Test-owned model (retained for inspection): {model_name}", flush=True)
    config = TrackingConfig(
        enabled=True, tracking_uri=tracking_uri, experiment_name=experiment_name
    )
    with track_run(config, run_name="skyulf-sm16-registry-spark") as run:
        uri = log_model(
            original, run_id=run.run_id, artifact_path="model", tracking_uri=tracking_uri
        )
        published = register_model(
            uri, model_name, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        resolved = resolve_model(
            model_name,
            version=published.version,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        downloaded = load_registered_bundle(
            resolved, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        if downloaded.semantic_digest != original.semantic_digest:
            raise AssertionError("Registry round-trip changed the test model digest.")
        check_predictions(
            list(zip([101, 102, 103], predict_local(gold, downloaded).prediction, strict=True))
        )
        frame = spark.createDataFrame(
            [(101, -1.0), (102, None), (103, 2.5)], "id long, x double"
        ).repartition(2)
        checks = {"local_gold": True}
        for mode in ("native_features", "python_pipeline"):
            output = predict_spark(
                frame,
                downloaded,
                frame_spec=FrameSpec(row_keys=("id",)),
                options=ExecutionOptions("spark", python_batch_rows=2),
                mode=mode,
            )
            # This fixed three-row probe is bounded; production batch data stays distributed.
            check_predictions([(row.id, row.prediction) for row in output.limit(4).collect()])
            checks[mode] = True
        run.log_metrics({"gold_rows": 3.0, "spark_modes_passed": 2.0})
    return {
        "stage": "registry_spark_parity",
        "platform_gate_complete": False,
        "checks": checks,
        "remaining_gates": [
            "wheel_on_clean_workers",
            "uc_alias_pinning",
            "delta_publication",
            "distributed_admission",
            "uc_permissions",
            "scale_measurement",
        ],
        "model_name": resolved.name,
        "model_version": resolved.version,
        "model_uri": resolved.model_uri,
        "model_digest": downloaded.semantic_digest,
        "mlflow_run_id": run.run_id,
        "training_engine": training_engine,
        "runtime": {
            "python": platform.python_version(),
            "spark": spark.version,
            "mlflow": version("mlflow"),
            "pyarrow": version("pyarrow"),
            "pandas": version("pandas"),
            "polars": version("polars"),
            "scikit_learn": version("scikit-learn"),
            "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        },
        "installation": installation_evidence(),
    }


def main() -> None:
    """Run only against explicitly selected stores and a caller-configured Spark runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--registry-uri", required=True)
    parser.add_argument("--training-engine", choices=("pandas", "polars"), default="pandas")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spark = importlib.import_module("pyspark.sql").SparkSession.builder.getOrCreate()
    report = run_smoke(
        spark,
        model_prefix=args.model_prefix,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        training_engine=args.training_engine,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Registry/Spark stage passed; complete platform gate remains pending. Report: {args.output}"
    )


if __name__ == "__main__":
    main()
