# Databricks notebook source
"""SM-58: materialized Spark inference, executor memory and worker-load benchmark.

Upload as a Databricks Python notebook on dedicated job compute. The benchmark
creates only its own experiment and registered models; it never changes existing
models, jobs or prediction tables. See initiative 171 for deployment and evidence.
"""

import io
import json
import math
import platform
import tempfile
import time
import zipfile
from functools import partial
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4


def validate_measurement(evidence, *, expected_rows, elapsed_seconds):
    """Reject incomplete, unmaterialized or incorrect results before reporting speed."""
    exact = {
        "rows": expected_rows,
        "keys": expected_rows,
        "min_id": 0,
        "max_id": expected_rows - 1,
        "invalid": 0,
    }
    if any(evidence.get(key) != value for key, value in exact.items()):
        raise ValueError(f"Invalid benchmark coverage: {evidence}")
    error = evidence.get("max_error")
    total = evidence.get("prediction_sum")
    if error is None or not math.isfinite(error) or error > 1e-7:
        raise ValueError(f"Invalid benchmark predictions: {evidence}")
    if total is None or not math.isfinite(total):
        raise ValueError(f"Invalid benchmark prediction checksum: {evidence}")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Invalid benchmark timer.")
    return {
        **evidence,
        "seconds": elapsed_seconds,
        "rows_per_second": expected_rows / elapsed_seconds,
    }


def record_case(report, destination, identity, run_case):
    """Checkpoint every completed action and retain errors without hiding job failure."""
    case = {**identity, "stage": "preparation", "actions": []}
    report["cases"].append(case)

    def checkpoint(update):
        """Keep the persisted record synchronized with the active measurement."""
        case.update(update)
        Path(destination).write_text(json.dumps(report, sort_keys=True))

    checkpoint({})
    try:
        outcome = run_case(on_progress=checkpoint)
    except Exception as error:  # noqa: BLE001 - persist diagnostics then preserve the exception
        checkpoint({"failure": {"type": type(error).__name__, "message": str(error)}})
        raise
    checkpoint({**outcome, "stage": "complete"})
    return case


def fit_fixture(width, directory, model_prefix):
    """Fit one deterministic model shared by every execution route for this width."""
    import mlflow
    import numpy as np
    import pandas as pd

    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.bundle import build_bundle
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
    from skyulf.integrations.mlflow.models.pipeline_model import log_pipeline_model
    from skyulf.pipeline import SkyulfPipeline

    columns = tuple(f"x{i}" for i in range(width))
    rng = np.random.default_rng(58)
    training = pd.DataFrame(rng.normal(size=(2048, width)), columns=columns)
    training.iloc[::17, 0] = np.nan
    means = training.mean().to_dict()
    weights = {name: (-1.0 if i % 2 else 1.0) * (i + 1) / width for i, name in enumerate(columns)}
    training["target"] = 7.0 + sum(
        training[name].fillna(means[name]) * weights[name] for name in columns
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": list(columns), "strategy": "mean"},
                },
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": list(columns)},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")
    artifact_path = directory / f"local_{width}"
    save_pipeline(pipeline, artifact_path)
    artifact = load_pipeline(artifact_path)
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=columns)
    with mlflow.start_run(run_name=f"sm58-width-{width}") as run:
        uri = log_pipeline_model(
            artifact_path, run_id=run.info.run_id, artifact_path="model", tracking_uri="databricks"
        )
    registered = mlflow.register_model(uri, f"{model_prefix}_{width}")
    return {
        "artifact": artifact,
        "bundle": bundle,
        "columns": columns,
        "means": means,
        "weights": weights,
        "model_uri": f"models:/{model_prefix}_{width}/{registered.version}",
    }


def raw_expression(functions, name, index):
    """Generate features from identity so SQL can independently verify each prediction."""
    value = ((functions.col("id") * (2 * index + 3) + index) % 1009) / 101.0 - 5.0
    return functions.when((functions.col("id") + index) % 37 == 0, None).otherwise(value)


def source_frame(spark, rows, partitions, columns):
    """Generate all rows on Spark; never materialize the scoring population locally."""
    from pyspark.sql import functions as F

    return spark.range(rows, numPartitions=partitions).select(
        "id", *(raw_expression(F, name, i).alias(name) for i, name in enumerate(columns))
    )


def consume_predictions(scored, fixture):
    """Force model evaluation and validate every prediction without collecting rows."""
    from pyspark.sql import functions as F

    expected = F.lit(7.0)
    for i, name in enumerate(fixture["columns"]):
        expected = (
            expected
            + F.coalesce(raw_expression(F, name, i), F.lit(fixture["means"][name]))
            * fixture["weights"][name]
        )
    prediction = F.col("prediction")
    invalid = prediction.isNull() | F.isnan(prediction) | (F.abs(prediction) == float("inf"))
    row = scored.agg(
        F.count("*").alias("rows"),
        F.countDistinct("id").alias("keys"),
        F.min("id").alias("min_id"),
        F.max("id").alias("max_id"),
        F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
        F.max(F.abs(prediction - expected)).alias("max_error"),
        F.sum(prediction).alias("prediction_sum"),
    ).first()
    return row.asDict()


def normalize_executor_metrics(metrics, *, process_metrics_enabled):
    """Preserve JVM counters while marking disabled or unobserved process RSS unknown."""
    process = {name: value for name, value in metrics.items() if name.startswith("ProcessTree")}
    observed = process_metrics_enabled and any(value > 0 for value in process.values())
    return {
        name: None if name in process and not observed else value for name, value in metrics.items()
    }


def executor_peaks(spark):
    """Read cumulative Spark executor process-tree high-water marks, excluding driver."""
    names = (
        "JVMHeapMemory",
        "JVMOffHeapMemory",
        "ProcessTreeJVMRSSMemory",
        "ProcessTreePythonRSSMemory",
        "ProcessTreeOtherRSSMemory",
    )
    if spark.__class__.__module__.startswith("pyspark.sql.connect"):
        return {"unavailable": "Serverless Spark Connect does not expose executor statusStore."}
    try:
        process_metrics_enabled = (
            spark.sparkContext.getConf()
            .get("spark.executor.processTreeMetrics.enabled", "false")
            .lower()
            == "true"
        )
        summaries = spark.sparkContext._jsc.sc().statusStore().executorList(True)
        records = []
        for i in range(summaries.size()):
            summary = summaries.apply(i)
            if summary.id() == "driver":
                continue
            peak = summary.peakMemoryMetrics()
            metrics = (
                normalize_executor_metrics(
                    {name: int(peak.get().getMetricValue(name)) for name in names},
                    process_metrics_enabled=process_metrics_enabled,
                )
                if peak.isDefined()
                else None
            )
            records.append(
                {
                    "executor_id": summary.id(),
                    "host": summary.hostPort(),
                    "total_tasks": summary.totalTasks(),
                    "failed_tasks": summary.failedTasks(),
                    "peak_bytes": metrics,
                }
            )
        return {
            "scope": "application_lifetime_peak_per_executor_not_case_delta",
            "process_metrics_enabled": process_metrics_enabled,
            "executors": records,
        }
    except Exception as error:  # noqa: BLE001 - optional JVM diagnostics must report unavailable
        return {"unavailable": f"{type(error).__name__}: {error}"}


def score_frame(spark, frame, fixture, mode, batch_rows, env_manager):
    """Use the public distributed routes without benchmark-specific scoring shortcuts."""
    from skyulf.core.execution import ExecutionOptions, FrameSpec
    from skyulf.inference.spark import predict_spark
    from skyulf.integrations.mlflow.spark.spark_model import predict_spark_pyfunc

    if mode == "pyfunc":
        return predict_spark_pyfunc(
            spark,
            frame,
            model_uri=fixture["model_uri"],
            artifact=fixture["artifact"],
            record_key_columns=("id",),
            env_manager=env_manager,
            prediction_batch_rows=batch_rows,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )
    return predict_spark(
        frame,
        fixture["bundle"],
        frame_spec=FrameSpec(record_key_columns=("id",)),
        options=ExecutionOptions("spark", python_batch_rows=batch_rows),
        mode=mode,
    )


def measure_case(spark, fixture, *, rows, partitions, batch_rows, mode, env_manager, on_progress):
    """Separate route preparation and two uncached fully evaluated actions."""
    from pyspark.sql import functions as F

    frame = source_frame(spark, rows, partitions, fixture["columns"])
    actual_partitions = (
        frame.select(F.spark_partition_id().alias("partition"))
        .agg(F.countDistinct("partition").alias("partitions"))
        .first()["partitions"]
    )
    started = time.perf_counter()
    scored = score_frame(spark, frame, fixture, mode, batch_rows, env_manager)
    preparation = time.perf_counter() - started
    case = {
        "mode": mode,
        "rows": rows,
        "features": len(fixture["columns"]),
        "partitions": actual_partitions,
        "requested_partitions": partitions,
        "prediction_batch_rows": batch_rows,
        "env_manager": env_manager if mode == "pyfunc" else None,
        "preparation_seconds": preparation,
        "actions": [],
    }
    for repeat in range(2):
        case["stage"] = f"repeat_{repeat}"
        on_progress(case)
        started = time.perf_counter()
        evidence = consume_predictions(scored, fixture)
        seconds = time.perf_counter() - started
        case["actions"].append(
            {
                "repeat": repeat,
                **validate_measurement(evidence, expected_rows=rows, elapsed_seconds=seconds),
            }
        )
        on_progress(case)
    case["executor_peaks"] = executor_peaks(spark)
    return case


def package_archive(model_uri):
    """Copy only our just-created trusted MLflow package for isolated worker probes."""
    import mlflow

    path = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(path.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(path).as_posix())
    return output.getvalue()


def worker_load_probe(spark, fixture, batch_rows):
    """Measure actual pyfunc restore and predict RSS in executor Python processes."""
    archive = package_archive(fixture["model_uri"])
    columns = fixture["columns"]

    def probe(batches):
        """Return one diagnostic per partition, never the prediction population."""
        import os
        import resource
        import socket

        import mlflow
        import numpy as np
        import pandas as pd
        import psutil

        process = psutil.Process()
        for _batch in batches:
            with tempfile.TemporaryDirectory(prefix="sm58-worker-") as directory:
                with zipfile.ZipFile(io.BytesIO(archive)) as saved:
                    saved.extractall(directory)  # self-generated trusted archive only
                before = process.memory_info().rss
                started = time.perf_counter()
                model = mlflow.pyfunc.load_model(directory)
                load_seconds = time.perf_counter() - started
                loaded = process.memory_info().rss
                query = pd.DataFrame(np.ones((batch_rows, len(columns))), columns=columns)
                started = time.perf_counter()
                output = model.predict(query)
                predict_seconds = time.perf_counter() - started
                if len(output) != batch_rows or not output.prediction.notna().all():
                    raise ValueError("Invalid worker benchmark predictions.")
                yield pd.DataFrame(
                    [
                        {
                            "host": socket.gethostname(),
                            "pid": os.getpid(),
                            "load_seconds": load_seconds,
                            "predict_seconds": predict_seconds,
                            "rss_before": before,
                            "rss_loaded": loaded,
                            "rss_after_prediction": process.memory_info().rss,
                            "process_lifetime_peak_rss": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            * 1024,
                            "rows": batch_rows,
                        }
                    ]
                )

    schema = (
        "host string, pid long, load_seconds double, predict_seconds double, "
        "rss_before long, rss_loaded long, rss_after_prediction long, "
        "process_lifetime_peak_rss long, rows long"
    )
    records = spark.range(8, numPartitions=8).mapInPandas(probe, schema).collect()
    return {
        "features": len(columns),
        "prediction_rows": batch_rows,
        "scope": "separate_worker_probe_not_udf_action_timing",
        "workers": [row.asDict() for row in records],
    }


def run_benchmark(spark, dbutils):
    """Run the finite matrix and retain partial evidence even when a case fails."""
    import mlflow

    from skyulf.integrations.mlflow.spark.spark_model import runtime_source_digest

    dbutils.widgets.text("schema", "workspace.skyulf_sm58_20261005")
    dbutils.widgets.text("experiment", "/Shared/skyulf-sm58-20261005")
    dbutils.widgets.text("env_manager", "virtualenv")
    dbutils.widgets.text("compute", "classic")
    dbutils.widgets.text("suite", "full")
    suite = dbutils.widgets.get("suite")
    if suite not in ("full", "smoke"):
        raise ValueError("suite must be full or smoke")
    schema = dbutils.widgets.get("schema")
    # Identifier input is validated before it reaches CREATE SCHEMA.
    if len(schema.split(".")) != 2 or any(
        not part.replace("_", "").isalnum() for part in schema.split(".")
    ):
        raise ValueError("schema must be a simple catalog.schema identifier")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {schema}.benchmark")
    report_path = f"/Volumes/{schema.replace('.', '/')}/benchmark/{uuid4().hex}.json"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(dbutils.widgets.get("experiment"))
    classic = dbutils.widgets.get("compute") == "classic"
    if classic:
        spark.conf.set("spark.sql.adaptive.enabled", "false")
        spark.conf.set("spark.sql.shuffle.partitions", "32")
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
    result = {
        "python": platform.python_version(),
        "spark": spark.version,
        "source_sha256": runtime_source_digest(),
        "versions": {
            name: version(name)
            for name in ("skyulf-core", "mlflow", "scikit-learn", "pandas", "numpy", "pyarrow")
        },
        "compute": dbutils.widgets.get("compute"),
        "suite": suite,
        "arrow_max_records_per_batch": 10000 if classic else "platform-managed",
        "cases": [],
        "load_probes": [],
        "timing_scope": "uncached source generation + inference + all-row correctness aggregate; no Delta write",
        "models": [],
        "complete": False,
        "report_path": report_path,
    }
    print(f"SM58_REPORT_PATH={report_path}")
    Path(report_path).write_text(json.dumps(result, sort_keys=True))
    with tempfile.TemporaryDirectory(prefix="sm58-driver-") as directory:
        for width in (2,) if suite == "smoke" else (2, 64):
            fixture = fit_fixture(width, Path(directory), f"{schema}.linear")
            result["models"].append(
                {
                    "uri": fixture["model_uri"],
                    "features": width,
                    "portable_digest": fixture["bundle"].semantic_digest,
                }
            )
            matrix = (
                [
                    (1_000_000, 8, 10000),
                    (5_000_000, 8, 10000),
                    (5_000_000, 32, 10000),
                    (5_000_000, 32, 1000),
                ]
                if width == 2
                else [(1_000_000, 32, 10000), (1_000_000, 32, 1000)]
            )
            if suite == "smoke":
                matrix = [(10000, 2, 1000)]
            for rows, partitions, batch_rows in matrix:
                for mode in ("native_features", "python_pipeline", "pyfunc"):
                    identity = {
                        "mode": mode,
                        "rows": rows,
                        "features": width,
                        "requested_partitions": partitions,
                        "prediction_batch_rows": batch_rows,
                    }
                    case = record_case(
                        result,
                        report_path,
                        identity,
                        partial(
                            measure_case,
                            spark,
                            fixture,
                            rows=rows,
                            partitions=partitions,
                            batch_rows=batch_rows,
                            mode=mode,
                            env_manager=dbutils.widgets.get("env_manager"),
                        ),
                    )
                    print(json.dumps(case, sort_keys=True))
            for batch_rows in (1000,) if suite == "smoke" else (1000, 10000):
                result["load_probes"].append(worker_load_probe(spark, fixture, batch_rows))
                Path(report_path).write_text(json.dumps(result, sort_keys=True))
    result["executor_peaks"] = executor_peaks(spark)
    result["complete"] = True
    Path(report_path).write_text(json.dumps(result, sort_keys=True))
    dbutils.notebook.exit(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run_benchmark(globals()["spark"], globals()["dbutils"])
