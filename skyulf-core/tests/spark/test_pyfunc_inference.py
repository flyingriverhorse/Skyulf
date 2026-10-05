"""Actual MLflow Spark UDF parity on named, distributed pandas-worker batches."""

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.data.dataset import SplitDataset  # noqa: E402
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline  # noqa: E402
from skyulf.inference.local_scoring import score_local_pipeline  # noqa: E402
from skyulf.integrations.mlflow.local_model import log_local_model  # noqa: E402
from skyulf.integrations.mlflow.spark_model import predict_spark_pyfunc  # noqa: E402
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_actual_named_spark_udf_matches_local_and_reuses_workers(
    spark, tmp_path, monkeypatch, task
):
    """Nulls, shuffled feature order, partitions and string labels retain local predictions."""
    monkeypatch.chdir(tmp_path)
    training = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "z": [10.0, 3.0, 9.0, 1.0, 7.0, 2.0]}
    )
    training["target"] = (
        ["no", "no", "no", "yes", "yes", "yes"]
        if task == "classification"
        else training.x * 3 + training.z
    )
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
            "modeling": {
                "type": "logistic_regression" if task == "classification" else "linear_regression"
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")
    local = tmp_path / "local"
    save_local_pipeline(pipeline, local)
    artifact = load_local_pipeline(local)
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
        mlflow.set_experiment("spark-pyfunc")
        with mlflow.start_run() as run:
            uri = log_local_model(local, run_id=run.info.run_id, artifact_path="model")
        registered = mlflow.register_model(uri, "spark_model")
        query = pd.DataFrame(
            {
                "z": [2.0, 8.0, np.nan, 1.0, 9.0, 0.0, 2.0],
                "x": [4.0, np.nan, 2.0, 8.0, 1.0, 3.0, 5.0],
            }
        )
        query.insert(0, "id", np.arange(len(query), dtype="int64") + 2**53)
        frame = spark.createDataFrame(query).repartition(3)
        actual = predict_spark_pyfunc(
            spark,
            frame,
            model_uri=f"models:/spark_model/{registered.version}",
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="local",
        )
        expected = score_local_pipeline(query[["x", "z"]], artifact)
        for _ in range(2):
            # Only this bounded fixture is collected; production adapter retains a Spark frame.
            rows = actual.orderBy("id").collect()
            assert [row.id for row in rows] == query.id.tolist()
            result = pd.DataFrame([row.asDict() for row in rows]).drop(columns="id")
            pd.testing.assert_frame_equal(
                result, expected, check_dtype=False, atol=1e-10, rtol=1e-10
            )
        assert "python" in actual._jdf.queryExecution().executedPlan().toString().lower()
    finally:
        mlflow.set_tracking_uri(previous)
