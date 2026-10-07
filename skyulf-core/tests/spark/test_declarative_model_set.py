"""Real Spark workers replay six fitted tree families and declarative arithmetic."""

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.data.dataset import SplitDataset  # noqa: E402
from skyulf.inference._manifest import ColumnSpec  # noqa: E402
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline  # noqa: E402
from skyulf.inference.model_set import (  # noqa: E402
    ComponentReference,
    load_model_set,
    save_model_set,
)
from skyulf.inference.model_set_scoring import predict_model_set  # noqa: E402
from skyulf.integrations.mlflow.models.model_set import log_model_set  # noqa: E402
from skyulf.integrations.mlflow.spark.spark_model import predict_spark_pyfunc  # noqa: E402
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


def _tree_set(tmp_path):
    """Build independent real Core components with string labels and fixed numeric fills."""
    records = {}
    for family in ("decision_tree", "random_forest", "extra_trees"):
        for task in ("regressor", "classifier"):
            branch = f"{family}_{task}"
            data = pd.DataFrame({"x": [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]})
            data["target"] = ["low"] * 4 + ["high"] * 4 if task == "classifier" else data.x * 2 + 1
            params = {"max_depth": 3, "random_state": 42}
            if family != "decision_tree":
                params.update(n_estimators=3, n_jobs=1)
            preprocessing = [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}
            ]
            if branch == "decision_tree_regressor":
                preprocessing.append(
                    {
                        "name": "range",
                        "transformer": "MinMaxScaler",
                        "params": {"columns": ["x"], "feature_range": [-2.0, 3.0]},
                    }
                )
            pipeline = SkyulfPipeline(
                {
                    "preprocessing": preprocessing,
                    "modeling": {"type": branch, "params": params},
                }
            )
            pipeline.fit(SplitDataset(train=data, test=data.iloc[:0]), target_column="target")
            path = tmp_path / branch
            save_local_pipeline(pipeline, path)
            digest = load_local_pipeline(path).manifest.pipeline_sha256
            records[branch] = (ComponentReference(name=branch, version="1", digest=digest), path)
    rules = [
        {
            "name": operation,
            "version": "1",
            "operation": operation,
            "params": {
                "inputs": [
                    "decision_tree_regressor__prediction",
                    "random_forest_classifier__probability_1",
                    "extra_trees_regressor__prediction",
                ],
                "weights": [1.0, 2.0, 3.0],
            },
            "columns": [{"name": operation + "_value", "dtype": "float64"}],
            "required_components": [
                "decision_tree_regressor",
                "random_forest_classifier",
                "extra_trees_regressor",
            ],
        }
        for operation in ("weighted_sum", "weighted_mean")
    ]
    artifact = save_model_set(
        tmp_path / "tree_set",
        records,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_config={"outputs": rules},
    )
    return load_model_set(artifact.directory)


def test_actual_spark_model_set_replays_all_tree_types_and_operations(spark, tmp_path, monkeypatch):
    """Spark partitions and bounded model calls must preserve labels, null fills and blends."""
    monkeypatch.chdir(tmp_path)
    artifact = _tree_set(tmp_path)
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
        mlflow.set_experiment("spark-tree-composition")
        with mlflow.start_run() as run:
            uri = log_model_set(artifact.directory, run_id=run.info.run_id, artifact_path="model")
        registered = mlflow.register_model(uri, "tree_composition")
        query = pd.DataFrame({"id": [9, 2, 7, 4, 1], "x": [np.nan, -2.0, 0.0, 9.0, np.nan]})
        expected = predict_model_set(query, artifact).sort_values("id").reset_index(drop=True)
        for partitions, batch_rows in ((1, 1), (3, 2)):
            source = spark.createDataFrame(query).repartition(partitions)
            result = predict_spark_pyfunc(
                spark,
                source,
                model_uri=f"models:/tree_composition/{registered.version}",
                artifact=artifact,
                record_key_columns=("id",),
                env_manager="local",
                prediction_batch_rows=batch_rows,
            )
            rows = result.orderBy("id").collect()
            actual = pd.DataFrame([row.asDict() for row in rows])
            pd.testing.assert_frame_equal(
                actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12
            )
            assert all(row["weighted_mean__scoring_status"] == "predicted" for row in rows)
        empty = predict_spark_pyfunc(
            spark,
            source.limit(0),
            model_uri=f"models:/tree_composition/{registered.version}",
            artifact=artifact,
            record_key_columns=("id",),
            env_manager="local",
        )
        assert empty.count() == 0
        assert empty.columns == expected.columns.tolist()
    finally:
        mlflow.set_tracking_uri(previous)
