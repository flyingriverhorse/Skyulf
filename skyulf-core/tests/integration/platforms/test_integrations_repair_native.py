"""Native Spark and Delta acceptance for repaired serving and identity boundaries."""

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

pytest.importorskip("mlflow")


@pytest.mark.parametrize(
    "dtype,values",
    [("float64", [1, None]), ("Int64", [2**53 + 1, None]), ("boolean", [True, None])],
)
def test_native_serving_sql_sample_preserves_fitted_values(delta_spark, dtype, values):
    """Native SQL construction must match the fitted dtype without rounding keys or nulls."""
    from skyulf.integrations.databricks.jobs.serving_endpoint_job import _sql_input_frame
    from skyulf.integrations.databricks.serving.contracts import (
        PinnedEndpointPlan,
        PinnedEndpointSpec,
    )
    from skyulf.integrations.databricks.serving.sql_functions import build_serving_sql_function

    plan = PinnedEndpointPlan(
        PinnedEndpointSpec(
            "native-test", "main.models.test", "1", "", "", "", logging_mode="disabled"
        ),
        {},
        ("value",),
        (("value", dtype),),
        (("prediction", "float64"),),
    )
    function = build_serving_sql_function(plan, "main.models.native_test")
    frame, ordinal = _sql_input_frame(delta_spark, function, [{"value": value} for value in values])
    actual = [row.value for row in frame.orderBy(ordinal).collect()]
    expected = [float(values[0]) if dtype == "float64" else values[0], None]
    assert actual == expected and actual[1] is None


def test_native_feature_snapshot_accepts_maintenance_rejects_update(delta_spark):
    """A Delta metadata-only commit may advance history; changed feature values may not."""
    from skyulf.integrations.databricks.feature_store.snapshots import (
        snapshot_table,
        validate_snapshots,
    )

    spark = delta_spark
    namespace = f"{spark.catalog.currentCatalog()}.{spark.catalog.currentDatabase()}"
    name = f"{namespace}.repair_features_{uuid4().hex}"
    try:
        spark.createDataFrame([(1, 2.0), (2, None), (1, 2.0)], "id int, value double").write.format(
            "delta"
        ).saveAsTable(name)
        saved = snapshot_table(spark, name)
        spark.sql(f"ALTER TABLE {name} SET TBLPROPERTIES ('repair_probe' = 'maintenance')")
        assert snapshot_table(spark, name)["version"] > saved["version"]
        validate_snapshots(spark, [saved])
        spark.sql(f"UPDATE {name} SET value = 3.0 WHERE id = 1")
        with pytest.raises(ValueError, match="changed.*retrain"):
            validate_snapshots(spark, [saved])
        assert spark.table(name).where("id = 1 AND value = 3.0").count() == 2
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {name}")


def test_native_serving_label_schema_accepts_request_outcomes(delta_spark):
    """Request-key labels pass native schema checks while business-key-only labels fail early."""
    from skyulf.integrations.databricks.jobs.serving_enrollment import (
        validate_serving_label_sources,
    )

    name = "repair_labels_" + uuid4().hex
    frame = delta_spark.createDataFrame(
        [("request-a", 0, "2026-10-20T00:00:00Z")],
        "databricks_request_id string, request_row_index int, available_at string",
    )
    frame.createOrReplaceTempView(name)
    config: Any = SimpleNamespace(label_table=name, result_available_at_column="available_at")
    try:
        validate_serving_label_sources(delta_spark, [config])
        frame.drop("databricks_request_id").createOrReplaceTempView(name)
        with pytest.raises(ValueError, match="business record keys"):
            validate_serving_label_sources(delta_spark, [config])
        assert delta_spark.table(name).count() == 1
    finally:
        delta_spark.catalog.dropTempView(name)


def test_native_mlflow_model_set_keeps_integer_keys_with_exclusions(delta_spark, tmp_path):
    """Actual MLflow Spark workers must preserve wide keys and widen small keys without float."""
    import mlflow

    types = pytest.importorskip("pyspark.sql.types")

    from skyulf import SkyulfPipeline
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.inference.model_set_scoring import model_set_output_schema
    from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
        seal_partition_probe,
    )
    from skyulf.integrations.mlflow.models.model_set import model_set_save_options
    from skyulf.integrations.mlflow.spark import spark_model

    training = pd.DataFrame({"x": [float(i) / 10 for i in range(-30, 30)]})
    training["target"] = (training.x > 0).astype("int64")
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "bounds",
                    "transformer": "ManualBounds",
                    "params": {
                        "bounds": {"x": {"lower": -3.0, "upper": 3.0}},
                    },
                }
            ],
            "modeling": {"type": "decision_tree_classifier", "params": {"max_depth": 3}},
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")
    component = tmp_path / "component"
    save_pipeline(pipeline, component)
    fitted = seal_partition_probe(load_pipeline(component), training, component, enabled=True)
    artifact = save_model_set(
        tmp_path / "set",
        {
            "first": (
                ComponentReference(
                    name="model", version="1", digest=fitted.manifest.pipeline_sha256
                ),
                component,
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    options_dir = tmp_path / "options"
    options_dir.mkdir()
    model_path = str(tmp_path / "pyfunc")
    mlflow.pyfunc.save_model(path=model_path, **model_set_save_options(artifact, options_dir))
    outputs = model_set_output_schema(artifact)
    result_type = spark_model._result_type(types, outputs, True, ("id",))
    udf = mlflow.pyfunc.spark_udf(
        delta_spark,
        model_path,
        result_type=result_type,
        env_manager="local",
        params={"spark_output": True},
    )
    for dtype, keys in (
        ("bigint", [1, 2**53 + 1, 2**63 - 1]),
        ("int", [1, 2, 3]),
        ("smallint", [1, 2, 3]),
    ):
        source = delta_spark.createDataFrame(
            list(zip(keys, [0.0, 100.0, -100.0], strict=True)),
            f"id {dtype}, x double",
        )
        named = spark_model._named_inputs(source, [("id", "int64"), ("x", "float64")])
        result = spark_model._keyed_result(source, ("id",), outputs, udf(named), nullable=True)
        rows = result.orderBy("id").collect()
        assert result.schema["id"].dataType.simpleString() == "bigint"
        assert [row.id for row in rows] == keys
        assert [row.first__prediction is None for row in rows] == [False, True, True]
