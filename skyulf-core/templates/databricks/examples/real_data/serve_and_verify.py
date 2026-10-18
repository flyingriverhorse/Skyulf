# Databricks notebook source
"""Create a pinned endpoint, then compare its results with real Spark UDF scoring.

Install the same Core build and requirements used by the training job first.
Run action=create once, wait for the endpoint to become ready, then run
action=verify. Existing endpoints and SQL functions are never overwritten.
The acceptance table includes explicitly marked altered inputs; the original
scoring table remains the unmodified real dataset.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.inference.model_set import load_model_set
from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    build_serving_sql_function,
    create_pinned_endpoint,
    create_serving_sql_function,
    prepare_pinned_endpoint,
    query_named_records,
)
from skyulf.integrations.mlflow.registration.registry import (
    download_registered_package,
    packaged_artifact_path,
)
from skyulf.integrations.mlflow.spark.spark_model import predict_spark_pyfunc

spark = SparkSession.builder.getOrCreate()
dbutils.widgets.dropdown("project", "diagnostic_single", ["diagnostic_single", "bike_demand_set"])  # noqa: F821
dbutils.widgets.text("model_version", "1")  # noqa: F821
dbutils.widgets.dropdown("action", "create", ["create", "verify"])  # noqa: F821
project = dbutils.widgets.get("project")  # noqa: F821
version = dbutils.widgets.get("model_version")  # noqa: F821
action = dbutils.widgets.get("action")  # noqa: F821
if project not in {"diagnostic_single", "bike_demand_set"}:
    raise ValueError("Select one of the two prepared projects.")
schema = f"workspace.{project}"
model_name = (
    f"{schema}.diagnostic_single_model"
    if project == "diagnostic_single"
    else f"{schema}.demand_set"
)
endpoint_name = f"{project.replace('_', '-')}-v{version}"
client = WorkspaceClient()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
spec = PinnedEndpointSpec(
    endpoint_name=endpoint_name,
    model_name=model_name,
    model_version=version,
    logging_catalog="workspace",
    logging_schema=project,
    logging_table_prefix=f"serving_v{version}",
    logging_mode="telemetry",
)
plan = prepare_pinned_endpoint(spec, tracking_uri="databricks", registry_uri="databricks-uc")
if action == "create":
    create_pinned_endpoint(client, plan)
    dbutils.notebook.exit(  # noqa: F821
        json.dumps(
            {"endpoint": endpoint_name, "model_uri": spec.model_uri, "status": "creation_requested"}
        )
    )  # noqa: F821

tracking = mlflow.tracking.MlflowClient()
package_path = Path(
    download_registered_package(mlflow, tracking, model_name, version, "databricks")
)
package_info = mlflow.models.Model.load(package_path)
kind = package_info.metadata["skyulf_artifact_kind"]
artifact_path = packaged_artifact_path(package_path, package_info.flavors, kind)
artifact = (
    load_pipeline(artifact_path) if kind == "local_pipeline" else load_model_set(artifact_path)
)
pyfunc = mlflow.pyfunc.load_model(str(package_path))

raw = spark.table(f"{schema}.observations_score").orderBy("record_id").limit(32).toPandas()
raw = raw[list(dict.fromkeys(["record_id", *plan.input_columns]))].copy()
edge = raw.iloc[:3].copy()
edge["record_id"] = [-1, -2, -3]
if project == "diagnostic_single":
    edge.loc[edge.index[0], "mean_radius"] = -1.0  # Reused pre-split filter.
    edge.loc[edge.index[1], "mean_area"] = np.nan  # Explicit eligibility rule.
    edge.loc[edge.index[2], "mean_texture"] = np.nan  # Saved imputer, still eligible.
else:
    edge.loc[edge.index[0], "hr"] = 99.0
    edge.loc[edge.index[1], "hr"] = np.nan
    edge.loc[edge.index[2], "hum"] = np.nan
sample = pd.concat([raw, edge], ignore_index=True)
inputs = sample.loc[:, list(plan.input_columns)]
expected = pyfunc.predict(inputs).reset_index(drop=True)
empty = pyfunc.predict(inputs.head(0))
assert len(empty) == 0 and list(empty.columns) == list(expected.columns)
reversed_predictions = pyfunc.predict(inputs.iloc[::-1]).iloc[::-1].reset_index(drop=True)
pd.testing.assert_frame_equal(expected, reversed_predictions)

distributed = spark.createDataFrame(sample).repartition(3)
scored = predict_spark_pyfunc(
    spark,
    distributed,
    model_uri=spec.model_uri,
    artifact=artifact,
    record_key_columns=("record_id",),
    env_manager="local",
    prediction_batch_rows=7,
    tracking_uri="databricks",
    registry_uri="databricks-uc",
)
spark_rows = (
    scored.toPandas()
    .set_index("record_id", drop=False)
    .loc[sample["record_id"]]
    .reset_index(drop=True)
)
pd.testing.assert_frame_equal(
    expected, spark_rows[expected.columns], check_dtype=False, atol=1e-10, rtol=1e-10
)
records = json.loads(inputs.to_json(orient="records", double_precision=15))
response = query_named_records(
    client, plan, records, client_request_id=f"task233-{project}-parity-v{version}"
)
rest = pd.DataFrame(response.predictions).loc[:, expected.columns]
pd.testing.assert_frame_equal(expected, rest, check_dtype=False, atol=1e-8, rtol=1e-8)

if project == "diagnostic_single":
    assert expected.iloc[-3:-1]["prediction"].isna().all()
    assert expected.iloc[-3:-1]["exclusion_reason"].notna().all()
    assert pd.notna(expected.iloc[-1]["prediction"])
else:
    total = expected["casual__nonnegative_demand"] + expected["registered__nonnegative_demand"]
    np.testing.assert_allclose(expected["total_demand"], total, equal_nan=True)
    assert expected.iloc[-3:-1]["total_demand"].isna().all()
    assert pd.notna(expected.iloc[-1]["total_demand"])
prediction_output = "prediction" if project == "diagnostic_single" else "total_demand"
assert scored.where(f"record_id IN (-1, -2) AND `{prediction_output}` IS NULL").count() == 2
scored.writeTo(f"{schema}.acceptance_predictions_v{version}").using("delta").create()
distributed.writeTo(f"{schema}.acceptance_inputs_v{version}").using("delta").create()

sql_plan = build_serving_sql_function(plan, f"{schema}.predict_v{version}")
create_serving_sql_function(spark, client, sql_plan)
sql_rows = spark.sql(
    f"SELECT record_id, {sql_plan.call_sql(table_alias='source')} AS result "
    f"FROM (SELECT * FROM {schema}.observations_score ORDER BY record_id LIMIT 5) source"
).collect()
assert len(sql_rows) == 5 and all(row.result is not None for row in sql_rows)
result = {
    "project": project,
    "model_uri": spec.model_uri,
    "endpoint": endpoint_name,
    "parity_rows": len(sample),
    "real_rows": len(raw),
    "altered_edge_rows": len(edge),
    "spark_partitions": 3,
    "spark_prediction_batch_rows": 7,
    "checks": [
        "raw_pyfunc",
        "empty",
        "reverse",
        "spark_udf",
        "rest",
        "pre_split_exclusion",
        "eligibility_exclusion",
        "saved_imputation",
        "sql_ai_query",
    ],
    "sql_rows": len(sql_rows),
    "output_columns": list(expected.columns),
    "logging_table": spec.inference_table,
    "components": (
        [{"name": model_name, "version": version}]
        if project == "diagnostic_single"
        else [
            {
                "name": component.reference.name,
                "version": component.reference.version,
                "branch": component.branch,
            }
            for component in artifact.manifest.components
        ]
    ),
}
evidence_path = Path(f"/Volumes/{schema.replace('.', '/')}/artifacts/acceptance-v{version}.json")
evidence_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
dbutils.notebook.exit(json.dumps(result))  # noqa: F821
