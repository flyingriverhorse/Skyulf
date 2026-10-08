"""Publish separate REST and SQL predictions from raw columns using one pinned endpoint."""

import json
import time

import numpy as np
import pandas as pd

from skyulf.inference.local_scoring import score_local_pipeline
from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    build_serving_sql_function,
    create_pinned_endpoint,
    create_serving_sql_function,
    prepare_pinned_endpoint,
    query_named_records,
    require_pinned_endpoint_ready,
)
from skyulf.integrations.mlflow.registration.registry import (
    load_registered_local_pipeline,
    resolve_model,
)

from .data import bounded_pandas, read_cohort
from .training import STORES


def prepare(namespace, endpoint_name, model_version):
    """Inspect one concrete registered pipeline; aliases never select inference versions."""
    catalog, schema = namespace.split(".")
    spec = PinnedEndpointSpec(
        endpoint_name,
        f"{namespace}.customer_churn",
        str(model_version),
        catalog,
        schema,
        "customer_churn",
    )
    return prepare_pinned_endpoint(spec, **STORES)


def deploy(client, plan):
    """Create a new endpoint and wait for readiness with visible progress and a deadline."""
    create_pinned_endpoint(client, plan)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        endpoint = client.api_client.do(
            "GET", f"/api/2.0/serving-endpoints/{plan.spec.endpoint_name}"
        )
        state = endpoint.get("state", {})
        print(json.dumps({"phase": "endpoint_build", "state": state}), flush=True)
        if state.get("config_update") == "UPDATE_FAILED":
            raise RuntimeError("Endpoint build failed; inspect the endpoint build logs")
        if state.get("ready") == "READY" and state.get("config_update") == "NOT_UPDATING":
            require_pinned_endpoint_ready(client, plan)
            return {
                "endpoint": plan.spec.endpoint_name,
                "model_uri": plan.spec.model_uri,
                "raw_inputs": list(plan.input_columns),
                "outputs": list(plan.output_schema),
            }
        time.sleep(20)
    raise TimeoutError("Endpoint was not READY within 20 minutes; inspect its build logs")


def write_predictions(spark, ids, predictions, table, source_version, model_uri="local_artifact"):
    """Retain exact integer keys and publish only after a complete prediction batch exists."""
    from pyspark.sql import functions as F

    if len(ids) != len(predictions) or len(set(ids)) != len(ids):
        raise ValueError("Predictions must preserve every unique source key")
    result = predictions.reset_index(drop=True).copy()
    result.insert(0, "customer_id", pd.Series(ids, dtype="int64"))
    output = (
        spark.createDataFrame(result)
        .withColumn("source_version", F.lit(source_version))
        .withColumn("model_uri", F.lit(model_uri))
        .withColumn("scored_at", F.current_timestamp())
    )
    output.write.format("delta").mode("error").saveAsTable(table)


def rest_predictions(spark, client, plan, namespace, source_version, config):
    """Send raw-only JSON batches through REST and write the separately requested sink."""
    source = bounded_pandas(
        read_cohort(spark, namespace, source_version, "score"), config["prediction_rows"]
    )
    raw = source[list(plan.input_columns)]
    records = json.loads(raw.to_json(orient="records", double_precision=15))
    responses = []
    for offset in range(0, len(records), 32):
        batch = records[offset : offset + 32]
        response = query_named_records(client, plan, batch)
        predictions = response.predictions
        if len(predictions) != len(batch):
            raise ValueError("REST changed the number of source rows")
        responses.extend(predictions)
    predictions = pd.DataFrame(responses)
    if set(predictions.columns) != {name for name, _ in plan.output_schema}:
        raise ValueError("REST output columns differ from the inspected model schema")
    table = f"{namespace}.predictions_rest"
    write_predictions(
        spark, source.customer_id.tolist(), predictions, table, source_version, plan.spec.model_uri
    )
    return {"table": table, "rows": len(predictions), "requests": (len(records) + 31) // 32}


def sql_predictions(spark, client, plan, namespace, source_version, *, function_ddl=None):
    """Use a typed UC SQL function over raw rows; Spark owns the output table write."""
    function = build_serving_sql_function(
        plan, f"{namespace}.predict_customer_v{plan.spec.model_version}"
    )
    if function_ddl is None:
        create_serving_sql_function(spark, client, function)
    else:
        require_pinned_endpoint_ready(client, plan)
        spark.sql(function_ddl).collect()
    call = function.call_sql(table_alias="raw")
    table = f"{namespace}.predictions_ai_query"
    query = f"""
      CREATE TABLE {table} USING DELTA AS
      SELECT customer_id, result.*, {source_version} AS source_version,
             '{plan.spec.model_uri}' AS model_uri, current_timestamp() AS scored_at
      FROM (
        SELECT raw.customer_id, {call} AS result
        FROM {namespace}.raw_customers VERSION AS OF {source_version} AS raw
        WHERE raw.cohort = 'score'
      )
    """
    print(query, flush=True)
    spark.sql(query).collect()
    return {
        "table": table,
        "rows": spark.table(table).count(),
        "function": function.function_name,
        "sql": query,
    }


def verify(spark, client, plan, namespace, source_version, config):
    """Compare persisted REST/SQL evidence with the registered artifact on identical raw rows."""
    require_pinned_endpoint_ready(client, plan)
    source = bounded_pandas(
        read_cohort(spark, namespace, source_version, "score"), config["prediction_rows"]
    )
    resolved = resolve_model(plan.spec.model_name, version=plan.spec.model_version, **STORES)
    artifact = load_registered_local_pipeline(resolved, **STORES)
    expected = score_local_pipeline(source[list(plan.input_columns)], artifact).reset_index(
        drop=True
    )
    errors = {}
    for method in ("rest", "sql"):
        table_name = "predictions_rest" if method == "rest" else "predictions_ai_query"
        actual = bounded_pandas(
            spark.table(f"{namespace}.{table_name}").orderBy("customer_id"),
            config["prediction_rows"],
        )
        assert actual.customer_id.tolist() == source.customer_id.tolist(), "Exact key mismatch"
        assert actual.model_uri.eq(plan.spec.model_uri).all(), "Model version mismatch"
        assert actual.source_version.eq(source_version).all(), "Delta version mismatch"
        for column in expected:
            np.testing.assert_allclose(
                actual[column],
                expected[column],
                atol=1e-12,
                rtol=0,
                err_msg=f"{method} column {column} differs from the registered artifact",
            )
        errors[method] = float(
            np.max(np.abs(actual[expected.columns].to_numpy() - expected.to_numpy()))
        )
    assert source[config["raw_columns"]].isna().any().any(), "Null-input case missing"
    assert source.segment.eq("NewSegment").any(), "Unseen-category case missing"
    assert list(plan.input_columns) == config["raw_columns"], (
        "Endpoint must accept only raw columns"
    )
    batch_exists = spark.catalog.tableExists(f"{namespace}.batch_predictions")
    assert batch_exists == config["write_batch_predictions"], "Unexpected batch publication"
    return {
        "status": "PASS",
        "rows_per_method": len(source),
        "maximum_absolute_error": errors,
        "batch_prediction_table_exists": batch_exists,
        "raw_columns": list(plan.input_columns),
        "feature_columns": list(artifact.manifest.feature_columns),
        "output_columns": list(expected.columns),
        "nulls_and_unseen_category": True,
    }
