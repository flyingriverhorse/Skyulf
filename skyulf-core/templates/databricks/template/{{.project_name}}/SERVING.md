# Serve a registered model through REST and SQL

This optional workflow deploys a **concrete registered model version**, then
creates a Unity Catalog SQL function calling its endpoint through `ai_query`.
The endpoint runs the saved preprocessing and prediction code. Callers supply
the model's input features, including any features produced by upstream joins
or aggregations. The function returns predictions; your application or SQL job
owns source selection and destination writes.

There is no extra default Bundle job. Run the steps below deliberately from a
Databricks Python notebook, or render the DDL and run it in Databricks SQL.
REST serving works independently of whether you create the SQL function.

For native entity-key feature lookup or gradual A/B traffic, see
[optional online lookup and daily rollout](#optional-online-lookup-and-daily-rollout)
below. Those workflows use separate opt-in configuration and guides.

## 1. Inspect the model and prepare the endpoint

Install the project's serving-compatible Skyulf wheel and MLflow dependencies
in your notebook environment. Use the **same Skyulf source runtime that packaged
the registered model**: admission checks source provenance, fitted artifact
digest, partition safety, named input/output signature and nullable transport.
Updating the local library does not silently upgrade an old model package.
Repackage and register a new version with the intended runtime when necessary.

Supported packages are certified local pipelines (including a selected
competition winner) and model sets. Admission is stricter than “any MLflow model”.
Spark-native models, stateful/batch-dependent preprocessing and native feature
lookup envelopes are not automatically admitted by this bridge.

Edit the explicit identifiers below. Use a separate endpoint and function name
for each release so existing callers continue to use their selected version.

```python
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    build_serving_sql_function,
    create_pinned_endpoint,
    create_serving_sql_function,
    prepare_pinned_endpoint,
    query_named_records,
    require_pinned_endpoint_ready,
)

client = WorkspaceClient()  # Notebook workspace identity, never embed a token.
spec = PinnedEndpointSpec(
    endpoint_name="risk-v7",
    model_name="models.risk.customer_risk",
    model_version="7",  # Concrete numeric version, not @champion or latest.
    logging_catalog="ops",
    logging_schema="serving",
    logging_table_prefix="risk_v7",
)
endpoint = prepare_pinned_endpoint(
    spec, tracking_uri="databricks", registry_uri="databricks-uc"
)
print("Inputs:", endpoint.input_schema)
print("Outputs:", endpoint.output_schema)
print("Endpoint configuration:", endpoint.config)

function = build_serving_sql_function(endpoint, "models.api.score_risk_v7")
print(function.create_sql)  # Inspect complete DDL before creating anything.
```

Preparation reads the registry but creates no endpoint or function. The generated
SQL uses the inspected schema, not an independently maintained feature list.
Fields must have simple names (letters, digits, underscores; no leading digit)
and must be unique ignoring case. Reserved words are quoted. Supported primitive
types are boolean, int32/int64, float32/float64 and string, including supported
pandas/Polars aliases. Unsupported types fail before SQL execution.

## 2. Create the REST endpoint and check readiness

```python
create_pinned_endpoint(client, endpoint)  # Run once for this new endpoint name.
```

Creation is asynchronous. Check the endpoint page, then run this in a later cell:

```python
require_pinned_endpoint_ready(client, endpoint)
```

This requires `READY` and `NOT_UPDATING`, and checks the exact model version,
workload and logging configuration. Retry the readiness check after provisioning;
do not repeatedly call creation. An existing endpoint name is rejected. The
helper does not overwrite endpoints or change model aliases.

The endpoint uses CPU/Small with scale-to-zero enabled. Idle endpoints can stop
serving compute after the platform's inactivity interval, and the next request
can incur a cold start. This is separate from SQL warehouse compute and billing.
See [Databricks custom model serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models).

For an inspected model whose only input is `x: float64`, a REST probe is:

```python
response = query_named_records(client, endpoint, [{"x": 1.5}])
print(response.predictions)
```

Replace that example with **all and only** your model's input fields. REST records
use native JSON scalars except fitted nullable `Int32`, `Int64`, `boolean` and
`Boolean` columns: those use canonical strings (e.g. `"42"`, `"true"`) or `None`,
matching the saved package. The SQL wrapper handles this encoding for you.

## 3. Create the SQL function

Use a Databricks Spark session and SDK client from the **same workspace**:

```python
create_serving_sql_function(spark, client, function)
```

The helper checks endpoint readiness/version before submitting `CREATE FUNCTION`.
It never uses `OR REPLACE` or `IF NOT EXISTS`: an existing function is an error,
so a stale implementation cannot be silently reused. SQL creation errors are
propagated without retries. After successful creation, rerun the SELECT below
as often as needed; creation is not part of every prediction query.

For a SQL warehouse, run `require_pinned_endpoint_ready` first and execute the
printed `function.create_sql` in the SQL editor. This path requires a Pro or
Serverless SQL warehouse; `ai_query` is unavailable on SQL Classic. Supported
Databricks Runtime compute requires 15.4 LTS or later. Verify workspace/region
availability in the [ai_query reference](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query).

The SQL wrapper calls the endpoint name, not a UC model URI. The model version is
checked at deployment, but an endpoint administrator can change it afterwards.
Restrict endpoint management rights and retain one endpoint per model release.
The SQL function cannot atomically enforce the version on each call. Run
`require_pinned_endpoint_ready` before a controlled batch as an additional check.

## 4. Call it from a table

```python
# These names stand for your existing table with the model's input feature columns.
expression = function.call_sql(table_alias="source")
print(expression)
scored = spark.sql(
    f"SELECT source.customer_id, {expression} AS result "
    "FROM analytics.features.customer_inputs AS source LIMIT 10"
)
display(scored)
```

Copy the printed expression into your own SQL job. It uses **named arguments**,
so source-table column order does not control feature order. The wrapper builds
`named_struct` in saved model order and declares `returnType` explicitly. SQL
arguments use the fitted types; nullable integer/boolean values are cast directly
to canonical strings without a floating-point conversion. Their nulls stay null.
For floating-point inputs, the wrapper represents SQL nulls as a same-width
`NaN` before calling `ai_query`, so the saved preprocessing handles missing
values. This prevents the numeric-null-to-zero behavior reproduced in native
testing. It does not fill values with a new SQL-side mean or constant.
Use matching source types: SQL's argument coercion is not a data validation layer.

`result` is the named per-row prediction struct, not the REST `predictions` array.
Its fields match `endpoint.output_schema`: e.g. `result.prediction`, probability
fields, model-set outputs and saved project-scoring status fields when present.
A single model, a registered competition winner and an admitted model set use
the same SQL API; the saved output schema determines the returned fields.

## Errors and privileges

Default `fail_on_error=True` stops the query on an endpoint error. To inspect
row-level endpoint failures, create a **different function** with capture enabled:

```python
capture = build_serving_sql_function(
    endpoint, "models.api.score_risk_v7_checked", fail_on_error=False
)
print(capture.create_sql)
create_serving_sql_function(spark, client, capture)
```

This function returns `{response: <prediction struct>, errorMessage: STRING}`.
On endpoint success, `errorMessage` is null. On an endpoint error, `response` is
null. Authorization, SQL and other non-endpoint failures can still fail the
whole query. Preserve and inspect errors before writing accepted predictions.
See [Databricks error behavior](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query).

- The deployer needs appropriate model/endpoint creation and logging destination
  permissions. Existing endpoint preparation also needs registry/artifact access.
- The function creator needs `USE CATALOG`, `USE SCHEMA`, `CREATE FUNCTION` in
  the function's schema, and endpoint `CAN QUERY`; the SDK readiness check also
  requires endpoint visibility. The endpoint must be in the SQL workspace.
- Function callers need `USE CATALOG`, `USE SCHEMA` and `EXECUTE` on the function,
  plus permissions for their own source reads, destination writes and compute.
  Databricks documents the `ai_query` definer's endpoint permission; verify the
  intended caller identity in your workspace before sharing the function.
- Skyulf does not grant permissions or create catalogs/schemas here. An owner
  can grant `EXECUTE ON FUNCTION models.api.score_risk_v7 TO ...` explicitly.

See [Unity Catalog UDF permissions](https://docs.databricks.com/aws/en/udf/unity-catalog)
and [SQL function creation](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function).

| Symptom | What to check |
| --- | --- |
| Preparation rejects source/digest/signature | Correct concrete model package and matching Skyulf runtime; do not bypass admission. |
| Endpoint not ready/config mismatch | Endpoint deployment state, build logs and exact version/configuration. |
| Function already exists | Use a new release name or deliberately manage the old function yourself. |
| SQL permission/unsupported feature error | Caller/definer privileges, warehouse/runtime and workspace preview availability. |
| Missing field or type error | Compare source fields with `endpoint.input_schema`; inspect SQL coercions. |
| Row returns errorMessage | Endpoint response/build logs; preserve the failed row for diagnosis. |
| First invocation is slow | Cold start and endpoint capacity; retry only under your workload's policy. |

## Validate your endpoint

Compare direct saved-model, REST and SQL predictions using the same representative
inputs. Include nulls, reordered named columns, integer limits and keys, and all
component/composition outputs used by your application. Check strict and captured
error behavior without treating an error result as a valid prediction.

Use the actual caller identity to verify access rules. For classification, inspect
labels and every probability column; for production capacity, measure cold starts,
scale-to-zero behavior and sustained load. Recipe admission does not establish
these endpoint-specific properties. Keep the concrete model version and request
schema fixed during the comparison.

Online monitoring enrollment is documented in
[README.md](README.md#pinned-serving-and-online-enrollment).

## Raw-input models and optional prediction tables

The model's saved input schema determines what callers send. If the complete
pipeline was fitted on raw columns, its admitted feature generation and fitted
preprocessing run inside the model. Callers do not repeat those transformations
or create a merged feature table. Spark joins and historical feature-table
calculations performed before that pipeline are not automatically included.

Training does not require a prediction sink. Keep `score_handoff: disabled`
and `scoring_mode: manual` for endpoint-only use; an output table name in
configuration does not itself create a table. REST calls and SQL functions
return results. A separate caller decides whether to publish them with a
DataFrame write or `CREATE TABLE ... AS SELECT`.

The repository example `skyulf-core/examples/databricks_raw_serving/` demonstrates
one synthetic customer classifier with imputation, interaction features, scaling
and category encoding. Its optional training-time batch write is disabled by
default. Separate tasks call the same concrete model through REST and a typed
SQL function, write two prediction tables and compare keys and probabilities.
Read its README for file responsibilities, commands and the small-data limits
of the example REST caller.

## Optional online lookup and daily rollout

Enable these jobs explicitly when generating a project. Both use
`config/serving.yml`, start with a paused schedule, and require `enabled: true`
after their resource settings are complete. Set `exclusive_writer: true` only
after restricting mutation permissions and serializing every writer to the
configured resources.

- `include_online_publication: "true"` adds an hourly job that publishes a
  latest-value feature table into an existing native online store. Follow the
  [online feature lookup guide](https://flyingriverhorse.github.io/Skyulf/user_guide/databricks_online_features/)
  to save a completeness and freshness policy with the model and admit it with
  `prepare_online_endpoint`. Supply the entity keys and any other fields in the
  inspected native request schema. A submitted sync does not establish that an
  endpoint has received the new feature values.
- `include_daily_rollout: "true"` adds a daily job for an explicitly initialized
  two-version endpoint and durable MLflow rollout run. The default policy moves
  traffic by 10 percentage points after at least 24 hours at each passing stage.
  The final passing 100% stage triggers guarded automatic `champion` promotion.
  Follow the [gradual serving rollout guide](https://flyingriverhorse.github.io/Skyulf/user_guide/databricks_serving_rollout/)
  for the saved comparison, request health, mature labels and recovery steps.

Keep the A/B endpoint separate from an endpoint used by a pinned SQL function.
Changing its traffic would change the function's prediction contract. Neither
opt-in job is generated by default, and neither requires a batch prediction sink.
