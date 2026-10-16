# Score raw trips with Spark, REST and SQL

This project trains a pandas random forest on up to 4,096 sampled January trips
from `samples.nyctaxi.trips`. The train job fits and saves the filters and five
preprocessing steps. Its admission checks run automatically because
`inference_mode: spark`. The separate `preprocessing_probe: true` adds diagnostics.

## Train and score

Use a Skyulf build that includes the recipe admission, excluded-row outputs and
endpoint compute settings shown here. When testing unreleased template changes,
replace the release pin in the deployment copy of `deployment/requirements.txt`
with the matching wheel and use that same build in the notebook. Updating the
template alone does not update a package already published on PyPI.

Follow the parent NYC examples README to load data, edit the workspace target
and deploy. Run `train`; automatic promotion requires RMSE <= 6 and at least
80% holdout coverage, below the standard 90% floor. This is deliberate: the
1.5-IQR rule on right-skewed trip distances can exclude valid long trips as
statistical outliers. Inspect their exclusion rate
and business impact before using this example's policy in production.
After promotion, the `score` job runs `mlflow.pyfunc.spark_udf` on Spark workers
and writes `trips_predictions`. It does not call a serving endpoint.

The fare range only selects training rows. The distance range is reused at
prediction time. A request needs no `fare_amount`. Out-of-range distances and
fitted IQR outliers retain their rows with `scoring_status = excluded`, an
`exclusion_reason` and an empty prediction. Missing numeric inputs pass through
Yeo-Johnson to the saved mean imputer. A zero distance is valid for Yeo-Johnson;
the preceding eligibility and IQR checks still apply.

SHAP explains sampled rows accepted by the saved preprocessing. Its JSON and
HTML report disclose sampled, accepted and excluded counts, including exclusion
reasons. Fewer than two accepted rows give `too_few_rows_after_exclusions`;
the sample is not topped up. Evaluation charts retain the original holdout
sample and plot its predicted rows, with excluded counts in captions and output.

Batch monitoring measures input drift, exclusion rates and observed performance.
This example's `deployment/targets.yml` binds `monitoring_label_table` to
`${var.catalog}.${var.input_schema}.trips_labels` and
`monitoring_result_available_at_column` to `result_available_at`. The training
configuration supplies the join key `trip_id` and target `fare_amount`. Run the
loader first: it creates outcomes with the trip's end time as availability time.
Only predictions with status `predicted` and an available, finite outcome enter
performance metrics; excluded rows are counted separately. Measuring performance
does not enable a performance-loss policy or automatic retraining.
Some taxi outcomes have a zero fare, which can make MAPE extremely large.
Inspect MAE and RMSE alongside coverage; do not treat MAPE alone as a useful
relative-error summary for this data.

The loader initially scores the same January population used for training, so
that first observation is a wiring check. To observe later data without retraining,
run only its commented February inserts for `trips_score` and `trips_labels`,
then run the score job again. Leave `trips_train` unchanged. The incremental batch
receipt identifies the new prediction window; inspect its row counts and source
versions instead of treating the entire prediction table as one observation.

`pickup_zip` is a string, not a numeric measurement. Categorical drift currently
supports at most 50 distinct non-null reference categories; this taxi reference
exceeds that limit. Its categorical evidence is therefore unavailable, and the
observation is degraded even when the four numeric features and performance are
measured. Do not interpret that status as proof of stable pickup-zone frequencies
or cast ZIP codes to numbers to bypass the limit. See `src/monitoring/README.md`
for monitoring setup and dashboard publication.

## Create the endpoint explicitly

No endpoint or online feature store is created by `bundle deploy` or `train`.
This example's `config/serving.yml` is read by the ordinary Python file
`src/serve.py`; it is not the
online-feature setup configuration. It uses the same public pinned-endpoint
SDK described in the generated `SERVING.md`.

1. Read the approved **concrete model version** from the training report. Fill
   `endpoint.model_version` in `config/serving.yml`. Check its three-part
   `model_name`, endpoint name and `sql_function`, especially on another target.
2. Set the four compute/readiness fields in that file if needed. The default is
   CPU / Small / scale to zero / 3,600 seconds. Workspaces can restrict these
   choices. With `logging_mode: disabled`, there is no endpoint payload table
   or serving monitoring; batch prediction monitoring remains separate.
3. Deploy the edited files. Open a Databricks Python notebook with the **same
   requirements as training** and run the following cells deliberately. Replace
   both paths with the deployed files' paths (Bundle root, then `files/`).
   Names must be new; creation never overwrites an existing endpoint or function.

```python
import runpy

helpers = runpy.run_path("/Workspace/<deployed project>/files/src/serve.py")
config_path = "/Workspace/<deployed project>/files/config/serving.yml"
function = helpers["prepare"](config_path)  # Read-only inspection.
print(function.create_sql)
```

In a second cell, create the endpoint, wait for readiness and create the SQL function:

```python
function = helpers["create"](spark, config_path)
```

A readiness timeout leaves the endpoint intact. Inspect its build logs, then
resume the existing endpoint using the `function` prepared in the first cell:

```python
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.serving import (
    create_serving_sql_function,
    wait_for_pinned_endpoint_ready,
)

client = WorkspaceClient()
wait_for_pinned_endpoint_ready(client, function.endpoint)
# A timeout during create() occurs before SQL function creation. If another
# run already created the function, omit this line after checking its identity.
create_serving_sql_function(spark, client, function)
```

Do not rerun endpoint creation blindly. Delete the endpoint when finished so
idle or active compute cannot keep charging; drop the example function after
its callers are stopped. The generated `SERVING.md` explains the API details.

## Compare REST and SQL

In that notebook, using the prepared `function` from the previous cells:

```python
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.serving import query_named_records

records = [
    {"pickup_hour": 10.0, "pickup_dow": 3.0, "trip_distance": 2.0,
     "duration_min": 12.0, "pickup_zip": "10001"},
    {"pickup_hour": 10.0, "pickup_dow": 3.0, "trip_distance": 99.0,
     "duration_min": 12.0, "pickup_zip": "10001"},
    {"pickup_hour": 10.0, "pickup_dow": 3.0, "trip_distance": None,
     "duration_min": None, "pickup_zip": "10001"},
]
print(query_named_records(WorkspaceClient(), function.endpoint, records).predictions)
```

The second row demonstrates a distance that passes the fixed 0-100 range but
is expected to exceed the **learned** IQR bound; inspect its actual reason.
The third demonstrates missing values. The fitted model, not the example code,
determines each result. SQL uses the typed function to preserve numeric NULLs
for imputation; a bare `ai_query` call can coerce a numeric NULL to zero.

```sql
SELECT s.trip_id,
  workspace.taxi_fare_robust.score_fare_v1(
    pickup_hour => s.pickup_hour, pickup_dow => s.pickup_dow,
    trip_distance => s.trip_distance, duration_min => s.duration_min,
    pickup_zip => s.pickup_zip) AS result
FROM VALUES
  (1, 10D, 3D, 2D, 12D, '10001'),
  (2, 10D, 3D, 99D, 12D, '10001'),
  (3, 10D, 3D, CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE), '10001')
AS s(trip_id, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip);
```

The function wraps `ai_query`; it returns a struct with prediction and status.
Your SQL can store that result with `CREATE TABLE ... AS SELECT ...`.
Never interpret an excluded row's empty prediction as a zero fare.
