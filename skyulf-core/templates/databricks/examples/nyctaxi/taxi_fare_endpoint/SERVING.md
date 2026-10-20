# Call the taxi endpoint and monitor its requests

First complete [deployment and training](README.md#run-the-example). Successful
champion promotion submits `serving_endpoint`, which creates or updates the
endpoint, verifies REST/SQL predictions against the saved model, and enrolls
the served version for monitoring. Check that child run, including its
`verification` and `monitoring` results, before making requests.

## 1. Call the generated SQL function

With the example's unchanged `test` bindings, the endpoint is
`taxi-fare-endpoint-test-endpoint` and its SQL function is
`workspace.taxi_fare_endpoint.taxi_fare_endpoint_score`. If you change bindings,
use the exact function name and query from the serving job's `examples.sql`.
Callers need function `EXECUTE`, endpoint `CAN_QUERY` and catalog/schema usage.

Run this query on a Databricks SQL warehouse supporting `ai_query`:

```sql
SELECT s.trip_id,
       workspace.taxi_fare_endpoint.taxi_fare_endpoint_score(
         pickup_hour => s.pickup_hour,
         pickup_dow => s.pickup_dow,
         trip_distance => s.trip_distance,
         duration_min => s.duration_min,
         pickup_zip => s.pickup_zip
       ) AS result
FROM (
  SELECT trip_id, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip
  FROM workspace.taxi_fare_endpoint.trips_score
  ORDER BY trip_id
  LIMIT 100
) AS s;
```

The function calls the endpoint through `ai_query`; `result` contains the model's
prediction and scoring fields. This `SELECT` displays results without creating
a batch prediction table. REST/SDK request examples are also in the serving run's
output. Requests use raw feature values; the endpoint applies saved preprocessing.

## 2. Read the captured traffic

`logging_mode: telemetry` captures REST and SQL requests. By default Databricks
creates `workspace.taxi_fare_endpoint.taxi_fare_endpoint_serving_payload` and its
backing `taxi_fare_endpoint_serving_otel_logs` table. Treat them as Databricks-managed
logging objects. Monitoring reads them; it does not create a batch prediction sink.

Log delivery is asynchronous. Once the payload view has received requests, run:

```bash
databricks bundle run monitoring -t test --profile YOUR_PROFILE
```

The job saves captured request counts, errors, latency and available feature-drift
measurements in `workspace.taxi_monitoring`. An empty observation immediately after
a request can mean logs have not arrived yet; rerun after delivery. Enrollment by
itself creates no traffic measurements. Monitoring is enabled on shared targets;
the generated hourly schedule starts paused. After the first successful manual
observation, set `monitoring_pause_status: UNPAUSED` in your target's variables
and redeploy to collect observations automatically.

## 3. Open Online serving

Reuse the dashboard for `workspace.taxi_monitoring` if one exists: copy the
commented `connect` example in `deployment/monitoring.yml` under `test` and paste
its published URL and ID. If this is the first project owning that store's
dashboard, copy the `create` example instead and set your SQL warehouse ID; the file
explains where to find it. Then deploy and run monitoring:

```bash
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run monitoring -t test --profile YOUR_PROFILE
```

Use the published link from the monitoring run and choose **Online
serving**, model `workspace.taxi_fare_endpoint.taxi_fare_endpoint_model`, then the
served version/context. Training charts and SHAP are available in the MLflow run.

**Performance needs actual outcomes.** This example leaves online label binding
and performance-loss policies off. The shared loader's `trips_labels` uses
`trip_id`; online monitoring needs `databricks_request_id` and zero-based
`request_row_index` to match each prediction. Do not bind that table directly.
See [serving monitoring](https://www.skyulf.com/manual/user_guide/databricks_serving_monitoring.html)
to supply matched labels and configure a performance policy.

Automatic champion following replaces the endpoint's single served version; it
does not enable gradual traffic rollout. To retire the example, disable
`serving_endpoint.follow_champion`, redeploy, then remove its endpoint and SQL
function explicitly. Destroying Bundle jobs alone does not remove those resources.
