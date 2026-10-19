# Taxi fare: endpoint, SQL and monitoring

Train one pandas Random Forest regressor on `samples.nyctaxi.trips`, promote it
when its holdout RMSE passes the 6.0 bound, and serve its saved preprocessing and
model through REST and SQL `ai_query`. No online feature store is needed.

**There is no batch score job or batch prediction table.** `trips_score` is an
input table used for verification and the SQL example. The telemetry payload
view contains captured requests and responses that monitoring reads; it is not
a prediction table maintained by a scoring job.

## Run the example

1. Copy this folder out of the repository. In the Databricks SQL editor, run the
   schema, `trips_base`, `trips_train` and `trips_score` statements from
   [load_nyctaxi.sql](../load_nyctaxi.sql) with `catalog=workspace` and
   `schema=taxi_fare_endpoint`. They read the built-in sample data and do not
   overwrite existing tables. The loader's label and zone-feature tables are
   not needed by this example.
2. Replace the test workspace URL in `deployment/targets.yml`. Check the test
   catalog/schema and `pipeline_owner` in `deployment/variables.yml`. The cost
   tag is `taxi-fare-endpoint-demo`; change it if your workspace requires another
   value. Monitoring uses the shared `workspace.taxi_monitoring` schema.
3. Run from this project folder, replacing `YOUR_PROFILE`:

   ```bash
   python src/tools/smoke.py
   python src/tools/preview.py --action train
   databricks bundle validate --strict -t test --profile YOUR_PROFILE
   databricks bundle deploy -t test --profile YOUR_PROFILE
   databricks bundle run train -t test --profile YOUR_PROFILE
   ```

4. Open the child `serving_endpoint` run linked from training and wait for its
   successful result. Training waits for submission, not endpoint readiness.
   Then follow [SERVING.md](SERVING.md) to call the SQL function, run monitoring
   and open the **Online serving** dashboard page.

Use the shared `test` target: its production Bundle mode enables automatic
serving updates. Personal development targets require manual serving and disable
shared monitoring. A rejected candidate or an unchanged champion does not submit
an automatic update. To retry after fixing deployment settings:

```bash
databricks bundle run serving_endpoint -t test --profile YOUR_PROFILE
```

This example follows the template in this checkout. For an unreleased checkout,
use a wheel built from that checkout in the job environment; an older published
Core package may not support these settings. See [deployment/README.md](deployment/README.md)
for dependencies, identities and permissions.

## What is configured

| File | Purpose |
| --- | --- |
| `config/training.yml` | Random Forest, six Optuna trials, three-fold CV, evaluation charts, SHAP and automatic promotion |
| `config/preprocessing.yml` | Saved mean imputation, fixed clipping, ZIP encoding and scaling |
| `config/serving.yml` | Follow champion, telemetry logging, SQL function and 20-row REST/SQL verification |
| `config/inference.yml` | Endpoint delivery and the source for verification; no prediction destination |
| `deployment/variables.yml` | Cost tag and shared monitoring namespace; no online outcome-label binding |

The adjacent `taxi_fare_endpoint.json` records initialization answers.
`build_example_project.py --check` verifies generated files while preserving
this guide, `SERVING.md` and the example's recipe files.
