# SM-15I real NYC taxi end-to-end rehearsal

Status: COMPLETE, 2026-09-23. See [the live report](15-sm15i-real-nyctaxi-live-report.md). This validates existing Skyulf Core contracts before SM-20a, not a Bundle implementation.

## Scope and data

- Use the previously selected `skyulf` Databricks CLI profile and create only `workspace.skyulf_nyctaxi_e2e_20260923` resources.
- Source is the public real `samples.nyctaxi.trips` table (21,932 rows, six original columns). Read it without modifying it.
- Keep fares $2.50-$100, distance 0.1-30 miles, duration 1-120 minutes, and complete ZIP/timestamp fields; data discovery found 21,723 eligible distinct rows.
- Materialize a deterministic bounded 3,800-row Delta copy with a stable hash key; use 3,000 train, 500 held-out test, 200 initial inference, and 100 later inserts. Fit statistics only on training rows.
- Predict completed-trip fare from distance, trip duration, pickup hour/day and pickup/dropoff ZIP, never from fare itself. Document that this is retrospective batch prediction, not pre-trip pricing.

## Execution and evidence

1. Create the isolated schema and MLflow experiment folder. Prepare the source/target/admission Delta tables in a one-time serverless job. Enable source Change Data Feed before the first inference run.
2. Read the bounded training table through Skyulf's UC reader. Fit a Skyulf preprocessing pipeline and regression model with `fit_local_workflow`; evaluate the saved artifact on 500 unseen labels with MAE/RMSE/R2. Log configuration, lineage, artifact and metrics to one MLflow run; register an immutable UC model version.
3. In a second job, resolve that version through `prepare_local_workflow` and call `run_incremental_local_batch` without date/version arguments. Verify exactly 200 persisted keyed predictions against direct model output.
4. Append 100 real held-back trips to the same source Delta table. In a third job, run the same inference call; verify exactly 100 new predictions, prior 200 unchanged, all persisted values match direct model output, and an immediate retry is a no-op.
5. Record schema, tables, model/run/metric identifiers, three job runs, row/version counts, and limitations in a live report. Run local lint/type/doc gates, then mark this rehearsal complete without starting SM-20a.

## Boundaries

The proof uses a small pandas local model, Spark for governed Delta reads/writes, insert-only CDF, stable unique keys, shared admission and explicit row/byte limits. It does not claim Spark-native model execution, online serving or production-ready model quality. Never update/delete the public source or alter existing test schemas.
