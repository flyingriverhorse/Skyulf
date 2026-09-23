# SM-15I real NYC taxi end-to-end live validation

Status: PASSED on 2026-09-23. This is a separate real-data rehearsal of the existing small-data Skyulf local pipeline and automatic Delta insert scoring. SM-20a (the first Bundle) remains unstarted.

## Data and isolated resources

- CLI profile: `skyulf`, workspace `https://dbc-45604623-c18b.cloud.databricks.com`.
- Public source: [`samples.nyctaxi.trips`](https://dbc-45604623-c18b.cloud.databricks.com/explore/data/samples/nyctaxi/trips?isDbOne=true&utm_source=one-chat&o=7474646244882000), 21,932 original rows; it was read only.
- Filters: fare $2.50-$100, trip distance 0.1-30 miles, trip duration 1-120 minutes and complete source fields. Data discovery found 21,723 eligible distinct rows. The job copied 3,800 of them, deterministically ordered by a SHA-256 key, into the new `workspace.skyulf_nyctaxi_e2e_20260923` schema.
- New managed Delta tables: `taxi_records` (3,800), `taxi_training` (3,500), `taxi_score_source` (200 initially, then 300), `taxi_predictions` (200 initially, then 300), and `taxi_admission` (one owner row). Change Data Feed was enabled on `taxi_score_source` before scoring.
- The first 3,000 training rows, next 500 held-out test rows, next 200 initial inference rows and last 100 later inserts have disjoint stable keys. Only training rows fit preprocessing/model statistics. Inference labels stayed in `taxi_records` and were joined for evaluation *after* predictions were persisted.
- The wheel and notebook are in `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_nyctaxi_e2e_20260923/r1`. Wheel SHA-256: `42ad85847e37980daf06c00fb2f0dfaabf928c62f25e968f4f1c27c73efe5a09`.

## Skyulf and MLflow model

The saved Skyulf pipeline fitted `SimpleImputer` (numeric median), `StandardScaler` (numeric), `OneHotEncoder` (pickup/dropoff ZIP, unknown categories ignored) and a 40-tree depth-9 `random_forest_regressor`. The inputs were trip distance, trip duration, pickup hour/day and pickup/dropoff ZIP; `fare_amount` was the label and was never an input. This is completed-trip batch fare estimation, not pre-trip pricing.

- MLflow experiment: `/Users/edwardwolfe99@gmail.com/skyulf_nyctaxi_e2e_20260923/experiment`, ID `3707030714488006`.
- Training run: `73c53f97d2de48cb9759ed0b3e531f5d`, `FINISHED`. The saved full-pipeline artifact, pipeline configuration and data lineage were logged in this run. The UC model `workspace.skyulf_nyctaxi_e2e_20260923.taxi_fare_skyulf` version **1** is `READY` and points to that same run. Pipeline digest: `09f0f0a4be0b25f46db9442db016362fc3d8ee3784880b962e96d2bbdb3668fa`.
- Independently queried MLflow metrics on 500 held-out real trips: **MAE 0.435143**, **RMSE 1.024078**, **R2 0.988277**. The training run also records source table, row counts, model type and fit engine.
- Independent MLflow artifact listing confirmed `skyulf_pipeline_config.json`, `data_lineage.json` and the `model/` directory on the training run; both inference runs contain `delta_publication_receipt.json`.

## Three one-time serverless jobs

All three runs had a 900-second timeout, zero retries, `STANDARD` performance target, the uploaded Skyulf 0.9.0 wheel and `mlflow==3.16.1`. Each ended `SUCCESS`.

| Stage | Parent run / task run | Source rows selected | Target rows / version | MLflow run and observed metrics |
| --- | --- | ---: | ---: | --- |
| Train and create data | [653952531609511](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/246102427177961/run/653952531609511) / `557281442176653` | 3,000 train + 500 held-out | 0 | `73c53f97d2de48cb9759ed0b3e531f5d`; metrics above |
| First inference | [564243921177963](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/354675847486158/run/564243921177963) / `616926090231392` | 200 snapshot rows | 200 / 1 | `ee754c66f6b34db3ab3df2d551bbaf6e`; observed MAE 0.627872, RMSE 3.374626 |
| Append and infer | [1103696172771856](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/649442154281950/run/1103696172771856) / `568639379332373` | 100 CDF inserts | 300 / 2 | `b1f9a0f27d904260926a6c90b1fc9f68`; cumulative observed MAE 0.636182, RMSE 3.062555 |

Both inference runs loaded the same pinned UC model version through `prepare_local_workflow` and called `run_incremental_local_batch` without a date or source-version argument. Every persisted keyed prediction was compared with a fresh direct call to the registered local model. The second run checked all first 200 complete target rows were unchanged. An immediate third scoring call selected zero rows and left the target Delta version at **2**. The source high-water mark advanced from version **1** to **2** with the matching target commit. Both inference MLflow runs were independently queried and contain input/output/target row counts plus observed MAE/RMSE.

## Interpretation and boundary

This confirms the requested first-all-rows, later-new-rows behavior on real data. The measured scores are for one historical public sample and one held-out split; they do not establish production fitness or stability under data drift. Trip duration and dropoff ZIP are available after a trip, so this model is suitable only for retrospective/batch scoring. The incremental writer still requires insert-only changes, CDF retention, globally unique keys, bounded local batches and shared admission. The job notebook is a one-time rehearsal; a scheduled and restartable generated project is SM-20a.
