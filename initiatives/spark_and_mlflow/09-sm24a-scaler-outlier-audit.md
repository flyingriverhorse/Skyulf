# SM-24a scaler and outlier inference audit

Status: completed on 2026-09-23 for the bounded pandas/Polars Databricks workflow. The executable probe is [`databricks_local_sm24a_scaler_outlier_audit.py`](../../skyulf-core/examples/databricks_local_sm24a_scaler_outlier_audit.py). This audit tests the existing implementation; it does not add a new inference endpoint or a Spark-native transform.

## Result

| Nodes | pandas | Polars | Inference behavior |
| --- | --- | --- | --- |
| MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler | Passed | Passed | Apply fitted scaling state and return one prediction per input, including an extreme value. |
| Winsorize | Passed | Passed | Apply fitted percentile clipping and return one prediction per input. |
| IQR, ZScore, ManualBounds, EllipticEnvelope | Guard passed | Guard passed | An extreme value is filtered by the fitted transform; prediction raises a row-count error instead of silently returning fewer predictions. An input on which no row is filtered can still score. |

The 62-ID matrix currently has 41 inference-eligible registrations, four conditional outlier registrations, seven train-only, two inspection, two optional-dependency and six unsupported registrations. Eight optional/unsupported registrations remain `not_run`; this audit does not claim they work in inference.

`train_only` in the earlier node matrix was too coarse for the four filtering outlier nodes. They are fitted and applied during inference, but arbitrary batches cannot be promised one output per input. The [node matrix](07-sm24a-node-matrix.csv) now records their conditional inference behavior and the live guard evidence.

## Evidence

- Local regression gate: 219 focused outlier, row-contract and numeric tests passed. A separate saved-artifact probe fitted, saved, loaded and predicted all five row-preserving nodes on both engines (10/10); each returned three predictions, and Winsorize clipped `x=1000`.
- Databricks training parent run `973375074755539`, task `176846010524557`: **SUCCESS** with a 900-second timeout and zero retries. It read 1,000 training rows from `workspace.skyulf_sm24a_20260923.skyulf_sm24a_train_source` at Delta version 0. It created a test-only two-month scoring table `workspace.skyulf_sm24a_20260923.skyulf_sm24a_scaler_outlier_score_source` at Delta version 0, with 20 rows per month and one `x=1000` row in each month.
- The training job fitted and registered 10 separate linear-regression pipelines: five preprocessing nodes times two engines. Each model was registered under a test-only Unity Catalog name at concrete version 1; no alias was changed. The receipt with exact names, versions, digests, run IDs, metrics and reference predictions is in `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/scaler_receipt.json` and locally in `.cache/sm24a-scaler-receipt.json`.
- All four filtering outlier nodes removed the extreme row in their direct transform. `SkyulfPipeline.predict` rejected the shortened batch for each node and both engines (8/8 guarded cases) in the Databricks training job. The existing local row-contract test also checks that normal rows still predict.
- Independent scoring parent run `629958980658605`, task `1090605491396608`: **SUCCESS**. It loaded all 10 concrete model versions via the MLflow/Unity Catalog workflow and scored January and February separately. All 20 model-month cases retained all 20 entity keys, including the extreme key, and matched the training-job saved-artifact reference at absolute tolerance `1e-9`. This reference is a replay comparison, not an independent mathematical oracle.
- The new MLflow runs log `heldout_mae`, `heldout_rmse`, `heldout_r2`, engine and preprocessing. The tracking API confirmed these are present for the MaxAbsScaler/pandas run `d3a87c86c7e94f6ab907ab656736164f`; its held-out R2 was 0.9473 on this synthetic fixture. Metric presence was checked directly, not inferred from the receipt.
- Focused Ruff and Ty checks passed for the probe script. The live wheel was the previously validated `skyulf-core 0.9.0` wheel in the isolated SM-24a workspace folder.

## Scope

Here, "live" means a Databricks serverless **batch job** reading bounded Delta rows into a pandas or Polars local pipeline and loading a saved model through MLflow. It does not mean a Model Serving HTTP endpoint or distributed Spark-native FE. The latter remains separate work in the Spark node matrix. The test did not write a prediction Delta table, deploy a Bundle, or certify every parameter combination. Filtering outlier nodes require an explicit product policy if prediction must return a result for every row: keep and flag, clip, or reject the batch. The current behavior is rejection when any row is removed.
