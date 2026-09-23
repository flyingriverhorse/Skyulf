# SM-24a live validation report

Status: completed for the documented small-data scope on 2026-09-23. This
report separates local proof from remote results. The
[plan](06-sm24a-live-validation-plan.md) states the accepted transport
boundary; the [62-ID matrix](07-sm24a-node-matrix.csv) keeps inapplicable and
untested nodes visible.

## Local evidence

- The SM-24a reader, training entry and five model configurations passed 29
  focused tests (one real Spark test skipped in the base environment).
- The requested-month read passed separately against real local Spark in the
  repository's Linux/Java test environment: 1 passed.
- Ruff, Ty and `mkdocs build --strict` passed for the changed files/docs.
- Five fit/save/load/predict cases passed locally: R1 pandas linear regression,
  R2 Polars random forest regression, C1 pandas logistic regression, C2 Polars
  random forest classification, R3 pandas gradient boosting regression.
  Eleven distinct preprocessing registration IDs appeared in those cases.
  Their predictions also matched direct calls to the fitted estimators (exact
  class labels; regression tolerance `1e-12`). The core unit suite passed
  3,911 tests with 71 optional skips. Separate targeted train-only and
  inspection suites passed 317 tests.
- Another 30 inference-eligible registrations passed local fit/save/load/
  predict probes on both pandas and Polars (60 probe runs). Each probe used a
  bounded, node-specific fixture and a nonempty fitted candidate state. This
  verifies those configurations, not every parameter combination.
- The Delta reader limits returned row count before driver iteration, then
  measures serialized decoded rows and local-frame memory. Spark wire framing
  is opaque to this API; this is not an exact network-byte measurement.

## Isolated Databricks resources

- Profile: previously selected `skyulf` on
  `https://dbc-45604623-c18b.cloud.databricks.com`.
- Schema: `workspace.skyulf_sm24a_20260923`, ID
  `e756f7db-08eb-4bd0-a897-28a81396a550`.
- Workspace folder:
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923`.
- Uploaded wheel: `skyulf_core-0.9.0-py3-none-any.whl`, SHA-256
  `fefdb71f020ee80975d507c8b55730e53e8c82d5206e3bd6eaa8cd967e07223b`.
- Uploaded notebook: `local_sm24a_job` in the same folder.
- Training parent run: `127884897177470`, task run `613571036409051`,
  900-second timeout, zero retries; **SUCCESS** after 142 seconds.
- Source tables: `workspace.skyulf_sm24a_20260923.skyulf_sm24a_train_source`
  (1,000 rows, Delta version 0) and
  `workspace.skyulf_sm24a_20260923.skyulf_sm24a_score_source`
  (160 rows, two 80-row months, Delta version 0).
- Model versions: `skyulf_sm24a_r1`, `r2`, `c1`, `c2`, `r3` in the same schema,
  each at concrete version `1`. The full name/digest/run ID receipt is retained
  in `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/training_receipt.json`
  and locally at `.cache/sm24a-training-receipt.json`.

| Case | Engine | Model class | MLflow run ID | Pipeline SHA-256 |
| --- | --- | --- | --- | --- |
| R1 | pandas | LinearRegression | `c96af1092692454d9c32aba21694285b` | `1c1cc2f41e23f718d56d02f44245cb2121fff34a6b6af8eb569396fe4e81cfb7` |
| R2 | Polars | RandomForestRegressor | `61896c2ed4aa449faa2b474ce62dcbea` | `5854d1ec07026200381d28d68d12833b7db2c8e47590d9fe7b8678a8525e3da2` |
| C1 | pandas | LogisticRegression | `af5fe6ea6ef242c78b84cbd30a80bc41` | `ed0f38b8b7ccb8ae83f7d3358acb72fb036ea15093ca7e912894578b5e46dac5` |
| C2 | Polars | RandomForestClassifier | `25a0008472724cee8094009f14fafa6c` | `a78530cdcba46509f88aec4949fb998fea66cd1122c1bf2d1e4afb3e6467fc1c` |
| R3 | pandas | GradientBoostingRegressor | `19db554ff23147a0a5445587763c7453` | `53b68cf37ae56e85f4c538dfdb463aa035bfadde92d65987617b3ed67035fd94` |

- Observed training runtime: `skyulf-core 0.9.0`, `mlflow 3.16.1`,
  `pandas 2.2.3`, `polars 1.44.2`, `scikit-learn 1.6.1`.
- Separate scoring parent run: `217135029658639`, 900-second timeout,
  zero retries; task `204469025040289`; **SUCCESS**. The Jobs API reported
  56 seconds of execution duration and 135 seconds of setup duration. All five
  pinned version-1 models scored 80 January and 80 February rows from source
  version 0. Each model-month output matched its training-job reference by
  entity key; class labels matched exactly and probability rows summed to one.
- Independent replay/negative parent run: `242324305571657`, task
  `1045750740187935`; **SUCCESS**. Jobs API execution duration: 134 seconds;
  setup duration: 175 seconds. Repeating R1 January returned an identical
  prediction frame and source/model diagnostics. Reordered or missing inputs,
  changed source version, row/byte budget breaches, wrong engine, wrong
  artifact kind and Spark runtime were each rejected before scoring.
- Per-node audit parent run: `262581913302220`, task `154157693596997`;
  **SUCCESS**, 900-second timeout and zero retries. Jobs API execution duration:
  38 seconds; setup duration: 135 seconds. All remaining 30 inference-eligible
  registrations passed an isolated fit/save/load/predict probe on both pandas
  and Polars (60/60). The [62-ID matrix](07-sm24a-node-matrix.csv) now records
  41 inference-eligible IDs with live evidence, 11 train-only, two inspection,
  two optional-dependency and six unsupported IDs. A passing fixture does not
  claim universal parameter or source-schema compatibility.
- Registry negative/pinning parent run: `460312914130071`, task
  `420698524557946`; **SUCCESS**, 900-second timeout and zero retries. Jobs
  API execution duration: 75 seconds; setup duration: 215 seconds. An unknown
  model version returned `registry_model_missing`. A disposable alias on R1
  moved from version 1 to newly registered test version 2 after preparation;
  the prepared workflow still scored 80 January rows with concrete version 1
  and matched its reference. The alias was deleted in the job's `finally`
  block. R1 version 2 remains as a test-owned registry record.

The training job created exactly two test-owned Delta source tables, registered
five test UC model versions and returned reference predictions. No alias was
moved and no prediction table was written. The separate scoring job proved
cross-job prediction parity on two months; neither job rescored the other
month during a monthly request. Reference predictions came from the saved local
artifact before registration, not from an independent pre-package estimator.
The latter was checked separately in local tests.

## Scope and follow-up

The reader enforces a Spark-side `max_rows + 1` limit, then decoded
serialized-row and local-frame byte caps. The selected Spark iterator does
not expose actual wire bytes, and a single wide row may arrive before the
decoded-byte check. The [plan](06-sm24a-live-validation-plan.md) explicitly
accepts this boundary for small data; [SM-24d](OPEN_QUEUE.md) tracks a hard
transport budget if larger or wide-row sources require it. The test did not
publish predictions to a UC Delta table or deploy a Bundle. Those are SM-15L
and SM-20a respectively. Unsupported/optional nodes remain visible in the
matrix, and the probes do not certify every model/parameter combination.
## Subsequent audits

The [scaler/outlier audit](09-sm24a-scaler-outlier-audit.md) refined four
row-filtering outlier registrations from the original `train_only` category to
`conditional_inference`; the current matrix has 41 inference-eligible, four
conditional and seven train-only registrations. The
[held-out metrics audit](10-sm24a-heldout-metrics-report.md) later logged
200-row test metrics in the same MLflow runs as five new test model versions.
The counts above describe the original audit snapshot.
