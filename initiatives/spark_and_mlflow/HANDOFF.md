# Session handoff - 2026-09-22

SM-11 is complete after the corrective validation gate. SM-12 and SM-13 are
now complete; resume with SM-14.
Target release: 0.9.0. Branch: `090`.

## Starting point

- SM-00 through SM-13 are complete; SM-14 is READY.
- SM-13 starts from `67315253`; its implementation, guide and evidence are in
  the delivery commit containing this handoff.
- Previous guide/queue commit: `80dc9ee9` (SM-11 closure and SM-12 handoff).
- Read [OPEN_QUEUE.md](OPEN_QUEUE.md), [ARCHITECTURE.md](ARCHITECTURE.md) and
  the SM-14 section of [03-mlflow-batch-delivery-plan.md](03-mlflow-batch-delivery-plan.md).
- User-facing guide: [How inference works](../../docs/user_guide/inference_flow.md).

## Confirmed terminology and intended workflow

`local` describes single-machine execution, not the user's personal computer.
The runtime environment and the execution engine are independent choices.
Local pandas/Polars FE and sklearn training may run inside Databricks; a later
Spark inference job may also run inside Databricks.

The user's intended workflow is:

```text
Databricks training job: pandas/Polars FE + sklearn model
    -> Save fitted FE, model and input contract
    -> Databricks scoring job: Spark FE + Python model on workers
    -> Write predictions to a Delta table, for example monthly
```

This is the intended deployment scenario, not a claim that the complete
Databricks integration has already been validated. Scheduling, Delta delivery,
MLflow/Unity Catalog, endpoints and templates retain their later queue stages.
Keep new documentation and diagram labels in English.

## Current execution boundaries

- `predict_local` consumes the new standalone bundle; the frontend does not
  call it yet. Frontend inference uses `POST /deployment/predict` and
  `DeploymentService`, which loads the existing artifact, applies FE, aligns
  model columns and predicts. The backend artifact bridge belongs to SM-18.
- `predict_spark(mode="native_features")` supports raw regression and
  classification bundles with supported native FE. Spark applies the saved
  rules; Python workers run the same fitted model on batches. There is no
  full-data driver collection.
- Portable FE currently supports SimpleImputer mean/constant, StandardScaler
  and an empty chain. Unsupported steps fail explicitly.
- Spark FE fit -> export -> restore -> Spark apply is available. This does
  not imply that end-to-end distributed model training is implemented.
- `mode="python_pipeline"` runs compatible fitted Python FE and model
  execution together inside workers, without refitting. It supports raw
  regression and classification, and rejects unsupported context-dependent,
  row-changing and non-portable FE operations.
- Classification output preserves manifest label types, class-ordered
  probability columns and saved threshold precedence in both Spark modes.

## Verification and workspace

Before parking, the English guide's example was checked with both pandas and
Polars training: local and Spark predictions matched `[400.0, 200.0]`. SM-10
passed its focused 14-test Spark lane; SM-11 classification and isolation
regressions passed in the corrective 63-test focused lane and the full Spark
gate passed 418 tests with 2 warnings. The standalone Spark
batch example completed in both modes with matching string labels and
probability columns.
The strict documentation build, four rendered diagrams and commit hooks passed.
A built 0.9.0 wheel imported the worker inference module in a separate process.
Synthetic local measurements recorded 10k/2 partitions at 3.929s and
50k/8 partitions at 8.089s with driver peak RSS 246.2/246.8 MB. These are local
evidence only; Databricks worker-wheel deployment remains a later gate. SM-12
also added optional MLflow tracking: the base environment keeps MLflow absent,
while an isolated MLflow 3.16.1 environment passed all 8 tracking tests for
explicit client-bound lifecycle, concurrency, caller-run preservation, config
digest/artifact logging, and warn-mode degradation.

SM-13 added MLflow pyfunc packaging around the immutable inference bundle.
The MLflow 3.16.1 lane passed 31 tests, including a separate Python `-I`
subprocess loading the final 0.9.0 wheel from site-packages. Consumer dependencies
were copied from the isolated MLflow environment; Skyulf was reinstalled from
the wheel. Base bundle/schema/integration tests passed 80 with 7 optional skips.
MLflow aligns named columns and safely casts compatible inputs before bundle
prediction; direct `predict_local` remains strict. Unsupported bundle integer
dtypes fail at packaging. Producer uv files and temporary paths are excluded.
Registry-to-G2 evidence is carried into SM-14; live UC/Databricks, scheduled
batch delivery, endpoints and templates remain later gates.

Pre-existing `.tmp-*` directories remain outside this change. Preserve them;
do not stage or delete them as part of the next task.
