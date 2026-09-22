# Session handoff - 2026-09-22

SM-00 through SM-15 are complete. Resume with SM-16 platform preparation;
live Databricks execution has not been validated.
Target release: 0.9.0. Branch: `090`.

## Starting point

- SM-00 through SM-15 are complete; SM-16 is READY for platform preparation.
- SM-15 baseline: `4a613cb5`. Its implementation and verification are in the
  delivery commit containing this handoff. No live Databricks validation yet.
- SM-13 starts from `67315253`; its implementation, guide and evidence are in
  the delivery commit containing this handoff.
- Previous guide/queue commit: `80dc9ee9` (SM-11 closure and SM-12 handoff).
- Read [OPEN_QUEUE.md](OPEN_QUEUE.md), [ARCHITECTURE.md](ARCHITECTURE.md) and
  the SM-16 section of [03-mlflow-batch-delivery-plan.md](03-mlflow-batch-delivery-plan.md).
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

SM-14 added explicit MLflow registry publication and resolution. The adapter
publishes only `runs:/...` artifacts, leaves alias promotion explicit, resolves
an alias once to a concrete `models:/name/version` URI, and returns the packaged
signature and bundle digest. MLflow 3.16.1 local validation passed 10 tests,
including separate tracking/registry stores, missing/access/dependency failures,
alias movement and Unity Catalog name validation. Live UC and Databricks
validation remain SM-16; the G2 Spark runner consumes this concrete artifact in
that later platform gate.

Pre-existing `.tmp-*` directories remain outside this change. Preserve them;
do not stage or delete them as part of the next task.

## SM-15 delivery and next platform boundary

The monthly runner now reads a pinned Delta snapshot, checks its availability
at `as_of`, runs either Spark inference mode and atomically replaces one period
in a precreated target. Required provenance includes the model digest and
installed code version. The source producer still owns historical feature joins;
the cutoff does not establish their point-in-time correctness.

Local Linux validation passed **43 tests** on Spark **4.0.3** / Delta **4.0.0**:
18 real Delta cases plus 25 contract/admission cases. Base regression passed
105 with 26 optional skips. See the queue for exact commands and package versions.
The English [batch guide](../../docs/user_guide/databricks_batch.md) includes the
schema, code example, diagram, retry policy and limits.

`LocalTableLock` uses cross-process OS locks and requires a common directory
on one host. Both public entry points reject it on distributed Spark masters.
SM-16 must implement or validate distributed publish admission; the provider
protocol is not proof that Databricks has a coordinator. Expiring leases cannot
protect this sink because it has no fencing-token mechanism. Every publisher
must use the same authority. Test actual Delta/UC permissions in that lane.

Keep the current source snapshot and original spec available for retries. A
receipt returns the old committed version even after a newer recomputation,
without writing again. New computation needs a new run ID and reviewed target
version. History and transaction retention must cover the allowed retry window.

Next, prepare the SM-16 wheel/job smoke and platform evidence record, including
the carried registry-to-Spark path, without creating a DAB template or endpoints.
Actual execution needs an explicitly selected runtime, permitted test namespace,
and authentication. The user retracted an accidentally pasted login command;
do not treat its host or editor-created `databricks.yml` as authorization to log
in, submit jobs or use that workspace. Those editor files remain outside SM-15.
