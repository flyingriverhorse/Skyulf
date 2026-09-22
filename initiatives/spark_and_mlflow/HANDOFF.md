# Session handoff - 2026-09-22

SM-00 through SM-16 are complete for their documented scopes. The selected
Databricks serverless regression workflow passed. SM-15L is READY and next.
Target release: 0.9.0. Branch: `090`.

## Starting point

- SM-00 through SM-16 are complete; local pandas/Polars Delta writing is next.
- The registry-bundle loader and first registry-to-Spark probe are implemented.
  Local validation passed 29 combined tests; the selected live platform gate is
  now complete. Tested 0.9.0 wheel hashes are recorded in the platform file.
- See [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md). The user authorized the
  supplied workspace and `skyulf` profile; OAuth authentication is verified.
  No classic clusters exist. The user explicitly approved the isolated resources
  and serverless probe. Run `245612415275039` failed on restricted
  `spark.sql.caseSensitive` access after the UC/local round-trip. The fix passed
  75 real Spark tests. The user approved the corrected rerun; parent run
  `447606109645160` completed SUCCESS. The Polars-trained registered bundle
  produced the expected gold predictions in both Spark modes on serverless
  Spark Connect 4.2.0 / Python 3.12.3 / MLflow 3.16.1.
- Shared `DeltaTableAdmission` is implemented. Its local real Delta gate covers
  forced acquisition races, lost acknowledgements and public batch replay.
  Live monthly admission and independent-job contention passed.
- Checkpoint `311547fc` contains that code and the serverless identifier fix.
  Restricted service-principal run `2921374308246` passed model allow/deny,
  worker imports and 10k/50k synthetic parity. A subsequent Delta/alias run
  completed worker-content and alias prechecks, then failed on serverless
  `REFRESH TABLE`. Narrow refresh/cache fixes passed a combined 48-test Delta
  gate; live retry `783094949884769` passed monthly/alias/worker-content checks.
  Winner `973709879452231` committed after contender `404334394214907` verified
  held-owner rejection. The contender then failed a test-only assertion about
  the permission error code; final restricted run `977447944071613` passed the
  denial, unchanged data/version and released ownership checks. That failed run
  remains documented, not relabeled. The aggregate report was generated from
  actual results and checked against the r3 wheel; all 225 current package files
  match that wheel. See the platform evidence file for retained resources.
- Local-engine pandas/Polars Delta writing is tracked as SM-15L. The current
  Spark writer does not implement it; templates remain SM-20. Start with explicit
  local source/target and memory/provenance contracts. A delta-rs filesystem test
  must not be presented as UC managed-table writer support; see the current
  platform document's external-client constraints.
- SM-15 baseline: `4a613cb5`. Its implementation and verification are in the
  delivery commit containing the original SM-15 handoff. Live evidence is SM-16.
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

SM-16 validated this regression workflow on the selected serverless environment.
It does not certify arbitrary FE nodes, classification or every runtime.
Scheduling configuration, endpoints and reusable templates remain later scope.
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
evidence only; the later SM-16 worker-wheel evidence is recorded above. SM-12
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
Registry-to-G2 evidence continued through SM-14 and the completed SM-16 live
gate. Scheduling configuration, endpoints and templates remain later work.

SM-14 added explicit MLflow registry publication and resolution. The adapter
publishes only `runs:/...` artifacts, leaves alias promotion explicit, resolves
an alias once to a concrete `models:/name/version` URI, and returns the packaged
signature and bundle digest. MLflow 3.16.1 local validation passed 10 tests,
including separate tracking/registry stores, missing/access/dependency failures,
alias movement and Unity Catalog name validation. SM-16 subsequently validated
the live UC-to-Spark workflow using the concrete registered bundle.

Pre-existing `.tmp-*` directories remain outside this change. Preserve them;
do not stage or delete them as part of the next task.

## SM-15 delivery and completed platform boundary

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
SM-16 added and validated shared Delta-table admission. Expiring leases cannot
protect this sink because it has no fencing-token mechanism. Every publisher
must use the same authority. Actual Delta/UC permission evidence is recorded
in the completed platform report.

Keep the current source snapshot and original spec available for retries. A
receipt returns the old committed version even after a newer recomputation,
without writing again. New computation needs a new run ID and reviewed target
version. History and transaction retention must cover the allowed retry window.

SM-16 is complete; see its evidence file for wheel checksums, approved namespace,
retained resources and finished runs. Start SM-15L without repeating these cloud
tests unless a new change requires them. Databricks CLI
1.17.0 is installed but absent from the current shell PATH; use its existing
WinGet executable or a refreshed shell. The user confirmed `databricks.yml` is
not needed in this repository; it is absent. DAB/templates and endpoints remain
later work. The user's subsequent explicit authorization covers the `skyulf`
profile and the named isolated serverless test resources.
