# Session handoff - 2026-09-24

SM-00 through SM-16, SM-15L/15I and SM-24a/25/26 are complete for their
documented scopes. The selected Databricks serverless local monthly UC
publication passed with 80 pandas January and 80 Polars February predictions.
Latest user direction: **Do not require manual period/source-version
values for recurring jobs. SM-15I now provides automatic new-row scoring
for the first small-data pandas/Polars Databricks Bundle (SM-20a). Spark handles table
I/O; Spark FE/model execution follows the Bundle. SM-18 and streaming remain
parked.**
The revised order completed SM-22a/b/c comparison and controlled promotion,
then SM-28a label-aware candidate training before SM-20a. SM-28b adds the
optional monthly schedule after the Bundle; SM-27 remains later. Target release: 0.9.0. Branch: `090`.

SM-22a has a locally verified, read-only comparison API and passed its
isolated Databricks metrics and UC comparison gate; see the
[local report](16-sm22a-local-validation-report.md) and
[live report](17-sm22a-live-metrics-report.md). SM-22b has explicit
version-checked promotion and rollback with shared admission. Its isolated
UC promotion, conflict, rollback and restricted-principal denial passed; see
[live evidence](18-sm22b-live-validation-report.md). SM-22c added challenger
and previous-champion aliases; see [its report](19-sm22c-lifecycle-alias-validation-report.md).
SM-28a now trains from a pinned label-aware Delta snapshot and compares a
registered candidate without promotion; see [its local and live report](20-sm28a-label-aware-retraining-report.md).
SM-20a now generates and deploys the first local-engine Bundle. Its own jobs
passed Polars training, 2+2 incremental scoring/no-op, read-only comparison,
challenger staging and explicit promotion; see [the report](21-sm20a-local-bundle-validation-report.md).
SM-20P now hardens the Bundle for a company environment before SM-28b. The
personal `dev` shape remains serverless; generated company `test/syst/prod`
targets use a looked-up job-cluster policy and `PayingRegNo`. Both templates
rendered, personal strict CLI validation passed, and company target shapes
passed strict CLI validation with a dummy policy ID in the personal workspace.
This is not a company deployment or policy execution check. Company profile,
host, catalog, schema, policy and cost-tag values still need confirmation;
one-time provisioning/preflight is open. See
[the production hardening plan](22-sm20p-production-bundle-plan.md).

## Starting point

- SM-00 through SM-16, SM-15L, SM-15I, SM-24a, SM-25, SM-26,
  SM-22a/b/c, SM-28a and SM-20a are complete. Continue SM-20P from
  [the open queue](OPEN_QUEUE.md), using the generated Bundle and the
  [pre-Bundle lifecycle plan](06-prebundle-model-lifecycle-plan.md).
  SM-15I [live evidence](13-sm15i-live-validation-report.md) proves automatic
  80+80 insert-only scoring with no date column or per-run version input.
  A later [real NYC taxi rehearsal](15-sm15i-real-nyctaxi-live-report.md)
  trained a Skyulf model with held-out MLflow metrics, registered UC model
  version 1, then passed 200 initial + 100 new-row predictions and a no-op replay.
  SM-26 added local artifact and MLflow packaging without cloud execution. The
  SM-25 SDK adds immutable local workflow configuration, local/remote preflight,
  explicit path or pinned registry artifact selection and caller-frame limits.
  It has no separate FE node or model-family allowlist: the fitted local
  pipeline's own prediction contract governs them. A representative sample can
  be probed before job submission. SM-24a added versioned UC reads, bounded
  local training and monthly pandas/Polars scoring. Its
  [live report](08-sm24a-live-validation-report.md) records two isolated UC
  source tables in `workspace.skyulf_sm24a_20260923`, five cross-job models,
  replay/negative checks and the 62-ID preprocessing matrix. Local-result
  Delta writes were subsequently proven in SM-15L; no Bundle was created.
  The current `max_bytes` limit measures decoded payload and local frame
  memory, not exact Spark wire bytes; SM-24d tracks a hard transport budget.
  previous Spark-first direction is superseded. Preserve the later
  [NODE_SUPPORT.md](NODE_SUPPORT.md) inventory and
  [gap review](reports/2026-09-22-spark-databricks-gap-review.md).
  The review enumerated all 100 source registration IDs and added model training,
  broader model inference, configuration usability and parked platform follow-ups.
  It did not implement those features. Missing node/model coverage must remain
  open unless implemented or explicitly deferred by the user. Broad SM-17
  follows the first local Bundle. The current Spark integration uses compatible bundles;
  pandas/Polars -> Spark is not support for arbitrary local FE or model artifacts.
- SM-26 separates fitted pandas/Polars pipeline packaging from native Spark
  porting. The versioned trusted-pickle artifact records fit engine, schemas,
  model class, runtime versions and a checksum; the MLflow pyfunc restores it
  without refitting. Its declared scope is whole-frame local prediction;
  row-local HTTP and Spark modes fail a scope check. SM-25, SM-24a, SM-15L
  and SM-20a follow this artifact contract.
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
- Local-engine pandas/Polars UC Delta writing passed SM-15L after SM-24a.
  The current Spark `run_batch` performs inference itself and cannot be used
  for local predictions unchanged. Prefer a bounded local-result -> Spark
  DataFrame bridge into guarded Delta publication; SQL Connector is optional.
  The later local-first Bundle request supersedes the earlier one-day deferral.
  See the [SM-15L report](11-sm15l-live-validation-report.md) for pinned
  models, monthly UC rows, replay, stale rejection and local negative tests. A delta-rs filesystem test
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

The first deliverable is:

```text
Databricks training job: pandas/Polars FE + sklearn model
    -> Save fitted FE, model and input contract through MLflow
    -> Databricks scoring job: pandas/Polars local batch on bounded data
    -> Spark reads bounded UC source changes and publishes UC Delta predictions
       (SM-15I automatic incremental path)
    -> Compare candidate/champion, promote explicitly, train challenger (SM-22a/b, SM-28a)
    -> Generate and run the first local-engine Bundle (SM-20a)
```

SM-16 validated an earlier compatible Polars-trained -> Spark regression path
on selected serverless compute. It does not validate the new local UC sink or
generated Bundle. Spark becomes a later optional Bundle choice in SM-20b.
Endpoint and online-lookup resources are separate optional work.
Keep new documentation and diagram labels in English.

## Current execution boundaries

- `predict_local_pipeline` consumes the new fitted local artifact, while
  `predict_local` consumes the portable standalone bundle. The frontend does not
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

SM-26 added `save_local_pipeline`/`load_local_pipeline`/`predict_local_pipeline`
and `log_local_model`. The pandas and Polars MLflow 3.16.1 lane proved
categorical encoding, null input, classification probabilities and tuned
decisions after load; an isolated subprocess also reloaded the logged model.
The changed 0.9.0 wheel was installed into the isolated environment and
`python -I` resolved the new module from site-packages.
Custom binning followed by encoding passed both local engines. The registry
adapter now resolves the local payload digest alongside the existing portable
bundle digest; no alias was promoted. This was local validation only. The
matching Skyulf wheel remains an explicit deployment dependency, and broader
model families still require parity evidence. See the MLflow model guide.

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
retained resources and finished runs. Start SM-26 without repeating these cloud
tests unless a new change requires them. Databricks CLI
1.17.0 is installed but absent from the current shell PATH; use its existing
WinGet executable or a refreshed shell. The user confirmed `databricks.yml` is
not needed in this repository; it is absent. DAB/templates and endpoints remain
later work. The user's subsequent explicit authorization covers the `skyulf`
profile and the named isolated serverless test resources.
