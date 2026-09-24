# Local-first Databricks integration and Bundle plan

Current sequencing update (2026-09-24): the first local Bundle has been
validated. The user now prioritizes [SM-30 through SM-43 improvements](37-local-bundle-improvement-program.md)
before broad Spark work. [OPEN_QUEUE.md](OPEN_QUEUE.md) is authoritative;
the earlier baseline and first-Bundle plan below are historical context.

Updated: 2026-09-23. SM-26, SM-25, SM-24a, SM-15L and SM-15I are implemented
and validated for their documented scopes; the first Bundle remains planned.
Baseline: `63dd3e21` and the completed, scoped SM-16 evidence.

## Current direction

The first deliverable is a small-data Databricks Bundle using pandas or Polars
for fitted FE and a Python model. Local means one Python process, which may run
inside a Databricks job; it does not mean the user's laptop. A bounded Spark
bridge may handle UC table I/O after local prediction; native Spark FE/model
execution remains a later task. Build and validate that project before extending
its prediction jobs with Spark. SM-15L handles explicit-period backfills;
SM-15I provides automatic new-row scoring required by the first Bundle.
SM-18 Backend/Canvas work and continuous streaming remain parked. Full SM-17
node/model expansion waits until the first local Bundle works.

SM-26 made broader existing fitted local pandas/Polars pipelines usable through
MLflow without waiting for native Spark node ports. SM-24a then validated the
first bounded local-engine Databricks training and scoring path; see its
[live report](08-sm24a-live-validation-report.md).

The first scenario is pandas/Polars FE and Python-model training, followed by
local Python batch prediction in a Databricks job and automatic incremental
publication to a UC Delta table. Existing Spark support is limited:
native portable FE covers mean/constant SimpleImputer, StandardScaler and an
empty chain; model bundles support the documented sklearn regression/classification
contract. The worker Python-pipeline mode is also capability-gated, not a generic
fallback for arbitrary local FE. SM-26 expands local-package coverage through
explicit per-pipeline/model tests; it does not expand Spark compatibility. External
model libraries, clustering and custom behavior need their own local coverage
checks. Backend legacy artifact bridging remains parked in SM-18.

Native Spark FE/model expansion stays in SM-17, after the local Bundle. The
first Bundle may reuse Spark solely to read/write UC tables, with local data
size limits and explicit schema conversion. Existing Spark inference remains
outside its first variant. Each integration gate uses a compatible fixture and rejects
unsupported configurations before submission. Existing local tests cover pandas
and Polars training; live SM-16 used a Polars-trained regression model. New local
UC publication and Bundle claims need their own cloud evidence.

## Boundaries and reuse

- Reuse existing `predict_local` and MLflow tracking/model/registry adapters for
  the first Bundle. SM-15L and SM-15I provide bounded local-result -> Spark
  DataFrame bridges with separate period-replacement and append contracts.
  Later Spark inference variants reuse `predict_spark` and `run_batch`.
- Keep platform, FE engine, model execution, sink and optional tracking/registry
  independent. Local execution can run inside a Databricks job.
- No implicit Spark-to-pandas collection, node substitution, refitting or alias
  promotion. Known artifact schema/order/digests should be derived once.
- Keep explicit period and `as_of` pins for backfills. Scheduled SM-15I jobs
  derive source versions and retry identity from committed Delta receipts;
  both writers use the verified shared admission.
- Config parsing and validation have no remote mutation. Resource provisioning,
  submission, endpoint updates, promotions and cleanup are explicit operations.
- Databricks optional dependencies belong in integration modules, not core nodes.
- No new Skyulf repository root `databricks.yml`; the generated Bundle project
  has its own required root `databricks.yml`. Templates call tested services
  rather than copy their business logic.
- The reference is a design input, not a certified dependency or code copied into
  Skyulf. See the [source comparison](reports/2026-09-22-spark-databricks-gap-review.md).

## Recommended sequence

| Order | Task | Depends on | State |
| --- | --- | --- | --- |
| 1 | SM-26: local pipeline MLflow packaging | SM-16; existing local persistence | DONE |
| 2 | SM-25: local-first SDK configuration and preflight | SM-26 | DONE |
| 3 | SM-24a: local training and bounded batch prediction | SM-25 | DONE |
| 4 | SM-15L: local prediction -> UC Delta monthly publication | SM-24a | DONE |
| 5 | SM-15I: automatic incremental local scoring | SM-15L | DONE |
| 6 | SM-22a/b: candidate/champion comparison and controlled promotion | Current evaluation/registry | ACTIVE |
| 7 | SM-28a: label-aware candidate retraining service | SM-22a/b; SM-24a | WAIT |
| 8 | SM-20a: first pandas/Polars Databricks Bundle/template | SM-24a; SM-15I; SM-22b; SM-28a | WAIT |
| After first Bundle | SM-28b: optional monthly schedule wiring | SM-20a; SM-28a | LATER |
| After first Bundle | SM-27: bounded full-history rescore and selectable Bundle mode | SM-20a; SM-15L | LATER |
| Later | SM-24b, SM-19, SM-21, SM-23: optional Jobs API, serving, feature lookup and monitoring | Relevant local adapters | LATER |
| After local Bundle | SM-24c and SM-17: Spark workflow then broader Spark coverage | SM-20a | LATER |
| After selected Spark gates | SM-20b: add tested Spark option to the Bundle | SM-24c; relevant SM-17 slices | LATER |

The first Bundle needs bounded local input, verified automatic new-row UC
output and the already-tested lifecycle services. Its promotion job is an
explicit operation, never an automatic result of evaluation. Online lookup,
HTTP, SQL, A/B, Canvas and Spark FE/model execution remain outside SM-20a. Spark may handle UC table I/O only.
The local-first follow-ups add full-history rescore and optional monthly
retraining before broad Spark inference expansion.

## SM-26 - Broader local pipeline MLflow packaging

Proposed areas: standalone pipeline persistence/prediction contracts,
`skyulf-core/skyulf/integrations/mlflow/`, integration tests and the MLflow guide.
Reuse existing fitted pipeline behavior; do not implement another FE engine.

- [x] SM-26a: audit existing standalone save/load/predict semantics, fitted FE,
  local engine selection, schema/order, labels/probabilities/thresholds and model
  dependencies. Define a versioned local artifact contract distinct from the
  existing portable bundle; retain backward compatibility and trusted loading.
- [x] SM-26b: package and restore the fitted local pipeline through MLflow pyfunc.
  Preserve learned state without refit. An MLflow pandas input boundary must not
  silently switch an engine-sensitive Polars pipeline; any conversion is explicit
  and preserves dtypes, nulls, column order and output semantics.
- [x] Include the required code/dependencies and a signature/input example.
  Use the current registry/version-pinning services; do not promote aliases.
- [x] SM-26c: prove original -> saved -> clean-environment MLflow predictions for
  pandas and Polars. Include representative FE beyond mean imputation/scaling,
  such as encoding/binning, and explicitly track each advertised model family.
- [x] Separate whole-frame local prediction eligibility, row-local HTTP eligibility
  and Spark-worker/native eligibility. Context-dependent or row-changing FE does
  not become safe for requests or partitions merely because it can be serialized.
- [x] Keep Python local batch prediction available without a Delta writer. No
  implicit Spark collection, new native Spark ports or backend artifact bridge.

Acceptance: new local-package paths reproduce the original fitted pipeline under
the declared contract, preserve the existing portable bundle, and reject unsupported
serving/Spark combinations explicitly. Local numerical/schema parity and dependency
isolation precede live endpoint claims. SM-17 and SM-18 retain their later scope;
SM-15L follows the local workflow.

## SM-25 - SDK configuration and preflight

Reference: `common/config_service.py.tmpl`, model factory and inference service.
Proposed implementation area: `skyulf-core/skyulf/integrations/databricks/`;
config/preflight/workflow modules, integration tests and an English SDK guide.
Names below describe responsibilities; settle public signatures in the task design.

- [x] Define immutable configuration for runtime, input source, FE/model execution,
  optional stores and output sink without embedding credentials.
- [x] Reuse bundle metadata to resolve feature order, model digest and output schema;
  resolve an alias once to a pinned version. Preserve explicit caller-owned inputs.
- [x] Provide a preflight result with node/config/model/runtime incompatibilities
  and actionable fixes. Distinguish local checks from optional read-only remote checks.
- [x] Document a short pandas/Polars fit -> local batch workflow that can run
  on Databricks job compute, plus an advanced custom-code entry point. Keep
  source size limits explicit and do not generate a project to use the SDK.
- [x] Select among verified local-package and portable-bundle contracts explicitly.
  Keep wider native engine/model choices in parked SM-17j; packaging a local
  pipeline is not a shortcut to claim broader Spark support.

Acceptance: valid local fixtures reach the selected local package and predictor;
unsupported FE/model/sink combinations fail before submission; config loading
performs no remote mutation; defaults never invent snapshot/period or promotion
decisions. Spark preflight expands only with later tested Spark options.

## SM-24a - Reusable training and batch workflows

Reference: inference/data/config services; reuse Skyulf's fitted local pipeline.
Delivered in `integrations/databricks/local_batch.py`, the
[local-engine guide](../../docs/user_guide/databricks_local_sdk.md), runnable
examples and integration tests. The [live validation report](08-sm24a-live-validation-report.md)
records separate UC training and inference tables, five cross-job model cases
and a per-node preprocessing audit.

- [x] Compose existing local pandas/Polars SkyulfPipeline fit, SM-26 local
  package export and optional tracking/registration as a thin training entry
  point. Accept explicit training/split inputs and a bounded local source.
- [x] Read a named UC table or other declared source through an explicit bounded
  local adapter. Filter the requested month at the pinned source version
  before collecting locally; enforce row/byte ceilings. A full two-month
  table download or an implicit unbounded Spark collect is not acceptable.
- [x] Expose pure Python batch prediction for verified SM-26 local packages;
  preserve required whole-frame context and input order. This task returns
  predictions; SM-15L handles publication.
- [x] Compose pinned registry load, package preflight and local prediction.
- [x] Return structured diagnostics with source, period and model identity;
  the separate job report records installed code versions. Do not claim the
  Spark runner's durable publication receipt.
- [x] Accept explicit period/backfill parameters and business timezone; retries
  reuse the same request rather than refreshing the target version silently.
- [x] Keep table/control provisioning an explicit setup step, separate from scoring.

Acceptance evidence: pandas/Polars training and local batch prediction preserve
the declared engine and cross-job model parity. Bounded reads reject excess
rows and decoded/local-frame bytes; Spark wire bytes and one-row peak transport
are not capped. Registration never promotes an alias implicitly; no table-write
success is claimed here. The stronger transport guarantee is SM-24d.

## SM-15L - Local monthly UC Delta publication

The earlier one-day deferral was superseded by the local-first Bundle
request. The verified adapter converts bounded local results to a Spark
DataFrame and uses guarded Delta period replacement. Databricks documents
[`spark.createDataFrame` from local data on serverless](https://docs.databricks.com/aws/en/compute/serverless/limitations)
and [atomic selective overwrite](https://docs.databricks.com/aws/en/delta/selective-overwrite).
The [Python SQL Connector](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector)
remains an alternative when a Spark bridge is unavailable; it is not required.

- [x] Choose and test a bounded UC source reader. After local pandas/Polars
  prediction, convert only the final result to a Spark DataFrame using an
  explicit target schema. Retain named columns, dtypes, nulls, row keys,
  labels and model-version metadata; avoid silent Polars-to-pandas semantics.
- [x] Publish exactly one requested period using a safe, verified transaction
  design. Test replay, empty period, stale model/source, concurrent writers,
  permission denial and failed staging; reject unsupported overwrite policies.
  A new month adds its predictions while prior-month rows and run metadata stay
  unchanged. Only an explicit backfill may rescore an earlier month.
- [x] Reuse the existing guarded `publish_replace_period` where its contract
  applies. Its current `BatchSpec` includes Spark inference modes, so define
  a truthful local publication request or extract shared publication fields;
  do not label local prediction as `native_features` or `python_pipeline`.
- [x] Require explicit `as_of`, source snapshot/version, period, target identity,
  logical run identity and maximum local row/byte limits. Table setup and
  cleanup are explicit, and credentials use supported providers. The publisher
  requires an explicit Spark session for UC I/O; the job or Bundle selects its
  compatible compute. No SQL warehouse is required for this tested path.
- [x] Record a durable publication receipt and compare rows by key after commit.
  A filesystem delta-rs test alone does not prove UC managed-table behavior.

Acceptance: pandas and Polars local predictions reach the monthly UC target
with tested replay/concurrency semantics on the chosen Databricks runtime.
Spark performs only bounded source/result I/O in this first path; it does not
apply FE or run the model. The output equals local gold predictions by row key,
and a second monthly run does not recompute the first month.
This gate passed in the isolated serverless jobs documented in the
[SM-15L live report](11-sm15l-live-validation-report.md). The first Bundle can
now use this path, subject to its own generated-project validation.

## SM-15I - Automatic incremental local scoring

The explicit-period SM-15L job and its live rehearsal supplied period/source
version values in code. They do not satisfy the user's no-manual-input
requirement. The separate
[SM-15I live report](13-sm15i-live-validation-report.md) validates the implementation:
first
bounded snapshot, later only newly inserted source rows, an append-safe Delta
writer, and a source watermark in the same committed output receipt. Do not
feed an insert-only change batch into `replaceWhere`, which would delete
earlier predictions in an overlapping event-time period. Reject update/delete
events until a separate policy is tested.

## SM-24b - Optional Databricks Jobs API operations

Reference: developer resource scope and cleanup patterns. The first Bundle
already defines/deploys/runs jobs; add this optional SDK adapter only if a
non-Bundle caller needs dynamic submit/status/cancel operations.

- [ ] Build reviewable pandas/Polars training and local batch job requests from
  validated config and installed wheel/version, without requiring Spark execution.
- [ ] Separate request construction from submit/status/cancel operations; use existing
  platform credential providers and keep tokens out of config/logs.
- [ ] Set bounded timeouts/retries, resource scope, run identity and optional schedule
  parameters. Never equate a scheduler retry with permission to overwrite new data.
- [ ] Record deployment/run receipts and owned-resource inventories; document orphan
  claim recovery and explicit cleanup without touching unrelated resources.

Acceptance: request generation is offline; scoped live tests validate submitted
parameters, runtime dependencies and identity. No notifications are sent by
merely configuring them. This task is not a gate for SM-20a.

## Model lifecycle before the first Bundle

SM-22a/b and SM-28a are prerequisites for SM-20a. They implement validation,
explicit promotion/rollback and label-aware candidate training as reusable
services. SM-28b wires an optional schedule after the Bundle exists. SM-19,
SM-21 and SM-23 remain later platform additions. Endpoint and `ai_query` work do not require Spark; feature-table
adapters must declare any Spark dependency rather than pulling it into the
first local path. The tasks below remain open with their own acceptance gates.

## SM-22a/b - Validation and promotion

Reference: `src/validation/services/model_validation.py.tmpl` and registry service.
Proposed area: platform-neutral validation reporting plus optional registry operations;
reuse Skyulf evaluation and explicit MLflow clients rather than duplicate metric code.

- [ ] SM-22a: evaluate pinned candidate/champion versions on the same documented
  dataset/split, with configured thresholds and original label/probability semantics.
- [ ] Produce a comparison report with model/data/code identities and reasons;
  evaluation alone must not mutate aliases or endpoints.
- [ ] SM-22b: separate explicit promotion/rollback operations, recording prior and
  requested versions. Require shared alias admission and restricted registry writers; re-read
  the expected prior version under admission before mutation. The MLflow/UC
  alias APIs do not expose compare-and-swap, so external bypass writers
  remain outside this guarantee.
- [ ] Test first-model behavior without automatic first-champion promotion.

Acceptance: deterministic report fixture, real registry alias/version tests,
permission failures and rollback identity checks. SM-17i's broader native-model
evaluation remains parked and does not block current compatible bundles.

## SM-27 - Full-history local rescore and selectable Bundle mode

Keep `incremental_append` as the first Bundle's scheduled default. Retain
SM-15L's `period_update` for explicit backfills. Add an explicit
`full_rebuild` choice after SM-20a; it reads one pinned, bounded source snapshot
and scores the requested historical scope with one pinned model version. Publish
to a new prediction generation or table, validate counts, keys, parity and
provenance, then activate it explicitly. Preserve the previous generation for
rollback and keep original decision-time forecasts auditable. An ordinary
scheduled run must never trigger a full rescore because an alias moved.

- [ ] Define generation identity, source/model/version receipt, active-view
  cutover and rollback; reject unbounded reads and overlapping writes.
- [ ] Require point-in-time feature/label semantics for historical forecasts;
  a current-state rescore must be labeled as such, not presented as an old
  decision-time prediction.
- [ ] Exercise both modes through the generated local-engine Bundle: two monthly
  runs leave month one unchanged; an explicit full rebuild creates a complete
  second generation; failed validation leaves the first active.

## SM-28a/b - Monthly retraining and challenger workflow

The reference project wires feature engineering, training, validation and
inference jobs, but its sample inference schedule is commented out. SM-28a
implements the label-aware training and challenger comparison before the
Bundle. SM-28b later wires an opt-in schedule. Both are independent of
`period_update` and `full_rebuild` scoring choices.

- [ ] Provide an opt-in monthly schedule with business timezone, label
  availability cutoff, pinned training snapshot and reproducible temporal
  holdout. Never train on data unavailable at the prediction cutoff.
- [ ] Register a candidate, compare it with a pinned champion on the same
  documented evaluation set using SM-22, and record thresholds and slice
  failures. Validation must not silently promote or change serving aliases.
- [ ] Require an explicit promotion step with prior-version receipt and rollback;
  a failed or unpromoted candidate leaves champion unchanged. A scoring job
  resolves its chosen alias once to a concrete version, then pins it for the
  whole run. Newly promoted models affect future periods unless SM-27 is
  requested separately.
- [ ] Generate and test the optional train/validate/promote/score dependency
  chain without conflating scheduler date, training cutoff and scoring period.

## SM-19a - Live HTTP serving

Reference: endpoint service and endpoint integration tests.
Proposed area: optional serving adapter, a query example and contract tests.

- [ ] Create/update an endpoint from an explicit pinned compatible MLflow model;
  observe build/readiness with deadlines and retain the last known configuration.
- [ ] Validate request/response schema, labels/probabilities/thresholds and model
  dependencies. Serving must not assume an available SparkSession.
- [ ] Test real prediction parity, malformed inputs, query permissions and timeouts.
- [ ] Treat history/window lookups as unsupported until an explicit context design
  exists; do not expand the parked node scope through the serving adapter.

Acceptance: local pyfunc isolation plus an authorized live endpoint probe. The
existing backend HTTP deployment is separate; no SM-18 Canvas bridge is required.

## SM-19b/d - SQL and serving operations

- [ ] SM-19b: invoke the existing endpoint through ai_query; named feature struct,
  output schema, query permission and row-level error handling with keyed parity.
- [ ] Keep request/model identity traceable; SQL endpoint calls do not inherit the
  Delta sink's idempotency automatically.
- [ ] SM-19d: explicit A/B routing, readiness, scaling/cold-start behavior, rollback,
  bounded retries, query logs and load tests. No automatic traffic changes.

Acceptance: HTTP and SQL use the same compatible model contract and gold fixture;
live evidence distinguishes endpoint infrastructure success from prediction parity.
SM-19c continuous streaming remains PARKED and optional.

## SM-21a/b - Optional feature lookup

Reference: feature-store service and training-set preparation.

- [ ] SM-21a: explicit UC feature-table access, primary/timestamp keys, point-in-time
  joins, lineage and packaged lookup metadata. Plain-column input stays supported.
- [ ] SM-21b: optional online publication and serving lookup with freshness,
  missing-key and permission contracts; no implicit online infrastructure creation.
- [ ] Define how lookup-aware model packaging composes with Skyulf's raw/features
  bundle boundary so preprocessing is applied exactly once.

Acceptance: leakage fixture and offline/online parity; dependency-free behavior
when disabled. Online availability is not a requirement for ordinary Spark batch.

## SM-23a/b - Monitoring and observability

Reference: data-quality/monitoring services and endpoint inference logging.

- [ ] SM-23a: batch run/quality/drift reporting using existing Skyulf metric contracts;
  optional Databricks monitor setup and refresh kept separate from prediction.
- [ ] SM-23b: endpoint inference-table integration, delayed labels and version-aware
  monitoring; payload retention/redaction and permissions are explicit.
- [ ] Test no-op behavior when disabled, metric/schema agreement and failure handling
  so monitoring failures do not masquerade as model training or prediction failures.

Acceptance: trustworthy data/model/run attribution and scoped platform evidence;
no requirement to move existing core metrics into Databricks-specific code.

## SM-20a - First local Databricks Bundle and project template

This is the first deployable milestone after SM-26, SM-25, SM-24a,
SM-15L, SM-15I, SM-22a/b and SM-28a.
Generate a project using tested local services, with a `databricks.yml` in the
generated project root, job resources, thin Python entry points, pinned package
dependencies, dev/prod targets and a short README. Do not add a YAML file to
the Skyulf repository root. The first verified cross-job artifact path uses
MLflow and a pinned UC model version. The user selects pandas or Polars, UC
source/target, schedule and limits; credentials remain in Databricks
authentication or secrets, never generated source. Tracking-off or registry-off
Bundle variants require a separately tested durable artifact handoff.

- [ ] Package separate local training, comparison, explicit promotion and
  automatic incremental batch jobs. Promotion remains opt-in and cannot
  follow evaluation automatically. Pin
  model selection in configuration; derive source versions from committed
  receipts. Scheduled runs require no manual period or source-version input.
  Serverless Python script/wheel tasks declare their required environment key
  and installed package dependencies in the generated job resources.
- [ ] Generate tests for config/imports, `databricks bundle validate`, a
  deployment probe, two consecutive local-engine monthly job runs and UC
  output parity/replay.
- [ ] Document how to edit custom FE and model code while preserving the
  artifact and prediction contract. The generated project calls SDK services;
  it does not copy their implementation.
- [ ] Leave endpoint, `ai_query`, online feature lookup, monitoring and Spark
  inference jobs absent unless their separate adapters have passed their gates.

Acceptance: a fresh generated project can be validated, deployed and run on
the chosen Databricks environment; its pandas/Polars model produces the
expected pinned incremental table rows. Generated-file checks alone are not runtime
evidence. A supported no-UC-output variant may be offered separately; it does
not satisfy the monthly table gate.
See [the dedicated SM-20 plan](05-sm20-bundle-plan.md) for the two-run incremental
source/output-table rehearsal and the later Spark inference variant.

## SM-24c / SM-17 / SM-20b - Spark enhancement after the first Bundle

Only after SM-20a passes, expose the already implemented compatible Spark
inference and Delta runner through a thin SM-24c workflow. Then resume the
SM-17 inventory and node/model-family gates in independently reviewed slices.
SM-20b adds a Spark execution choice to the generated Bundle only for tested
artifact/runtime combinations. Local training and local batch remain valid
choices; a model saved through SM-26 does not become a Spark artifact by default.
Every Spark variant needs keyed prediction parity, runtime dependency checks,
monthly Delta replay/concurrency evidence and a generated Bundle run probe.
