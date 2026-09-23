# Local-first Databricks integration and Bundle plan

Updated: 2026-09-22. Planning only; no implementation or new cloud execution.
Baseline: `63dd3e21` and the completed, scoped SM-16 evidence.

## Current direction

The first deliverable is a small-data Databricks Bundle using pandas or Polars
for fitted FE and a Python model. Local means one Python process, which may run
inside a Databricks job; it does not mean the user's laptop. A bounded Spark
bridge may handle UC table I/O after local prediction; native Spark FE/model
execution remains a later task. Build and validate that project before extending
its prediction jobs with Spark. The latest direction reopens
SM-15L as a prerequisite for monthly UC table output in the first Bundle.
SM-18 Backend/Canvas work and continuous streaming remain parked. Full SM-17
node/model expansion waits until the first local Bundle works.

Latest refinement: **SM-26 comes first**, making broader existing fitted local
pandas/Polars pipelines usable through MLflow without waiting for native Spark
node ports. This is planned packaging work, not existing support.

The first scenario is pandas/Polars FE and Python-model training, followed by
local Python batch prediction in a Databricks job and monthly publication to a
UC Delta table. Existing Spark support is limited:
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
  the first Bundle. Evaluate a bounded local-result -> Spark DataFrame bridge
  to reuse guarded Delta publication, with a distinct local batch contract.
  Later Spark inference variants reuse `predict_spark` and `run_batch`.
- Keep platform, FE engine, model execution, sink and optional tracking/registry
  independent. Local execution can run inside a Databricks job.
- No implicit Spark-to-pandas collection, node substitution, refitting or alias
  promotion. Known artifact schema/order/digests should be derived once.
- Keep `as_of`, period, source snapshot, target version and logical retry identity
  explicit. The local publication path needs its own verified retry/transaction
  guard before the template advertises monthly UC output.
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
| 1 | SM-26: local pipeline MLflow packaging | SM-16; existing local persistence | READY |
| 2 | SM-25: local-first SDK configuration and preflight | SM-26 | WAIT |
| 3 | SM-24a: local training and bounded batch prediction | SM-25 | WAIT |
| 4 | SM-15L: local prediction -> UC Delta monthly publication | SM-24a | WAIT |
| 5 | SM-20a: first pandas/Polars Databricks Bundle/template | SM-24a; verified SM-15L | WAIT |
| After first Bundle | SM-27: bounded full-history rescore and selectable Bundle mode | SM-20a; SM-15L | LATER |
| After first Bundle | SM-22 then SM-28: validation/promotion and optional monthly retraining | SM-20a; SM-22 for SM-28 | LATER |
| Later | SM-24b, SM-19, SM-21, SM-23: optional Jobs API, serving, feature lookup and monitoring | Relevant local adapters | LATER |
| After local Bundle | SM-24c and SM-17: Spark workflow then broader Spark coverage | SM-20a | LATER |
| After selected Spark gates | SM-20b: add tested Spark option to the Bundle | SM-24c; relevant SM-17 slices | LATER |

The first Bundle needs a working, bounded local input and a verified monthly
UC output; it does not need model promotion, online lookup, HTTP, SQL, A/B,
Canvas, or Spark FE/model execution. Spark may handle UC table I/O only.
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

- [ ] Define immutable configuration for runtime, input source, FE/model execution,
  optional stores and output sink without embedding credentials.
- [ ] Reuse bundle metadata to resolve feature order, model digest and output schema;
  resolve an alias once to a pinned version. Preserve explicit caller-owned inputs.
- [ ] Provide a preflight result with node/config/model/runtime incompatibilities
  and actionable fixes. Distinguish local checks from optional read-only remote checks.
- [ ] Document a short pandas/Polars fit -> local batch workflow that can run
  on Databricks job compute, plus an advanced custom-code entry point. Keep
  source size limits explicit and do not generate a project to use the SDK.
- [ ] Select among verified local-package and portable-bundle contracts explicitly.
  Keep wider native engine/model choices in parked SM-17j; packaging a local
  pipeline is not a shortcut to claim broader Spark support.

Acceptance: valid local fixtures reach the selected local package and predictor;
unsupported FE/model/sink combinations fail before submission; config loading
performs no remote mutation; defaults never invent snapshot/period or promotion
decisions. Spark preflight expands only with later tested Spark options.

## SM-24a - Reusable training and batch workflows

Reference: inference/data/config services; reuse Skyulf's fitted local pipeline.
Proposed files: a workflow module in the Databricks integration, a thin executable
example, integration tests and `docs/user_guide/databricks_batch.md` updates.

- [ ] Compose existing local pandas/Polars SkyulfPipeline fit, SM-26 local
  package export and optional tracking/registration as a thin training entry
  point. Accept explicit training/split inputs and a bounded local source.
- [ ] Read a named UC table or other declared source through an explicit bounded
  local adapter. Filter the requested month at the pinned source version
  before collecting locally; enforce row/byte ceilings. A full two-month
  table download or an implicit unbounded Spark collect is not acceptable.
- [ ] Expose pure Python batch prediction for verified SM-26 local packages;
  preserve required whole-frame context and input order. This task returns
  predictions; SM-15L handles publication.
- [ ] Compose pinned registry load, package preflight and local prediction.
- [ ] Return structured diagnostics with source, period, model and code identity;
  do not claim the Spark runner's durable publication receipt.
- [ ] Accept explicit period/backfill parameters and business timezone; retries
  reuse the same request rather than refreshing the target version silently.
- [ ] Keep table/control provisioning an explicit setup step, separate from scoring.

Acceptance: pandas/Polars training and local batch prediction preserve the
declared engine and clean-environment model parity, including tracking-off
behavior. Bounded reads fail before exhausting process memory. Registration
never promotes an alias implicitly; no table-write success is claimed here.

## SM-15L - Local monthly UC Delta publication

The earlier one-day deferral is superseded by the local-first Bundle request.
Proposed area: a local prediction -> UC output adapter and integration tests.
Prefer an explicit local-result -> Spark DataFrame bridge followed by guarded
Delta period replacement, subject to live validation. Databricks documents
[`spark.createDataFrame` from local data on serverless](https://docs.databricks.com/aws/en/compute/serverless/limitations)
and [atomic selective overwrite](https://docs.databricks.com/aws/en/delta/selective-overwrite).
The [Python SQL Connector](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector)
remains an alternative when a Spark bridge is unavailable; it is not required.

- [ ] Choose and test a bounded UC source reader. After local pandas/Polars
  prediction, convert only the final result to a Spark DataFrame using an
  explicit target schema. Retain named columns, dtypes, nulls, row keys,
  labels and model-version metadata; avoid silent Polars-to-pandas semantics.
- [ ] Publish exactly one requested period using a safe, verified transaction
  design. Test replay, empty period, stale model/source, concurrent writers,
  permission denial and failed staging; reject unsupported overwrite policies.
  A new month adds its predictions while prior-month rows and run metadata stay
  unchanged. Only an explicit backfill may rescore an earlier month.
- [ ] Reuse the existing guarded `publish_replace_period` where its contract
  applies. Its current `BatchSpec` includes Spark inference modes, so define
  a truthful local publication request or extract shared publication fields;
  do not label local prediction as `native_features` or `python_pipeline`.
- [ ] Require explicit `as_of`, source snapshot/version, period, target identity,
  logical run identity and maximum local row/byte limits. Table setup and
  cleanup are explicit, and credentials use supported providers. State the
  selected Spark compute or SQL warehouse requirement in config and preflight.
- [ ] Record a durable publication receipt and compare rows by key after commit.
  A filesystem delta-rs test alone does not prove UC managed-table behavior.

Acceptance: pandas and Polars local predictions reach the monthly UC target
with tested replay/concurrency semantics on the chosen Databricks runtime.
Spark performs only bounded source/result I/O in this first path; it does not
apply FE or run the model. The output equals local gold predictions by row key,
and a second monthly run does not recompute the first month.
The first Bundle cannot advertise a monthly UC output until this gate passes.

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

## Optional local/platform additions after the first Bundle

SM-22, SM-19, SM-21 and SM-23 can extend a generated project after SM-20a.
They are not prerequisites for the first local training and monthly batch
Bundle. Endpoint and `ai_query` work do not require Spark; feature-table
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
  requested versions. Detect conflicting state instead of overwriting silently.
- [ ] Test first-model behavior without automatic first-champion promotion.

Acceptance: deterministic report fixture, real registry alias/version tests,
permission failures and rollback identity checks. SM-17i's broader native-model
evaluation remains parked and does not block current compatible bundles.

## SM-27 - Full-history local rescore and selectable Bundle mode

Keep `period_update` as the first Bundle's default: score only the requested
closed month and replace that period idempotently. Add an explicit
`full_rebuild` choice after SM-20a; it reads one pinned, bounded source snapshot
and scores the requested historical scope with one pinned model version. Publish
to a new prediction generation or table, validate counts, keys, parity and
provenance, then activate it explicitly. Preserve the previous generation for
rollback and keep original decision-time forecasts auditable. An ordinary
monthly run must never trigger a full rescore because an alias moved.

- [ ] Define generation identity, source/model/version receipt, active-view
  cutover and rollback; reject unbounded reads and overlapping writes.
- [ ] Require point-in-time feature/label semantics for historical forecasts;
  a current-state rescore must be labeled as such, not presented as an old
  decision-time prediction.
- [ ] Exercise both modes through the generated local-engine Bundle: two monthly
  runs leave month one unchanged; an explicit full rebuild creates a complete
  second generation; failed validation leaves the first active.

## SM-28 - Monthly retraining and challenger workflow

The reference project wires feature engineering, training, validation and
inference jobs, but its sample inference schedule is commented out. Monthly
retraining is therefore an explicit Skyulf follow-up, not inherited behavior.
It is independent of `period_update` and `full_rebuild` scoring choices.

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

This is the first deployable milestone after SM-26, SM-25, SM-24a and SM-15L.
Generate a project using tested local services, with a `databricks.yml` in the
generated project root, job resources, thin Python entry points, pinned package
dependencies, dev/prod targets and a short README. Do not add a YAML file to
the Skyulf repository root. The first verified cross-job artifact path uses
MLflow and a pinned UC model version. The user selects pandas or Polars, UC
source/target, schedule and limits; credentials remain in Databricks
authentication or secrets, never generated source. Tracking-off or registry-off
Bundle variants require a separately tested durable artifact handoff.

- [ ] Generate local training and monthly local batch jobs. Pass data period,
  timezone, source snapshot and model version as explicit job parameters.
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
expected pinned monthly table rows. Generated-file checks alone are not runtime
evidence. A supported no-UC-output variant may be offered separately; it does
not satisfy the monthly table gate.
See [the dedicated SM-20 plan](05-sm20-bundle-plan.md) for the two-month
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
