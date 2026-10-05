# SM-57 — Spark pyfunc inference implementation plan

> For agentic workers: use `executing-plans` or `subagent-driven-development` to
> implement the independently testable steps below. Follow current AGENTS.md
> test ownership and focused-test rules; do not repeat full suites.

**Goal:** New projects can choose distributed inference independently of local
training, and score through the actual `mlflow.pyfunc.spark_udf` API.

**Architecture:** Keep pandas training and the trusted fitted pipeline artifact.
Admit only verified partition-safe inference behavior, run named-input pyfunc
batches on Spark workers, and keep keyed predictions distributed through existing
Delta publication contracts. Existing local pandas/Polars routes remain the
default for older configurations.

**Tech stack:** Skyulf Core, pandas, MLflow pyfunc, Spark/Arrow, Databricks Bundles,
Unity Catalog and transactional Delta receipts.

**Spec:** User-agreed SM-57 criteria in [OPEN_QUEUE_updated.md](OPEN_QUEUE_updated.md)
and the [actual apply-body inventory](165-sm57-partition-safety-inventory.md).
Prepared against `42bf826f` on 2026-10-05.
Implementation record: [170](170-sm57-spark-pyfunc-delivery.md).
Implementation and live acceptance are complete in the initial admitted scope.
The delivery record lists runtime evidence and remaining SM-56/58 boundaries.

## Global constraints

- Ask once about expected inference input **per run** and suitable local capacity.
  Rows, bytes and worker resources inform the choice; no universal two-million-row
  cutoff. Do not add another monitoring-engine question: monitoring already uses Spark.
- Large-data mode uses pandas training/preprocessing and Spark pyfunc inference.
  Local mode retains the pandas/Polars choice. Persist training engine and inference
  mode separately, and reject contradictory explicit setup values.
- New projects only; no silent conversion of existing fitted Polars artifacts.
  Training row/byte budgets remain local and independent of distributed scoring.
- No whole prediction population may pass through `toPandas`, `toLocalIterator`
  or a driver Python list in Spark mode. Bounded scalar diagnostics are permitted.
- Reject unsupported preprocessing, custom scoring rules, temporal history and
  model-set composition before publication; a pickle/pyfunc wrapper is not proof.
- Pin the concrete model/version/digest once, preserve CDF/recovery and atomic
  receipt semantics, and retain the existing Spark monitoring child-job handoff.
- Do not silently fall back to local scoring after a distributed-route failure.
- SM-58 owns large-scale throughput/executor-memory benchmarks; ordinary UDF
  correctness and all three generated layouts still require real Databricks runs.

## 1. Partition-safety gate and first admitted nodes

**Files:** `skyulf-core/skyulf/core/capabilities.py`,
`skyulf-core/skyulf/inference/partition_safety.py` (new), selected preprocessing
registration modules, and `skyulf-core/tests/unit/test_partition_safety.py` (new).
Direct consumers: `inference/local_pipeline.py`, `inference/local_scoring.py`,
`preprocessing/pipeline.py`, `integrations/mlflow/local_model.py`.

**Interface:** A single preflight accepts a loaded fitted local artifact and returns
detached evidence of admitted steps, configurations, fitted engine and pipeline
digest, or raises a structured unsupported-execution error identifying the step.
Do not initialize Spark or execute callbacks while checking metadata.

- [x] Inspect actual apply bodies and distinguish portable codec support from
  trusted pickle transport. Record counterexamples instead of broad allowlists.
- [x] Write failing tests for unknown nodes, wrong fitted engine, config/state
  mismatch, row-changing/outlier behavior, carry history and custom callbacks.
- [x] Add explicit pandas-worker capability matching for execution kind/context/
  row effect. Reuse existing metadata validation; do not equate native Spark
  support with pandas-batch support. Normalize defaults from fitted semantics.
- [x] Begin with SimpleImputer mean/constant and StandardScaler, then admit further
  candidates only alongside whole/split and composition parity. This first set is
  a release boundary, not a permanent architectural allowlist.
- [x] Validate the **effective** inference chain: some train-only/row-filter steps
  are skipped under `preserve_rows`, while other outlier steps execute and can fail.
  Any admitted skip requires an explicit, tested skip contract.
- [x] Pin support to reviewed artifact/key versions and output schemas. Treat
  power-transform/replacement/binning counterexamples as unsupported until fixed
  or constrained by tests. Do not change local fallback semantics incidentally.
- [x] Run the new test file plus affected capability/registry and local-artifact
  consumers. Record exact files/commands and obtain independent review.

## 2. Named-input Spark UDF adapter

**Files:** `skyulf-core/skyulf/integrations/mlflow/spark_model.py` (new),
`local_model.py`, `_nullable_transport.py`, `registry.py`, model-set adapter as
needed; focused `skyulf-core/tests/integration/platforms/test_mlflow_spark_model.py` (new)
and `skyulf-core/tests/spark/test_pyfunc_inference.py` (new).

**Interface:** A verified, pinned registered pyfunc plus a Spark frame produces
a Spark frame of raw keys and all declared prediction outputs. This layer does
not write a table or choose a champion.

- [x] Test failure-before-UDF-creation for unsupported artifacts and missing inputs.
- [x] Call real `mlflow.pyfunc.spark_udf` with a named `struct` and an explicit
  output struct covering predictions, probabilities and admitted scoring outputs.
  Do not use a scalar double result that loses classification/extra columns.
- [x] Cast only declared nullable transport inputs to string before Arrow and
  restore their saved dtypes inside pyfunc. Keep record keys outside the model
  input unless the model-set signature explicitly includes them.
- [x] Preserve exact runtime dependencies and artifact/source hashes. Record an
  additive safety certificate tied to the payload; revalidate the loaded artifact.
  Do not relabel legacy whole-frame artifacts as safe without inspection.
- [x] Make worker environment choice explicit and validate it against the selected
  Databricks compute. Avoid assuming notebook-installed wheels are inherited by
  isolated UDF workers. Verify worker reload/reuse and bounded prediction calls (see serverless adaptation).
- [x] Run actual local-vs-UDF regression/classification parity with nulls and large integral inputs, multiple partitions and worker reuse. Compare
  portable `predict_spark` for its supported subset.

The current [MLflow Spark UDF API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)
documents named structs, output structs and environment options. Prebuilt runtime
environments are platform/runtime specific. Serverless UDF sandbox constraints
must be checked on the actual target; distributed row count does not remove the
per-worker model and batch memory requirement.

## 3. Distributed source-to-Delta route

**Files:** `integrations/databricks/spark_scoring.py` (new), `local_workflow.py`,
`local_incremental.py`, `model_set_batch.py`, `scoring_recovery.py`,
`workflow_config.py`, `model_set_project.py`; focused new
`tests/integration/platforms/test_spark_scoring.py` plus direct affected consumers.
All paths above are under `skyulf-core/skyulf/` except the test path.

**Interface:** Explicit inference-mode dispatch returns the existing batch result
and receipt shapes. Single/competition pin one winner; model-set pins one set and
validates every component and combined output contract.

- [x] Reuse source snapshot/CDF selection, bootstrap/no-op and table-identity checks.
  Validate null/duplicate keys with distributed checks; do not impose local row caps.
- [x] Keep source, UDF outputs, provenance and key joins distributed. Preserve
  exclusion counts/schema contracts and one outcome per requested key.
- [x] Reuse the receipt protocol and transactional write guards. Extract shared
  publication primitives only where necessary; do not duplicate a second lifecycle.
- [x] Cover new inputs, no-op replay, changed model, full rebuild, CDF loss/recovery,
  concurrent target changes, failed worker and atomic model-set failure behavior.
- [x] Verify Spark scoring still publishes the same monitoring handoff. Monitoring
  configuration, delayed-label schedule and policy semantics remain independent.

## 4. New-project setup and generated jobs

**Files:** `templates/databricks/schema/project.json`, schema topic files if needed,
`templates/databricks/build_schema.py`, generated `databricks_template_schema.json`,
`template/{{.project_name}}/config/workflow.json.tmpl`, generated README and job
runtime/environment definitions. Paths relative to `skyulf-core/`.

- [x] Add one early `inference_mode` choice, with user-facing local-capacity wording.
  Large mode fixes training engine to pandas; only local mode asks pandas/Polars.
  Keep existing config `engine` as training engine for compatibility; persist the
  independent inference setting and default older configs to local explicitly.
- [x] Validate noninteractive init contradictions rather than silently ignoring
  `engine=polars` with Spark mode. Regenerate the schema from topic files.
- [x] Generate single, competition and model-set layouts for both routes with real
  CLI init and strict validation. Keep training budgets separate from UDF batches.
- [x] Update operator docs with chosen mode, supported-node errors, environment
  packaging, training limits and exact graph node/output expectations.

## 5. Live acceptance and completion gate

- [x] Deploy isolated generated single, competition and model-set examples using
  the already selected `skyulf` profile. Inventory ownership and capture job IDs.
- [x] Run each train-to-score flow with a genuinely distributed source/UDF plan,
  verify task outcomes and persisted keyed predictions against independent local
  predictions, then run the existing monitoring child job/native dashboard refresh.
- [x] Exercise at least one unsupported-node rejection before source materialization
  and one no-op replay without duplicated Delta output.
- [x] Record model/environment versions, Arrow settings, partitions, worker evidence,
  row counts, receipts and limitations. Do not substitute fake UDF mocks or a
  successful Bundle validation for live inference acceptance.
- [x] Run deduplicated affected tests and full CI static scopes, review, pre-commit
  and DCO commit. Update queue status only to the evidence actually achieved.

SM-57 implementation, local checks, real UDF parity, generated lifecycle and
no-op acceptance passed. This plan is closed with the local DCO commit and the
evidence and support boundaries in delivery 170.

## Live adaptation — serverless batch control

The first real serverless run rejected Arrow configuration reads. Use worker-side
`spark_udf_prediction_batch_rows` to bound each model call instead. Serverless
manages incoming Arrow allocation independently; no transport memory bound is
claimed. This supersedes the plan's references to a configurable Arrow batch
limit for the generated serverless route. Keep distributed partitions and test
prediction slices independently; do not fall back to driver scoring.
