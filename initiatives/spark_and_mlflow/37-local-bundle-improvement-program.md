# Local Bundle improvement program

Date: 2026-09-24. Target release: 0.9.0.
Status: SM-30 and SM-31 delivered; SM-32 next. Company-compatible tags and
SM-30 passed [clean live verification](41-company-tags-and-clean-live-validation.md).
See [SM-31 evidence](42-sm31-refactor-plan-and-evidence.md) for the extraction,
110 tests and successful deployed score notebook. Remaining tasks are planned.
Source review: [reference comparison](36-bundle-reference-comparison-and-readiness.md).
Execution authority: [OPEN_QUEUE.md](OPEN_QUEUE.md).

## Outcome and boundaries

Build a generic, maintainable pandas/Polars Databricks Bundle suitable for a
real small-data batch workload, with explicit model approval, predictable
prediction refresh, understandable configuration and production operations.
Reuse the existing Skyulf pipeline, evaluation, tuning, registry and publication
services. Do not copy the reference project's parallel implementations.

Keep the default two persistent jobs, train/lifecycle and score. An approval
action uses the same serialized lifecycle job as training; it does not create
a second alias writer or retrain a registered candidate. Keep admission tables
out of the default Bundle. Keep English documentation and examples.

The first local Bundle gate already passed for its documented scope. These
tasks extend that baseline; historical evidence must not be rewritten to claim
the new behavior was deployed. Company production readiness requires its own
identity, compute and real-data validation. Do not deploy to company workspaces
or activate schedules as a side effect of creating these tasks.

Broad Spark execution, HTTP serving, ai_query, A/B, feature lookup and full
monitoring stay explicitly tracked after the local Bundle work. Backend/Canvas
and continuous streaming remain parked. Existing task IDs are reused for
those extensions rather than creating duplicate backlogs.

## Decisions confirmed by the user

1. Challenger means the latest successfully trained and registered contender,
   not only a model that passed promotion gates. A tied or worse v3 remains
   challenger while v2 remains champion. A newer v4 replaces the challenger
   pointer; v3's evaluation history stays in its version/run records.
2. Challenger assignment and champion promotion are separate operations.
   Evaluation failure or insufficient quality must never move champion.
3. Scoring selection and approval are independent configuration choices:
   `score_model_selection = pinned_version | champion` and
   `promotion_policy = manual_approval | automatic`.
4. Manual approval of an existing candidate does not train or republish its
   artifact. It rechecks pinned evidence, records a controlled transition and
   can call the existing score job. It does not bypass quality checks.
5. Append keeps historical predictions; full rebuild publishes a complete
   generation for a different model. Both policies survive these changes.

The proposed field names above become public only through SM-32/SM-33.
Existing `model_selection_mode` configurations need an explicit migration,
not silent reinterpretation.

## Work packages and acceptance

### SM-30 — Challenger identity and visible evaluation status

Files: `skyulf-core/skyulf/integrations/mlflow/promotion.py`,
`skyulf-core/skyulf/integrations/databricks/local_retraining.py`,
`skyulf-core/templates/databricks/template/{{.project_name}}/src/workflow.py`,
`skyulf-core/tests/integrations/test_mlflow_promotion.py`,
`skyulf-core/tests/integrations/test_databricks_local_retraining.py`,
`tests/integration/test_sm20a_bundle_template.py`.

- Separate contender nomination from validated promotion. Preserve the
  read-only comparison API and do not hide alias mutation inside generic fit.
  The serialized lifecycle orchestrator owns nomination after registration.
- Publish `validation_status` (`pending`, `passed`, `rejected`, `error`) and
  `validation_reason` on the model version; preserve the complete comparison
  in the training run. A first candidate can pass its absolute gate even
  though a relative report says `no_champion`.
- Persist `promotion_status` separately; evaluation is not activation. Keep
  model-version status writes consistent with existing receipts and report
  incomplete/unknown registry outcomes without claiming successful promotion.
- Handle no champion, rejected candidates, replacement of an existing
  challenger, repeated nomination and failed evaluation. A stale version
  must not displace a newer challenger without an explicit expected version.
- Revisit rollback's current refusal when any challenger exists: a retained
  rejected challenger must not make recovery unusable. Preserve or explicitly
  reconcile that pointer under the same writer; never silently lose evidence.
- Test real local MLflow versions: v1 -> v2 promotion, tied v3 retained as
  challenger, v4 replacement, failed quality, evaluation error, stale writer,
  partial tag/alias writes, rollback with retained contender. Check scoring
  still uses champion and does not refit.
- Acceptance: the v3 UI explanation matches aliases plus durable status;
  promotion checks remain strict. Local completion and UC verification are
  recorded separately; SM-43a is the combined deployment gate.

### SM-31 — Thin template and reusable orchestration

Files: current template `src/workflow.py`; create
`skyulf-core/skyulf/integrations/databricks/local_workflow.py` for orchestration
and `prediction_output.py` for table/generation/view publication; retain
`local_retraining.py` as the training-window/data service. Add focused tests
under `skyulf-core/tests/integrations/`; retain generated-template tests.

- Move reusable behavior out of the 516-line generated notebook. The notebook
  reads widgets/config, calls the library and serializes its result.
- Preserve both engines, target binding, source bounds, aliases, row keys,
  generation provenance, grant-preserving view switches and no-op commits.
- Keep business pipeline configuration and documented customization points in
  the project. Avoid a new service layer per helper or duplicated model code.
- Acceptance: behavior-preserving tests for every existing action, installable
  wheel import test and generated-project test; no extra persistent job/table.

### SM-32 — Independent scoring selection and promotion policy

Detailed contract: [policy design](38-model-selection-and-approval-design.md).
Files: `local_workflow.py`, `promotion.py`, template config/schema/job resources
and `docs/user_guide/databricks_bundle.md`.

- Support all four policy combinations explicitly. Pinning a scoring version
  must remain effective even when registry promotion is automatic.
- Add approve/reject/rollback lifecycle actions for concrete versions, using
  the same serialized lifecycle job and authorized identity as training.
- Approval revalidates the saved dataset/version/split, model digest, metric
  policy and expected champion. Rejected or stale evidence cannot promote.
- Champion-following score resolves the controlled alias once per run.
  Approval can hand off to the existing score job without uploading model/code.
- Preserve score-only retry after a committed promotion. No implicit forced
  approval or raw UI-alias adoption; unexpected aliases need reconciliation.
- Acceptance: four-combination tests, approval without fit/log/register calls,
  rejected/stale/duplicate approvals, first champion, rollback and pinned-run
  stability. Live validation occurs in SM-43a.

### SM-33 — Validated configuration and runtime parameters

Files: create `skyulf-core/skyulf/integrations/databricks/workflow_config.py`;
template `databricks_template_schema.json`, config, job YAML and entry point;
generated-project and configuration tests.

- Separate environment variables, project/model configuration and per-run
  inputs. Expose a concrete scoring-version override and lifecycle action,
  candidate version and expected champion as validated run parameters.
- Add task-aware initialization for regression/classification, existing source
  keys (including composite keys), features/target and editable pipeline.
  Do not promise arbitrary FE can be inferred automatically.
- Validate metric/task/threshold compatibility, resource-name boundaries,
  stale example dates/source version, incompatible key/schema changes and
  model limits before expensive work or writes.
- Migrate `model_selection_mode` explicitly. Validate the generated job graph
  against policy configuration; editing JSON alone must not silently change
  the expected score handoff.
- Acceptance: strict generated Bundle validation, defaults/migration tests,
  actionable negative tests and a per-run version change without redeployment.

### Pre-SM-34 prerequisite: SM-33A through SM-33E

The user approved the [training data contract extension](54-training-data-contract-plan.md).
Complete the direct field rename, explicit date/timezone normalization, date-free
training with optional result availability, Core CV integration and combined
live acceptance before scheduling work. Existing SM-33 completion does not
imply that these additional behaviors are implemented.

### SM-34 — Independent schedules and explicit training windows

Files: template schema, job resources, target variables, workflow config and
`local_retraining.py`/`local_workflow.py`; schedule/window tests.

- Add independent optional train and score Quartz cron/timezone/pause inputs;
  start paused. Scoring new data must not require retraining or manual dates.
- Expose and document lookback, holdout duration and label-availability cutoff.
  Preserve the existing monthly UTC behavior as an explicit default. Changing
  cron must not silently imply a different data-split policy.
- Pin source snapshot and window per training run, including retry evidence.
- Test month/year boundaries, DST/timezones, score-only schedule, manual and
  automatic promotion, no-new-data runs and queued overlap with train handoff.
- Acceptance: predictable scheduled jobs and clear distinction between
  queueing, scheduling, data progress and model version.

### SM-35 — Multiple quality gates and understandable metrics

Files: MLflow `validation.py`, `promotion.py`, existing
`skyulf-core/skyulf/inference/local_evaluation.py`, workflow config and tests.

- Keep one selection metric and add optional additional absolute guardrails;
  reuse existing metric computation and direction definitions.
- Validate metric domains, finite values, class/probability availability and
  task compatibility. Keep probability decision thresholds distinct from
  model-quality thresholds. No generic default RMSE threshold for real data.
- Compare models on the identical held-out rows; keep minimum improvement
  explicitly absolute. Explain every failed gate in run/version evidence.
- Acceptance: regression/classification gates, conflicting secondary gate,
  tie, missing metric and first-champion tests; backward-compatible single gate.

### SM-36 — Existing Core tuning and model search in the Bundle

Files: `local_workflow.py`, `local_retraining.py`, existing
`skyulf-core/skyulf/pipeline/_pipeline.py` and `modeling/_tuning/` as consumers,
template configuration/examples and integration tests.

- Expose an optional finite model list/search space/trial/time budget using
  existing Core tuning, with train/validation/holdout separation and FE fitted
  inside tuning folds where required. Do not optimize on the final holdout.
- Log configuration, trials, selected candidate and final metrics to MLflow.
  Retain simple one-model training as the default.
- Add optional explainability artifacts only through existing supported
  model APIs, with explicit cost/sample limits and capability checks.
- Acceptance: bounded regression/classification searches, no holdout leakage,
  fitted preprocessing persistence and prediction parity for the selected model.

### SM-37 — Deployment identities and enforced writer ownership

Files: Bundle targets/job resources, configuration preflight, deployment guide
and identity/permission integration fixtures.

- Configure environment-specific `run_as`, job permissions, model ownership,
  source read and output write privileges, workspace roots and profiles.
- Keep lifecycle writes in one serialized job. Scoring identity executes the
  registered model without owning its aliases. Model/API permissions and job
  edit permissions must support the claimed exclusive-writer contract.
- Validate test/syst/prod host/catalog/schema mappings and development suffixes;
  keep company values out of generic defaults. Preserve selected dev usability.
- Acceptance: intended writer succeeds, scoring/restricted principal cannot
  change aliases, concurrent lifecycle requests queue, target isolation holds.
  Personal and company evidence are separate; config presence alone is not proof.

### SM-38 — Operations, retries and basic run visibility

Files: job resources/variables, workflow result logging, docs and tests.

- Expose job/task timeouts, bounded retries/backoff and failure notifications;
  retain concurrency one and queueing as correctness requirements.
- Surface selected model, source snapshot/progress, input/output counts,
  no-op reason, policy/gate decision and prediction generation without leaking
  raw records. Reuse SM-23a metric contracts instead of a competing system.
- Document retry after failed score, interrupted publication, pending alias
  receipt and permission failure. Do not silently retry uncertain promotion.
- Acceptance: failure injection and retry tests, no duplicate predictions or
  unintended retraining, sanitized diagnostics, notification configuration.

### SM-39 — Reproducible packaging and configurable compute

Files: template artifact/dependency and compute resources, initialization
schema, release/build manifests and packaging tests.

- Centralize compatible wheel/runtime/MLflow version inputs with pinned
  dependencies; retain model artifact dependency requirements.
- Make policy-cluster worker bounds, approved policy/runtime/node/cost settings
  configurable per target. Keep serverless dev validation available.
- Prefer a reproducible build/copy/validate path over manually editing wheel
  filenames across resources. Distinguish model replacement from library upgrades.
- Acceptance: clean-environment install/load/score, matching wheel release,
  serverless and policy target generation/strict validation. Live policy
  compatibility remains part of the company gate.

### SM-40 — Generated-project tests and generic CI/CD

Files: generated `tests/`, dependency manifest, CI workflow/example adapter
and template documentation; existing repository template tests.

- Generate project smoke/config/pipeline tests plus validate/build/test steps
  that run outside this repository. Pin a release artifact for deployments.
- Provide generic dev/test/syst/prod promotion with explicit environment
  approvals. Company managed-pipeline integration is an optional adapter,
  not a hardcoded private dependency.
- Do not deploy or run chargeable jobs on pull requests by default. Document
  scoped credentials, cleanup and how to add an authorized live test stage.
- Acceptance: a generated project passes its own CI locally and in the chosen
  CI environment; company deployment is tested separately.

### SM-41 — Source recovery and retained generation lifecycle

Files: `local_incremental.py`, `local_publish.py`, `prediction_output.py`,
workflow config/operator actions and recovery tests.

- Define explicit recovery for expired CDF history, source replacement,
  changed keys/schema and oversized initial/full rebuilds. Fail before writes
  when a supported recovery cannot be proven; do not silently reset progress.
- Add an explicit bounded full-refresh/recovery path with audit evidence;
  define its interaction with the same model version and rollback.
- Define optional update/delete semantics separately from the insert-only
  default; never present append mode as generic CDC support without tests.
- Add opt-in generation retention/cleanup preview. Never delete active or
  protected rollback generations; deletion requires the scoped operator action.
- Acceptance: no duplicates/lost source progress, safe interrupted recovery,
  preserved active view/grants, protected-generation tests and clear limits.

### SM-42 — User-facing examples and first-run documentation

Files: generated README/examples/config and `docs/user_guide/databricks_bundle.md`.

- Provide editable regression and classification examples for pandas/Polars;
  explain source keys, labels, artifact FE/model and UC object responsibilities.
- Walk through manual approval and automatic promotion, both output policies,
  score-only reruns, rollback, scheduling and runtime version selection.
- Document which settings need redeploy, which are per-run, and which are
  safety contracts. Explain RMSE units, quality versus probability thresholds,
  challenger versus approval, model version versus Delta version and no-op.
- Acceptance: a clean generated project can follow the guide without editing
  library code; commands and output expectations match tests and live evidence.

### SM-43a / SM-43b — Final acceptance, not inferred readiness

SM-43a: personal-workspace rehearsal after the local improvements. Exercise
both engines across representative regression/classification models, real
fitted FE, manual and automatic selection, retained failed challenger,
score-only new data, both output policies, retry/rollback and concurrency.
Use scoped resources and bounded compute; keep a resource/run inventory.

SM-43b: company target rehearsal after actual profiles/hosts, catalogs/schemas,
run identities, policies and representative data are confirmed. Verify UC
access, policy compute, dependencies, CI deployment and operational ownership.
Do not guess or hardcode missing company identifiers. An unavailable company
environment does not prevent completion of the local engineering tasks.

Acceptance for both: exact code/wheel/config identity, run IDs, metrics, rows,
aliases/statuses, grants, recovery results and limitations recorded. A written
plan, unit tests or personal serverless success cannot close the company gate.

## Existing extension tasks retained

| Existing task | Coverage retained from the comparison | Order |
| --- | --- | --- |
| SM-19a | HTTP endpoint deployment, pyfunc request/output parity and load/cold-start checks | After local Bundle acceptance |
| SM-19b | ai_query contract, named inputs, permissions and endpoint error handling | After compatible serving |
| SM-19d | A/B/canary traffic, endpoint updates, readiness and rollback; batch rollout only as a separate explicit option | After serving parity |
| SM-21a/b | UC feature lookup, point-in-time correctness, optional online publication/freshness | After local Bundle; online depends on serving |
| SM-23a/b | Batch quality/drift/delayed labels and endpoint monitoring/dashboard options | Basic operational summaries are SM-38; managed monitoring remains optional |
| SM-17/24c/20b | Existing node/model Spark coverage, Spark workflow and Bundle choice | After the local improvement gate |
| SM-24b/24d | Optional dynamic Jobs operations and exact transport budget | Only when current workflow/decoded-row bounds are insufficient |
| SM-18/19c | Backend/Canvas/legacy bridge and continuous streaming | Remain parked |

## Coverage and completion rules

The reference review's configuration, packaging, identity, CI, data recovery,
candidate visibility and workflow-size findings map to SM-30 through SM-43.
Its serving, online features and monitoring findings map to the existing
extension tasks above. No finding is treated as complete merely because a
related library primitive exists.

For each task: reproduce the relevant missing behavior, write focused tests,
implement through existing services, update English examples/docs and record
commands/results. Run Ruff/type checks and relevant integration suites; run
strict generated Bundle validation when configuration changes. Record local
implementation and live validation separately. Update OPEN_QUEUE and HANDOFF
after each verified slice. Do not mark the whole program done after SM-30.
