# Local Bundle comparison and readiness review

Date: 2026-09-24. Skyulf baseline: `7c3a0866`.
Reference: `C:/Users/Murat/Downloads/codes-main (2)/codes-main/dbml-mlops-template`.

Follow-up: the user approved the improvements and corrected the desired
challenger meaning. The current code observations below remain historical;
the [approved program](37-local-bundle-improvement-program.md) and
[selection/approval design](38-model-selection-and-approval-design.md) define
the next behavior. In particular, a rejected contender will retain challenger
alias; challenger assignment will no longer require promotion eligibility.

This is a static comparison of the initialization schema, generated project
configuration, job resources, lifecycle services, inference, deployment,
monitoring and CI/CD entry points. The reference was not deployed or executed.
Its classification and example projects were inventoried; the regression
skeleton was the primary implementation inspected. This is not a claim that
every file or optional scenario in the reference has been tested.

Skyulf's prior live evidence is in
[the SM-27/SM-29 report](35-sm27-sm29-live-validation-report.md).
No cloud resources were changed during this review.

## Readiness decision

The existing local-engine Bundle is a working foundation for a supervised
small-data batch pilot. It is not yet a turnkey company production package.
Personal serverless validation does not establish company policy-compute,
service-principal, data-access or deployment-approval readiness.

Production prerequisites for this scope are real source/label/key contracts,
business-defined quality gates, supported data volumes, CDF availability,
explicit job identities and grants, a single serialized alias writer, score
scheduling and operational recovery. Serving, feature-store integration and
Spark-native FE are optional extensions, not prerequisites for this batch use case.

## What v3 and manual selection mean today

- v3 is registered and evaluated, but its improvement over v2 is zero. The
  automatic workflow returns before `stage_challenger`, so v3 has no alias.
- In this implementation, `challenger` means a validated candidate eligible
  to replace champion, not simply the most recently trained version. A passing
  automatic candidate is staged then promoted immediately, so the challenger
  alias is transient. A version's durable result is in its training run and
  `candidate_comparison.json`; readable status tags on rejected UC versions
  are a usability gap.
- `pinned_version` scores the concrete `model_version` in deployed JSON.
  Training does not change that value or automatically invoke score. Edit the
  version, deploy the configuration and run the existing score job. The model
  artifact is loaded from MLflow; it is not rebuilt or uploaded again.
- `auto_champion` resolves a controlled champion once at the beginning of
  each score run and pins that version. A later approved promotion is picked
  up by the next run without redeploying score code. A running score keeps
  its pinned version. The input/output contract must remain compatible.
- Direct Catalog UI alias edits bypass Skyulf's receipt records. They can
  cause automatic scoring to stop with an alias/receipt conflict; this is
  stricter than plain MLflow alias loading.
- Manual stage/promote/rollback APIs exist in the library, but a convenient
  manual-approval Bundle action is not implemented. Staging and promotion
  still require valid comparison evidence; the current APIs do not turn a
  rejected tie into an approved champion by operator preference alone.

Recommended extension, not current functionality: separate model selection
from promotion policy. Keep pinned scoring and controlled champion scoring;
allow promotion to be automatic or explicitly approved. A manual action should
reference an existing version, recheck evidence, write the same receipt and
invoke the existing serialized score job without retraining. It must share
the alias-writer serialization path rather than introduce a competing writer.
Any deliberate quality-gate override needs its own explicit policy and audit.

## Quality gates

`quality_threshold: 5.0` with `metric: heldout_rmse` means RMSE must be at most
5 target units on the held-out data. It is an example, not a percentage or a
classification probability threshold. With `heldout_accuracy`, a threshold
such as `0.90` means accuracy must be at least 90 percent instead.

`min_improvement: 0.1` means an absolute metric difference of at least 0.1;
ties never pass even when the minimum is zero. If champion RMSE is 4.0,
candidate RMSE 3.8 passes both example gates; 3.95 fails the improvement gate.
The absolute gate applies to subsequent champions as well as initialization.
Many metrics are logged, but only one selected metric controls this decision.

## Comparison

"Present in reference" below means source/configuration was found, not that
its runtime behavior was validated in this review.

| Capability | Reference template | Current Skyulf Bundle | Assessment |
| --- | --- | --- | --- |
| Project initialization | Classification/regression skeletons, examples, many infrastructure prompts | Generic local project; pandas/Polars, engine/compute/selection/scoring choices | Skyulf needs friendlier task, columns and pipeline setup |
| Core FE and model reuse | Project-specific feature notebooks and training services | Existing `SkyulfPipeline`, saved FE/model, local artifact and MLflow wrapper | Skyulf avoids reimplementing node/model behavior in each project |
| Small-data local execution | Local Python model training plus Spark-oriented inference | Bounded pandas/Polars fit/predict; Spark handles UC I/O | Selected Polars lifecycle live; both-engine local parity; not all nodes/models certified |
| Distributed batch inference | `spark_udf` and optional feature-store `score_batch` paths | Library Spark inference exists; first Bundle exposes local execution only | Intentional deferred Bundle extension |
| Metrics and model validation | MLflow evaluation, extra metrics, multiple thresholds and SHAP configuration | Core held-out metrics, one decision metric, absolute gate and minimum improvement | Multi-metric approval and explainability are not first-class Bundle options |
| Model search | Optuna, model list, search spaces and trial settings | Core tuning exists; template presents one `pipeline.modeling` configuration | Add an explicit tested tuning/model-search configuration, not duplicate trainers |
| Candidate visibility | Training sets `candidate`; validation stages `challenger` or bootstraps champion | Registered candidate plus comparison artifact; only eligible candidates get `challenger` | UC version status/rejection tags would improve Skyulf inspection |
| Promotion and rollback | Alias assignment and rolling-promotion services | Checked transitions, previous champion, committed receipts and rollback API | Skyulf has stronger explicit recovery checks; manual Bundle approval UX is missing |
| Prediction refresh | Date/backfill controls and optional champion/challenger rollout | CDF insert-only append or new complete model generation behind a stable view | Both Skyulf policies live-tested; updates/deletes and CDF-expiry recovery not covered |
| Retry/no new data | Project publication helpers | Commit-bound progress, duplicate checks, no-op, serialized score job | Skyulf retry/no-op and queue evidence is explicit |
| Resource footprint | Main workflow delegates to feature/train/validation/inference jobs; other optional jobs | Two jobs, no admission tables; full mode retains versioned tables plus a view | Keep the smaller default; retained generations need an eventual cleanup policy |
| Runtime parameters | Job parameters for dates, candidate version, smoke mode and model choices | Widgets read action/config/target; model version and gates come from deployed JSON | Missing convenient per-run model selection/approval parameters |
| Scheduling | Inference schedule example is commented; broader workflow resources | Optional paused train cron/timezone; score follows automatic training or runs explicitly | Add independent optional score cadence; new data alone does not launch a job |
| Multi-environment deployment | ci/test/syst/prod catalogs and developer suffixes | dev/test/syst/prod targets and output/model suffixes | Company hosts/grants/policies still require configuration and live validation |
| Compute and dependencies | Job cluster policies, cost tags, runtimes, requirement/lock files | Serverless or policy cluster, explicit wheel and MLflow pin | Version pinning is useful; packaging inputs, worker bounds and per-target compute need improvement |
| CI/CD and project tests | Generated GitHub workflows call company-managed pipelines; unit/integration suites | Repository tests and live proofs; generated project has no CI workflow or test scaffold | Production adoption gap; make generic CI with optional company adapters |
| Identities and operations | Resource permissions and company deployment integration | No generated run_as, notification, timeout or retry settings | Define target-specific identities/grants and operator settings; reference integration does not prove company readiness either |
| Serving and A/B rollout | Endpoint services, traffic configuration, rolling inference and load-test resources | No endpoint deployment/traffic controls in first Bundle | Deferred optional capability; batch-only production does not need it |
| Online feature lookup | Optional Feature Store/Lakebase path | Not in first Bundle | Deferred optional capability |
| Monitoring | Data-quality/model-performance jobs and dashboard resources | Core metrics/profiling exist; no deployed monitoring workflow | Start with failure alerts and scoring summaries; dashboards can follow |

The reference is not free of configuration issues: the inspected regression
inference job repeats `feature_store_model_enabled` in its parameter list,
and test/syst/prod all use the same `input_test_workspace_host` template
value. It also contains explicit dataset/model placeholders. Copying it
unchanged would not supply the user's three company workspace hosts or a
validated business model. No production claim follows from file presence.

## Configuration audit

| Setting | Current surface | Recommendation |
| --- | --- | --- |
| Engine, key, metric, selection mode, model-change mode | Initialization prompts and generated JSON | Keep editable, add task-aware validation |
| Source/output/model names, features, label columns, preprocessing, model parameters | Generated `config/workflow.json` | Clearly mark required dataset edits; these are editable examples, not fixed library limits |
| Training snapshot and fixed dates | JSON defaults: version 0 and 2025 example windows | Require deliberate data-specific values before first deployment |
| Monthly lookback | JSON value 4; previous complete UTC month is holdout | Expose/document window policy; cron alone does not change the split algorithm |
| Row/memory budgets | JSON defaults: 10,000 rows, 64 MiB | Tune for real workload; this is not an exact network-byte or total process-memory cap |
| Metric gates | Initialization and deployed JSON | Add optional multiple guardrails; first keep task-compatible threshold examples |
| Per-run version/manual approval | No dedicated job parameter/action | Add validated run parameters and approval orchestration |
| Selection mode changes after generation | JSON plus generated job graph | JSON-only changes do not add/remove the auto score handoff task; validate consistency |
| Score cadence | No generated schedule | Optional score cron/timezone/pause setting, independent of training |
| Model/schema locations and environment hosts | Bundle target variables/placeholders | Keep per environment; do not hardcode company values into the generic template |
| Dependencies | Wheel `0.9.0`, MLflow `3.16.1`, serverless client 4 in resource template | Preserve reproducibility but centralize release/runtime inputs; upgrading a model does not require upgrading the wheel |
| Policy cluster workers | YAML constants 2 to 4 | Make configurable; worker scale does not distribute local pandas/Polars model execution |
| Concurrent runs = 1 and queue enabled | Job YAML | Keep as correctness constraints, not casual performance knobs |
| Champion/challenger/previous aliases and receipt format | Lifecycle library contract | Keep stable conventions unless a separate compatibility design justifies changes |
| Timeouts/retries/notifications/run_as | Missing from generated defaults | Add before unattended production; rehearsal-only 900-second limits are not template defaults |

Bundle variables resolve during deployment; job parameters can be overridden
at execution. This distinction should drive the proposed configuration
surface. See [Databricks job parameter documentation](https://docs.databricks.com/aws/en/dev-tools/bundles/job-parameters).
Plain model aliases can redirect a future load without changing the inference
code; Skyulf adds its controlled transition check. See
[Databricks model lifecycle documentation](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle).

## Workflow size and next order

The generated `src/workflow.py` is 516 lines. It mixes notebook/widget I/O,
target binding, monthly windows, model lifecycle and prediction table/view
management. The concern is responsibility concentration, not a need for more
Databricks jobs. Keep two jobs and a short user-editable entry point; move
reusable orchestration and publication behavior into tested
`skyulf.integrations.databricks` modules. Keep business FE/model choices in
the project configuration and explicit extension points.

Recommended next work before expanding Spark (proposals, not delivered features):

1. Simplify that boundary with behavior-preserving tests and configuration validation.
2. Add manual approval of an existing candidate and readable candidate result tags;
   keep all alias writes serialized and retain score-only recovery.
3. Add runtime version parameters and independent score scheduling, with explicit
   compatibility checks between configuration and the generated job graph.
4. Add production identities, timeouts, failure notifications and generic project
   CI/smoke checks, then validate on the approved company target and real data.
5. Add model-search/multiple-metric gates if required by the use case. Keep
   serving, online features, broad monitoring and Spark expansion separately optional.

## Meaning of QUEUED and no-op

Two score requests targeted the same job. Concurrency one caused the second
to wait rather than write alongside the first. With no new source inserts,
the first produced `noop=true`: no predictions were appended and the output
Delta version stayed at 2. Here Delta version 2 is the table transaction
version, not model version 2. After the first run, the queued run also checked
for work. Rejected v3 training still triggered the existing score job because
training succeeded; that score used champion v2 and also found no new rows.
Queueing does not periodically check for new data or schedule future runs.

## Verification in this review

The focused Bundle/promotion suite initially had two failures caused by stale
test references to `skyulf_pending_alias_event`. The library and archived live
wheel both use `pending_alias_event`; their promotion module lines match.
Only those two test literals were corrected. No production code was changed.
The initial default temporary directory was inaccessible in the sandbox;
reruns used a dedicated repository-local pytest directory.

- `tests/integration/test_sm20a_bundle_template.py` and
  `skyulf-core/tests/integrations/test_mlflow_promotion.py`: **58 passed**.
- Ruff on the changed test file: **passed**.
- The earlier live run evidence remains historical; no new live execution
  or company validation was performed in this review.

Primary Skyulf code: `skyulf-core/templates/databricks/`,
`skyulf-core/skyulf/integrations/databricks/local_retraining.py`,
`local_batch.py`, `local_incremental.py`, and
`skyulf-core/skyulf/integrations/mlflow/{promotion,validation,local_model}.py`.

Primary reference code: `databricks_template_schema.json`, regression
`databricks.yml.tmpl`, `config/{training,validation,inference}.yml.tmpl`,
`resources/jobs/`, `src/{training,validation,inference,deployment,monitoring}/`,
and generated `.github/workflows/`.
