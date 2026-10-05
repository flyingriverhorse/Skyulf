# SM-23d Spark monitoring implementation and delivery

> For agentic workers: use subagent-driven-development for focused implementation
> and review, with the controller responsible for integration and cloud acceptance.

**Goal:** Calculate production monitoring on Spark without whole-population driver
materialization, preserving policy, provenance and guarded retraining semantics.

**Architecture:** Dedicated monitoring execution reads original Delta predictions,
pinned feature snapshots and mature outcomes. Spark calculates metrics; existing
Delta reports feed both policy decisions and the Databricks AI/BI dashboard.
Local monitoring remains an explicit compatibility mode. Model fitting is unchanged.

**Tech stack:** PySpark SQL, Delta, existing Skyulf policy contracts, MLflow, DAB.

**Spec:** User-approved design in OPEN_QUEUE_updated.md SM-23d and the monitoring
example README, accepted for implementation on 2026-10-05.

**Delivery:** Implemented and locally/cloud validated on branch 092. The native
[isolated dashboard](https://dbc-45604623-c18b.cloud.databricks.com/sql/dashboardsv3/01f1c090238e1b6da5d633032ad9960b/published)
shows the test results. Production rollout and inherited release gates remain separate.

## Global constraints

- No full observation collect/toPandas/toLocalIterator, including label joins.
- Bounded summaries only on driver; explicit limits for model metadata/cardinality.
- No silent approximation, unsupported metric, local fallback or raised safety cap.
- Preserve model version/digest, physical table identity, source snapshots, UTC
  availability, record keys, replay identity and independent drift/performance.
- New version needs its own baseline. Invalid evidence cannot trigger training.
- Separate monitoring compute, scheduling, retry and timeout budgets.
- Single, activated competition winner, each model-set component retain scope.
- Fresh-data controls cannot treat weight-only changes as new training values.
- Stay branch 092; preserve unrelated AGENTS.md/.claude/.tmp-review-model edits.
- Explicit test files only; Ruff/full CI Ty/CCN <= 10 and pre-commit before commit.
- Cloud profile skyulf already selected. Use isolated acceptance resources only.

## File and interface map

| Unit | Files | Contract |
| --- | --- | --- |
| Distributed metrics | spark_monitoring_metrics.py, spark_monitoring_drift.py, spark_monitoring_quality.py | Spark frames -> bounded existing report dictionaries |
| Provenance reads | spark_monitoring_sources.py, monitoring_sources.py | Pinned distributed frames + existing receipt evidence |
| Baseline lifecycle | spark_monitoring_reference.py, local_retraining.py | Version-bound saved reference evidence; no per-observation local replay |
| Observation/policy | monitoring.py, monitoring_performance.py, monitoring_config.py | Explicit engine dispatch, existing report identity/history |
| Fresh-data guard | spark_retraining_data.py, retraining_data.py | Distributed eligible-population comparison, same request guards |
| Separate execution | monitoring_tasks.py, producer templates, central example | Durable handoff and scheduled late-label observation |
| Presentation | monitoring example dashboard, READMEs | Existing metric tables plus engine/freshness/coverage evidence |

## Task 1: Spark metric kernels

- [x] Write focused parity/eligibility tests in
  `skyulf-core/tests/spark/test_monitoring_metrics.py` and execute RED.
- [x] Implement `build_spark_performance_report(predictions, labels, *,
  record_key_columns, target_column, result_available_at_column, as_of, task,
  classes=()) -> dict` with the existing build_performance_report result shape.
  Distributed key validation and joins precede aggregate calculation.
- [x] Implement exact regression aggregates and confusion-matrix classification
  metrics; never materialize saved prediction/label rows. Validate probability
  contracts; expose unsupported probability statistics explicitly if necessary.
- [x] Implement `build_spark_monitoring_report(reference, current, predictions,
  labels, *, feature_columns, record_key_columns, target_column,
  result_available_at_column, as_of, task, classes=(), thresholds=None) -> dict`.
  Quality and drift retain statistic/evidence distinctions. Approximate algorithms
  require separately named evidence and must not silently inherit exact thresholds.
- [x] GREEN: parity tests, malformed outputs/keys/labels, no-label and excluded
  rows, class order, constant regression targets, partition invariance, bounded
  result collection. Controller reviews implementation and missing cases.

## Task 2: Distributed observations and baseline evidence

- [x] RED: source snapshots, physical replacement, matching original features,
  late-label eligibility and model-set version failures in explicit platform tests.
- [x] Build Spark source readers reusing receipt validation and window helpers;
  maintain frames on Spark across source joins and outcome joins.
- [x] Persist exact training-reference and holdout evidence at preparation time;
  bind cache to model/training/spec identity and verify it on read. Existing
  artifacts need an explicit preparation/migration path, not silent re-splitting.
- [x] Add explicit engine configuration and dispatch from observe_model and
  performance measurement; preserve legacy default payload identity.
- [x] GREEN and review source/identity/cutoff/late-label/replay behavior.

## Task 3: Distributed training eligibility

- [x] RED: unchanged values, new labels, duplicate multiset counts, weight-only
  edits, advancing temporal boundary and random holdout reshuffle.
- [x] Compare eligible feature-target populations in Spark; bind request evidence
  to concrete source/model contracts. Preserve training-side eligibility rather
  than counting arbitrary CDF changes. Unsupported custom selection must fail
  explicitly rather than silently authorize a different training population.
- [x] GREEN and review guard integration through the existing request writer.

## Task 4: Separate job, templates and dashboard

- [x] RED: explicit engine propagation, independent budgets and receipt handoff;
  retain local job behavior for existing projects and development opt-out.
- [x] Wire separate Spark monitoring execution with its own schedule, retries and
  timeout; late labels do not require new predictions. Keep report-only vs retrain.
- [x] Update single/competition/model-set templates and central example; surface
  measurement time, engine, coverage and failure evidence in existing reports.
- [x] Validate every changed dashboard SQL query before isolated deployment.

## Task 5: Acceptance and delivery

- [x] Run deduplicated affected test file union once at final source state.
- [x] Run CI Ruff, formatting, full Ty, Lizard; inspect diff and pre-commit.
- [x] Build source-verified wheel and validate generated Bundle layouts.
- [x] Isolated Databricks acceptance: >1M prediction rows, regression and
  classification parity, delayed-label contracts, replay, actual policy evidence
  and guarded request dedup. Record runtime and driver RSS. New scale deployment
  is single-model; model-set/competition scope has focused contract coverage.
- [x] Independent final review, address findings with focused reproductions.
- [x] Update queue/delivery evidence accurately, DCO commit, no push.

## Verification ledger

2026-10-05 baseline: HEAD 38c094e0, branch 092. Documentation edits from the
preceding turn are authorized and retained. No implementation or cloud acceptance
had passed yet for SM-23d at that baseline.

2026-10-05 implementation checkpoint (uncommitted branch 092):

- Spark source joins, quality/drift/performance kernels, cached exact training
  references, distributed freshness comparison and independent producer jobs
  implemented. New producer generation defaults to Spark; existing config
  payloads retain local defaults and identities. Scheduled monitoring starts
  paused; scoring dispatch is asynchronous and receipt-idempotent.
- Reference preparation verifies a new original-source content artifact and
  performs one training-budget bounded replay. Ongoing observations never replay
  local source populations. Legacy artifacts without original content proof
  require retraining or separately recovered original evidence. Physical identity
  is pinned at preparation; identical-content replacement before preparation
  cannot be distinguished by the content artifact.
- Freshness supports native row eligibility and exact sklearn split metadata on
  one bounded executor. Fixed normalizers/custom filters/custom weight recipes
  without proven native parity reject explicitly. Training budgets still apply;
  the conservative Spark byte estimate can reject near-budget inputs.
- Independent review reproduced and fixed changed-source reference acceptance,
  nullable integer transport mismatch, normalized timestamp reparsing, and
  all-null reference schema inference. Review evidence lives in ignored
  tmp_repro_artifacts/sm23d/review1.md and review2.md.
- Real serverless run 425712942066945: 24 focused Spark checks passed. Additional
  boundary regressions reproduced three failures in run 1036509447292338
  (minimum LONG key overflow, excluded NullType probabilities, future-only label
  keys); remaining 24 passed. Fixes submitted in run 803894795332431, pending at
  this checkpoint. Typed-null preparation run 13718870322221 pending.
- Local final affected union part 1: 104 passed in nine named config/reference/
  performance/freshness files; direct consumers part 2: 205 passed in ten named
  registration/task/output/training files. Two new window tests and 24 task tests
  passed after the complexity-only task extraction. Commands/results are saved
  in final-union1.log, final-consumers.log and windows-final.log under the ignored
  evidence directory. Template owner ran 14 focused tests; all 12 freshly
  generated variants passed strict CLI Bundle validation with profile skyulf.
- Full CI Ruff scope, full CI Ty scope and Lizard CCN<=10 pass at checkpoint.
  Wheel build succeeds, packaged module bytes verified before every submission.
- Scale acceptance run 154946598841844 uses a new isolated model/schema and
  1,100,000 keyed predictions. Still pending; no cloud-delivery claim yet.

## Verified cloud evidence (2026-10-05)

- Run `803894795332431`: **37/37** focused Spark cases passed, including the
  three reproduced metric edge cases and both freshness review regressions.
  Run `13718870322221`: typed all-null reference-frame regression passed.
- Scale fixture `workspace.skyulf_sm23d_20261005_dc93e08c` uses a newly trained
  real registered linear model, its original source proof and prepared Delta
  reference tables. The 1,100,000 prediction keys repeat 80 distinct inputs whose
  outputs were produced by the saved model. This is monitoring acceptance,
  not an acceptance of distributed scoring (SM-57 remains separate).
- Final scale continuation `969179067578873` **passed**: all 1,100,000 rows
  observed/labeled, MAE **5.0**, baseline **0.0**, coverage **1.0**, verdict
  `degraded`, action `report`. Repeating the same cutoff retained report ID
  `963683b455afed9b62e7bd40d46fad57a90759b8e4aa8908dcbb9e42ffccf04b`.
  A separate 1,100,000-row classification population gave accuracy and weighted
  F1 **1.0**. Observation/replay, both freshness checks and classification took
  **301.68 seconds** together. Peak Python-process RSS was **2,231,100 KiB**;
  this includes runtime/model/query overhead and is not a memory-growth benchmark.
  Spark `toPandas`/`toLocalIterator` were forbidden during these checks; the
  largest `collect` result was **10 aggregate/metadata rows**.
- Same-source freshness: **0 changed rows**, `no_new_training_data`. After 20
  labeled source rows were appended, the exact new split had 80 training rows
  and **17 newly eligible feature/target rows**, `ready`. This passed after
  removing DataFrame cache/persist, which serverless does not support
  ([Databricks limitation](https://docs.databricks.com/aws/en/compute/serverless/limitations)).
- Initial scale attempts exposed harness mistakes (an unreceipted empty WRITE,
  then expecting `none` instead of the existing report policy's `report` action).
  Production validation stayed strict. They are not counted as acceptance passes.
- Dashboard **01f1c090238e1b6da5d633032ad9960b**: all **12 dataset queries**
  executed successfully against the real scale evidence before publication;
  readback verified 12 datasets, three pages and the isolated catalog/schema.
  Policy SQL independently matched 1,100,000 labels, coverage 1.0, MAE 5.0 and
  the degraded/report decision. This is the existing native AI/BI presentation
  over Spark-produced Skyulf tables, not a second InferenceLog calculator.
- Actual asynchronous dispatch to job **918715320090438** returned the same run
  for repeated receipt/invocation identity. Different producer run identities
  allow subsequent no-op scores to revisit late labels. A real job run exposed
  timezone-free `job.start_time.iso_datetime`; the template now passes
  `job.start_time.timestamp_ms`, explicitly parsed as UTC. The repaired live
  three-task chain is being verified separately.
- Static/template evidence: full Ruff, full CI Ty, CCN<=10 and pre-commit passed.
  All 12 producer variants passed strict CLI validation after the invocation
  parameter change. After the common epoch parameter change, the shared template
  passed one representative generated test plus strict CLI validation. No full
  repository test suite or frontend/MkDocs build was run; affected Python and
  Bundle contracts own this change, and documentation is Markdown-only.

## Operational boundaries

New generated projects default to a dedicated serverless monitoring job; its
schedule is initially paused. Set the central namespace, label table and policy,
deploy the producer, then unpause its monitoring schedule for delayed-label runs.
Scoring submissions still trigger it asynchronously. Use `monitor_model ->
drift_report -> retrain_on_drift` in the separate job for completed results.

Single models, activated competition winners and pinned model-set components
share these contracts. All layouts have generated/static and existing integration
coverage; the new scale fixture is single-model. Existing SM-23c real performance
request acceptance remains evidence for that shared policy path. New Spark
request acceptance is tracked separately below; do not infer a new multi-target
cloud deployment or production release from the scale result.

Exact ordered distribution/probability curves can concentrate distinct-value
processing on one executor. Summary class/category limits and unsupported
temporal/nested drift remain explicit. There is no silent sampling or local
fallback. This delivery removes monitoring's local observation cap; it does
not remove training budgets, inherited SM-37/43b production gates or approval
requirements, and it does not enable continuous streaming.

### Final job and request acceptance

- Repaired job run **364358743271992** passed all three actual tasks:
  `monitor_model`, `drift_report`, `retrain_on_drift`. Task outputs were read
  individually: failed model count **0**, saved report
  `6eb604b35cc33b2cb90c42ae96446f6022077aa4901820f8764379c6c9c3bdc6`,
  and retraining `disabled` for the report-only setup. The job remains scheduled
  **PAUSED**. The real epoch parameter resolved successfully. A later empty
  performance window remains unavailable rather than inheriting an old degraded
  decision; the full saved scoring batch's feature report is independently healthy.
- Request acceptance **494290923825803** passed using a separate monitor identity,
  80 genuinely scored shifted inputs, native Spark drift evidence, and the same
  **17** newly eligible training rows. It submitted train run **853036605815046**
  through the existing request writer. Replaying the candidate returned
  `already_submitted` with the same run and durable request
  `c792a6a708b1f03fd90575b9fa5d9d9b38f6f4bfa252e06a0d533d4637a1aa48`.
  This new test exercises the drift trigger into the shared Spark freshness/request
  path; SM-23c's earlier live evidence owns the performance trigger acceptance.
- Child train run **853036605815046** completed successfully and registered
  candidate model **v2** from baseline **v1**. `automatic_promotion=false`:
  creating a retraining request did not bypass the existing comparison gate.
- The final independent review found no remaining blockers, including typed null
  caches, no-op invocation deduplication, and UTC job-start handling. Source wheel
  SHA for the repaired job/request is
  `1020eb7bf92680943613817bb94d71fe12832e1fc0feb117dd363034c8a665ad`.
  Scale/freshness kernels were unchanged from passed continuation wheel
  `9992ec1ce7885aba5199d16a5d6df3a9323b60dec6e23beb7f2aa741b71089b3`;
  only the job-start parser changed afterward.

Final verification inventory: **329 distinct focused local test cases** (including
14 generated-template cases), **38 focused real Spark cases**, 12 strict Bundle
variants plus the common epoch-parameter recheck, actual scale/replay/freshness
acceptance, all three monitoring tasks, shared training request/replay and child
training, and all 12 dashboard SQL datasets plus publish/readback. Passing groups
were only repeated after relevant changes; the counts do not add repeated runs.
All applicable pre-commit hooks passed. No push was requested or performed.
