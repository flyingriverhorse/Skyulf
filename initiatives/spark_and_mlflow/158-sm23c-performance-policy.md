# SM-23c performance policy implementation and evidence

Date: 2026-10-05. Branch: 092. Baseline: 3db281c3.
Status: performance-policy scope complete; local gates and isolated live acceptance passed.
Implementation commit: e9588f07 (DCO signed, branch 092; not pushed).
Spec: OPEN_QUEUE_updated.md, SM-23c performance policy acceptance (lines 552-592).
User approved proceeding with SM-23c before SM-57 and serving.

## Design and context map

Reuse Core performance calculations and the existing retraining request writer.
Keep distribution drift and performance separate. A default-off per-model policy
uses fixed UTC windows, an explicit label-maturity delay, minimum labeled count
and coverage, and consecutive distinct completed failing windows. Baselines bind
to a concrete model version: replay the verified training holdout with the same
metric evaluator, or read an explicitly selected production report. A production
baseline must have matching metric/class/population contracts. A changed version,
policy, invalid window or gap cannot inherit a failure streak.

The generated variable `monitoring_performance_policies` is a JSON object keyed
by fully qualified component model name (single/competition/model-set compatible).
Each policy has mode off/report/retrain; metric; direction higher/lower;
baseline {kind: training_holdout|production_window, model_version, report_id for
production_window}; tolerance; tolerance_mode absolute/relative; window_hours;
label_delay_hours; minimum_labeled_rows; minimum_label_coverage; consecutive_windows.
No implicitly selected metric, baseline, or degradation tolerance.

| Files | Responsibility |
| --- | --- |
| New performance_policy.py, test_performance_policy.py | Strict policy, pure verdict and distinct-window streak evaluation |
| monitoring_config.py, monitoring_registration.py, monitoring_model_set.py | Enrollment and per-model policy propagation; legacy digests remain unchanged when policy absent |
| New monitoring_performance.py; monitoring_reference.py, monitoring.py | Fixed-window measurements, pinned baseline, bounded history and persisted evidence |
| retraining_task.py | Combine independent signals through existing fresh-data/request guards |
| monitoring_output.py, monitoring_store.py | Saved performance report and SQL projection |
| examples/databricks_monitoring/src/monitoring.lvdash.json | Performance status, baseline/tolerance trend, evidence and request drill-down |
| templates/databricks/template/{{.project_name}} | Optional variables, all producer/task parameters, development isolation and documentation |

No new dependency or existing Delta table migration is needed. Existing report_json
and evidence_json retain the new evidence. One additive owned performance_actions
table records request and skip outcomes; performance_history joins those receipts
without rewriting immutable monitoring reports. The central initializer provisions
both objects and rejects foreign/incompatible objects.

## Execution plan and ownership

- [x] Task 1: strict pure policy and verdict. Child implementer owns new policy
  module/test only. Test off defaults, invalid/nonfinite values, higher/lower and
  absolute/relative thresholds, zero relative baseline, coverage, stale evidence,
  replays, gaps, version/policy resets. Root reviews integration contract.
- [x] Task 2: root owns enrollment, Core metric reuse, baseline replay, UTC windows,
  bounded persisted history and shared retraining integration. Focused failing
  tests first; preserve request identity based on training data, not trigger type.
- [x] Task 3: child owns saved HTML/report and dashboard changes after interface
  freeze. Tests verify exposed fields, filter bindings, separate units and missing
  labels; root handles SQL evidence projection and request joins.
- [x] Task 4: root owns generated settings, docs, focused integration union,
  Ruff, CI Ty scope, Lizard, template schema check, review and live acceptance.

Use `python -m pytest` with explicit files and no coverage; no full suite.
CI static commands: `ruff check .`; `ruff format --check backend skyulf-core tests
run_skyulf.py celery_worker.py`; `ty check backend skyulf-core/skyulf
skyulf-core/tests run_skyulf.py celery_worker.py`; `lizard skyulf-core/skyulf
--CCN 10 -w`; `python skyulf-core/templates/databricks/build_schema.py --check`.

## Decisions and evidence

- Existing drift monitoring observes a one-microsecond scoring commit window.
  Performance therefore needs an independent fixed window and current label
  cutoff; otherwise late labels could never affect that committed observation.
- Preserve absent-policy serialized payloads to avoid invalidating existing
  inventory/report digests on upgrade.
- Keep existing on_drift cooldown/minimum-row controls as shared request controls;
  independent signal modes must not create independent request writers.
- Local offloader unavailable: launcher has no registered Python; repository
  Python health probe cannot start server because server.log is outside writable
  workspace. No local-model output used.
- Databricks CLI uses the authorized elevated executable path. User explicitly
  selected profile skyulf; isolated live acceptance is in progress.

## Validation ledger

Baseline 3db281c3 plus working-tree implementation, before cloud acceptance:

- Explicit union of 20 affected test files: 420 passed, 9 failed. The failures
  were stale mocks missing label-table identity and a no-op receipt fixture.
  After correcting those fixtures, the two affected files passed all 36 tests.
  This is 429 passing cases across the union and focused repair run, not one
  clean 429-test invocation. Logs: tmp_repro_artifacts/sm23c/final-union-r1.log.
- Generated Bundle template: 13 passed with SKYULF_BUNDLE_CLI_TEST_PROFILE=skyulf,
  including 12 layout/recovery/compute combinations. Replaced obsolete direct
  package assertions with checks of the generated requirements-file contract.
- Full Ruff lint/format, full CI Ty scope, Core Lizard CCN 10 and generated
  template schema check passed. No frontend source changes.
- Independent final code review found no critical/important findings; reviewed
  policy, replay/gaps, population contracts, labels, shared requests and SQL paths.
- Built wheel: SHA256 10fcb6ac619d2608bd0d260f594fcef1399ae10259c4a14c237aa92ecc0d41f8.
  Upload preparation verified all 343 packaged Python modules against checkout.
- Live acceptance run 160617670562676 submitted using profile skyulf. Cloud
  acceptance and dashboard/request evidence remain pending at this entry.
- Pre-commit on all changed implementation/test/template/README/changelog files
  passed: whitespace, JSON, Ruff, formatting, backend/Core Lizard and full Ty.
  Frontend hooks correctly had no matching changes.
- All 12 generated Bundle variants passed `bundle validate --strict -t test`
  after binding their temporary target host, built wheel and existing Job Compute
  policy. Initial attempts only failed on unset fixture deployment bindings;
  no production/template edits were needed to pass strict validation.
- All 12 dashboard dataset SQL queries passed on warehouse d047a4d9aa276958
  with model/version/metric parameters and the isolated acceptance namespace.
  Published test dashboard 01f1c086b24d1390b910d312bf88b6c4; readback retained
  all dataset/page definitions and workspace.skyulf_sm23c_20261005_09f1862e
  bindings. This is API/SQL validation, not a browser visual review.
- Live measurement acceptance 160617670562676 / task 316654667794486 succeeded
  in 656.934 seconds. Independently computed NumPy MAE matched saved policy
  values: 5.0 without drift and 0.0 with drift. Exact replay retained one report
  and failure count 1; wrong-version production baseline was unavailable; late
  labels changed unavailable to degraded for the same fixed window. Real source
  freshness assessment returned no_new_training_data with changed_rows=0.
- Real-request follow-up 954212243279154 used isolated train job
  152873613181291 (skyulf_sm23c_09f1862e_train), with max_concurrent_runs=1 and
  no schedule. It measured actual performance-only and combined-trigger reports
  against a new CDF source/prediction pair and verified one shared request/run.
- Final request acceptance 954212243279154 succeeded in 304.633 seconds.
  Unchanged training data was skipped; after 20 upstream rows were appended,
  17 newly eligible training rows justified request
  d0815f9a1a97b648098fcb2301b02499a2abe71c7774326286c092a53650401c.
  Actual training run 936266752709528 succeeded in 101.549 seconds, producing
  candidate model version 2 without automatic promotion. Existing comparison
  rules remained authoritative. Ordinary replay and a separately measured
  combined drift/performance report both returned already_submitted with the
  same request/run. The Jobs API lists exactly one run for this train job.
  Both policy MAEs matched independent NumPy MAE 5.0; combined drift count was 1.
- Final published-dataset read verified all eight policy rows, including independent
  MAE values, unavailable/healthy/degraded states and exact request/run identities.
  Two HTML reports rendered from live stored evidence also retained the matching
  request/run and degraded verdict. No browser visual review was performed.

### Focused union command and repairs

`python -m pytest -q --no-cov --tb=short` with these explicit files under
`skyulf-core/tests/integration/platforms/` and workspace `--basetemp`:

```text
test_performance_policy.py test_performance_actions.py test_monitoring_performance.py
test_monitoring_config.py test_monitoring_registration.py test_monitoring_model_set.py
test_monitoring_output.py test_monitoring_dashboard.py test_monitoring_metrics.py
test_monitoring_sources.py test_monitoring_reference.py test_monitoring_store.py
test_monitoring_tasks.py test_monitoring_job.py test_retraining_task.py
test_retraining_requests.py test_retraining_data.py test_monitoring_review_batch15.py
test_monitoring_drift_review_batch16.py test_monitoring_evidence_batch17.py
```

Result 420 passed / 9 failed before fixture repair. Then the explicit deduplicated
repair scope `test_monitoring_model_set.py test_monitoring_review_batch15.py`
passed all 36 cases. No production source changed after the union. The enrollment
agent also verified the direct `test_model_set_project.py` consumer as part of its
77-pass focused enrollment group. Generated template command:
`python -m pytest skyulf-core/tests/integration/platforms/test_monitoring_template.py
-q --no-cov --tb=short -p no:cacheprovider` with selected CLI profile: 13 passed.

## Operator walkthrough

1. Build/deploy the updated Core wheel. Run the shared monitoring initializer
   before updating an existing dashboard; it adds the owned action table/view.
2. In the producer target, set monitoring_label_table and
   monitoring_result_available_at_column. Configure monitoring_performance_policies
   with the exact registered component name and concrete baseline model version.
   Start with report mode; the README example lists every required field.
3. Score a batch, provide keyed actual outcomes, and run monitoring after the
   configured fixed window plus label delay. Read the performance section in
   the drift report; distribution drift and performance have separate verdicts.
4. In the shared dashboard, choose the model/version and policy metric. Compare
   current, baseline and tolerance-boundary series, then inspect the policy table
   for report ID, coverage, failing windows, reason and request/run receipt.
5. When enabling retrain mode, keep the configured upstream training table current.
   Monitoring labels alone do not create eligible training rows. Unchanged training
   data is skipped; fresh eligible rows still pass the shared active-run/cooldown
   gate and ordinary training evaluation/approval policy.

Personal-workspace acceptance does not close SM-37 separate-identity/concurrency
or SM-43b company-target production readiness. Existing producer deployments
were not updated by this isolated acceptance. No browser visual review was run.
