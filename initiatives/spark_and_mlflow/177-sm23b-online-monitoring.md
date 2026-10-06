# SM-23b online monitoring implementation and acceptance

Status: **FUNCTIONAL DONE — native acceptance and shared-reader rollout passed**.
Actual endpoint capture, Spark drift,
late-label performance, exact report replay and shared native dashboard refresh
passed. Full SM-19a serving qualification and SM-37/43b company/identity gates
remain separate. This report records implementation and native acceptance;
commit and PR validation are tracked separately during publication.

> Agentic workers: use subagent-driven-development with separate file ownership,
> focused failing tests and independent review. Root owns integration and live acceptance.

Goal: complete the remaining SM-23 online monitoring contract using actual
Databricks serving logs, Spark metrics and the existing shared dashboard.

Architecture: a pinned optional endpoint records requests through native Databricks
telemetry inference logging (explicit legacy AI Gateway also supported). A Spark reader selects actual served model versions,
deduplicates delivery retries, expands named request/response rows and feeds the
existing monitoring/performance pipeline. Event keys are
`databricks_request_id` and `request_row_index`; delayed labels use those keys.

Spec: existing 39-serving-and-feature-lookup-delivery-plan.md and SM-23b in
OPEN_QUEUE_updated.md, with the concrete source contract below.

## Scope and decisions

- User explicitly requested continuing until SM-23 is finished. Batch SM-23a/c
  is already delivered; add online SM-23b and its necessary serving bridge.
- Keep the current 092 branch, preserve unrelated local edits. No push or new
  commit requested for this task. No production/company-target claim.
- Native CPU endpoint logging uses `telemetry_config`, documented at
  https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-model-serving-uc-logs.
  The generated payload VIEW projects a managed backing `_otel_logs` Delta table.
  The reader validates its canonical definition and `otel.sourceLogsTable`, then
  reads the backing table at a pinned version and verifies its identity afterwards.
- Spark only for observation. No raw population toPandas/collect; collect bounded
  aggregates or metadata. Databricks owns the raw table; never mutate it.
- Actual `served_entity_id` joins `system.serving.served_entities` by endpoint,
  fully qualified model and concrete version. Never infer from current alias.
- Each endpoint/model/version gets its own enrollment; batch IDs and payload
  hashes stay backward compatible when serving fields are absent.
- Preserve at-least-once exact duplicate requests once; conflicting copies fail.
  Failed HTTP requests stay in operational counts, not model metric populations.
- Named dataframe_records input and record-oriented predictions are explicit
  initial transport contracts. Other shapes fail visibly, never vanish silently.
- Logging errors, partial capture, row-count mismatch and ambiguous identities
  fail closed. No-data and unavailable labels are distinct from healthy.
- A request can include multiple rows; labels identify request plus zero-based
  ordinal, never a reused customer key. Input types/output classes come from the
  saved artifact. Model-set branch outputs retain their parent serving version.
- Pinned table read at as_of, event-time windows, late-log replay and late labels.
  Reuse existing performance policies and guarded training requests; no automatic
  endpoint rollout after retraining or bypass of promotion/ownership gates.
- Endpoint helper requires a partition-safe fitted artifact and explicit version;
  Small CPU, scale-to-zero for isolated verification. Both readiness fields must
  settle, and actual config must match. No unrelated endpoint update/deletion.

## Context map and interfaces

1. Parser: new monitoring/serving/serving_payloads.py and __init__.py. Function
   `parse_serving_payloads(payloads, entities, *, endpoint_name, model_name,
   model_version, input_columns, output_columns, output_prefix, start, end)`
   returns `(current, predictions, summary, observed_at)`; columns are tuples of
   `(name, Spark scalar type)`; predictions rename selected prefix to standard
   prediction/probability names. No MonitorConfig dependency.
2. Serving control: new integrations/databricks/serving/ package owns validated
   pinned endpoint config, SDK operations/readiness and explicit inference logging.
   It reuses trusted MLflow artifacts, partition safety and exact runtime pins.
3. Root integration: MonitorConfig optional `serving_endpoint`, source/prediction
   must be same raw table, concrete version and Spark required. Serving reader
   uses artifact input/output schema, explicit request keys and source evidence.
   local/monitoring.py and monitoring_performance.py select reader/keys while
   retaining batch behavior and training-reference identity.
4. Template/operator flow and shared native dashboard: optional serving choices,
   configured enrollments and scheduled monitoring reuse existing job/report.
   Endpoint context and request errors/latency belong in observability views.
5. Native acceptance: isolated synthetic training/model(s), pinned serving,
   HTTP/local parity, actual platform log capture, Spark observation, late labels,
   version isolation, replay, native dashboard SQL/refresh and readback.

## Execution and tests

- [x] Parser red/green tests: duplicate delivery, conflicting payloads, request
  ordinal alignment, failed requests, entity/version isolation, logging errors,
  unsupported envelopes, null features and probability outputs.
- [x] Endpoint red/green tests: aliases/unsafe artifacts rejected, Small/scale-zero
  config, logging schema, exact version readiness, pending/failed config behavior.
- [x] Root config/integration tests: unchanged legacy IDs/payloads, serving IDs
  separated by endpoint/version, event-key labels, performance source identity.
- [x] Generated-template validation and targeted integration consumers.
- [x] Spark parser/metrics tests on actual Databricks; local Spark unavailable.
- [x] Review source/data integrity independently; repair confirmed findings.
- [x] Ruff, format, full CI Ty, Lizard, schema generation; record commands.
- [x] Native accepted results and tracker closure; disclose separate gates.

## Ownership / interface preflight

| Tasks | Shared boundary | Ruling |
| --- | --- | --- |
| Parser / root | signature above, two event keys | Root passes declared typed columns; no parser artifact loads |
| Endpoint / root | concrete model and payload table identity | Actual platform logging and saved model versions only |
| Root / dashboard | existing report_json and evidence_json | Add serving evidence without changing Delta storage schema |
| Each task | focused tests, imports optional, CCN <= 10 | Preserve batch contracts; test failures precede implementation |

## Progress

- Planning: current queue and source verified at b75ba035. No online adapter
  exists. CLI1.17 supports required serving API. Profile skyulf remains selected.

### Integration verification before native launch

- Root config/legacy reference identity tests: 12 passed with existing Spark config consumers.
- Root online routing, event-key labels, late-log cutoff, parent attribution and replaced-table rejection: 5 passed.
- Full CI Ty passed after typed config dictionaries and explicit online-reader guard.
- Independent review identified scalar JSON coercion, incomplete dimension identity,
  missing timestamps and endpoint dtype normalization/float32 overflow; repair work
  is covered by dedicated regressions. Native Spark execution remains pending.
- Read-only native preflight: skyulf authenticated user edwardwolfe99@gmail.com;
  system.serving.served_entities query succeeded. Existing shared inventory has 12
  enrollments. Shared dashboard ID 01f1c090238e1b6da5d633032ad9960b, warehouse
  d047a4d9aa276958, namespace workspace.skyulf_sm23d_20261005_dc93e08c verified.
- Native test source/models/raw logs use workspace.skyulf_sm23b_20261006;
  monitoring results and references reuse the existing common namespace.

### Frozen source and local gates

- Wheel SHA256 e7bfeb115939ddd63fe281f3d898fbf3cb330cc1936e00cc676e415914316aec;
  518 Python files byte-identical to source, 1,149,820 bytes. Uploaded to the
  personal skyulf-sm23b-20261006 directory; worker UDF source included.
- Affected union: 101 passed, 23 Spark-only skipped. Added disabled-enrollment
  regression subsequently reproduced and fixed; affected job file 12 passed.
- Generated Bundle CLI checks: 12 passed across single/competition/model-set,
  serverless/policy-cluster and recovery modes; only local pytest-cache ACL warning.
- Full Ruff, format (1525 files), full CI Ty and backend/Core Lizard CCN<=10 passed.
- Independent review repaired nested/nonfinite scalar payloads, null event times,
  incomplete relevant entity maps, signature aliases, float overflow and disabled
  enrollment preparation. Inline online policy plus job-level retrain opt-in
  documented; endpoint rollout is never implicit.
- Shared dashboard gained aggregate-only online serving current/history datasets.
  Focused dashboard contracts 19 passed. No raw request/response dataset.

### Native acceptance checkpoints

- Initial run 541289144452930: native parser task 954244463355309 running;
  training task 113960769769084 failed due test experiment/directory name collision.
  Platform auto retry 176527556466906 failed before writes because bootstrap tables
  already existed. This is harness setup, not a training runtime defect.
- Source and empty label tables were created in isolated schema. Retry verifies
  source version0 and 96 rows, retains tables, uses a child training_experiment.
  Training-only retry run 27591856224861; retry/auto optimization disabled.
- Auto approval review rejected exporting complete dashboard result sets as
  possible company-data download. Safe metadata-only path: all33 queries compiled
  in both default/NULL modes (66 successful); new serving dataset COUNT wrappers
  executed successfully. Full aggregate-only execution is ongoing. No raw results
  exported and dashboard not yet changed/published.

### Native telemetry and SDK compatibility findings

- Native Spark parser task 954244463355309 completed: **33 passed, zero skipped**,
  Spark 4.2.0, 240.732 seconds. The parser source has not changed since that run.
- Training retry 27591856224861 succeeded (task348079934499296): actual synthetic
  source96 rows, train72/holdout24, holdout accuracy1.0, registered classifier
  versions1 and2. Both packages were admitted using their exact producer wheel;
  no aliases changed. Experiment3101611862942171. Model digest
  `d39cf8d7b5a8c6c63cfedbb9693c3224887c1a436db1a8694809586ceaa9d009`.
- Legacy AI Gateway create was rejected by this workspace before creation. Modern
  telemetry create succeeded for `skyulf-sm23b-20261006-cls-v1` and `...-v2`,
  CPU Small, scale-to-zero, sample fraction1. Databricks requires all three output
  table names at creation; with only inference capture enabled readback retains
  logs_table. The payload object is a VIEW, not a time-travelable Delta table.
- Modern source reader16 tests passed after13 red cases; endpoint24 tests passed.
  Combined changed-file unit tests57 passed. Full Ruff, format1527 files, full CI
  Ty and backend/Core Lizard<=10 passed. Original source wheel was superseded by
  the modern telemetry wheel, then the SDK compatibility wheel below.
- Run865804738021952 stopped before HTTP invocation: typed SDK readback omitted
  telemetry. Read-only probe394310714408658/task407439618636348 proved native SDK
  **0.49.0** returns typed_telemetry=null while raw API telemetry is correct.
  The adapter now uses authenticated `api_client.do` for telemetry GET/POST,
  retaining strict checks of model/version/workload/logging/READY state.
  Regression red then25 endpoint tests green; full Ty/Lizard and Ruff/format passed.
- Current native monitoring wheel SHA256
  `ff5c5e5847daaa0d43f45f9a8a6d3e0ae2560dd8fdc0288d487d8d2c65eedd65`,
  519 Python sources byte-identical, 1,152,829 bytes. Uploaded under personal
  `skyulf-sm23b-20261006/rest-runtime`. Retry run596651861441521 is in progress.
- Shared dashboard final source hash
  `84f492921f73c78701e320f678917aec347d5eedc0b0863afc91a5f7e500a8eb` published
  **2026-10-06T15:48:02.840Z**, same ID01f1c090238e1b6da5d633032ad9960b and
  warehouse. All33 dataset queries executed in default and NULL modes through
  COUNT wrappers:66 succeeded, no company/result rows exported. The later
  config_digest fix reran the four changed queries;62 identical SQL results reused.
  Note-only edit reused66 exact SQL results. Etag-protected update and normalized
  API readback passed; four existing pages/defaults unchanged, embed_credentials=false.
  New online current/history datasets exclude stale policy digests and performance-only
  replay records, and label counts as captured traffic because native logs are
  asynchronous and can omit custom 4xx/5xx responses.

Evidence files are temporary under `tmp_repro_artifacts/sm23b/`; durable run IDs,
hashes and acceptance conclusions belong in this document. At this checkpoint,
native HTTP/log/late-label acceptance was still pending; final evidence follows.

### Final adapter and retraining review

- Retry596651861441521 exposed the second SDK0.49 mismatch: invocation lacks
  `client_request_id`. Native read-only signature probe835826607226308 confirmed
  both query methods lack it. The adapter now sends authenticated raw POST to
  `/serving-endpoints/{name}/invocations`, with unchanged named rows and top-level
  client request ID, and deserializes the response into the existing SDK response
  type. Full strict endpoint readback precedes every send. Endpoint tests25 passed.
- Independent review reproduced a real retraining dedup defect introduced by
  multiple monitoring contexts: duplicate identical model/version/source/content
  tuples changed the training request ID. `_submit_candidates` now hashes the
  sorted set of distinct tuples, preserving legacy singleton hashes and all
  individual monitor evidence. Mixed batch/online and two-endpoint regressions
  failed before the fix; `test_retraining_task.py`46 passed after it, including
  distinct model/version/source/content negative cases.
- Final native code wheel SHA256
  `df3c5bee678b350536f9f2e6592cc97ddc1bce1086696de139e0a2fbcf149533`,
  519 Python sources byte-identical, 1,153,042 bytes; personal `final-runtime/`.
  HTTP parity retry768767010216146/task289880271263666 started. Both actual
  endpoint-to-version mappings now exist in `system.serving.served_entities`.
- Final full Ruff, format1527 files, CI Ty and backend/Core Lizard<=10 passed.
  Readme privilege/schedule clarification subsequently passed its focused
  generated-template contract test. Production Python source remains frozen.
- Operator docs explicitly require SELECT on payload view **and** backing Delta
  plus the served-entity system dimension, with usual usage privileges. They
  distinguish asynchronous HTTP capture, unpaused Spark observation schedule,
  completed-window late-label replay and configured native Dashboard refresh.

### Native capture and label-commit cutoff evidence

- HTTP retry768767010216146 succeeded in210.842s. Both concrete versions returned
  five predictions matching the independently loaded local artifact (10 total).
  Real captured rows joined one actual system served entity per version; first
  Spark reports for both versions had current_rows5, labeled_rows0 and no error.
- Run256755343455622 appended10 synthetic outcomes keyed to those real request
  IDs and row ordinals. Version1 uses the true synthetic target; version2 deliberately
  inverts it to test degradation. This does not represent real customer outcomes.
- The immediate post-write test cutoff was `16:11:37.732003Z`; actual Delta commit1
  is timestamped `16:11:38.000Z`. Correct as-of logic therefore retained snapshot0.
  Native saved evidence proves label_version0, rather than a join/snapshot defect.
  This is a 268ms test clock boundary. Root canceled that isolated test before its
  knowingly invalid after-label assertion and prepared a continuation that checks
  the committed label snapshot before observing. No library cutoff was relaxed.
- Continuation506262826849957 verifies the existing10 labels and saved before-label
  evidence, writes no additional outcomes, and reruns only after-label observations,
  historical policy checks, exact report replay and native Dashboard refresh.
  Original endpoint payload views/log tables remain untouched.

## Final native acceptance

[Continuation job 506262826849957](https://dbc-45604623-c18b.cloud.databricks.com/jobs/111258888463178/runs/506262826849957)
finished **SUCCESS** in 1,076.738 seconds. Both `observe_labels` (task
270304432962933) and `refresh_monitoring_dashboard` (task250325089789284)
completed successfully. Observation execution was 969 seconds; this small-data
functional rehearsal is not a throughput or load qualification.

| Evidence | Version 1 | Version 2 |
| --- | --- | --- |
| Actual captured HTTP requests | 1 | 1 |
| Predictions matching the saved local artifact | 5 | 5 |
| Labels before arrival | 0 | 0 |
| Labels after commit | 5 | 5 |
| Label coverage | 100% | 100% |
| Accuracy with controlled outcomes | 1.0 | 0.0 |
| Confusion matrix total | 5 | 5 |
| Historical policy verdict | healthy | degraded |

Both full drift observations remain healthy; the controlled outcome shift is
detected independently by performance measurement. The policy replay uses the
actual completed request-time window. Newer empty windows remain unavailable;
backfill does not pretend to be current evidence or trigger training. The test
policy is report-only, and the historical verdict action is `none`. Each replay
report was persisted twice and verified to exist once. Existing batch/performance
training guards remain in place, with the duplicate-context request repair above.

The continuation reused the 10 actual-key labels and wrote **zero** new labels.
It preserved both endpoint logs/views and all original company jobs, tables,
model aliases and dashboard identity. Only isolated synthetic resources and
their entries/references in the existing shared monitoring store were added.

### Published dashboard verification

[Shared dashboard](https://dbc-45604623-c18b.cloud.databricks.com/dashboardsv3/01f1c090238e1b6da5d633032ad9960b/published)
retains the existing four pages and adds Online serving. Its 33 queries passed
66 default/NULL COUNT executions before publication. After the native refresh,
**9 additional shipped SQL checks passed** with explicit synthetic model/version
and monitoring-context selectors: two current online rows, five full observation
rows, accuracy 1.0/0.0, four confusion cells summing to five per version, and
healthy/degraded historical decisions. No company result rows were exported.

Source hash remains
`84f492921f73c78701e320f678917aec347d5eedc0b0863afc91a5f7e500a8eb`;
publication was `2026-10-06T15:48:02.840Z`, with `embed_credentials=false`.
The native task refreshed that published revision. Verification used actual SQL,
job state and normalized API readback; no new browser visual inspection was run.

### Retained resources and boundaries

- Model: `workspace.skyulf_sm23b_20261006.classifier`, versions 1 and 2.
- Endpoints: `skyulf-sm23b-20261006-cls-v1` and `...-v2`, both Small CPU with
  scale-to-zero, verified READY/NOT_UPDATING. IDs are
  `02d45a5a730244788090899c18af005f` and `0f4ad913a3fe427e9fa7062124822cdb`.
- Synthetic source and labels: `training_classifier` and `labels_classifier` in
  `workspace.skyulf_sm23b_20261006`. Each endpoint has its own
  `sm23b_20261006_cls_vN_payload` view and `_otel_logs` backing Delta table.
- Inventory, results and prepared references reuse
  `workspace.skyulf_sm23d_20261005_dc93e08c`. No second shared dashboard or
  central monitoring namespace was created.
- Personal notebooks/wheels: `/Users/edwardwolfe99@gmail.com/skyulf-sm23b-20261006`.
  Final native code wheel matches all 519 production Python files. Model packaging
  and initial admission used the earlier matching producer wheel, as recorded
  above; the final monitoring/HTTP adapter wheel was verified separately.
- Broader serving qualification, continuous streaming, endpoint promotion/canary,
  production identity authorization and load tests were not claimed by this task.
  No new scheduled production project or automatic endpoint rollout was enabled.

### Shared-reader compatibility closeout

Final inventory review found older deployed SM-23e/SM-57/proof runtimes sharing
this namespace. Their loader parses every enrollment before filtering the project;
an old MonitorConfig therefore cannot read the new serving field. Downloaded
copies of all seven deployed wheel paths (three distinct hashes) reproduced that
rejection. Disabling online entries would not avoid it. All 19 owned job settings
were backed up, with zero active runs, and 33 relevant notebooks were inspected.

The generated score-job `prepare_monitoring` task reads only its producer receipt;
it does not load the shared inventory and needs no reader upgrade. All training
and scoring environments therefore remain at their original package versions.

[Read-only compatibility run 1057743313713804](https://dbc-45604623-c18b.cloud.databricks.com/jobs/504375781673557/runs/1057743313713804)
finished **SUCCESS**, seven tasks, in 300.101 seconds. Each task used its original
monitoring dependency spec with only the Skyulf wheel replaced by the final native
code wheel. All seven legacy notebook imports resolved. Each environment read all
14 enrollments, including the two online versions, and selected the correct batch
project independently for observation and retraining evaluation. Their inventory
digests matched. Inventory Delta version32 and results version96 were unchanged:
these probes did not measure customer data, save reports or request training.

The following seven monitoring jobs now reference the verified final wheel:

| Project | Job ID | Batch contexts validated |
| --- | --- | --- |
| sm23e_single | 849383177102322 | 1 |
| sm23e_competition | 382789798978890 | 1 |
| sm23e_model_set | 52053718085306 | 2 |
| sm57_single | 185291625939783 | 2 |
| sm57_competition | 781331399342724 | 2 |
| sm57_model_set | 1050459754065826 | 2 |
| sm23d proof | 918715320090438 | 1 |

Full API comparisons verified **only the wheel dependency changed**. Existing
environment keys, other requirements, notebook paths, task graphs, policies,
parameters, schedules, Bundle deployment metadata and UI locks were preserved.
All 12 training/scoring jobs remained byte-equivalent after keyed-list ordering
normalization. No old wheel, registered artifact, alias or raw table was replaced.

Two API details were handled explicitly during this migration: the local SDK
drops `environments` from run responses, so proof binding uses the authenticated
raw Jobs API; an update omitting `edit_mode` resets a Bundle's UI lock. The first
job's lock was restored immediately and all subsequent updates explicitly retained
the original mode. Final comparisons include that field and passed for every job.

Independent review confirmed the environment-only migration and required binding
the successful native probe to the exact proposed dependency spec. The runner now
verifies the remote wheel SHA, a fixed seven-job allowlist, exact wheel-only diffs,
native task/environment correspondence, fresh inactive settings and full readback.
Evidence/backups are under `tmp_repro_artifacts/sm23b/compat/` and `rollout/`, with
final `migration-result.json` reporting seven verified upgrades and 12 untouched
jobs. These are local diagnostic artifacts, not files to commit.

The README requires upgrading all shared inventory readers before the first online
enrollment. Future deployments of old rehearsal Bundle sources must retain these
reader overrides or regenerate from the current template; deploying an old wheel
again would reintroduce the incompatibility. This migration intentionally did not
redeploy whole historical Bundles or change their certified train/score runtimes.

### Verification commands and final state

Full repository CI static scopes passed after the last production Python change:

```powershell
.venv/Scripts/ruff.exe check .
.venv/Scripts/ruff.exe format --check backend skyulf-core tests run_skyulf.py celery_worker.py
.venv/Scripts/ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
.venv/Scripts/lizard.exe backend skyulf-core/skyulf --CCN 10 -w
git diff --check
```

Focused test batches are recorded above and overlap; their counts are not summed.
The last behavior repairs passed `test_serving_endpoints.py` (25) and
`test_retraining_task.py` (46); native Spark parser tests passed 33 with zero skips.
Template generation covered all three layouts and both supported compute modes.
No full local suite/coverage was run. SM-23 itself changed no frontend source;
no MkDocs build was needed for its generated-template README. Native acceptance
preceded the follow-up request to commit the project changes and validate a PR.
