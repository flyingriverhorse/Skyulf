# Task218: daily serving rollout and online feature lookup

Status: FUNCTIONAL DONE. Implementation, independent review, local/native
regression checks, actual endpoint acceptance and temporary capacity cleanup
are complete. Production-duration boundaries remain explicit below.
Branch `093`, baseline `48bee839`; user requested a separate Task218 commit
before the streaming follow-up on 2026-10-11. No push requested.

The user selected SM-19d and SM-21b, approved automatic guarded champion
promotion after the final observation, and authorized Databricks tests without
repeated permission. The latest approval permits deletion to free serving
capacity. Internal execution evidence belongs here, not in public guides.

## Delivered behavior

- Daily rollout defaults to 10 percentage points with at least 24 hours per
  stage. Missed schedules never cause catch-up jumps. Missing, stale or
  insufficient evidence holds; confirmed health/quality failure rolls traffic
  back to the retained champion. A passing final 100% stage invokes the existing
  guarded approval flow, including saved holdout re-evaluation and alias pins.
- Two concrete, compatible, admitted model packages share a new A/B endpoint.
  Durable MLflow PREPARED/COMMITTED receipts bind endpoint identity, configuration
  revision, routes and policy. Uncertain writes reconcile without blind resend.
  Existing SQL endpoints retain their separate pinned contract.
- Existing monitoring supplies version/stage-specific observations, response
  attribution, delayed-label joins and Core quality metrics. There is no second
  monitoring engine or configurable PASS flag.
- Online publication uses native Feature Engineering TRIGGERED synchronization
  with explicit store/table selection, source Delta identity, CDF, declared
  non-null unique keys and fresh completed native update IDs.
- Saved online policy rejects null/missing and stale/future feature values before
  preprocessing. Raw saved pipeline/model-set guards cover pandas and Polars;
  native endpoint admission retains its existing pandas row-independent
  certificate and non-temporal scalar request boundary.
- Optional generated daily rollout/publication jobs start disabled and PAUSED.
  Existing batch-only projects retain their output. Jobs serialize writers,
  disable automatic optimization at task level and have no automatic retry.
- Public guides explain configuration, ownership, retry, freshness, native
  overrides and historical training-snapshot restrictions. No internal task or
  machine paths were added to public documentation.

## Review ledger

- [x] R01: pure policy, timing, HOLD/FAIL/COMPLETE, identity and replay validation.
- [x] R02: exact two-model endpoint admission and native GET/readiness contract.
- [x] R03: durable state, shared writer admission, unknown-write reconciliation.
- [x] R04: existing monitoring adapter and quality/health evidence integration.
- [x] R05: native publication, source/store/key checks and sync-status receipts.
- [x] R06: saved missing/freshness guard and pandas/Polars serialization parity.
- [x] R07: native feature envelope admission and exact key-only request schema.
- [x] R08: optional Bundle schema/templates, real notebook return contracts.
- [x] R09: guides, SDK/template links and changelog.
- [x] R10: affected local/native regression checks and static analysis.
- [x] R11: actual online refresh/stale/restore and capacity cleanup, fresh readback.
- [x] R12: independent review, source/evidence matching and bounded closure.

Independent reviews reproduced and closed a private helper import, a missing
required documentation argument, notebook dictionary/string return mismatch,
publication update-ID ambiguity, and several native readback mismatches.
Ponytail review retained separate policy/controller/storage responsibilities;
existing approval, metrics and native publication APIs are reused. The final
repair deletes a redundant telemetry-sink checker instead of adding an adapter.
Reports: `.cache/t218-final-review.md`, `t218-job-review.md`,
`t218-independent-review.md`, `t218-online-implementation-report.md`,
`t218-rollout-controller-report.md`, `t218-promotion-report.md`,
`t218-serving-template-report.md`, `t218-native-readback-fix.md`,
`t218-telemetry-fix.md` (final section supersedes its initial proposal).

## Final source identity and test evidence

Final package: `task218_native/final6/build_manifest.json`.

- Wheel SHA256: `0e2a3ec53cfff871a92004dbd7340c7173c55d680ef26d717f237b11152785e4`.
- Runtime digest: `ea6397144f010e3b26a95a901fd761de50e7be156fc61f6a4e3041e05933ecc3`.
- Tests ZIP SHA256: `16039f08410416765d08de0b4815c01c24db42e0959ad39f6d35e270ca32ddea`.
- All 539 runtime Python files match the final wheel manifest. Only
  `serving/endpoints.py` and `serving/rollout_endpoints.py` differ from the original
  immutable final2 endpoint model packages. Model prediction, online guard,
  publication, evidence and approval code are byte-identical.
- Final2 model package wheel: `fb2a86f068ae5668f7a6dca21e2a1985bd052ac2cdeaac0141b02af4dbfc516a`;
  runtime: `2dd57f267ef0e2d7ab805600ca117d41a9cc467bfed1e0c681fa560fbe765190`.
  Later controllers reuse previously admitted plans. Models were not falsely
  readmitted under a different source; source/certificate guards remain intact.

| Verification | Result | Evidence |
| --- | --- | --- |
| Frozen local affected consumer batch | 344 passed | `.cache/t218-final-focused.xml`, `.txt` |
| Actual notebook/config/template entrypoints | 15 passed | `.cache/t218-job-review.md`, corrected tests |
| Private API boundary after single-definition rename | 16 passed | Final boundary rerun; related 50 policy/binding passed |
| Real saved online models and policy | 48 passed | pandas/Polars pipeline/model-set serialization |
| Actual offline CLI/template generation, opted in | 11 passed | Required task-level `disable_auto_optimization` verified |
| Final6 local five-file direct consumer union | 197 passed, 8 expected warnings | `.cache/t218-telemetry-fix.md` |
| Final2 native 16-file regression | 496 passed, 6 explicit CLI skips | Job `307810894757104`; `final2/test_artifacts/tests/` |
| Final3 native endpoint/controller repair | 103 passed, no skips | Job `384217610269959` |
| Final4 native direct consumers | 183 passed, no skips | Job `141103492085475` |
| Final5 native pending-config repair | 118 passed, no skips | Job `116077791132826` |
| Final6 native five-file direct consumers | 197 passed, no skips, 42.307s | Job `997835652213496`; `final6/test_artifacts/tests/` |

Do not sum overlapping batches. Final6 covers `test_serving_endpoints.py`,
`test_serving_rollout_endpoints.py`, `test_serving_rollout.py`,
`test_online_feature_model.py` and `test_serving_sql_functions.py`.
The six native skips are CLI-only opt-in generation cases executed locally.

Whole-repository Ruff, full CI Ty (`backend skyulf-core/skyulf skyulf-core/tests
run_skyulf.py celery_worker.py`), Backend/Core CCN10 and whitespace checks passed
on final6. The 513-file integration/runtime format check passed. Strict MkDocs
returned 0 (`.cache/t218-docs-exit.txt`); the later prose-only native error-log
paragraph adds no links or code. Frontend was untouched. No full local suite or
coverage run was performed. Commit hooks are recorded in the commit checkpoint
below. No push was requested/performed.

Old job `583422954765999` had 479 passed/6 skipped persisted before notebook
communication finalization failed. Its recovered JUnit is kept under `final/`;
it is neither a successful job nor final-source evidence. Superseded preparation
runs were canceled. Do not describe this as a test-initialization failure.

## Actual native A/B and automatic champion proof

Host: `https://dbc-45604623-c18b.cloud.databricks.com`, profile `skyulf`.
Synthetic model: `workspace.skyulf_t218_fb080aba.promotion_final_model`.
Endpoint: `skyulf-t218-fb080aba-promotion-final`, ID
`6f77c302b50640d594d91236e0eff5a2` (deleted after completed verification).
Fixture MLflow run `0063567549f94af4b29708ea35c2d4e2`;
comparison SHA256 `afcc7924159275ea9d2a1200a1dd53bcf8964718dcdfea4a0d97836339052363`.

1. Actual native 0 -> 10 traffic update succeeded, config2/90-10. An unexpected
   native pending `start_time` was safely retained as PREPARED. Reconciliation
   verified and committed it without another PUT.
2. A real attributed challenger response exceeded a deliberately restrictive
   0.001ms test SLO. FAIL evidence triggered actual 10 -> 0 rollback, followed by
   verified config3/100-0 and ROLLED_BACK. Champion alias remained version1.
   Evidence: `final5/latency_failure.json`, `latency_rollback_result.json` and
   `final4/latency_reconcile_result.json`.
3. A new receipt run `0fda26de0b1c452bb6bb7e69b7fe49d0` reused the same pair after
   verified rollback. It reached 100%; eight actual routed challenger responses
   with synthetic labels met RMSE <= 1e-6. Guarded holdout re-evaluation promoted
   champion1 -> 2; a repeated promotion returned the identical receipt.
   Job `1058639195166473` SUCCESS; event
   `724c8464b4b548f9a1dc3d0ac810f920`.
   Evidence: `final5/promotion_complete_output.json`.
4. Fresh ready/100% readback and successful job state were checked before deleting
   this temporary endpoint. Fresh NotFound confirms removal:
   `final5/promotion_endpoint_cleanup.json`.

Native findings fixed with focused failing tests:

- Equivalent `served_entities`/deprecated `served_models` views and route name
  aliases are normalized only when every field agrees; bool/int remain distinct.
- Pending `start_time` is bounded integer metadata; unknown settings still fail.
- Native CREATE requires all three telemetry table names; GET retains only the
  active log sink. Final6 preserves mandatory CREATE fields and reuses the exact
  active inference-log/name/sampling/enabled-feature checks. The intermediate
  logs-only CREATE proposal was disproved by native execution and superseded.
- This workspace rejects explicit legacy AI Gateway mode for this endpoint type.
  Only the fixture explicitly selected supported telemetry before creation,
  preserving before/after evidence. No silent production fallback was added.

## Online native acceptance

Source: `workspace.skyulf_t218_fb080aba.features`, 32 synthetic rows, declared
NOT NULL primary key `entity`, CDF, `value` and `feature_time` DOUBLE columns.
UC object ID `ab894d5f-efba-4a9c-bff2-a4425f841917` is distinct from Delta ID
`13aa9447-cc8c-48f3-b5fd-f93ad61d837f`.
Online table `workspace.skyulf_t218_fb080aba.online_features`, store
`skyulf-t218-fb080aba`, pipeline `0a663335-cfeb-4012-aa18-70fc0bf610b8`.
Model `workspace.skyulf_t218_fb080aba.online_final_model` version1;
fixture MLflow run `45f6a7926824416080d5d69c9756b110`.
Endpoint `skyulf-t218-fb080aba-online-final`, ID
`734c0f9c046e490893ed535ae2abe9a3`.

Native baseline prediction passed. Missing entity was rejected by the exact
saved Skyulf non-null guard, proven in server logs. Native Feature Engineering
wraps it in a generic REST model-evaluation error. Initial test job
`412196415871536` failed its assumed client error text before source mutation;
`194313790636998` was canceled for a harness field correction. Production code
was unchanged. Replacement **`1008163388816411` finished SUCCESS** (task
`956049335016609`). It requires BadRequest plus a new matching server traceback;
generic HTTP failure alone cannot pass. Results:

| Scenario | Native result |
| --- | --- |
| Entity1, source value1 | Actual prediction2.9999999999999867, expected3 |
| Missing entity | Fresh saved-model non-null guard traceback; no prediction |
| Value11, fresh timestamp, publish/sync | Actual22.999999999999996, expected23 |
| Timestamp0, publish/sync | Fresh saved-model stale/future guard traceback; no prediction |
| Restore value1 with fresh timestamp, publish/sync | Actual2.9999999999999867, expected3 |

All successful responses identify `online_final_model-1`. Publication source
versions2/3/4 each have a distinct completed native update:
`ff06b1c3-b710-4cca-86a7-18b1aebb9c71`,
`258faad8-56a8-464f-912f-398492079d24`,
`82fdb808-c870-4d22-801e-8bd325bdff45`.
Evidence: `final2/online_workflow_guard_{request,run,output}.json`,
`final6/online_manifest.json`, MLflow `rehearsal/guard_logs/` artifacts.

## Capacity release and restoration

User-approved capacity release saved full configurations in
`capacity_release.json` before deleting the old test endpoint and the failed
owned A/B attempt. Only serving capacity resources were removed.

Original `skyulf-raw-d35721fd-v1` was recreated with the same model version1,
CPU Small/scale-to-zero settings and active inference-log destinations.
New ID `941edf74e4c146c5a00fef83169a636c`; READY/NOT_UPDATING. A real request
(age42, income50000, tenure5, segmentMass) returned prediction0 and probabilities
0.5610695615300878 / 0.4389304384699121. Evidence:
`preserved_existing_endpoint.json`, `restore_existing_request.json`,
`restored_existing_latest.json`, `restored_existing_prediction.json`.
The original resource ID/history was not recreated.

After online acceptance, fresh endpoint/table/store/pipeline identity and idle
publication checks passed. The owned online endpoint, online table and CU_1
store were deleted through their native APIs. The publication pipeline was
removed by native online-table cleanup. Every resource returned fresh NotFound;
receipt `final6/online_capacity_cleanup.json` includes full before snapshots.
Combined with the earlier A/B cleanup, no Task218 serving endpoint or online
store remains. Offline synthetic source/training tables, registered test models,
MLflow evidence and one-time job history remain for review; no production
schedule was enabled. Existing non-test resources were not broadly deleted.

Final independent API audit at `2026-10-10T20:54:52Z` confirms zero owned
endpoints, zero active Task218 jobs, absent online table/store/pipeline,
restored original endpoint READY/NOT_UPDATING and retained test champion2.
Seven offline data/log tables and six test registered-model names remain.
Full inventory and harness hashes: `final6/final_native_audit.json`.
Both fresh guard tracebacks were downloaded to `final6/guard_logs/`.
Final source manifest matched all 539 runtime files at acceptance on baseline
`48bee839`. No commit or push was made during that acceptance run.

## Commit checkpoint (2026-10-11)

The user requested committing Task218 before adding continuous streaming.
Pre-commit comparison confirms 539 runtime files match the native final6 wheel
manifest and all 676 Python tests in its ZIP match the working tree. No runtime
or test edits occurred after that validation, so passed pytest batches were not
repeated. Whole-repository Ruff passed; strict MkDocs returned0 after the final
operator paragraph (`.cache/t218-commit-docs-exit.txt`). The commit runs the
configured schema/whitespace/JSON/YAML/Ruff/format/full-Ty/CCN10 hooks.
Explicit staged pre-commit run passed every applicable hook; frontend hooks
correctly skipped because no frontend source changed. Staged diff check passed.
Task218's internal task record is included; temporary packages, cloud snapshots,
test caches and model artifacts are excluded. Streaming changes are separate.

## Acceptance boundaries

These are short functional canaries using a one-second stage, not 24-hour
production observation or native production telemetry/delayed-label acceptance.
The default policy remains 24 hours. No production rollout schedule was enabled.
SingleWriterAdmission still requires externally enforced exclusive ownership;
readback is not distributed CAS. Latest-value sync does not claim snapshot-exact
or exactly-once publication. Direct native REST callers can override enriched
features; saved guards validate supplied values, not their origin.
Automatic promotion supports single pipelines; original training_snapshot
reconstruction may block promotion after feature-source advancement. That guard
is not relaxed. There is no claim that every combined online/rollout recipe or
all feature/preprocessing configurations have native acceptance.

Platform references used:
- https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-multiple-models-to-serving-endpoint
- https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store
- https://docs.databricks.com/aws/en/machine-learning/feature-store/automatic-feature-lookup
