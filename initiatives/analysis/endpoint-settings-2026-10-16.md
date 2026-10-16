# Configurable serving endpoints — 2026-10-16

Status: implementation, local checks, native Databricks proof and independent
reviews complete. The user authorized this endpoint-settings commit on
2026-10-16. Branch `0941`, parent `4bc3f6d8`; no push or G7 work.

## Authorized checkpoint

The user's instruction was “ilk bunlari commit et sonra ekle”. Completed
previous work was committed before this change:

- `1b8b3a1e`: coverage, exclusion and invalid training-row policies.
- `4bc3f6d8`: the four accompanying user guides.

Their applicable pre-commit gates passed. The endpoint settings below were
first left uncommitted for review, then explicitly approved for commit.
Prior G5 cloud resources were subsequently removed by Claude under the user's
separate cleanup instruction (M465); G5 run evidence remains in the existing
`g5-implementation-2026-10-16.md` report and ignored local records. The G5 guide
claims were released in M464 before commit `4bc3f6d8`.

## Behavior

| Field | Default | Validation |
| --- | --- | --- |
| `workload_type` | `CPU` | `CPU`, `CPU_MEDIUM`, `CPU_LARGE` |
| `workload_size` | `Small` | `Small`, `Medium`, `Large` |
| `scale_to_zero_enabled` | `true` | Strict boolean |
| `ready_timeout_s` | `3600` | Strict positive integer, excluding boolean |

All four belong to `PinnedEndpointSpec` and `online_serving.endpoint` in
`config/serving.yml`. Hidden answer-file keys prefix each name with `serving_`.
Initialization answers are strings; rendering produces typed YAML values.

Creation refuses an existing name, sends exactly one create request, and waits
for the exact model, compute and logging configuration to reach `READY` plus
`NOT_UPDATING`. Setup reports `READY` after that check. Failed native updates,
configuration mismatch and API failures propagate. A timeout does not delete,
overwrite or recreate anything. `wait_for_pinned_endpoint_ready` can resume an
explicit wait; request and SQL paths retain immediate readiness checks.

The existing outer setup job budget (`monitoring_timeout_seconds`, default
7200) also controls its task. This leaves preparation time around the default
3600-second wait. Operators increasing the wait must also allow sufficient
total job time. Other monitoring/publication task budgets are unchanged.

Rollout requires identical compute settings for both versions and rejects
mutable config values that differ from their admitted specs. The exact served
entity allowlist remains in place. Historical default plans can omit the four
new fields: only those fields are normalized during composition checking,
after the original durable plan hash is checked. Caller input is not mutated.

## Local verification

Commands use `uv run --no-sync`; pytest uses explicit files, not a full suite.

| Scope | Result |
| --- | --- |
| Initial regression selection | 34 failures, 1 pass before implementation |
| `test_serving_endpoints.py`, `test_serving_rollout_endpoints.py`, `test_serving_job.py`, `test_serving_sql_functions.py`, `test_serving_rollout.py` | 251 passed |
| Additional job override forwarding and timeout propagation | 4 passed, including 2 new cases; no production changes after the preceding union |
| `test_online_feature_model.py`, `test_probed_builtin_steps.py` | 89 passed; 10 existing fixture/model-signature warnings |
| Setup task budget regression | Failed before template correction |
| Final `test_serving_jobs_template.py` and `test_databricks_render_snapshots.py` with offline CLI opt-in | 39 passed; 14 snapshots checked and unchanged |
| Generated schema and six prepared examples | Checks passed; only online example serving config/resource changed |
| `test_databricks_documentation_contracts.py` | 10 passed |
| Full Ruff, CI Ty scope and Lizard CCN10 | Passed |
| Ruff format on CI scope | 1773 files already formatted |
| Commit gate: `mkdocs build --strict` into the ignored proof directory | Passed, 9.86 seconds |

The initial template test invocation skipped CLI cases without the opt-in;
the later explicit offline invocation executed them. An attempted consumer
command named a nonexistent test file and collected nothing; the corrected
explicit command produced the 251 passes above. Neither attempt is counted as
successful test evidence.
Full repository pytest was not run; CI keeps that gate. The changed guide
passed the documentation contract tests and a fresh strict documentation build
before committing. Fresh full Ruff/format/Ty/Lizard checks also passed.

## Native Databricks proof

- Host: `https://dbc-45604623-c18b.cloud.databricks.com`, profile `skyulf`.
- Separate namespace/folder: `endpoint_settings_20261016`; deleted G5 resources
  are not reused.
- Wheel: `0.9.3+endpoint.20261016r1`.
- SHA256: `f9f1e729cbfdee910242638e09ed8e41643e4df38f428de78636c9d02a9a8a11`.
- Test ZIP SHA256: `d87cb9ded509c27990d26a14f6d55ff529e379da22f78d5c1968526bb6c7f34b`.
- Run `263379170668649`, task `75769796778761`.
- First attempt: **162 tests passed, zero failures/errors/skips** on the installed
  wheel. Real diabetes training and registration produced model version 1,
  MLflow run `4d1b8d62260c47cc85a14f4ff00c6fc1`.
- The workspace rejected `scale_to_zero_enabled=false` with
  `InvalidParameterValue: Scale to zero must be enabled for this workspace`.
  This proves forwarding and native error propagation, not successful warm
  endpoint provisioning. No endpoint was created. The platform started an
  automatic serverless retry despite task `max_retries=0`; root cancelled the
  owned run and disabled auto-optimization on the completion task.
- Completion run `816620858652267`, task `551873165581781`, reuses the same
  wheel and registered model without repeating the 162 passing tests. It uses
  the workspace-supported `scale_to_zero_enabled=true` explicitly in the
  verification script; production has no fallback or silent setting change.
- Native timeout proof passed: `ready_timeout_s=1` raised after 1.95 seconds
  including the create API call. A fresh read found the same endpoint still
  present and `NOT_READY/IN_PROGRESS`; explicit waiting then continued without
  another create. Durable progress: MLflow `5c95119706e64de4a40c60a99a797161`,
  local `progress.json`.
- Completion run and task both **SUCCESS**. The endpoint became
  `READY/NOT_UPDATING` after 543.65 seconds including creation. Exact native
  readback confirmed CPU/Small, scale-to-zero true and the pinned model version.
  All five real REST predictions matched saved-model inference at
  `rtol=1e-9`, `atol=1e-12`, with exact row count and column order.
- Cleanup verified: one temporary endpoint deleted and freshly `NotFound`;
  one owned registered model and its schema removed; one experiment soft-deleted;
  one owned workspace folder removed. This proof created no Delta tables or
  persistent scheduled jobs. Ad-hoc run history and local result records remain.
- The final 475 source-file hashes still matched the tested wheel before cleanup.

Raw records, source hashes and scripts are under the ignored local directory
`.superpowers/sdd/endpoint-settings-2026-10-16/`: `first-result.json`,
`progress.json`, `final-output.json`, `final-result.json`, `cleanup-inventory.json`
and `cleanup.json`. The two MLflow result artifacts were downloaded before
cloud cleanup. There is no claim that the initial scale-to-zero-false run passed;
its native rejection and the successful completion run are distinct evidence.

## Review and limits

Ponytail: no new dependency or configuration framework. Existing SDK transports,
immutable spec, template generator and budget variable are reused. Two small
compute validators keep existing complexity limits without widening admission.

Independent Codex review and Claude M473 review found no blockers. Additional
independent probes passed all nine SDK compute combinations, JSON rollout round
trips, per-role timeout differences, four boolean config tampering cases, legacy
defaults, unrelated missing-field rejection and original durable hash rejection.

Review caveats: readiness is a polling budget, not an interruptible HTTP deadline.
A read already in progress can finish after it, including a ready response;
the guide now states this explicitly. Installed SDK transport supplies its own
retry and HTTP timeouts, so no second retry framework is added. The existing
monitoring-named job budget is retained to avoid another variable; the guide
names it and explains its use for setup. GPU support, automatic
updates of an existing endpoint and new NYC examples are not part of this task.
Config/SDK tests cover supported CPU types and sizes; the native proof uses
only CPU/Small and must not be described as live verification of every tier.
This change does not alter rollout scheduling, promotion policy or feature lookup.

Platform references checked for the supported compute/readiness contract:
[custom model serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models)
and [endpoint configuration](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints).
