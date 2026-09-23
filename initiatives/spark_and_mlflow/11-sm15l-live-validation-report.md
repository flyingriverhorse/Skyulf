# SM-15L local-engine monthly UC Delta publication

Status: completed for the documented small-data scope on 2026-09-23.
This report records the specific tested runtime and data; it does not claim
arbitrary local artifacts are Spark-compatible.

## Contract

`run_local_batch` reads one pinned source Delta snapshot and month under
`LocalSourceSpec` row/byte limits. It applies the saved pandas or Polars fitted
pipeline in one Python process. Only final keyed predictions become a Spark
DataFrame, with the precreated target's explicit output types. The existing
`publish_replace_period` transaction writes exactly the requested period
using `BatchSpec(mode="local_pipeline")`, source/model digest, expected target
version and shared admission. Spark handles UC I/O only; it does not perform
feature engineering or model inference in this path. No SQL Connector is used.

## Local evidence

- Focused SDK/reader tests: 32 passed, 1 optional Spark skip; batch-contract
  tests: 25 passed.
- Real local Spark/Delta test in Linux: 6 passed in 49.92 seconds, including
  admission denial, invalid target schema, and stale model/source pins with
  unchanged target.
- The direct local reader rejects null/duplicate keys, source or final-result
  memory over-budget, and accidental empty periods. The publisher rejects
  mismatched source/model/period/code/target identities before writing.
- The shared SM-15 writer's real Delta tests separately cover invalid rows,
  held admission and denied publication without target mutation.
  [SM-16 platform evidence](PLATFORM_VALIDATION.md) records live independent
  writer contention and target-write permission denial for this same writer
  and admission implementation. SM-15L reuses those components unchanged.
  The uploaded wheel predates a later type-only empty-frame constructor change
  and SDK docstring correction; both live months exercised the unchanged
  nonempty scoring and publication path.

## Isolated Databricks resources

- Profile: `skyulf`; schema: `workspace.skyulf_sm24a_20260923`.
- Existing source fixture: `skyulf_sm24a_score_source` at Delta version 0,
  with 80 January and 80 February rows.
- Test-owned source: `skyulf_sm15l_source_r1`; test-owned output:
  `skyulf_sm15l_predictions_r1`; test-owned control:
  `skyulf_sm15l_admission_r1`.
- Test notebook and wheel:
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/sm15l_r1`.
  Uploaded wheel SHA-256:
  `d6b38c52f3a52ebf7f15336561abea850c667acc90253d68ac13b47e2e1ac952`.
  Both one-time jobs declare a 900-second timeout and zero retries.
- January pandas LinearRegression model: pinned UC
  `workspace.skyulf_sm24a_20260923.skyulf_sm24a_metrics_r1` version 1,
  digest `e4281f2feb9349b2dd745bcdd894128c47d434f0bd8779eb42d474adb8aed662`.
- February Polars RandomForestRegressor model: pinned UC
  `workspace.skyulf_sm24a_20260923.skyulf_sm24a_metrics_r2` version 1,
  digest `efb2174b158d5702be9cc741be3f790721cc43b419f6bc3d0fe3891413f40490`.

## Monthly job observations

| Stage | Parent/task run | Source version | Target version | Result |
| --- | --- | ---: | ---: | --- |
| January | `1055645331479835` / `486320682365717` | 0 | 1 | SUCCESS: 80 keyed output rows equal direct local gold; no replay |
| February | `105734608770943` / `850744412284306` | 1 | 2 | SUCCESS: 80 keyed output rows equal direct Polars gold; 80 January rows and metadata unchanged; exact replay kept version 2; stale request rejected |

The February job appended 80 source rows to version 1 and published only
those predictions. Its live assertions compared all persisted values to direct
local gold by key, checked the 80 January rows and metadata for exact equality,
verified 160 total target rows, replayed without another target version, and
rejected a new logical request with the old expected version. The two-month
scenario intentionally uses two already registered local models to exercise
both pandas and Polars publication. SM-20a will separately test one fixed
pinned model across its generated Bundle's two monthly jobs. No Bundle was
generated or deployed in SM-15L.
