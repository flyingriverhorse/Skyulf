# SM-57 — Distributed pyfunc inference

Status: **DONE in the initial admitted scope** (2026-10-05). Implementation,
local checks, real UDF parity, all three generated lifecycles and no-op acceptance passed.
Worktree: `.worktrees/sm57`, branch `sm57-spark-inference`, base `d6e6585d`.
Plan: [166](166-sm57-spark-udf-plan.md). Safety inventory: [165](165-sm57-partition-safety-inventory.md).

## Delivered implementation

- Independent `inference_mode=local|spark`, with explicit pandas training for Spark
  inference; old configuration defaults to local. Spark uses the actual MLflow
  pyfunc UDF, named struct inputs and complete structured outputs.
- Fail-closed fitted-artifact gate for mean/constant SimpleImputer,
  StandardScaler, exact LinearRegression/LogisticRegression and reviewed tuner
  wrappers. Effective train-only skips are checked. Independent model sets are
  admitted; custom scoring, composition, temporal history and unknown nodes are
  rejected before publication.
- Exact source wheel and dependency pins accompany certified MLflow packages.
  Driver and worker validate saved state, signature, certificate and source hash.
  Nullable integer/boolean inputs are encoded before Arrow conversion; raw keys
  remain exact. An opt-in worker transport preserves SQL NULL outcome reasons.
- Distributed source validation and keyed predictions reuse existing Delta
  publication, receipt, CDF, recovery, source-correction and monitoring handoffs.
  No whole prediction population is collected on the driver. Training budgets
  and worker prediction batch size are independent of the scoring population.
- Generated single, competition and model-set projects expose the same route.
  User guides and generated operator docs explain the initial supported scope.

## Local verification

Tests run with the main checkout's virtual environment and `PYTHONPATH` pointing
at this worktree's `skyulf-core`. Temporary results are under ignored
`tmp_repro_artifacts/sm57`; full repository suites remain in CI.

- Baseline: 94 passed (execution capabilities, local artifact, local workflow).
- Gate: final 72 passed, including actual tuner regression/classification parity.
  Initial affected capability/registry/artifact union: 346 passed.
- Setup: 159 workflow/template tests passed; 22 SDK tests passed. Six real CLI
  init/strict-validate combinations passed; contradictory Spark/Polars init failed.
- Publication union: 162 passed in `test_spark_scoring.py`,
  `test_local_cdf_recovery.py`, `test_model_set_batch.py`,
  `test_model_set_source_changes.py`, `test_databricks_local_workflow.py`,
  `test_databricks_local_sdk.py`, and `test_model_set_project.py`.
- Independent review fixed GI-1 callable estimator overrides and GI-2 local SDK
  calls silently ignoring Spark selection. Eleven bounded rereview probes passed.
  SR-1 supported-scope documentation finding was closed by independent rereview.
- UR-1 model-set null reasons converted to literal `<NA>` by MLflow was reproduced
  using its actual installed converter, repaired and independently rechecked.
  Final MLflow consumer union after worker slicing: 55 passed, two local Spark
  tests skipped.
- Final independent integration review found no additional actionable defect;
  bounded multiclass model-set parity and receipt-identity probes passed.
- Full Ruff, formatting (1,346 files), Ty and CCN 10 passed after the UR-1 repair.
  `mkdocs build --strict` passed after user-guide changes.
- Actual local Spark tests skipped because local PySpark is absent; these are not
  evidence of real Spark execution. Databricks acceptance below is required.

## Isolated Databricks acceptance

Profile `skyulf`; CLI 1.17.0. New schema
`workspace.skyulf_sm57_20261005_a1`, workspace root
`/Workspace/Users/edwardwolfe99@gmail.com/skyulf-sm57-20261005-a1`.
Seed run `1051072804470735` succeeded: 160 training rows, 4,096 scoring rows,
four source partitions and CDF enabled. Local training limit is 500 rows;
distributed inference prediction batch size is 64, environment manager `local`
using the declared serverless Bundle task environment.
Databricks manages Arrow transport allocation separately.

| Layout | Train job | Score job | Monitoring job |
| --- | --- | --- | --- |
| Single | 405840346656296 | 811175016445828 | 185291625939783 |
| Competition | 733604897231626 | 625002612005468 | 781331399342724 |
| Model set | 671841462981787 | 519805788037772 | 1050459754065826 |

The projects reuse the shared monitoring store
`workspace.skyulf_sm23d_20261005_dc93e08c` and existing native dashboard
`01f1c090238e1b6da5d633032ad9960b`. Existing project resources are preserved.
Real UDF parity, generated publication, no-op replay, monitoring and dashboard
refresh passed after the two lifecycle repairs documented below.

### Live serverless correction

The first parity run `637718034912147`, task `731302980064584`, reproduced
`CONFIG_NOT_AVAILABLE` when reading `spark.sql.execution.arrow.maxRecordsPerBatch`.
The [Databricks serverless configuration allowlist](https://docs.databricks.com/aws/en/spark/conf)
does not permit configuring that property. The initial three train runs
`326621006030536`, `23791816526453` and `1112512627368950`, plus the parity run,
were canceled before accepting their results.

The implementation therefore uses `spark_udf_prediction_batch_rows` to slice
worker model calls, without reading or setting Arrow configuration. Incoming
Arrow allocation remains platform-managed and is not claimed to have this bound.
Both single and model-set paths reproduced the forbidden-setting defect in
focused tests, then passed after forwarding the worker limit. The deduplicated
publication/config/template union passed **323 tests** after the change.
Worker slicing passed the final 55-test MLflow consumer union (two local Spark
tests skipped), plus independent real `[3,3,1]` single/set scorer probes.

The isolated virtualenv packaging probe `36640884014088` failed at Databricks
SafeArchiveExtractor: MLflow 3.16.1's environment archive contains an absolute
Python-interpreter symlink. No archive security setting was disabled. A separate
explicit `env_manager=local` probe `524586272089618` passed worker/driver source
and dependency equality, real named UDF repeated regression parity, then exposed
only a fixture column-order mismatch at the separate portable API comparison.
That fixture was corrected to follow the portable API's existing order contract.
Final acceptance uses the declared serverless task environment, not driver-side
scoring or an implicit fallback. Policy-cluster template defaults retain the
`virtualenv` choice; its actual cloud validation is not
claimed by these serverless tests.

### Successful real UDF acceptance

Run `1006604118230130`, task `30615952321595`: **SUCCESS**. All three registered
fixtures (regression, string-label classification, nullable Int64/boolean) passed
independent local parity with shuffled input columns, nulls and exact keys above
`2**53`. Each used requested partitions 3/2 and model-call batches 3/64, repeating
the same distributed action twice. Regression/classification also matched the
portable `python_pipeline` API using its required input order. Nullable transport
was compared with the saved local artifact, outside that portable subset.
Physical UDF plans were recorded. Two actual fitted RobustScaler probes rejected
the unsupported artifact before source schema or rows were accessed.

The companion four-row/two-partition worker probe matched driver Python,
source hash and all ten measured package versions exactly:

| Runtime | Version |
| --- | --- |
| Python / Spark | 3.12.3 / 4.2.0 |
| Skyulf / MLflow | 0.9.1 / 3.16.1 |
| scikit-learn / pandas | 1.8.0 / 2.2.3 |
| NumPy / SciPy | 2.1.3 / 1.15.1 |
| Polars / joblib | 1.44.2 / 1.4.2 |
| PyArrow / Pydantic | 25.0.1 / 2.10.6 |

Source SHA256: `5e2fed28750cdeeb829cda4656f812b0c6c4c99230a8ffc4095924ef67a1b8c3`.
Wheel SHA256: `0e55549da5e196dd2afabca51ffc88a1457247280965096c14586623c4a383f4`.
The native dashboard refresh overlay was explicitly generated for all three
monitoring jobs with shared dashboard `01f1c090238e1b6da5d633032ad9960b` and
warehouse `d047a4d9aa276958`, then strictly validated and deployed.

### Live lifecycle budget repairs

The generated single score run `96215438589607` reached an older table-bootstrap
check that still applied the local 500-row cap. Model-set train
`904058806550788` reached a functional-approval reader that tried to collect the
complete 4,096-row inference population. Both defects were reproduced locally.

- Bootstrap checks the local row budget only in local mode. Existing admission,
  source identity, schema, CDF and distributed global key checks remain.
- Spark model-set approval certifies partition safety before source access,
  validates every source key on the pinned Spark snapshot, then takes a
  deterministic key-ordered functional probe bounded by the local budget. Every
  component must still produce a prediction. Complete saved holdout evaluation
  remains independent; actual publication scores every row with Spark.
- Workflow/Spark affected union: **89 passed**. Model-set project/release union:
  **22 passed**. Initial Windows default-temp permission errors were environmental;
  the explicit workspace basetemp rerun passed. Two Spark approval tests and two
  oversized bootstrap cases failed before their respective fixes.
- Independent scoped review passed three additional probes: global-key failure
  aborts before sampling, integer snapshot pinning is preserved, and implicit/
  explicit local approval still receives the full bounded source relation.
  No additional actionable findings.
- Full Ruff, formatting (1,346 files) and full CI Ty passed again after these
  final Core edits. Source wheel was rebuilt and all three Bundles redeployed.
  Fresh `_r2` model identities avoid mixing initial failed rehearsal evidence
  with the final package; sources, output destinations and job IDs are unchanged.
  Final train runs: single `707100261489642`, competition `426025910098660`,
  model set `433382604278522`; final package parity `431173550903318`.

Final package parity run `431173550903318`: **SUCCESS**. All original regression,
classification, nullable/large-key, repeated-action and unsupported-node checks
passed again, with exact worker/driver runtime equality. This supersedes the
previous source hash as final-artifact evidence.

Final source SHA256: `d6a8ca3a2448682e51136800b519366edd150b21e1e6ba2c5cf5e139ede4ec1b`.
Final wheel SHA256: `038f4b329a1dea73bc522d6ad35cc74a80c6ab1bc72e832ed0e1cbb2eaff4b41`.
Full CI complexity and strict documentation build passed after the lifecycle
repairs. Pre-commit also passed all applicable hooks on the final source.

### Generated lifecycle progress

Single train `707100261489642`, child score `1091418275073336` and monitoring
`303956752536131`: **SUCCESS**, including native dashboard refresh.
Single committed 4,096 inputs/outputs to Delta version 1 with an explicit Spark
receipt, model version 1 and prediction batch 64.
Competition train `426025910098660`, child score `506249927872628` and monitoring
`466090344081351`: **SUCCESS**, including native dashboard refresh.
Model-set train `433382604278522` passed complete quality evaluation, approval
and monitor enrollment. Child score `872987644544739` wrote
4,096 rows; monitoring `812521997480072` and its native dashboard refresh passed.
The parent train run is also **SUCCESS**. Competition also wrote 4,096
rows. All three publication receipts report Delta commit version 1. Independent
full-table comparison run `995128019062420`: **SUCCESS**. Each table contains
4,096 unique non-null keys at Delta version 1, and every prediction/probability
column matches the separately loaded local artifact exactly (maximum absolute
error 0.0). Every provenance column and receipt hash matched. Both model-set
exclusion-reason columns contain 4,096 SQL NULL values and zero literal null
strings; every branch has 4,096 predicted statuses.

No-op score reruns: single `470174856265625`, competition `510306103901695`,
model set `1069330558499886`. Single and competition completed **SUCCESS**:
`noop=true`, zero inputs/outputs, commit version still 1. Their monitoring runs
`323256275563473` and `988244140261261` also passed native dashboard refresh.
Model-set score replay also returned `noop=true`, zero inputs/outputs and commit
version 1. Its monitoring run `176391483291396`, native dashboard refresh and
parent score run `1069330558499886` also completed **SUCCESS**. Final comparison
`124335514146443`: **SUCCESS**, `noop_baseline_compared=true`. All three source
and output identities, versions, row counts, table hashes and receipt hashes
match the first successful publication exactly.

Final unchanged output fingerprints:

| Layout | Delta version | Rows | Table SHA256 |
| --- | --- | --- | --- |
| competition | 1 | 4096 | `2897ec3eeef9d781009a44e665bcdf71f7282c1487567a40b7173a47ddcf0075` |
| model_set | 1 | 4096 | `d58115b1c481ada82193f40bfe4fced04a796236533a4f1edac96dfdd39d79e8` |
| single | 1 | 4096 | `653aab54037efbda6e6a1cf5e15725089f793c0e5c820580ca86f191e1c8b131` |

## Closure and boundaries

All three initial train-to-score chains, their monitoring/native-refresh jobs,
and all three no-op score/monitoring/native-refresh chains passed. The first
monitoring outputs reported four healthy contexts (single, competition, two set
components), zero failed and zero disabled, in the shared monitoring store.

The final template test file passed **69 tests**, including all four local/Spark
and serverless/policy-cluster environment combinations. Two fresh real CLI
initializations and strict validations also passed for the environment correction.
Final affected repair tests were `test_databricks_local_workflow.py` plus
`test_spark_scoring.py` (89 passed), and `test_model_set_project.py` plus
`test_model_set_auto_release.py` (22 passed), using `pytest -q --no-cov` with an
explicit workspace basetemp. Full CI Ruff/format/Ty/CCN and strict docs passed;
all applicable pre-commit hooks passed. No frontend source changed.

This is correctness/lifecycle acceptance on serverless with explicit local worker
environment reuse, not a large-data benchmark. Policy-cluster virtualenv execution
has not been cloud-tested. SM-58 owns throughput, executor memory and model-load
benchmarks; SM-56 still owns broader node/codec support. Existing local projects
were not silently migrated. Earlier failed isolated rehearsal runs remain audit
history; the successful final model identities use `_r2`.

This closure record accompanies the local DCO commit; no remote push is requested.
