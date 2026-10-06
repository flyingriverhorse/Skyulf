# SM-58: Spark inference scale and memory validation

Status: PARTIAL — full serverless matrix passed; classic-compute acceptance blocked.
Base revision: `e35632f2` (SM-57).

## Scope and method

Compare the existing `native_features`, `python_pipeline` and certified MLflow
pyfunc routes using one fitted pandas SimpleImputer/StandardScaler/LinearRegression
pipeline. Generate deterministic raw data in Spark, including missing features.
Use 1,000,000 and 5,000,000 rows, multiple partition and model-call batch sizes,
and repeat actions. Verify predictions against an independent SQL expression for
the fitted model over every row; `count()` alone is not an inference benchmark.
Report preparation, first action and repeat action separately. Do not claim that
repeat actions guarantee model reuse or that model-call batches bound Arrow RAM.

The original plan used an isolated, finite Databricks job with two workers and
the selected `skyulf` profile. The workspace restriction below required managed
serverless execution instead; no fixed worker-count claim applies to that run.
Record cluster/runtime, source/package/model identities, actual partition
counts, task results, throughput, executor process-tree memory where available,
and a separate worker model-load probe. Never substitute driver RSS for executor
RSS. Missing metrics remain missing rather than zero. Retain raw evidence outside
version control and summarize observed results here.

Production behavior is unchanged unless the benchmark exposes a reproduced bug.
No universal row-limit or default tuning change will be inferred from a small
linear estimator. Unsupported preprocessing remains governed by SM-56/57.

## File map

- `skyulf-core/benchmarks/bench_spark_inference.py`: executable benchmark notebook.
- `skyulf-core/tests/unit/test_spark_inference_benchmark.py`: measurement validity.
- `docs/user_guide/spark.md`, `databricks_bundle.md`: measured operator guidance.
- This record and `OPEN_QUEUE_updated.md`: evidence and acceptance status.

## Acceptance checklist

- [x] Measurement checks reject incomplete or incorrect predictions.
- [x] All three routes execute on distributed data above local caps (serverless).
- [x] First/repeat timings, batch/partition matrix and correctness recorded.
- [x] Separate worker model-load and process RSS observations recorded.
- [ ] Full executor memory observations on classic compute.
- [ ] Policy-cluster environment/wheel packaging exercised.
- [x] Focused tests, full CI static scopes, review and final limitations recorded.

## Environment discovery and review

CLI 1.17.0, previously selected profile `skyulf`; no existing classic clusters.
Creating a finite two-worker `i3.xlarge`, DBR 17.3 LTS dedicated job through
Job Compute policy `0012C0EE65D441E4` returned:
`Only serverless compute is supported in the workspace.` No classic cluster or
classic job was created. This is a platform restriction, not a scoring failure.

The supported serverless adaptation uses the existing SM-57 exact source wheel,
MLflow 3.16.1 and declared worker dependencies with `env_manager=local`.
Arrow transport remains platform-managed; executor statusStore is unavailable.
Isolated job: `1024399528200149`; first full run: `465986663682428`, task
`1076130538560776`. Only own schema `workspace.skyulf_sm58_20261005`, experiment
`/Shared/skyulf-sm58-20261005`, registered models and benchmark volume are created.
No existing training, scoring, monitoring or dashboard resource is changed.

Independent Codex review found two benchmark reporting defects, reproduced and
fixed: a failed second repeat lost earlier action evidence; disabled process-tree
metrics could look like zero memory. The final runner checkpoints actions/errors
and preserves job failure, and emits null for disabled/unobserved process metrics.
Nineteen focused tests passed after red/green reproduction, as did full Ruff and
the full CI Ty scope. The initial cloud run uses the pre-review notebook snapshot;
its successful action timings remain valid, but it does not validate these later
reporting fixes. The final-runner smoke pass below validates that integration.

Timing includes distributed source generation, model evaluation and the full-row
correctness aggregate. It excludes route preparation (reported separately), model
training/registration, dependency setup, and Delta publication. Routes run in a
fixed order; the second action is not proof of model reuse. No output is explicitly
cached, and serverless resource allocation/cache effects are platform-managed.

## Sources

- [Spark monitoring](https://spark.apache.org/docs/latest/monitoring.html):
  process-tree metrics require `spark.executor.processTreeMetrics.enabled`.
- [MLflow Spark UDF API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf):
  environment recreation and serverless sandbox limits are compute-specific.

## Full serverless results (2026-10-05)

Run `465986663682428`, task `1076130538560776`: **SUCCESS**.
18 configurations, two actions each, **108,000,000 scored row evaluations**.
All row counts, unique/non-null identities, finite predictions and independent
prediction expectations passed. Maximum absolute error: 1.794120407794253e-13.
Actual source partitions matched every requested 8/32 setting.

Python 3.12.3, Spark 4.2.0, MLflow 3.16.1, sklearn 1.8.0, pandas 2.2.3,
NumPy 2.1.3, PyArrow 25.0.1, Skyulf Core 0.9.1.
Exact runtime source SHA256: `d6a8ca3a2448682e51136800b519366edd150b21e1e6ba2c5cf5e139ede4ec1b`.
Full raw report: `/Volumes/workspace/skyulf_sm58_20261005/benchmark/ef875e95e0c24e87a64f38a314a5c7fa.json`.
Report byte SHA256: `d5c917c8491306bde500386bf2cabb86d004ef7a125fe721091f7333c47a3403`.

| Features | Rows | Partitions | Model batch | Route | Prepare s | First s | Repeat s | Repeat rows/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1,000,000 | 8 | 10,000 | native_features | 7.25 | 31.38 | 2.04 | 490,416 |
| 2 | 1,000,000 | 8 | 10,000 | python_pipeline | 1.08 | 1.87 | 1.88 | 533,256 |
| 2 | 1,000,000 | 8 | 10,000 | pyfunc | 45.30 | 43.78 | 2.74 | 365,325 |
| 2 | 5,000,000 | 8 | 10,000 | native_features | 5.69 | 3.66 | 3.31 | 1,509,046 |
| 2 | 5,000,000 | 8 | 10,000 | python_pipeline | 1.02 | 3.24 | 3.15 | 1,587,622 |
| 2 | 5,000,000 | 8 | 10,000 | pyfunc | 52.51 | 4.02 | 3.95 | 1,265,644 |
| 2 | 5,000,000 | 32 | 10,000 | native_features | 5.51 | 4.98 | 4.81 | 1,039,335 |
| 2 | 5,000,000 | 32 | 10,000 | python_pipeline | 0.86 | 5.01 | 4.90 | 1,020,537 |
| 2 | 5,000,000 | 32 | 10,000 | pyfunc | 49.18 | 9.80 | 10.07 | 496,705 |
| 2 | 5,000,000 | 32 | 1,000 | native_features | 5.64 | 7.46 | 6.80 | 734,884 |
| 2 | 5,000,000 | 32 | 1,000 | python_pipeline | 0.81 | 6.40 | 6.63 | 754,414 |
| 2 | 5,000,000 | 32 | 1,000 | pyfunc | 48.45 | 12.10 | 12.23 | 408,720 |
| 64 | 1,000,000 | 32 | 10,000 | native_features | 9.20 | 8.79 | 8.35 | 119,792 |
| 64 | 1,000,000 | 32 | 10,000 | python_pipeline | 1.41 | 8.45 | 8.07 | 123,930 |
| 64 | 1,000,000 | 32 | 10,000 | pyfunc | 49.43 | 16.03 | 15.61 | 64,066 |
| 64 | 1,000,000 | 32 | 1,000 | native_features | 9.12 | 8.62 | 8.72 | 114,691 |
| 64 | 1,000,000 | 32 | 1,000 | python_pipeline | 1.39 | 8.68 | 8.94 | 111,812 |
| 64 | 1,000,000 | 32 | 1,000 | pyfunc | 43.64 | 18.96 | 19.00 | 52,633 |

### Separate worker package-load and memory probes

Eight partition observations per width/batch combination; not eight guaranteed
distinct hosts/processes. The environment/imports are already warm. Timed load
starts after archive extraction and excludes network transfer and dependency
installation. RSS is whole Python-process resident memory, not model size.
Peak RSS is the process lifetime high-water mark and can include earlier work.

| Features | Probe rows | Median load s | Before RSS MiB | After prediction RSS MiB | Lifetime peak max MiB |
| --- | --- | --- | --- | --- | --- |
| 2 | 1,000 | 1.728 | 643.9–645.5 | 645.6–647.4 | 903.3 |
| 2 | 10,000 | 1.728 | 644.0–645.6 | 645.6–647.4 | 903.3 |
| 64 | 1,000 | 2.224 | 652.2–654.1 | 655.8–658.6 | 912.4 |
| 64 | 10,000 | 2.135 | 652.2–654.1 | 662.6–665.1 | 912.4 |

### Operator conclusions and limits

- The scoring population exceeds the local training caps; all three existing
  distributed routes completed 5M rows without collecting that population.
- Keep 10,000 model rows as the existing starting value for this small linear
  workload. A 1,000-row call increased overhead; this is not a safe default
  claim for large models or different feature widths.
- More partitions did not help this workload: 32 was slower than 8 at 5M rows.
  Measure against available compute rather than increasing partitions blindly.
- Pyfunc preparation took 43.64–52.51 s, including package acquisition,
  certificate verification and UDF setup. These measurements do not isolate
  which substep dominates. Do not describe the 3.95 s repeat action as total
  score-job latency: that case also had 52.51 s preparation before its actions.
- Native FE and Python preprocessing are close on this synthetic fixture.
  This does not establish a hot FE node or justify SM-59 implementation yet.
- The fixed route order, two actions, managed serverless compute, synthetic
  linear estimator and correctness-aggregate overhead limit generalization.
  No controlled cold-start, concurrency, large-model or full Delta-publication
  throughput claim is made. Single/competition/model-set lifecycle correctness
  remains the separate SM-57 acceptance.
- The serverless worker wheel is exercised, but full executor RSS, compute
  sizing and policy-cluster virtualenv remain blocked on a classic-enabled
  workspace. **Do not mark the full SM-58 contract DONE.**

## Final runner verification and handoff

Final notebook smoke run `547865259868912`, task `1060752823519926`:
**SUCCESS**. All three routes completed two actions on 10,000 rows, and the
separate worker-load probe passed. The report recorded intermediate active and
completed action stages before its final `complete=true`. Failure/re-raise and
missing-metric behavior are covered by the focused local tests.
Smoke report: `/Volumes/workspace/skyulf_sm58_20261005/benchmark/1061840de11a491d950b4280af995466.json`.
Final notebook SHA256: `d544a64f6b1892df4e9403664e3439ad11ee429719c708da81b9ff946dd147e1`.

Validation on the final runner:

- `pytest skyulf-core/tests/unit/test_spark_inference_benchmark.py -q --no-cov`
  with an explicit workspace basetemp: **19 passed**. The initial measurement
  contract failed before implementation; both review repairs have red/green
  evidence. No unrelated suites were run.
- Full `ruff check .`, CI formatting scope (1,348 files), full
  `ty check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py`
  and `lizard backend skyulf-core/skyulf --CCN 10 -w`: passed.
- `mkdocs build --strict` for the changed user guides: passed.
- Applicable pre-commit hooks passed. Frontend hooks were correctly skipped;
  no frontend, production Core or backend code changed.
- Independent Codex review verified reporting repairs, every saved result table,
  coverage totals, hashes and the explicit classic-compute boundary. Minor stale
  queue/scope wording was corrected. Local llama.cpp review was treated as draft;
  its inference that reusing an uncached Spark DataFrame caches results was rejected.

The visible job `skyulf_sm58_scale_memory_20261005` has no schedule, one concurrent
run, no retries, a 5,400-second job timeout and `suite=full` by default. Use its
`suite=smoke` job parameter for a small rerun. Both executed runs are terminal;
the job and own model/report artifacts remain available for inspection.

Remaining acceptance requires a workspace that permits classic job compute:
repeat with fixed worker resources, observe complete executor process-tree RSS,
and validate the policy-cluster `virtualenv` package. This environment cannot
satisfy those items. No worker-memory ceiling, universal batch/partition optimum
or SM-59 hot-node finding is inferred. The delivery is a local DCO commit on
`092`; no remote push is requested.
