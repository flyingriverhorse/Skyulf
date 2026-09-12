# 0.8.22 pipeline verification — 2026-09-13

Actual pipelines exercise the refactored code together, with numeric assertions
over learned predictions, saved artifacts, preview output and persisted drift
reports. The experiments found one pre-existing serving defect, now OC-320.

## Scenarios and results

Every scenario runs under pandas and Polars, with both Grid and Halving Grid.
These are controlled synthetic datasets for correctness checks, not an estimate
of model accuracy on real customer data.

| Scenario | Data and execution | Result |
|---|---|---|
| Mixed customer classification | 300 rows; 240 train / 60 test; age casting → split → median imputation → log transform → scaling → OneHot city → TF-IDF note → tuned Random Forest | CV accuracy 1.0, held-out accuracy 1.0; all 60 raw held-out rows predicted correctly in each of four combinations |
| Saved model serving | Actual temporary SQLite job and deployment; `DeploymentService.predict` loads the trained file; reload and 0.5/0.5 threshold override | Identical predictions; bundle digest unchanged |
| Data preview and inspection | Actual preview partitioner/runner, separate artifact store, real source CSV | 300 source rows, 240/60 feature and label rows; scaler output receipts and path provenance retained; training excluded from preview |
| Drift | Actual drift upload parsing, saved raw source, job lookup and SQLite alert persistence | Same data: zero drifted columns; `usage + 200`: exactly one drifted column (`usage`); distinct persisted alert IDs |
| Three-branch classification | Split → numeric imputation/log/scaling, city OneHot, note TF-IDF → merge into Random Forest | Training, CV and preview pass; 10 model features retained; serving fails as OC-320 below |
| Standalone Core regression | 240 rows, `y = 3x² + 5`; casting → split → square → scaling → Ridge tuning → save/load → model card | All four combinations satisfy the numeric formula and retain identical predictions/fingerprints after reload |

For the Core example, `x = [3, 9, 21]` has independent expected outputs
`[32, 248, 1328]`. The pandas/Grid run produced
`[32.0000000286, 248.0000000166, 1327.9999999566]`; all combinations are checked
with `rtol=1e-6, atol=1e-6`. Reloaded predictions must match exactly.

The scaler spy calls the real implementation and records the actual unique
usage values seen during fitting. Every fit must use a subset of the 240 outer
training rows, and at least one inner fit must use fewer than 240 rows. This is
checked independently of the pipeline's `fold_refit_audit` metrics.
Observed Grid fits used 160 rows per inner fold; Halving Grid used 53 rows in
its early rounds and 160 in later folds. Full-training fits used 240 rows.
None contained an outer test observation.

## OC-320: existing branched serving failure

The three branches share the same split. The numeric branch explicitly drops
the original city/note columns; the other branches generate their OneHot and
TF-IDF features. Merging into the training node succeeds and retains all ten
model features.

The saved bundle also records city/note as explicitly dropped columns.
`DeploymentService._predict_with_bundled_artifact` removes them from raw input
before calling the fitted feature engineer. Consequently its encoders cannot
recreate seven generated features, and serving rejects the resulting three
numeric columns with `missing after transform`.

This is a prediction failure with an explicit exception. It was reproduced in
all four current engine/search combinations and independently using production
sources exported from pre-refactor commit **e1bc7b44** (pandas/Grid). The current
working tree was not replaced. This work records the defect and its regression
test; it does not change production serving behavior.

The four serving tests have `xfail(strict=True, raises=ValueError)` with assertions
on the missing encoded feature names. They are reported as expected failures,
never as successful inference. A different error type/message or an unexpected
pass fails the check. OC-246, the separate old local-only smoke-test problem,
remains open.

## Verification and artifacts

| Check | Result |
|---|---|
| `tests/integration/test_ccn_release_pipeline.py` | 8 passed, 4 expected failures (one OC-320 defect × four combinations) |
| `skyulf-core/tests/integration/test_release_tuning_roundtrip.py` | 4 passed |
| Ruff check, Ruff format check and ty on both added test files | Passed |

Run the suites separately from the repository root:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe -m pytest tests/integration/test_ccn_release_pipeline.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/test_release_tuning_roundtrip.py -q -p no:cacheprovider
```

Ignored local evidence is in `tmp_repro_artifacts/ccn_pipeline_verified/` and
`tmp_repro_artifacts/ccn_core_roundtrip/`. Each customer run saves its CSV,
execution graph, model, execution log, Mermaid diagram and training summary.
The full serving scenarios also save held-out predictions, preview inspection
receipts, drift reports and an isolated SQLite database. Core runs save their
pipeline pickle, model card and expected/actual predictions.

Logs: `ccn_pipeline_verified.log`, `ccn_core_roundtrip.log`, and
`ccn_pipeline_baseline.log` under `tmp_repro_artifacts/`.

The tests run actual Core/backend computations and service functions. Catalog
construction and monitoring artifact discovery are directed to test-local
storage; learning, prediction, preview execution, drift metrics and database
persistence are not mocked. This does not exercise browser interaction,
Celery/Redis workers or a live production deployment.

The live audit queue is now **59 open / 4 parked**, including OC-320.
