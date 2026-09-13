# 0.8.23 second queue batch — verification

Verified on 2026-09-13 after committing the preceding thirteen-finding batch
as `0ca10968` with DCO sign-off and passing hooks. This batch closes twelve
findings, reducing the live queue from **46 open / 4 parked** to
**34 open / 4 parked**. Nine belong to Qwen OC-271–318; **39/48** of those
findings are now closed. OC-71/72/73/185 remain parked, and OC-319 remains open.

## Changes and examples

| Findings | Previous behavior | Verified result |
|---|---|---|
| OC-227/315 | A completed node drag could have no undo entry, while selecting an edge added one. | Single/group gestures produce one undo step; undo/redo restores positions. Edge selection preserves history and redo; actual edge edits remain recorded. |
| OC-314/316 | Ctrl+V edited a read-only graph; following an existing source link inserted another dataset node. | Paste checks the effective mode at invocation, copy remains usable, and source links reuse the registered dataset node. |
| OC-306 | A 17-byte upload against a 16-byte limit returned 400 instead of 413. Default validation requested 10 GiB plus one byte in one read. | The multipart endpoint retains 413. Validation reads at most 1 MiB per call and parsing uses the existing upload spool after rewind. |
| OC-309 | Negative sample limits could read the source and return all but its last row; jobs accepted excessive limits or negative offsets. | HTTP requests return 422 before SQL/file reads. Direct service/connector calls raise ValueError. Samples permit 1–50,000 rows, page limits use configuration, and full ingestion with `None` remains supported. |
| OC-259 | Decimal, Time and List columns reached text aggregation and aborted profiling. | The profile remains usable; unsupported type-specific statistics are explicitly unavailable, with an alert, missing counts and original samples. Supported numeric neighbors still have correct statistics. |
| OC-260 | A 1,000-timestamp plot with 500 observed metric values failed or misaligned observations. | All metrics use the same valid timestamp grid, with NaN gaps preserving each metric's missingness, including metrics absent from the first point. |
| OC-288 | Returned Date/Datetime/Time groups failed when reused as drill-down filters. | JSON bucket values select their original rows. Date/time nanoseconds, timezone offsets, DST folds, missing values and membership filters are covered. Fractional Time labels no longer merge distinct groups. |
| OC-303 | A 25-column profile returned 20 correlation columns, so the frontend's greater-than-20 warning never appeared. | Optional total/omitted metadata survives serialization. Desktop/mobile render the omission warning and do not suggest that the omitted data exists in the table. |
| OC-272 | Fitted boosting/SGD pipelines could not export fingerprints or model cards. | Five public classifier/regressor configurations fit, predict, save/load and retain fingerprints on both engines. Known compiled-loss parameters and built-in Generator state are canonicalized; unsupported RNG subclasses and arbitrary reducible state fail explicitly. |
| OC-313 | Four concurrent same-key embedding requests constructed four models. | One in-process load serves every waiter. Failures release waiters; immediate retry works before the failed loader returns, and its cleanup cannot remove the retry. Different keys still load concurrently. |

## Verification

| Check | Result |
|---|---|
| Final full Core suite, sklearn 1.9.1 | **9,009 passed, 80 skipped**, 3 snapshots; 226.87 s |
| Core statement/branch coverage | **97.43%**, above the 90% CI floor |
| Final fingerprint/cache compatibility subset | **102 passed** on sklearn 1.8.0 and **102 passed** on 1.9.1 |
| Final temporal compatibility subset | **12 passed**, including three native Python scalar cases added after full-suite collection |
| Backend full suite, pandas | **3,748 passed**, 7 snapshots; 177.18 s |
| Backend full suite, Polars | **3,748 passed**, 7 snapshots; 156.11 s |
| Frontend full suite | **2,547 passed** across 195 files |
| Final history simplification subset | **72 passed** |
| Chromium Canvas | **4 passed**, real pointer and keyboard interactions |
| Chromium EDA | **20 passed**, including desktop/mobile omission and temporal drill cases, missing groups, causal graph and label controls |
| Python static checks | Ruff, normal Ty scope and full production Core/backend CCN ≤10 pass |
| Frontend static/build checks | ESLint, CCN ≤10, TypeScript/Vite production build and bundle size gate pass |

The full Core run includes the final canonical loss identities. The subsequent
temporal cases extend tests without changing production code. With that coverage
appended, combined statement/branch coverage is **97.44%**, and
**86/87 changed executable Core lines** are covered. The remaining
line rejects an unexpected internal reduction format from a trusted compiled
loss; both formats produced by the installed sklearn versions are exercised.
One additional partial branch is the analyzer's defensive fall-through for a
semantic bucket outside its supported dispatch cases. No mocks manufacture an
unreachable analyzer state solely to raise coverage.

Initial failure evidence included 26 Core seal/cache failures, 11 profiling
failures, 10 Canvas failures and 34 backend failures, with passing controls.
Independent review additionally reproduced a waiter retry race and custom RNG
state omission. The final compatibility check found and repaired two Cython
wheel differences: multiclass state uses different reduction layouts, and loss
classes expose different module names. Two golden digests pin the resulting
canonical format. Fresh processes under sklearn 1.8 and 1.9 now produce the
same digest for five advanced RNG states and ten compiled loss objects:

`01f84ab59c8ab7640db57b16d1b8b9fc325340b168c47f6fdcbb296397553019`

All frontend assets were rebuilt in `static/ml_canvas`. The first build was
five gzip bytes above the existing main-bundle limit. Replacing the node
comparison loop with the same `every` pattern used for edges passed the history
tests and the unchanged size gate; no budget was raised.

Root-wide Ruff without exclusions also scans the unrelated, untracked
`tmp_polars_e2e` investigations and reports their lint findings. The recorded
Ruff check excludes that folder; no application or test files are exempted and
the user's investigation files were not modified.

## Durable regression coverage

- [Model cards, compiled losses and RNG state](../../skyulf-core/tests/integration/test_model_seal_supported_state.py)
  exercise both dataframe engines and public pipeline persistence. Existing
  seal golden, recursion, cycle, threshold and round-trip tests remain green.
- [Concurrent embedding cache](../../skyulf-core/tests/unit/test_sentence_embedder_concurrency.py)
  uses real threads and Futures with deterministic synchronization, including
  ValueError and KeyboardInterrupt failures and immediate waiter retry.
- [Unsupported native types](../../skyulf-core/tests/integration/test_profiling_unsupported_dtypes.py),
  [time-series alignment](../../skyulf-core/tests/unit/test_profiling_timeseries_alignment.py),
  [temporal decomposition](../../skyulf-core/tests/integration/test_profiling_temporal_decomposition.py)
  and [correlation omissions](../../skyulf-core/tests/integration/test_profiling_correlation_omissions.py)
  assert profile values, public plots and serialized results.
- [Temporal HTTP decomposition](../../tests/integration/test_eda_temporal_decomposition.py)
  exercises real Parquet files through DataService and Core;
  [upload and pagination bounds](../../tests/integration/test_intake_pagination_bounds.py)
  exercise real routers, SQL and source files, including validation before reads.
- [Canvas browser tests](../../frontend/ml-canvas/e2e/canvas-editing-history.spec.ts)
  verify real drag/history, clipboard and source navigation;
  [EDA browser tests](../../frontend/ml-canvas/e2e/eda-omissions-temporal.spec.ts)
  verify rendered warnings and exact temporal request payloads on desktop/mobile.
  Existing store, hook, page and EDA component tests cover their neighboring paths.

## Follow-up: invisible undo entries after selecting nodes

The user's report of repeated undo steps after moving/selecting nodes exposed
an additional case within the OC-227 history work. Actual Chromium interaction
with a configured advanced Classification node reproduced **10 invisible history
entries after five settings-panel open/close cycles**. Development StrictMode
replays the defaults effect; each equivalent response previously wrote a new
node data object, and history interpreted its identity as an edit. Hover,
ordinary selection, zoom and pan controls produced **zero** entries.

`updateNodeData` now compares the merged configuration by value and skips
equivalent writes before notifying subscribers or recording history. Plain
objects and arrays retain key/order/type distinctions: reordered object keys
are equivalent, while array reordering, numeric/string changes, NaN versus
null, absent versus undefined fields and array length changes remain edits.
Opaque objects are not silently treated as equivalent. Existing redo survives
repeated equivalent writes; real settings edits remain reversible.

The regression changed from **10 entries to 0** in the browser and from ten
subscriber notifications to none in the store. **80 store tests, six Chromium
Canvas tests and the full 2,555-test frontend suite pass**. The six browser
cases retain the original single/group drag, edge selection, clipboard and
source-navigation checks. ESLint, CCN 10, TypeScript, build and bundle checks
pass; generated assets were rebuilt. The new equality code put the main chunk
about 140 gzip bytes above the former ceiling, so the documented budget changed
from **325 KiB to 326 KiB**. Other budgets are unchanged.

This follow-up does not change queue counts: **34 open / 4 parked**. The table
above preserves the initial batch's test results. Backend/Core source did not
change in this follow-up, so their already-passing suites were not rerun.
Existing undo entries in an already-open browser tab are not retroactively
removed; reload to use the rebuilt code and begin a fresh history.

## Compatibility and limits

- No model retraining is required for fingerprint or cache fixes. Supported
  models that formerly failed to seal can generate fingerprints/model cards.
  Canonical compiled states match across the two verified sklearn versions;
  this does not promise that an entire estimator's learned state is identical
  across arbitrary dependency versions.
- Decimal, Time and List **type-specific profiling statistics remain unavailable**.
  They are marked Unknown with visible warnings; values are not coerced into
  text or imprecise Decimal floats. Temporal decomposition remains usable.
  Regenerate saved profiles to obtain corrected plots and correlation metadata.
- Correlation metadata adds optional fields. Old saved matrices validate and
  retain their existing UI behavior. No new required request fields or schema
  snapshot rewrites were needed.
- Upload validation avoids a second full byte buffer; accepted CSV/Parquet
  parsing still materializes a DataFrame. The configured 10 GiB total policy
  was not changed. No huge-file allocation/OOM experiment was performed.
- Model-cache sharing is per process. Controlled constructors validate thread
  behavior; real downloads, GPU memory behavior and cross-process sharing were
  not claimed. Browser routes are stubbed; separate backend tests validate
  actual file and API behavior.
- Package versions remain 0.8.22; these release-note entries are under v0.8.23.
  Live S3/Docker/external-service CI jobs were not run locally.

## Reproduction commands

Core and backend discovery remain separate. Backend engine runs are sequential
to avoid shared fixture races. Tests use Python 3.12.10, the existing sklearn
1.9.1 dependency overlay, offline Hugging Face settings and distinct temporary
directories; the focused compatibility run also uses installed sklearn 1.8.0.

```powershell
$env:PYTHONPATH=(Resolve-Path tmp_repro_artifacts/ci_nan_sk19).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:LOKY_MAX_CPU_COUNT='2'
$env:SKYULF_ENGINE='polars'
$env:COVERAGE_FILE='tmp_repro_artifacts/0823b_core_final.coverage'
.venv/Scripts/python.exe -m pytest skyulf-core/tests -q --tb=short -p no:cacheprovider --basetemp=tmp_repro_artifacts/0823b_core_final --cov=skyulf-core/skyulf --cov-branch --cov-fail-under=90 --cov-report=xml:tmp_repro_artifacts/0823b_core_final.xml --cov-report=term
$env:SKYULF_ENGINE='pandas'
.venv/Scripts/python.exe -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=tmp_repro_artifacts/0823b_backend_pandas_full
$env:SKYULF_ENGINE='polars'
.venv/Scripts/python.exe -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=tmp_repro_artifacts/0823b_backend_polars_full
.venv/Scripts/ruff.exe check . --extend-exclude tmp_polars_e2e
.venv/Scripts/ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
```

From `frontend/ml-canvas`: `npm test -- --maxWorkers=4 --reporter=dot --silent`,
`npm run lint`, `npm run complexity:check`, `npm run build`, `npm run size-check`,
and `npm exec playwright -- test` with the Canvas/EDA specifications above.
Local logs and coverage files use the ignored `tmp_repro_artifacts/0823b_*`
prefix; durable tests and this report are the reviewable evidence.
