# 0.8.23 queue batch verification

Date: 2026-09-13. Base commit: `7966d0e9` on branch `0822`.
The release notes are opened as `v0.8.23`; package versions remain
0.8.22 until the version bump is requested.

## Reproduced defects and resulting behavior

| Finding | Reproduction and repair | Regression evidence |
|---|---|---|
| OC-320 | Global drop metadata removed raw `city`/`note` before upstream encoders. Retain them through preprocessing, then enforce final drops and feature order. | All four formerly strict xfails now pass: pandas/Polars × Grid/Halving, plus promotion, reload, preview and drift. |
| OC-278 | Missing/corrupt/malformed/unfitted candidates replaced an active model. Validate before deactivation and roll back failed database promotion. | Real SQLite lifecycle and HTTP failure tests preserve prior predictions; successful replacement retains lineage. |
| OC-311 | Legacy abstract artifact references used a hardcoded root; schema loading bypassed the common local containment check. | Prediction, details and promotion use the configured permitted root; traversal remains rejected. |
| OC-246 | Old smoke depended on another workspace, deleted external artifacts and returned without checking tuple artifacts. | `tmp_path` exercises preprocessing, saved artifacts, reload and actual predictions on both engines; no external cleanup. |
| OC-279 | A real 60-scaler preview stalled the HTTP heartbeat for 0.953 s. Move synchronous graph work and resource ownership to a worker. | Real CSV, SQL sessions and engine execution allow concurrent HTTP requests; cancellation cannot remove live worker artifacts. |
| OC-307 | A direct loader → Data Preview graph returned success with no executed nodes. Remove preview sinks before partitioning. | The upstream loader/scalers execute; normal and sink-connected graphs agree on results and branch identities. |
| OC-308 | Cyclic preview graphs returned HTTP 500 and persisted a critical error event. Validate before resolution/execution. | HTTP 400, no critical ErrorEvent, no worker session/artifacts allocated. |
| OC-274 | Constant columns consumed causal discovery feature slots. Filter finite positive variance before the cap and rank deterministically. | Constants before/after useful variables, target eligibility and existing directed-edge tests pass. |
| OC-287 | One constant column suppressed all VIF results. Exclude constants before complete-case filtering and recheck the remaining observations. | Variable-feature VIF survives; existing alerts explain exclusions and unavailable calculations. |
| OC-304 | Reused analyzers kept filtered rows but forgot the active filter metadata. Persist cumulative filters with copied snapshots. | Repeated calls, additional filters, mutable filter values and prior profile independence are asserted. |
| OC-298 | Selected numeric values became text on Polars. Use the existing text-alias contract consistently on both engines. | Numbers, bools, dates and mixed objects retain values/types; textual `1`/`true` still map to `Yes`. |
| OC-301 | Saved custom estimators and parallelism settings never reached resampler construction. Forward method-supported arguments. | Generated SVM/KMeans SMOTE and NearMiss rows match direct imbalanced-learn runs on both engines; invalid estimators raise. |
| OC-312 | Arbitrary objects, dicts and SplitDataset became zero-dimensional object arrays. Validate supported containers at the adapter. | Invalid X and y fail clearly; mixed pandas/Polars frames, wrappers, arrays and Python sequences continue to convert. |

## Focused verification

- Core alias/bridge/resampling: **173 passed** on sklearn 1.8 and 1.9.1.
  Initial regression run had
  24 failures; a stronger SVC dataset also reproduced both engine failures
  before the repair. Independent review also reproduced rounding in an all-numeric
  pandas object column with nulls; the final tests pin exact values and nullable/
  categorical text behavior. Generated rows match the direct sampler.
- Profiling: **614 passed**; the new cases initially produced 12 failures and
  one passing control. Focused 99-case compatibility runs pass on sklearn 1.8
  and 1.9.1. Causal/numeric/analyzer statement coverage: 98% / 95% / 99%.
- Preview: **151 passed**; statement coverage for `preview.py`: 99%.
- Deployment: **175 passed**, plus **30 passed** with sklearn 1.9.1.
- Root Ruff scope and the repository's normal Ty command pass. Lizard reports
  no function above CCN 10 in either complete production tree.

## Full-suite verification

| Check | Result |
|---|---|
| Core, sklearn 1.9.1, Polars default | **8,949 passed, 80 skipped**, 3 snapshots; 222.84 s |
| Core statement + branch coverage | **97.40%**; the 90% CI floor passes |
| Backend, pandas | **3,694 passed**, 7 snapshots; 140.31 s |
| Backend, Polars | **3,694 passed**, 7 snapshots; 224.34 s |
| Final Core compatibility subset | **173 passed** on sklearn 1.8 and 1.9.1 |
| Final deployment helper subset | **77 passed**, including array-returning custom preprocessing |
| Final profiling fallback subset | **45 passed** on sklearn 1.9.1; separate broader audit **105 passed** |

The final Core subset includes two extension-text cases added after full-suite
collection; the deployment subset adds an array-output compatibility case.
Six profiling cases cover empty/non-finite candidates and numerical fallbacks.
These extend coverage without further production changes. Backend runs now have no
expected failures or deselected external smoke tests. Existing Core optional
skips remain; live S3/Docker/external-service CI jobs were not run locally.

After appending these focused runs, combined Core statement/branch coverage is
**97.44%**. All **139/139 changed executable production lines** are covered.
One changed branch exit remains defensive: causal variance ranking repeats the
eligibility check already applied by its caller, so normal selection cannot feed
it a nonpositive/non-finite variance. That guard was retained without bypassing
the caller solely to manufacture coverage. VIF's changed numerical/error paths
are covered, including real non-finite input and a forced inversion failure with
asserted residual-based results.

Core and backend were discovered separately. The backend matrix ran sequentially
to avoid shared fixture resources racing. Environment: Python 3.12.10, sklearn
1.9.1 through the existing `tmp_repro_artifacts/ci_nan_sk19` dependency overlay,
with Hugging Face/transformers offline and `LOKY_MAX_CPU_COUNT=2`.

Full-suite commands (logs and temporary directories use distinct `0823_*` paths):

```powershell
$env:PYTHONPATH=(Resolve-Path tmp_repro_artifacts/ci_nan_sk19).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:LOKY_MAX_CPU_COUNT='2'
$env:SKYULF_ENGINE='polars'
$env:COVERAGE_FILE='tmp_repro_artifacts/0823_core_final.coverage'
.venv/Scripts/python.exe -m pytest skyulf-core/tests -q --tb=short -p no:cacheprovider --basetemp=tmp_repro_artifacts/0823_core_final --cov=skyulf-core/skyulf --cov-branch --cov-fail-under=90 --cov-report=xml:tmp_repro_artifacts/0823_core_final.xml --cov-report=term
$env:SKYULF_ENGINE='pandas'
.venv/Scripts/python.exe -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=tmp_repro_artifacts/0823_backend_pandas --no-cov
$env:SKYULF_ENGINE='polars'
$env:COVERAGE_FILE='tmp_repro_artifacts/0823_backend_polars.coverage'
.venv/Scripts/python.exe -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=tmp_repro_artifacts/0823_backend_polars --cov=backend --cov-branch --cov-report=xml:tmp_repro_artifacts/0823_backend_polars.xml --cov-report=term
```

## Compatibility and remaining scope

- No frontend or API schema change is required. Existing alert rendering shows
  VIF explanations. Existing feature-engineer artifacts need no retraining for
  the deployment replay fix.
- Regenerate profiles for corrected diagnostics. Rerun training to use resampler
  settings formerly ignored. Alias standardization now preserves non-text
  values, including numeric 0/1; cast to strings to request text interpretation.
  Rebuild affected downstream training if it relied on the previous coercion.
- Promotion validates loading, interfaces and sklearn fitted state. Arbitrary
  custom predictors still need representative input to establish valid behavior.
  Live S3 and concurrent PostgreSQL promotions were not exercised.
- A cancelled preview worker completes before releasing its own resources.
  Existing single-training/dangling-branch partition behavior is outside this batch.
- Parked findings OC-71/72/73/185 and other open findings are unaffected.
  These repairs do not close the entire audit queue.

## Evidence locations

New regression files:

- `skyulf-core/tests/integration/test_alias_nontext_contract.py`
- `skyulf-core/tests/integration/test_resampling_advanced_settings.py`
- `skyulf-core/tests/integration/test_profiling_constant_columns.py`
- `skyulf-core/tests/integration/test_profiling_repeated_filters.py`
- `tests/integration/test_preview_execution.py`
- `tests/integration/test_deployment_promotion.py`

Extended existing tests include `test_engines_sklearn_bridge.py`,
`test_ccn_release_pipeline.py` and `test_full_inference_pipeline.py`.
Local detailed logs and coverage files live under ignored
`tmp_repro_artifacts/0823_*`; the tracked tests above are the durable evidence.
