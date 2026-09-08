# Skyulf core review — 2026-09-08

Reviewed the working tree based on `28f12473`, including its existing staged and
unstaged changes. This records the original review and its executed reproduction
evidence. The existing Opus queue and archive were
consulted to distinguish newly discovered defects from already filed issues.

**Follow-up 2026-09-08:** OC-210 (post-training thresholds), OC-212 (similarity
with duplicate indexes), and the two configured ty-gate diagnostics are fixed.
The fixes and validation are recorded in [the archive](opus_core_analysis-tracker.md).
OC-208, OC-209 and OC-211 remain in [the open queue](opus_core_analysis-open_queue.md).
The observations and counts below describe the original review before those fixes.

## New findings

### P2 — A failed refit leaves new preprocessing attached to the old model

Locations: `skyulf-core/skyulf/pipeline/_pipeline.py:329` and
`skyulf-core/skyulf/modeling/base.py:427`.

`fit()` updates the feature engineer before the replacement model has fitted
successfully. If model fitting raises, subsequent predictions remain enabled
and combine the new preprocessing state with the previous fitted model.

Executed through `SkyulfPipeline` with StandardScaler and LinearRegression:

1. Fit `x=0..23`, `target=2*x`; prediction for `x=25` is **50**.
2. Refit with `x += 100` and all-missing targets. Fitting raises
   `ValueError: Input y contains NaN`.
3. Predict `x=25` again: the result is now **-150**, despite no replacement model
   having fitted successfully.

The fit failure must either leave the previous complete fitted state intact or
invalidate prediction until another successful fit. This differs from OC-168's
stale thresholds and closed OC-164's mutation through `get_fitted_split()`.

### P2 — Time-series tuning with a validation partition fails on clean data

Locations: `skyulf-core/skyulf/modeling/_tuning/engine.py:353` and
`skyulf-core/skyulf/modeling/_tuning/splitters.py:91`.

Time-series preparation drops the time column from training features but leaves
it in the explicit validation features. The frame-based holdout path concatenates
those different schemas, creating missing values in the training rows.

Executed a Ridge grid search through `SkyulfPipeline.fit()` with numeric `x`,
`time`, and `target=2*x`, and separate train/validation/test partitions. Setting
`cv_type="time_series_split"`, `cv_time_column="time"` gives:

```text
Hyperparameter tuning failed: All trials failed.
First trial error: Input X contains NaN.
```

The same data and partitions with ordinary CV succeed. The input contains no
missing values. Training and validation must use the same selected feature
schema. This is separate from OC-162 and OC-194.

### P2 — Post-training threshold optimization rejects tuned classifiers

Location: `skyulf-core/skyulf/pipeline/_pipeline.py:511`.

`optimize_thresholds()` reads `classes_` from `model_estimator.model` directly.
For a hyperparameter tuner, that value is `(fitted_model, TuningResult)`, so the
lookup is performed on a tuple.

Executed a successful logistic-regression grid search, followed by
`pipeline.optimize_thresholds(raw_X, y, accuracy_score)`. It raises:

```text
The fitted model does not expose class labels (classes_);
threshold tuning requires a classifier.
```

Resolve the underlying fitted classifier consistently with the tuning applier.
This finding concerns the standalone core post-training API; automatic threshold
tuning and the backend job endpoint use separate paths. It is not OC-168 or
closed OC-207.

### P2 — Date features depend on unrelated rows in the prediction batch

Locations:
`skyulf-core/skyulf/preprocessing/feature_generation/_pandas_ops.py:179` and
`skyulf-core/skyulf/preprocessing/time_series/date_features.py:64`.

Both pandas paths infer the date format again from each input batch with
`pd.to_datetime(..., errors="coerce")`. The fitted artifact does not retain a
format, and the parsing policy is not independent for each value.

Executed both `FeatureGeneration.datetime_extract` and `DateFeatures`:

| Applied dates | Months for the original two dates |
|---|---|
| `2024-01-02`, `2024-03-04` | `1, 3` |
| Prepend `04/05/2024` to that same batch | missing, missing |

Year and day also become missing for the original valid rows. Thus the same raw
inference row can produce different features depending on its batch companions.
Define a stable parsing policy and pin batch invariance. These existing parsing
paths predate the current staged changes. This is distinct from OC-30 and OC-174.

### P2 — Duplicate pandas indexes silently remove a similarity feature

Location: `skyulf-core/skyulf/preprocessing/feature_generation/_common.py:137`.

`_vectorised_similarity()` iterates index labels and passes `a_str.at[i]` and
`b_str.at[i]` to a scalar helper. Repeated labels return Series. The exception is
caught by the operation dispatcher, which omits the entire output feature.

Executed a similarity operation producing `score` from identical text columns
`["apple", "pear"]`:

| Index | Output |
|---|---|
| `[0, 1]` | `score=[100.0, 100.0]` |
| `[7, 7]` | `score` column absent |

The log reports `The truth value of a Series is ambiguous`. A downstream model
can consequently receive a different feature schema. Use positional pairing
without discarding the caller's index. No matching existing OC entry was found.

## Existing queue reconciliation

The file currently contains **82 OC rows: 77 open, 4 parked, 1 marked done**.
The 77 open rows are 18 orange, 49 yellow and 10 white. Its original “100 open”
split note is historical. These counts describe recorded statuses, not 77 bugs
independently reproduced by this review.

Three existing reproductions no longer fail on the current working tree:

| ID | Executed current behavior |
|---|---|
| OC-24 | Public group aggregation with keys `['a', None, None]` and values `[10,4,8]` returns `[10,6,6]` in both pandas and Polars. |
| OC-170 | `TextCleaning`, `DateFeatures`, default `Casting`, and `feature_target_split` before the row splitter are accepted by the leakage validator. |
| OC-173 | Duplicate-index EllipticEnvelope removes the outlier and keeps aligned `x=[0,NaN,1]`, `y=[10,30,40]`. |

These rows need reconciliation with the current changes and regression coverage
before archival. OC-03 is already marked done but remains in the open file.

Six known open defects were re-executed and remain present:

| ID | Observed behavior |
|---|---|
| OC-63 | Cyclic artifact input raises `RecursionError`. |
| OC-167 | Distinct string/object sequences produce equal artifact digests. |
| OC-162 | Time-series CV removes the real `__cv_y__` feature. |
| OC-194 | Missing-date sorting produces row order `[1,3,2,0]` for dates Jan 3, missing, Jan 1, Jan 2. |
| OC-204 | Tuple features containing the explicit target fit one feature but predict two, raising a feature-count error. |
| OC-168 | Refitting with `no/yes` labels retains `0/1` thresholds and thresholded prediction raises. |

No queue statuses or parked deployment/auth decisions were changed.

## Verification and editor Problems investigation

| Check | Result |
|---|---|
| Full `skyulf-core/tests` | **5,494 passed, 56 skipped, 281 warnings**, 197.32 seconds; 3 snapshots passed. |
| `ruff check .` | Passed. |
| Configured `ty` gate | **2 diagnostics**, both in one newly added test. |
| `ruff format --check .` | **4 files** would be reformatted; 685 already formatted. |
| Frontend ESLint | Passed. |
| Frontend TypeScript | Passed. |
| Frontend production build | Passed; output written to the review's temporary directory. |

The two gate diagnostics are at
`skyulf-core/tests/integration/test_repeated_split_boundary.py:36` and `:38`.
The code narrows `expected_frame` to DataFrame or Series, but `actual_frame`
remains `DataFrame | Series`. The comparison helpers require the corresponding
concrete type. The runtime tests pass, but this is a real type-check gate failure.
Assertions checking the corresponding actual type would establish the contract.

The four formatting files are:

- `skyulf-core/examples/09_leakage_safety.ipynb`
- `skyulf-core/examples/09_leakage_safety.py`
- `skyulf-core/tests/integration/test_leakage_fixture_contract.py`
- `tests/integration/test_leakage_submission.py`

The editor's exact **78 Problems** could not be matched without the actual panel
entries/source labels. This environment exposes no direct editor-diagnostics
tool. Local VS Code activation logs confirm ty, Pylance and SonarQube for IDE
were activated; Pylance is configured with basic type checking. The ty language
server uses the same `0.0.75` version as the CLI and resolves the project venv.
These observations do not identify which provider owns the 78 entries.

There is also a scope difference: root `pyproject.toml` includes backend,
core library and core tests in ty, but excludes root `tests/` and core examples
from its include list. Explicitly expanding that include list for a separate
diagnostic run reports **297 diagnostics** in those additional areas. This
number is a separate measurement, not a reconstruction of the editor's 78.
For example, the new leakage notebook has 16 diagnostics, its Python counterpart
has 6, and `tests/integration/test_fold_preprocessing_refit.py` has 4 nullable
string checks. Many concern unresolved DataFrame/tuple unions or optional values;
they must be assessed individually before changing types or runtime code.

## Reproduction and verification commands

Run from the repository root in PowerShell. Reproduction scripts and raw logs
are retained locally under `tmp_repro_artifacts/core-review-20260908/`.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe tmp_repro_artifacts/core-review-20260908/modeling/repro_pipeline.py
.\.venv\Scripts\python.exe tmp_repro_artifacts/core-review-20260908/modeling/repro_known.py
.\.venv\Scripts\python.exe tmp_repro_artifacts/core-review-20260908/preprocessing/probe.py
.\.venv\Scripts\python.exe -m pytest skyulf-core/tests -q --no-cov --tb=short -o addopts='' -p no:cacheprovider --basetemp=tmp_repro_artifacts/core-review-20260908/pytest-temp
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ty check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py --output-format concise
.\.venv\Scripts\python.exe -m ty check tests skyulf-core/examples --config "src.include=['tests','skyulf-core/examples']" --output-format concise
```

From `frontend/ml-canvas/`:

```powershell
npm run lint
node node_modules/typescript/bin/tsc --noEmit --pretty false
npm run build -- --outDir ../../tmp_repro_artifacts/core-review-20260908/frontend-build
```

The initial build hit a sandbox parent-directory read restriction in esbuild;
the authorized retry completed successfully. No frontend source or served
static assets were changed by this review. No backend pytest suite or frontend
Vitest suite was run.

## Coverage limits

Source review concentrated on pipeline lifecycle, CV sorting and tuning,
thresholds, serialization, and the current preprocessing/leakage changes.
The small reproductions were independently rerun by the main reviewer.
The entire core test suite ran, but every core source file was not independently
audited and the 77 existing open rows were not all re-tested. Passing tests
therefore do not establish absence of defects; the five new reproductions above
exercise behavior missing from the current assertions.
