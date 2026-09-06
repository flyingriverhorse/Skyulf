# Core source review — 2026-09-05

Scope: the 188 Python files under `skyulf-core/skyulf/` (34,383 lines in the
initial inventory). Tests, examples, packaging files, and other repository
layers are not part of this file-by-file count.

Status: **188 files read; 0 remaining in the original core-source inventory.**
Last updated: 2026-09-06. The final continuation read the remaining 47
modeling and 23 profiling files. A checked file means its full source was
read, not that it is proven bug-free. This ledger records historical reads;
it does not certify later concurrent edits or cover backend/frontend files.

## Outliers/time-series batch — 10 files

- `skyulf-core/skyulf/preprocessing/outliers/__init__.py`
- `skyulf-core/skyulf/preprocessing/outliers/elliptic.py`
- `skyulf-core/skyulf/preprocessing/outliers/iqr.py`
- `skyulf-core/skyulf/preprocessing/outliers/winsorize.py`
- `skyulf-core/skyulf/preprocessing/outliers/zscore.py`
- `skyulf-core/skyulf/preprocessing/time_series/__init__.py`
- `skyulf-core/skyulf/preprocessing/time_series/_common.py`
- `skyulf-core/skyulf/preprocessing/time_series/date_features.py`
- `skyulf-core/skyulf/preprocessing/time_series/lag.py`
- `skyulf-core/skyulf/preprocessing/time_series/rolling.py`

## Cleaning/encoding continuation — 10 files (2026-09-06)

- `skyulf-core/skyulf/preprocessing/cleaning/__init__.py`
- `skyulf-core/skyulf/preprocessing/cleaning/_common.py`
- `skyulf-core/skyulf/preprocessing/cleaning/alias.py`
- `skyulf-core/skyulf/preprocessing/cleaning/invalid_value.py`
- `skyulf-core/skyulf/preprocessing/cleaning/text.py`
- `skyulf-core/skyulf/preprocessing/cleaning/value_replacement.py`
- `skyulf-core/skyulf/preprocessing/encoding/__init__.py`
- `skyulf-core/skyulf/preprocessing/encoding/_common.py`
- `skyulf-core/skyulf/preprocessing/encoding/dummy.py`
- `skyulf-core/skyulf/preprocessing/encoding/hash.py`

Filed OC-177–182: batch-dependent numeric dummy encoding, inconsistent missing
hash buckets, singleton drop-first width mismatch, nullable-text normalization
crash, unsafe boolean replacement-key coercion, and missed pandas StringDtype
auto-selection. Each was reproduced through the public Calculator/Applier API.
The seven targeted cleaning/encoding suites passed **218 tests** with cache
disabled: `test_cleaning_text.py`, `test_cleaning_alias.py`,
`test_cleaning_invalid_value.py`, `test_value_replacement.py`,
`test_encoding_dummy.py`, `test_encoding_hash.py`, and `test_encoding_common.py`.
No implementation code was changed. Existing OC-18/20/122/140/141 were not
duplicated. The ledger records historical reads; it does not certify later
concurrent edits to already-reviewed files.

## Findings and verification

New entries are documented in [the tracker](opus_core_analysis-tracker.md):
OC-170–172 from the preceding source pass, and OC-173–176 from the final
10-file batch. Existing OC-163/165/166 were not filed again. No implementation
code was changed. The review used the Skyulf codebase map to trace shared
helpers and decorators, and reproduction-driven checks to separate suspected
bugs from confirmed findings.

Executed checks: 141 existing tests passed across `test_outlier_failure_branches.py`,
`test_outliers_elliptic_winsorize_zscore_iqr.py`,
`test_preprocessing_time_series_common.py`, `test_time_series_nodes.py`, and
`test_time_series_gaps.py`; pytest warned that it could not write its cache.
Separate in-memory probes reproduced all seven new findings, with pandas/Polars,
unique-index, and native sklearn controls as recorded in the tracker. No full
suite rerun or implementation fix is claimed for this batch.

## Modeling/profiling completion — 70 files (2026-09-06)

Completed full-source reads of all 47 `modeling/` and 23 `profiling/` files:
20 modeling/evaluation/explainability files and 23 profiling files were read
by separate reviewers; the main review read all 11 hyperparameter files and
the remaining 16 tuning/CV files, and independently executed the filed bug
reproductions. Truncated source output was recovered before counting files.

Added **OC-187–206** directly to
[the tracker](opus_core_analysis-tracker.md) as findings were verified:
20 new findings (**5 high / 14 medium / 1 low**), all open. They cover
incorrect rule labels and support counts, nullable-date row corruption,
profiling exclusions and name collisions, model feature-selection mismatches,
ineffective LightGBM subsampling, invalid tuning scores/search candidates,
and caller-configuration mutation. Each entry records executed evidence and
a fix/verification target. Existing findings were checked for duplication;
no implementation code or permanent regression tests were changed.

Verification: **909 passed, 142 warnings** across **45 existing test files**
in 48.15 seconds, cache and coverage disabled. An earlier focused run of the
three hyperparameter suites passed 49 tests; those are included in the 909,
not additional unique tests. Reproduction probes are separate from passing
suite tests and demonstrate uncovered behavior, not fixes.

Exact test selection and command (PowerShell, repository root):

```powershell
$env:OMP_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
$reviewTests = @(rg --files skyulf-core/tests/unit skyulf-core/tests/integration |
    Where-Object {
        $_ -match 'test_(profiling_|modeling|evaluation_|tuning|cross_validation|cv_per_fold_refit|ensemble|hyperparameters)' -and
        $_ -like '*.py'
    })
./.venv/Scripts/python.exe -m pytest @reviewTests -q --no-cov -p no:cacheprovider -o addopts='' --tb=short
```

This completes the original 188-file source-reading pass. Backend, frontend,
tests, examples and packaging remain outside that inventory. No full
repository test/build result or implementation fix is claimed.

## File-by-file ledger

Continuation checkpoint (2026-09-06): all preprocessing files have now been
read. All 3 `data/` and 11 `core/` files were re-read in their current state;
no additional bug was confirmed in those folders. Their focused tests passed
78 cases. Modeling (47 files) and profiling (23 files) are now fully read;
their continuation findings and verification are recorded above.


### package root

- [x] `__init__.py` (47 lines)
- [x] `_validation.py` (12 lines)
- [x] `config_validation.py` (111 lines)

### core

- [x] `core/__init__.py` (47 lines)
- [x] `core/artifacts.py` (542 lines)
- [x] `core/compute.py` (79 lines)
- [x] `core/deprecation.py` (89 lines)
- [x] `core/meta/__init__.py` (1 lines)
- [x] `core/meta/decorators.py` (50 lines)
- [x] `core/model_registry.py` (83 lines)
- [x] `core/protocols.py` (73 lines)
- [x] `core/schema.py` (289 lines)
- [x] `core/serialization.py` (91 lines)
- [x] `core/warnings.py` (52 lines)

### data

- [x] `data/__init__.py` (0 lines)
- [x] `data/catalog.py` (40 lines)
- [x] `data/dataset.py` (56 lines)

### engines

- [x] `engines/__init__.py` (24 lines)
- [x] `engines/pandas_engine.py` (121 lines)
- [x] `engines/polars_engine.py` (147 lines)
- [x] `engines/protocol.py` (136 lines)
- [x] `engines/registry.py` (163 lines)
- [x] `engines/sklearn_bridge.py` (53 lines)

### package root

- [x] `leakage.py` (164 lines)

### modeling

- [x] `modeling/__init__.py` (103 lines)
- [x] `modeling/_boosting_progress.py` (90 lines)
- [x] `modeling/_evaluation/__init__.py` (45 lines)
- [x] `modeling/_evaluation/classification.py` (173 lines)
- [x] `modeling/_evaluation/clustering.py` (286 lines)
- [x] `modeling/_evaluation/common.py` (77 lines)
- [x] `modeling/_evaluation/metrics.py` (494 lines)
- [x] `modeling/_evaluation/regression.py` (87 lines)
- [x] `modeling/_evaluation/schemas.py` (88 lines)
- [x] `modeling/_evaluation/thresholds.py` (246 lines)
- [x] `modeling/_explainability/__init__.py` (7 lines)
- [x] `modeling/_explainability/shap_explanation.py` (413 lines)
- [x] `modeling/_sklearn_compat.py` (47 lines)
- [x] `modeling/_tuning/__init__.py` (4 lines)
- [x] `modeling/_tuning/engine.py` (721 lines)
- [x] `modeling/_tuning/fold_pipeline.py` (145 lines)
- [x] `modeling/_tuning/grid_random.py` (292 lines)
- [x] `modeling/_tuning/metrics.py` (150 lines)
- [x] `modeling/_tuning/params.py` (81 lines)
- [x] `modeling/_tuning/refit.py` (220 lines)
- [x] `modeling/_tuning/reporter.py` (81 lines)
- [x] `modeling/_tuning/schemas.py` (65 lines)
- [x] `modeling/_tuning/splitters.py` (173 lines)
- [x] `modeling/_tuning/strategies/__init__.py` (7 lines)
- [x] `modeling/_tuning/strategies/halving.py` (91 lines)
- [x] `modeling/_tuning/strategies/optuna.py` (242 lines)
- [x] `modeling/_tuning/strategies/runner.py` (177 lines)
- [x] `modeling/base.py` (620 lines)
- [x] `modeling/classification.py` (772 lines)
- [x] `modeling/clustering.py` (268 lines)
- [x] `modeling/cross_validation.py` (663 lines)
- [x] `modeling/ensemble.py` (662 lines)
- [x] `modeling/fold_preprocessing.py` (33 lines)
- [x] `modeling/hyperparameters/__init__.py` (106 lines)
- [x] `modeling/hyperparameters/_bayes.py` (68 lines)
- [x] `modeling/hyperparameters/_calibration.py` (51 lines)
- [x] `modeling/hyperparameters/_clustering.py` (121 lines)
- [x] `modeling/hyperparameters/_ensemble.py` (196 lines)
- [x] `modeling/hyperparameters/_field.py` (65 lines)
- [x] `modeling/hyperparameters/_linear.py` (241 lines)
- [x] `modeling/hyperparameters/_neighbors.py` (40 lines)
- [x] `modeling/hyperparameters/_registry.py` (616 lines)
- [x] `modeling/hyperparameters/_svm.py` (39 lines)
- [x] `modeling/hyperparameters/_tree.py` (651 lines)
- [x] `modeling/naive_bayes.py` (95 lines)
- [x] `modeling/regression.py` (543 lines)
- [x] `modeling/sklearn_wrapper.py` (279 lines)

### pipeline

- [x] `pipeline/__init__.py` (17 lines)
- [x] `pipeline/_pipeline.py` (520 lines)
- [x] `pipeline/diagram.py` (129 lines)
- [x] `pipeline/seal.py` (148 lines)

### preprocessing

- [x] `preprocessing/__init__.py` (254 lines)
- [x] `preprocessing/_artifacts.py` (121 lines)
- [x] `preprocessing/_helpers.py` (208 lines)
- [x] `preprocessing/_schema.py` (11 lines)
- [x] `preprocessing/base.py` (294 lines)
- [x] `preprocessing/bucketing.py` (548 lines)
- [x] `preprocessing/casting.py` (460 lines)
- [x] `preprocessing/cleaning/__init__.py` (35 lines)
- [x] `preprocessing/cleaning/_common.py` (48 lines)
- [x] `preprocessing/cleaning/alias.py` (170 lines)
- [x] `preprocessing/cleaning/invalid_value.py` (235 lines)
- [x] `preprocessing/cleaning/text.py` (208 lines)
- [x] `preprocessing/cleaning/value_replacement.py` (223 lines)
- [x] `preprocessing/dispatcher.py` (275 lines)
- [x] `preprocessing/drop_and_missing/__init__.py` (29 lines)
- [x] `preprocessing/drop_and_missing/_common.py` (48 lines)
- [x] `preprocessing/drop_and_missing/deduplicate.py` (97 lines)
- [x] `preprocessing/drop_and_missing/drop_columns.py` (128 lines)
- [x] `preprocessing/drop_and_missing/drop_rows.py` (136 lines)
- [x] `preprocessing/drop_and_missing/missing_indicator.py` (132 lines)
- [x] `preprocessing/encoding/__init__.py` (44 lines)
- [x] `preprocessing/encoding/_common.py` (140 lines)
- [x] `preprocessing/encoding/dummy.py` (175 lines)
- [x] `preprocessing/encoding/hash.py` (133 lines)
- [x] `preprocessing/encoding/label.py` (309 lines)
- [x] `preprocessing/encoding/one_hot.py` (233 lines)
- [x] `preprocessing/encoding/ordinal.py` (338 lines)
- [x] `preprocessing/encoding/target.py` (363 lines)
- [x] `preprocessing/encoding/woe.py` (355 lines)
- [x] `preprocessing/feature_generation/__init__.py` (31 lines)
- [x] `preprocessing/feature_generation/_common.py` (178 lines)
- [x] `preprocessing/feature_generation/_pandas_ops.py` (241 lines)
- [x] `preprocessing/feature_generation/_polars_ops.py` (272 lines)
- [x] `preprocessing/feature_generation/generation.py` (48 lines)
- [x] `preprocessing/feature_generation/interaction.py` (194 lines)
- [x] `preprocessing/feature_generation/polynomial.py` (132 lines)
- [x] `preprocessing/feature_selection/__init__.py` (25 lines)
- [x] `preprocessing/feature_selection/_common.py` (359 lines)
- [x] `preprocessing/feature_selection/correlation.py` (197 lines)
- [x] `preprocessing/feature_selection/facade.py` (79 lines)
- [x] `preprocessing/feature_selection/model_based.py` (99 lines)
- [x] `preprocessing/feature_selection/univariate.py` (100 lines)
- [x] `preprocessing/feature_selection/variance.py` (68 lines)
- [x] `preprocessing/fold_adapter.py` (268 lines)
- [x] `preprocessing/geo/__init__.py` (20 lines)
- [x] `preprocessing/geo/distance.py` (193 lines)
- [x] `preprocessing/geo/h3_index.py` (159 lines)
- [x] `preprocessing/imputation/__init__.py` (25 lines)
- [x] `preprocessing/imputation/_common.py` (158 lines)
- [x] `preprocessing/imputation/iterative.py` (93 lines)
- [x] `preprocessing/imputation/knn.py` (87 lines)
- [x] `preprocessing/imputation/simple.py` (195 lines)
- [x] `preprocessing/inspection.py` (157 lines)
- [x] `preprocessing/outliers/__init__.py` (32 lines)
- [x] `preprocessing/outliers/_common.py` (24 lines)
- [x] `preprocessing/outliers/elliptic.py` (175 lines)
- [x] `preprocessing/outliers/iqr.py` (114 lines)
- [x] `preprocessing/outliers/manual_bounds.py` (103 lines)
- [x] `preprocessing/outliers/winsorize.py` (115 lines)
- [x] `preprocessing/outliers/zscore.py` (120 lines)
- [x] `preprocessing/pipeline.py` (624 lines)
- [x] `preprocessing/resampling.py` (385 lines)
- [x] `preprocessing/scaling/__init__.py` (29 lines)
- [x] `preprocessing/scaling/_common.py` (21 lines)
- [x] `preprocessing/scaling/maxabs.py` (106 lines)
- [x] `preprocessing/scaling/minmax.py` (113 lines)
- [x] `preprocessing/scaling/robust.py` (138 lines)
- [x] `preprocessing/scaling/standard.py` (157 lines)
- [x] `preprocessing/split.py` (517 lines)
- [x] `preprocessing/time_series/__init__.py` (23 lines)
- [x] `preprocessing/time_series/_common.py` (66 lines)
- [x] `preprocessing/time_series/date_features.py` (181 lines)
- [x] `preprocessing/time_series/lag.py` (137 lines)
- [x] `preprocessing/time_series/rolling.py` (180 lines)
- [x] `preprocessing/transformations/__init__.py` (19 lines)
- [x] `preprocessing/transformations/_ops.py` (93 lines)
- [x] `preprocessing/transformations/_power_common.py` (67 lines)
- [x] `preprocessing/transformations/general.py` (199 lines)
- [x] `preprocessing/transformations/power.py` (153 lines)
- [x] `preprocessing/transformations/simple.py` (86 lines)
- [x] `preprocessing/vectorization/__init__.py` (31 lines)
- [x] `preprocessing/vectorization/_common.py` (267 lines)
- [x] `preprocessing/vectorization/count_vectorizer.py` (139 lines)
- [x] `preprocessing/vectorization/hashing_vectorizer.py` (126 lines)
- [x] `preprocessing/vectorization/sentence_embedder.py` (187 lines)
- [x] `preprocessing/vectorization/tfidf_vectorizer.py` (132 lines)
- [x] `preprocessing/vectorization/tokenizer.py` (166 lines)

### profiling

- [x] `profiling/__init__.py` (28 lines)
- [x] `profiling/_analyzer/__init__.py` (36 lines)
- [x] `profiling/_analyzer/_utils.py` (91 lines)
- [x] `profiling/_analyzer/categorical.py` (35 lines)
- [x] `profiling/_analyzer/causal.py` (133 lines)
- [x] `profiling/_analyzer/column.py` (347 lines)
- [x] `profiling/_analyzer/dates.py` (171 lines)
- [x] `profiling/_analyzer/decomposition.py` (194 lines)
- [x] `profiling/_analyzer/geo.py` (159 lines)
- [x] `profiling/_analyzer/multivariate.py` (406 lines)
- [x] `profiling/_analyzer/numeric.py` (64 lines)
- [x] `profiling/_analyzer/recommendations.py` (213 lines)
- [x] `profiling/_analyzer/rules.py` (370 lines)
- [x] `profiling/_analyzer/target.py` (229 lines)
- [x] `profiling/_analyzer/temporal.py` (260 lines)
- [x] `profiling/_analyzer/text.py` (126 lines)
- [x] `profiling/analyzer.py` (651 lines)
- [x] `profiling/correlations.py` (170 lines)
- [x] `profiling/distributions.py` (82 lines)
- [x] `profiling/drift.py` (541 lines)
- [x] `profiling/expect.py` (209 lines)
- [x] `profiling/schemas.py` (322 lines)
- [x] `profiling/visualizer.py` (824 lines)

### package root

- [x] `registry.py` (108 lines)
- [x] `types.py` (44 lines)
- [x] `utils.py` (399 lines)
