# 18 — Per-file audit matrix for `skyulf-core/skyulf/`

> Answers *"which files did you check, which had bugs, and which did you not check?"*
> Generated mechanically: every `.py` file under `skyulf/` enumerated from disk, then
> matched against the **File:** references inside every finding across all reports.
> Not written from memory.

## Summary

| Status | Files | Lines | Meaning |
|---|---:|---:|---|
| 🐛 **Checked — bug found** | 92 | 17,100 | At least one filed finding cites this file |
| ✅ **Checked — no bug found** | 88 | 9,419 | Read and/or behaviourally exercised; nothing filed |
| ➖ **Re-export shim** | 8 | 420 | `__init__.py` containing only imports/`__all__` — no logic to audit |
| ❌ **Not checked** | **0** | 0 | — |
| **Total** | **188** | **26,939** | |

**Coverage: 188 / 188 files (100%).** 92 files carry 161 finding references.
The last remaining file, `types.py` (25 lines), was read while producing this
matrix — it is pure `TypedDict`/constant declarations with no executable logic.

### File coverage ≠ line coverage

100% of files were touched. **~85% of lines were actually read.** The remainder
was validated behaviourally rather than by reading:

| Area | Files | Lines | Evidence |
|---|---:|---:|---|
| `modeling/hyperparameters/` | 11 | 2,023 | **Strong (non-reading).** All 389 declared search ranges executed against their real estimators — arguably better evidence than reading, since it proves the ranges are *accepted*, not just plausible. |
| `modeling/_tuning/` | 14 | 1,899 | **Weakest area in core.** Spot-checked only: sampler seeding verified correct, the rest not line-read. |
| Everything else | 163 | 23,017 | Read, and/or exercised by the 34-node cross-engine parity + determinism harnesses. |

So the honest residual risk is concentrated in **`modeling/_tuning/` (~1,899
lines, 7% of core)** — the one area with neither a close read nor systematic
executable coverage behind it.

---

## 1. Files where a bug was found

Sorted by worst severity in that file. Severity shown per finding.

| # | File | Lines | Findings |
|---:|---|---:|---|
| 1 | `modeling/_evaluation/metrics.py` | 380 | 🔴 OC-146, 🟠 OC-35, 🟡 OC-37, 🟡 OC-67, ⚪ OC-144, ⚪ OC-38 |
| 2 | `modeling/_tuning/metrics.py` | 113 | 🔴 OC-146, 🟡 OC-67 |
| 3 | `pipeline/_pipeline.py` | 415 | 🔴 OC-62 |
| 4 | `pipeline/seal.py` | 130 | 🔴 OC-62, 🟠 OC-63, 🟡 OC-09 |
| 5 | `preprocessing/casting.py` | 329 | 🔴 OC-58, ⚪ OC-57 |
| 6 | `preprocessing/drop_and_missing/deduplicate.py` | 59 | 🔴 OC-12 |
| 7 | `preprocessing/drop_and_missing/drop_rows.py` | 81 | 🔴 OC-12, 🟠 OC-13 |
| 8 | `preprocessing/split.py` | 402 | 🔴 OC-75, ⚪ OC-57, ⚪ OC-90 |
| 9 | `__init__.py` | 40 | 🟠 OC-01 |
| 10 | `core/compute.py` | 58 | 🟠 OC-64 |
| 11 | `core/serialization.py` | 66 | 🟠 OC-64, 🟡 OC-79, 🟡 OC-91 |
| 12 | `engines/__init__.py` | 21 | 🟠 OC-120 |
| 13 | `engines/polars_engine.py` | 90 | 🟠 OC-59, 🟡 OC-65, 🟡 OC-74 |
| 14 | `engines/registry.py` | 121 | 🟠 OC-64, 🟡 OC-74 |
| 15 | `modeling/_evaluation/classification.py` | 123 | 🟠 OC-35, ⚪ OC-144 |
| 16 | `modeling/_evaluation/clustering.py` | 233 | 🟠 OC-149, ⚪ OC-81 |
| 17 | `modeling/_evaluation/thresholds.py` | 166 | 🟠 OC-36, ⚪ OC-147 |
| 18 | `modeling/_tuning/engine.py` | 553 | 🟠 OC-66, 🟡 OC-09 |
| 19 | `modeling/_tuning/params.py` | 56 | 🟠 OC-66 |
| 20 | `modeling/_tuning/refit.py` | 175 | 🟠 OC-36, 🟡 OC-67 |
| 21 | `modeling/classification.py` | 610 | 🟠 OC-66, 🟡 OC-101 |
| 22 | `modeling/cross_validation.py` | 532 | 🟠 OC-68 |
| 23 | `modeling/ensemble.py` | 542 | 🟠 OC-68 |
| 24 | `modeling/fold_preprocessing.py` | 26 | 🟠 OC-68 |
| 25 | `modeling/hyperparameters/_calibration.py` | 47 | 🟠 OC-66, 🟡 OC-101 |
| 26 | `modeling/sklearn_wrapper.py` | 183 | 🟠 OC-66, 🟡 OC-101 |
| 27 | `preprocessing/_helpers.py` | 150 | 🟠 OC-120, ⚪ OC-121 |
| 28 | `preprocessing/base.py` | 221 | 🟠 OC-03 |
| 29 | `preprocessing/bucketing.py` | 420 | 🟠 OC-60, 🟡 OC-04, 🟡 OC-61 |
| 30 | `preprocessing/cleaning/invalid_value.py` | 193 | 🟠 OC-140, ⚪ OC-141, ⚪ OC-144 |
| 31 | `preprocessing/feature_generation/_polars_ops.py` | 218 | 🟠 OC-23, 🟠 OC-24, 🟠 OC-40, 🟡 OC-30 |
| 32 | `preprocessing/feature_selection/_common.py` | 265 | 🟠 OC-143, 🟠 OC-25, 🟡 OC-53, ⚪ OC-144 |
| 33 | `preprocessing/imputation/_common.py` | 82 | 🟠 OC-14, 🟠 OC-17 |
| 34 | `preprocessing/imputation/iterative.py` | 66 | 🟠 OC-16 |
| 35 | `preprocessing/imputation/knn.py` | 62 | 🟠 OC-16 |
| 36 | `preprocessing/inspection.py` | 116 | 🟠 OC-59 |
| 37 | `preprocessing/scaling/_common.py` | 15 | 🟠 OC-16 |
| 38 | `preprocessing/scaling/minmax.py` | 93 | 🟠 OC-15 |
| 39 | `preprocessing/scaling/robust.py` | 116 | 🟠 OC-15 |
| 40 | `preprocessing/transformations/general.py` | 162 | 🟠 OC-03, 🟠 OC-27 |
| 41 | `preprocessing/transformations/power.py` | 121 | 🟠 OC-28, 🟡 OC-05 |
| 42 | `preprocessing/vectorization/hashing_vectorizer.py` | 97 | 🟠 OC-26 |
| 43 | `profiling/_analyzer/column.py` | 301 | 🟠 OC-110, 🟡 OC-148 |
| 44 | `profiling/_analyzer/multivariate.py` | 325 | 🟠 OC-40 |
| 45 | `profiling/_analyzer/numeric.py` | 46 | 🟠 OC-113, ⚪ OC-131 |
| 46 | `profiling/_analyzer/recommendations.py` | 186 | 🟠 OC-42, 🟡 OC-111, 🟡 OC-50, 🟡 OC-51 |
| 47 | `profiling/_analyzer/target.py` | 177 | 🟠 OC-142, 🟠 OC-41, ⚪ OC-144 |
| 48 | `profiling/analyzer.py` | 521 | 🟠 OC-110, 🟠 OC-113, 🟠 OC-39, 🟠 OC-40, 🟠 OC-41, 🟠 OC-42 |
| 49 | `profiling/correlations.py` | 67 | 🟠 OC-43 |
| 50 | `profiling/drift.py` | 381 | 🟠 OC-40, 🟠 OC-44, 🟠 OC-45, 🟡 OC-47, ⚪ OC-131 |
| 51 | `utils.py` | 290 | 🟠 OC-120 |
| 52 | `config_validation.py` | 76 | 🟡 OC-80, ⚪ OC-81 |
| 53 | `core/__init__.py` | 45 | 🟡 OC-91 |
| 54 | `core/deprecation.py` | 73 | 🟡 OC-91 |
| 55 | `core/model_registry.py` | 63 | 🟡 OC-91 |
| 56 | `engines/pandas_engine.py` | 77 | 🟡 OC-74 |
| 57 | `modeling/_sklearn_compat.py` | 40 | 🟡 OC-80, ⚪ OC-81 |
| 58 | `modeling/_tuning/strategies/runner.py` | 150 | 🟡 OC-79 |
| 59 | `preprocessing/cleaning/alias.py` | 123 | 🟡 OC-19, ⚪ OC-121 |
| 60 | `preprocessing/cleaning/value_replacement.py` | 184 | 🟡 OC-20, 🟡 OC-80, ⚪ OC-81 |
| 61 | `preprocessing/encoding/dummy.py` | 134 | 🟡 OC-18 |
| 62 | `preprocessing/encoding/one_hot.py` | 184 | 🟡 OC-18 |
| 63 | `preprocessing/encoding/woe.py` | 288 | 🟡 OC-21 |
| 64 | `preprocessing/feature_generation/_common.py` | 144 | 🟡 OC-29 |
| 65 | `preprocessing/feature_generation/_pandas_ops.py` | 200 | 🟡 OC-30 |
| 66 | `preprocessing/feature_generation/generation.py` | 39 | 🟡 OC-07 |
| 67 | `preprocessing/feature_generation/interaction.py` | 148 | 🟡 OC-33 |
| 68 | `preprocessing/feature_generation/polynomial.py` | 107 | 🟡 OC-07 |
| 69 | `preprocessing/feature_selection/model_based.py` | 79 | 🟡 OC-53 |
| 70 | `preprocessing/feature_selection/variance.py` | 56 | 🟡 OC-32 |
| 71 | `preprocessing/vectorization/count_vectorizer.py` | 110 | 🟡 OC-34, ⚪ OC-10 |
| 72 | `preprocessing/vectorization/tfidf_vectorizer.py` | 106 | 🟡 OC-34, ⚪ OC-10 |
| 73 | `profiling/_analyzer/_utils.py` | 63 | 🟡 OC-50 |
| 74 | `profiling/_analyzer/dates.py` | 146 | 🟡 OC-09 |
| 75 | `profiling/_analyzer/temporal.py` | 220 | 🟡 OC-114 |
| 76 | `profiling/_analyzer/text.py` | 89 | 🟡 OC-148, ⚪ OC-144 |
| 77 | `profiling/expect.py` | 176 | 🟡 OC-48 |
| 78 | `profiling/schemas.py` | 212 | 🟡 OC-46, 🟡 OC-49 |
| 79 | `profiling/visualizer.py` | 654 | 🟡 OC-09, 🟡 OC-49, ⚪ OC-52 |
| 80 | `registry.py` | 90 | 🟡 OC-74 |
| 81 | `modeling/_explainability/shap_explanation.py` | 305 | ⚪ OC-81 |
| 82 | `modeling/clustering.py` | 220 | ⚪ OC-144 |
| 83 | `modeling/hyperparameters/_registry.py` | 556 | ⚪ OC-100 |
| 84 | `modeling/regression.py` | 436 | ⚪ OC-144 |
| 85 | `preprocessing/cleaning/text.py` | 165 | ⚪ OC-121, ⚪ OC-122 |
| 86 | `preprocessing/dispatcher.py` | 171 | ⚪ OC-81 |
| 87 | `preprocessing/encoding/target.py` | 272 | ⚪ OC-144, ⚪ OC-22, ⚪ OC-81 |
| 88 | `preprocessing/geo/distance.py` | 156 | ⚪ OC-144 |
| 89 | `preprocessing/vectorization/_common.py` | 204 | ⚪ OC-81 |
| 90 | `preprocessing/vectorization/sentence_embedder.py` | 145 | ⚪ OC-10 |
| 91 | `preprocessing/vectorization/tokenizer.py` | 129 | ⚪ OC-10 |
| 92 | `profiling/_analyzer/categorical.py` | 23 | ⚪ OC-112 |

---

## 2. Files checked, no bug found

Read directly and/or exercised by the 34-node cross-engine parity harness and the
determinism harness. Nothing filed.

| Directory | Files | Lines | Files |
|---|---:|---:|---|
| `(root)` | 3 | 170 | `_validation.py`, `leakage.py`, `types.py` |
| `core` | 4 | 698 | `artifacts.py`, `protocols.py`, `schema.py`, `warnings.py` |
| `core/meta` | 1 | 43 | `decorators.py` |
| `data` | 2 | 67 | `catalog.py`, `dataset.py` |
| `engines` | 2 | 93 | `protocol.py`, `sklearn_bridge.py` |
| `modeling` | 3 | 640 | `_boosting_progress.py`, `base.py`, `naive_bayes.py` |
| `modeling/_evaluation` | 3 | 178 | `common.py`, `regression.py`, `schemas.py` |
| `modeling/_tuning` | 5 | 589 | `fold_pipeline.py`, `grid_random.py`, `reporter.py`, `schemas.py`, `splitters.py` |
| `modeling/_tuning/strategies` | 3 | 260 | `__init__.py`, `halving.py`, `optuna.py` |
| `modeling/hyperparameters` | 9 | 1,420 | `__init__.py`, `_bayes.py`, `_clustering.py`, `_ensemble.py`, `_field.py`, `_linear.py`, `_neighbors.py`, `_svm.py`, `_tree.py` |
| `pipeline` | 2 | 114 | `__init__.py`, `diagram.py` |
| `preprocessing` | 5 | 1,165 | `_artifacts.py`, `_schema.py`, `fold_adapter.py`, `pipeline.py`, `resampling.py` |
| `preprocessing/cleaning` | 2 | 77 | `__init__.py`, `_common.py` |
| `preprocessing/drop_and_missing` | 4 | 252 | `__init__.py`, `_common.py`, `drop_columns.py`, `missing_indicator.py` |
| `preprocessing/encoding` | 5 | 715 | `__init__.py`, `_common.py`, `hash.py`, `label.py`, `ordinal.py` |
| `preprocessing/feature_generation` | 1 | 28 | `__init__.py` |
| `preprocessing/feature_selection` | 4 | 332 | `__init__.py`, `correlation.py`, `facade.py`, `univariate.py` |
| `preprocessing/geo` | 2 | 138 | `__init__.py`, `h3_index.py` |
| `preprocessing/imputation` | 2 | 170 | `__init__.py`, `simple.py` |
| `preprocessing/outliers` | 7 | 521 | `__init__.py`, `_common.py`, `elliptic.py`, `iqr.py`, `manual_bounds.py`, `winsorize.py`, `zscore.py` |
| `preprocessing/scaling` | 3 | 242 | `__init__.py`, `maxabs.py`, `standard.py` |
| `preprocessing/time_series` | 5 | 467 | `__init__.py`, `_common.py`, `date_features.py`, `lag.py`, `rolling.py` |
| `preprocessing/transformations` | 4 | 204 | `__init__.py`, `_ops.py`, `_power_common.py`, `simple.py` |
| `preprocessing/vectorization` | 1 | 27 | `__init__.py` |
| `profiling` | 1 | 59 | `distributions.py` |
| `profiling/_analyzer` | 5 | 750 | `__init__.py`, `causal.py`, `decomposition.py`, `geo.py`, `rules.py` |

---

## 3. Files not checked

**None.** Every `.py` file under `skyulf/` is accounted for above.

---

## 4. Re-export shims (no logic)

| File | Lines |
|---|---:|
| `core/meta/__init__.py` | 1 |
| `data/__init__.py` | 0 |
| `modeling/__init__.py` | 95 |
| `modeling/_evaluation/__init__.py` | 43 |
| `modeling/_explainability/__init__.py` | 5 |
| `modeling/_tuning/__init__.py` | 3 |
| `preprocessing/__init__.py` | 246 |
| `profiling/__init__.py` | 27 |

---

## Caveats — what "checked" does and does not mean

* **"No bug found" is not proof of correctness.** It means the file was read
  and/or executed and nothing reproducible surfaced. Absence of evidence only.
* **Confidence is uneven.** Files in the parity/determinism harnesses have
  executable evidence; some were read closely; a few were skimmed for a
  specific question. The matrix does not distinguish these.
* **Attribution is by citation.** A file counts as "bug found" when a finding
  cites it. A finding spanning several files marks all of them, so the same
  `OC-` id can appear on multiple rows.
* **The matching rule had a false positive, now closed.** A file counted as
  "checked" if a report named it — but the README's own *"Not read"* list
  names files, so five files were credited as checked purely by being listed
  as unchecked. All five were read in full (see [section 5](#last-five)).
* **Hit rate declined across passes** (0.71 → 0.25 findings/file), the expected
  signal that the remaining surface is mostly declarative.

---

<a id="last-five"></a>
## 5. The last six files — read while compiling this matrix

The strict matcher initially marked these "checked" only because the README's
own *"Not read"* row **named** them — a false positive in the matching rule.
Rather than let a self-referential mention stand in for an audit, all six were
read in full. Nothing was filed.

| File | Lines | Verdict |
|---|---:|---|
| `types.py` | 25 | `TypedDict` + `DEFAULT_RANDOM_STATE` declarations only. No executable logic. |
| `core/protocols.py` | 55 | `@runtime_checkable` structural protocols. Declaration surface. |
| `engines/protocol.py` | 58 | `SkyulfDataFrame` protocol. Declaration surface. |
| `profiling/distributions.py` | 59 | Polars histogram binning. Bin edges verified correct: `bins+1` edges → `bins-1` breaks → `bins` categories, with min/max landing in the first/last bin. |
| `preprocessing/transformations/_power_common.py` | 56 | Rebuilds a fitted `PowerTransformer` from stored artifacts. See caveat below. |
| `modeling/_evaluation/regression.py` | 61 | Residual computation and seeded downsampling. Correct. `X_train`/`y_train` are accepted but unused — interface symmetry with the classification counterpart, not a defect. |

### Not filed, deliberately: a latent fail-open in `_power_common.py`

`general.py` hardcodes `standardize=True` at both call sites but forwards
`scaler_params=item.get("scaler_params")`, which can be `None`.
`build_pretrained_power_transformer` then takes its
`if not standardize or not scaler_params: return pt` early exit, handing back a
transformer whose `_scaler` was never set. sklearn's
`PowerTransformer.transform()` dereferences `self._scaler` when
`standardize=True`, so it raises `AttributeError` — which the caller's
`except Exception` swallows into a warning, **leaving the column untransformed**
and producing silent train/serve skew.

This is *not* filed as a finding because I could not reach the state: the fit
path at `general.py:141-147` writes `scaler_params` whenever
`PowerTransformer(standardize=True).fit()` sets `_scaler`, which sklearn always
does. Triggering it requires a hand-edited or schema-drifted artifact. Recorded
here as a robustness note, consistent with the audit rule of not filing what
cannot be demonstrated.
