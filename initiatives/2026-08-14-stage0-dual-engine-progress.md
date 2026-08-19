# Stage 0 (Trust Floor) — Execution Progress

**Date:** 2026-08-14
**Branch:** `078` (all changes in the working tree — **nothing committed yet**)
**Scope:** Stage 0 of `initiatives/growth/2026-08-11-growth-plan.md`, executing
the Tier 1 findings of `initiatives/dual-engine-correctness/2026-08-11-audit-findings.md`
plus the plan's T-items.

**Status legend:** ✅ fixed + verified with red-green tests · 🟡 implemented,
covered by existing suites, own regression tests still owed · ⏳ not started ·
⏭️ descoped by plan decision.

## Stage 0 findings — status

| ID | Defect (short) | File(s) | Fix (one line) | Status | Verification |
|---|---|---|---|---|---|
| F-01 | `DummyEncoder` emits null dummies on Polars → training hard-fails | `skyulf/preprocessing/encoding/dummy.py` | Fill dummy columns from the null-flag instead of letting nulls propagate | ✅ | red-green test; full suite green |
| F-02 | Inference depends on JSON key order; reordered keys silently change predictions | `backend/ml_pipeline/deployment/service.py` | Reindex transformed X to the recorded training `feature_columns` order before predict | 🟡 | 102 existing deployment tests pass; dedicated key-order regression test owed |
| F-03 | `feature_columns` recorded post-transform but validated pre-transform | `backend/ml_pipeline/deployment/service.py` | Validate incoming requests against the feature engineer's **pre-transform** input columns (`_extract_features_from_engineer`) | 🟡 | 102 existing deployment tests pass; dedicated regression test + frontend form check owed |
| F-04 | Polars imputer: NaN mean → NaN fill (no-op) and false "0 missing" report | `skyulf/preprocessing/imputation/simple.py`, `imputation/_common.py` | `fill_nan` in applier; NaN-excluded stats and NaN-counted missing counts in fit (residual gap closed 2026-08-14, see below) | ✅ | `test_simple_imputer_polars_float_nan_parity_with_pandas` + `..._most_frequent_ignores_float_nan` (new, red→green) |
| F-05 | Missing-counts computed with Polars semantics that diverge from pandas `isna()` | `skyulf/preprocessing/imputation/_common.py` | `_polars_missing_counts` now counts null + NaN for float columns, matching `isna()` | ✅ | same parity tests as F-04 |
| F-06 | Outlier nodes (`iqr`, `zscore`) use `is_null()` on Polars → NaN rows never flagged; announced fix never applied | `skyulf/preprocessing/outliers/iqr.py`, `zscore.py`, `manual_bounds.py` | Flag null **or** NaN per engine (`is_nan` on float columns) | ✅ | red-green parity tests |
| T1 | Lag/Rolling return stale, misaligned `y` when rows are sorted/dropped | `skyulf/preprocessing/time_series/lag.py`, `rolling.py` | Route `(X, y)` tuple through `apply_dual_engine` (pattern from `deduplicate.py`) so row drops propagate to `y` | ✅ | red-green test |
| T2 | `FeatureSelection` default `variance` unknown to its own dispatch table | `skyulf/preprocessing/feature_selection/facade.py` | Accept `variance` as alias for `variance_threshold` | ✅ | red-green test |
| T3 | `GeneralBinning` default `uniform` produces no bins | `skyulf/preprocessing/bucketing.py` | Accept `uniform` as alias for `equal_width` | ✅ | red-green test |
| T4 | Demo `/health` reports `0.0.0-dev` (backend not installed in image) | — | — | ⏭️ | descoped to Stage 1 demo backlog by plan |
| T5 | No registry-wide contract test (the class of bug keeps recurring) | `skyulf-core/tests/test_registry_contract.py` (new) | Parametrised contract over every registered node: (1) own defaults work, (2) `y` length/order follows `X`, (3) engine parity on float NaN **and** wrapped frames | ✅ | 175 cases, all passing |
| T6 | `FeatureMath` silently drops datetime features on mixed-offset input | `skyulf/preprocessing/feature_generation/_pandas_ops.py` | `pd.to_datetime(..., utc=True)` + fail-loud instead of swallowed warning | ✅ | red-green test |

## Session fixes — test infrastructure and environment (2026-08-14)

| Area | Problem | Fix | Evidence |
|---|---|---|---|
| `tests/test_encoding_hash.py` | Subprocess died on Windows with `WinError 10106` (stripped env broke Winsock DLL loading) | Inherit full `os.environ`, override only `PYTHONHASHSEED` | test passes |
| `skyulf/engines/polars_engine.py` | `to_numpy()` raised "need at least one array to concatenate" on 0-column frames | Return `np.empty((height, 0))` for `width == 0`, mirroring pandas | `test_evaluation_clustering_polars[raw_polars]` passes |
| `skyulf/modeling/_evaluation/metrics.py` | 0-feature input raised the misleading row-count error (polars collapses 0-col frames to shape `(0,0)`) | sklearn-style "0 feature(s)" guard **before** the row-count check | both engines now raise the same message |
| `skyulf/modeling/_evaluation/clustering.py` | `pl.DataFrame(np.empty((n,0)))` silently collapses to `(0,0)`; comment claimed otherwise | Replaced with `pl.DataFrame()` + correct comment; error raised by metrics layer | clustering suite green |
| `imputation/_common.py` (F-04 residual) | Polars fit still included NaN in mean/median/mode and undercounted missing values | `drop_nans()` before stats; NaN counted in `_polars_missing_counts` for float columns | 2 new parity tests, red→green |
| Root `.venv` / `uv.lock` | `.venv` carried a **stale pip-installed editable `skyulf-core` 0.3.4** outside uv's management while `uv.lock` tracks the editable 0.5.8 — the visible "uv.lock problem" | `uv sync --locked` (root); `.venv` now shows `skyulf-core 0.5.8` editable; both lockfiles pass `uv lock --check` | verified via `importlib.metadata` |

Notes on the items the owner asked to check:

- **metrics.py** — the only uncommitted change is the 0-feature guard above;
  reviewed and covered by tests. No other issue found.
- **bucketing.py** — the only change is the `uniform` alias (T3); covered by
  red-green tests. No other issue found.
- **uv.lock** — both lockfiles (`./uv.lock`, `skyulf-core/uv.lock`) are in sync
  with their `pyproject.toml` (`uv lock --check` exit 0). The real problem was
  the stale root `.venv` install; fixed. `skyulf-core`'s packaged version is
  `0.5.8` (`setup.py`), deliberately reverted per commit `efdbda48`.

## Test evidence (2026-08-14)

- `skyulf-core` full suite: **3183 collected · 3113 passed · 0 failed**
  (70 skipped — all opt-in benchmark gates; 1 xfailed).
- Includes `test_registry_contract.py` (175 cases) and the 2 new NaN-parity
  imputation tests.
- Backend deployment tests: **102 passed** (covers the F-02/F-03 code paths
  via existing fixtures).

## Still open

1. **Commit.** Everything above is uncommitted on `078`; a `git checkout`
   would lose it. Owner confirmation pending.
2. **F-02/F-03 regression tests** (red-green, JSON-key-order and
   pre-transform validation) + the F-03 frontend form check required by
   repo policy. This is what keeps them at 🟡.
3. **Wave T1 remainder from the audit:**
   - F-33 (stale evaluation-response clobber) — ✅ fixed: `useEvaluationFetch`
     hook with monotonic request-sequence guard (see audit doc).
   - F-34 (threshold-tuning 500 on string labels) — ✅ fixed: per-request
     `_build_scorer()` with a label-aware `roc_auc` scorer (see audit doc).
   - F-35 (weighted-scorer ambiguity in binary threshold tuning) — ✅ fixed:
     positive-class scorers (`average="binary"`, `pos_label=classes[1]`) for
     `f1`/`precision`/`recall` on 2-class jobs (see audit doc).
   - F-37 (SHAP/feature-importance coverage) — ⏳ not started.
   Wave T2b (F-36, F-38–F-44) is later still.
4. **A2.2 template fix** on `078`, then demo cherry-pick (Stage 1a) — ⏳.
5. **PyPI release** of the fixed `skyulf-core` (`core-v*` tag) — pending
   commit.
6. **Disposition of 39 staged deletions** (`PRODUCT.md`, `landing-iterations/`,
   `redesign/`) present in the index — intent unconfirmed; untouched here.
