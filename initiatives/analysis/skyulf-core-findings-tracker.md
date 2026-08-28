# skyulf-core findings — fix tracker

**Source audit:** [`skyulf-core-findings.md`](skyulf-core-findings.md) (run on an older
snapshot, `/Users/BH7043/Skyulf`).
**Re-verified:** 2026-08-28 on branch `086` (v0.8.5), Polars 1.43.2, full suite green
(3,457 passed, 56 skipped).

The audit predates the 080–086 fix waves. 7 of 31 findings no longer reproduce;
3 are partial; 21 are live. This file tracks the fix work, few-at-a-time for easy
items, one-at-a-time for hard ones.

**Status key:** ⬜ open · 🟨 in progress · ✅ done · ⏭️ parked

---

## Already fixed before this tracker (verified 2026-08-28)

| ID | Finding | Why it no longer applies |
|---|---|---|
| F-01 | Polars `DataFrame.gather` missing | Polars 1.43.2 has `DataFrame.gather`; both split paths green |
| F-03 | Test suite red (11 failures) | 3,457 passed, 0 failed on `086` |
| F-04 | No class-imbalance handling | `class_weight` in tree-ensemble search spaces (`hyperparameters/_tree.py`); `sklearn_wrapper.py` sample-weight shim for non-native models |
| F-06 | All-folds-failed returns a model | `_evaluate_search_candidates` returns `best_params=None` → `ValueError("All trials failed…")` with first error |
| F-16 | mypy Optional-handling bugs | `SplitPayload` non-optional in `SplitDataset`; isinstance guards in `modeling/base.py`, `preprocessing/pipeline.py` |
| F-25 | Test deps undeclared | `hypothesis`, `pytest-benchmark` in `setup.py` dev extra |
| F-26 | Subprocess can't import skyulf | Test inherits full env; passes |
| F-27 | Stub seams silently no-op | Real local defaults + `NotImplementedError` on abstract methods |

## Partial

| ID | Status | State |
|---|---|---|
| F-13 | ⬜ | Threshold tuning reachable via `Pipeline.optimize_thresholds()` / `predict(use_tuned_thresholds=True)`, but not wired into `TuningConfig`/default path |
| F-14 | ⬜ | `global` statements 9 → 3; no contextvar/context-manager scoping yet |
| F-31 | ⬜ | `_AnalyzerState` now used; dispatcher `logger.exception` consolidated but relies on caller context |

## Live — fix queue

Easy items batched; hard items one at a time.

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| F-24 | ⚪ | Keep all fold errors, not just the first | 1 line | ✅ |
| F-20 | 🟡 | 8× `logger.error` in except → `logger.exception` | mechanical | ✅ |
| F-30 | ⚪ | 3× ValueError-for-type-validation → TypeError | mechanical | ⬜ open — tests pin `ValueError` (e.g. `test_evaluation_clustering.py:214`, `test_pipeline_split_extraction.py`); flipping to `TypeError` is a breaking change, deferred pending a compat call |
| F-02 | 🔴 | `TuningConfig.random_state` ignored at refit | small | ✅ |
| F-21 | 🟡 | 33 hardcoded `random_state: 42` defaults | half day | ✅ |
| F-05 | 🔴 | Metrics computed in one try-group, swallowed silently | ~2 h | ✅ |
| F-12 | 🟠 | KS drift decision on p-value only; statistic discarded | ~2 h | ✅ |
| F-22 | 🟡 | Default engine (pandas) undocumented | doc | ✅ |
| F-28 | ⚪ | Import-as-probe → `importlib.util.find_spec` | small | ✅ |
| F-29 | ⚪ | Stale/blanket noqa cleanup | mechanical | ✅ |
| F-17 | 🟡 | 20 mutable class attrs need `ClassVar` | ~2 h | ✅ |
| F-10 | 🟠 | `_HARDCODED_MODEL_MAP` shadows registry | half day | ✅ |
| F-15 | 🟡 | Pickle-based reproducibility digest | ~1 day | ⬜ |
| F-07 | 🟠 | `._df` unwrapping → public `to_native()` | ~1 day | ⬜ |
| F-09 | 🟠 | Dual-engine dispatch mapping (Spark prereq) | ~2 days | ⬜ |
| F-08 | 🟠 | Split the `SkyulfDataFrame` protocol | ~3 days | ⬜ |
| F-11 | 🟠 | Break import cycles (196 deferred imports) | ~2 days | ⬜ |
| F-18 | 🟡 | Split `_tuning/engine.py` (1,572 lines) | ~2 days | ⬜ |
| F-19 | 🟡 | Split `pipeline.py` responsibilities | ~1 day | ⬜ |
| F-23 | ⚪ | Enable BLE001 / broad-catch rule | half day | ✅ |
| F-13 | 🟡 | Wire threshold tuning into TuningConfig | ~1 day | ⬜ |
| F-14 | 🟡 | contextvar scoping for engine/backend globals | ~1 day | ⬜ |

## Log

### 2026-08-28 — Batch 1 (branch 086)

- **F-02** fixed: `_refit_best_model` now lets `TuningConfig.random_state` win over the
  calculator default, while a seed in the search space still takes precedence
  (`modeling/_tuning/engine.py`). Reproduced the bug first (seed 1 and 999 both refit at 42),
  then verified the fix.
- **F-21** (partial): same seed overlay threaded through the remaining tuning construction
  paths — per-candidate CV folds and the halving/optuna base estimator — so the whole tuning
  run (search + refit) honors the caller's seed consistently.
- **F-24** fixed: every fold failure is appended (not just the first); the all-trials-failed
  error now reports how many additional failures were suppressed.
- **F-20** fixed: 8 `logger.error`-in-except sites → `logger.exception` (tuning engine,
  resampling, power transform, causal discovery, correlations, histograms).
- Regression tests added in `tests/unit/test_tuning_engine.py` (seed propagation, seed
  precedence, fold-error surfacing). Full tuning suite green (98 passed).
- **F-30** deliberately left: flipping those three `ValueError`s to `TypeError` would break
  behaviour pinned by existing tests and downstream `except ValueError` callers.
- **F-05** fixed: metric computation restructured around `_try_add_metric`
  (`modeling/_evaluation/metrics.py`) — each metric fails in isolation with a logged warning
  and is omitted (not nan: `sanitize_metrics` strips non-finite values and nan would poison
  tuning comparisons). `predict_proba` failures in `classification.py` now log at info level.
  Tests pin isolation + omission-not-nan. Full suite green (exit 0) as the regression gate.

### 2026-08-28 — F-12 (branch 086)

- **F-12** fixed: the KS drift decision now thresholds the statistic (max CDF distance,
  sample-size robust) instead of the p-value (sample-size dependent). New default
  `ks_statistic: 0.1` in `_merge_thresholds`; both `ks_statistic` (decision) and
  `ks_test_p_value` (diagnostic) are emitted (`profiling/drift.py`).
- Blast radius beyond skyulf-core: `backend/monitoring/router.py` mirrors the calculator's
  defaults and passes overrides straight through — updated the mirror, the override routing,
  the effective-threshold recording, and added `ks_statistic` to the alert column summary.
  DB columns (`DriftThresholdVersion.ks`, `DriftCheckResult.threshold_ks`) keep their names;
  they now store the statistic threshold.
- Tests: `test_profiling_drift.py` updated (metric sets, threshold keys) + new regression test
  pinning "significant p-value with tiny statistic must not flag drift" (26 passed);
  `tests/integration/test_drift_alerts_ops003.py` mirror-defaults updated (25 passed).
- **Follow-up (frontend, cosmetic):** `frontend/ml-canvas` drift pages still label the KS
  threshold/value as "KS p-value" (`ThresholdsPanel.tsx`, `DriftTable.tsx`,
  `DriftAlertModal.tsx`, `csvExport.tsx`) — relabel to KS statistic, widen the input range
  (0.01–0.5-ish instead of 0.001–0.2), and surface the statistic in the table/alert detail.

### 2026-08-28 — Batch 2 (branch 086)

- **F-22** fixed (docs): `engines/registry.py` default now carries a deliberate-default
  comment; README gained a "Compute engines" section stating pandas is the fallback default
  and Polars is opt-in (`EngineRegistry.set_active_engine("polars")` or just pass Polars data).
- **F-28** fixed: the four try/except import probes in `profiling/_analyzer/_utils.py`
  (11+ class imports used only to set availability booleans) replaced with
  `importlib.util.find_spec`. Side benefit: importing the analyzer no longer imports
  sklearn/scipy/statsmodels eagerly.
- **F-29** fixed: 35 stale `noqa`s removed via RUF100, two blanket `noqa`s (sklearn
  experimental side-effect imports) dropped, and the blanket `# type: ignore` in
  `modeling/base.py` replaced with a proper isinstance narrowing that `ty` verifies clean.
  Caveat found along the way: running RUF100 with `--select RUF100` disables every other
  rule, so genuinely-needed noqas look stale — 2 of the 35 (B010 in `test_tuning_engine.py`)
  were restored as real fixes (setattr → direct assignment) instead.
- Verification: ruff clean, `ty check skyulf` clean, targeted suites green (tuning 98,
  split/registry/analyzer 62, drift 26, drift-alerts 25). Full suite gate passed (exit 0).

### 2026-08-28 — F-17 (branch 086)

- **F-17** fixed: all 20 RUF012 sites annotated `ClassVar` — `NodeRegistry` dicts
  (`skyulf/registry.py`), `EngineRegistry` registry/dispatch maps (`engines/registry.py`),
  `_METRIC_ALIAS_MAP` (`_tuning/engine.py`), `_SOLVER_PENALTIES` + calibration
  `BASE_ESTIMATORS` (`classification.py`), ensemble `BASE_ESTIMATORS` base (`ensemble.py`),
  and the 12 transformer-type set constants in `preprocessing/pipeline.py`.
  Annotation-only change (all plain classes, no pydantic/dataclass semantics involved).
- Verification: RUF012 now empty, ruff + `ty check skyulf` clean; targeted suites green
  (registry/engines/classification/pipeline/tuning/ensemble: 399 passed). Full suite gate
  passed (exit 0).

### 2026-08-28 — F-10 (branch 086)

- **F-10** fixed: `_HARDCODED_MODEL_MAP` and `_resolve_from_hardcoded_map` deleted from
  `pipeline.py` (plus the 8 model-class imports that existed only to feed the map).
  The registry is now the only source of truth — verified first: all 100 registered nodes,
  including the 4 previously-hardcoded types, resolve complete calculator+applier pairs.
- Failure paths now distinguish "unknown model type" from "partially registered
  (calculator without applier)" with explicit errors; tuner base-model resolution lost
  its fallback too.
- Guard added: `test_every_registered_node_resolves_calculator_and_applier`
  (`tests/unit/test_registry_contract.py`) walks the whole registry so the gap can't reopen.
- Verification: ruff + ty clean; registry-contract + pipeline unit suites (10) and the
  three pipeline integration suites (35) green. Full suite gate passed (exit 0).

### 2026-08-28 — F-23 (branch 086)

- **F-23** done: `BLE` (blind-except) enabled in `[tool.ruff.lint] select` (root
  `pyproject.toml`). The audit's "6 bare try/except/pass" sites were already gone from
  current code — only the `except Exception` sites remained (72 in skyulf-core; the
  repo-wide count is 193, the delta being backend/ and root tests/).
- Staged rollout: `backend/**` and root `tests/**` carry per-file `BLE001` ignores so
  this pass stays in the audit's scope (skyulf-core). Backend triage is a follow-up.
- Triage of all 72 skyulf-core sites: every one is deliberate — best-effort profiling
  sections that log and return None, per-column/per-operation isolation that logs and
  skips, optional-dependency guards (`predict_proba`, shap, vader, sentence-transformers),
  or test harnesses that convert failures into strings/skips/fails. None was a masking
  bug, so all sites got justified `# noqa: BLE001 - <reason>` waivers instead of narrowed
  catches (narrowing would regress the isolation each site documents). New code must now
  justify any broad catch.
- Verification: `ruff check .` clean repo-wide (no leftover violations, no stale noqas),
  `ruff format --check` clean, `ty check` clean (exit 0).
- Gate correction: the full-suite run unmasked 11 stale test pins left behind by F-10 and
  F-28 (earlier gates piped pytest through `tail`, whose exit code hid the failures).
  - `test_pipeline_coverage.py`: the "manual dispatch"/"hardcoded fallback" tests pinned
    the `_HARDCODED_MODEL_MAP` F-10 deleted. Rewritten to pin the new contract: known
    types resolve via the registry (compared against live `NodeRegistry` classes), and
    partial registration raises the explicit "only partially registered" /
    "Unknown base model type for tuner" errors.
  - `test_profiling_utils.py`: the missing-dependency test still forced `__import__`
    failures; rewritten for F-28's `importlib.util.find_spec` probes
    (`test_optional_dependency_flags_flip_to_false_when_package_missing`).
  - Both files green (22 passed). Full suite gate passed with a real exit code:
    3464 passed, 56 skipped (exit 0).

### 2026-08-28 — F-21 (branch 086)

- **F-21** done: single seed owner. `DEFAULT_RANDOM_STATE = 42` now lives in
  `skyulf/types.py` (leaf module — no import cycle) and is injected at exactly one
  merge point: `SklearnCalculator._resolve_fit_params` → `_inject_default_seed`,
  which only sets `random_state` when the resolved params don't already carry one
  and the estimator's signature (or `**kwargs`) accepts it. Precedence: explicit
  node `params.random_state` > `TuningConfig.random_state`/`cv_random_state` >
  the fallback; an explicit `"random_state": null` still means "unseeded".
- Scope was bigger than the audit's "33": 31 `"random_state": 42` default-dict
  entries removed (classification 13, regression 12, clustering 6) plus 41 literal
  `42`s replaced with the constant (calibration + ensemble `BASE_ESTIMATORS`
  lambdas, cross-validation signatures, `TuningConfig` defaults, hyperparameter
  defaults, split/outlier/encoding/profiling paths) across 13 files. Side benefit:
  meta-estimators (voting/stacking/calibrated) now get seeded via injection even
  when their config didn't carry a seed.
- Tests: 4 new regression tests in `test_modeling_sklearn_wrapper.py` (seed
  injected when unconfigured, user override wins, explicit `None` respected, no
  injection for estimators without a `random_state` param). Two stale pins in
  `test_classification_gaps.py` (SGD `default_params` and registry metadata
  contained the literal) re-pinned to the new contract — they now assert the
  injection instead.
- Docs: `docs/user_guide/configuration.md` gained a "Reproducibility and seeds"
  section (precedence, `null` opt-out, example config).
- Found along the way: two F-29 `unresolved-attribute` ty errors in
  `test_tuning_engine.py` (masked earlier by piping ty through `tail`) fixed via
  `fake_module.__dict__[...]` assignment — no suppressions.
- Verification: ruff check + format clean, `ty check` exit 0, full suite 3468
  passed, 56 skipped (pytest_exit=0).
- Follow-up (frontend, noted not fixed): `ClassificationNode.tsx` /
  `EnsembleNode.ts` mirror the `random_state: 42` UI default, and
  `pipelineConverter.ts:224` seeds IterativeImputer with `?? 0` — inconsistent
  with core's 42 convention.
