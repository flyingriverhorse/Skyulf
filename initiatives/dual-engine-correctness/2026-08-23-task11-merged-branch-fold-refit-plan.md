# Task #11 — Per-fold refit for merged preprocessing branches (fork-join scope)

Date: 2026-08-23 · Branch: 082 · Status: implemented (all phases complete, red→green)

## Context

F-15 gave us leakage-free CV/tuning: every CV/tuning fold re-fits preprocessing on fold-train rows only. Today, that works for linear chains (loader → FE/transformer nodes → training). But when a training node has **multiple inputs** (merged branches), `_resolve_fold_preprocessing` bails and logs the skip-fallback warning — those graphs still leak.

User-approved scope: **fork-join topology only** —

```
data → [shared trunk ... TrainTestSplitter]  ← fork point F (last step of its node)
            ├── branch 1 (single-transformer nodes)
            ├── branch 2 ...
            └── branch N
                    → merge → training node
```

No subgraph-replay mechanism exists in the engine (nothing re-executes arbitrary DAG fragments), so we re-run the branch step lists inside a new `FoldPreprocessor` adapter and reproduce the engine's pure-strategy column merge. Any graph that doesn't match this exact shape falls back to the existing skip-with-warning path (never fail a run, never silent leak).

## Design (validated, with 6 refinements)

1. **Fork point F** = a `TrainTestSplitter` (or `Split`) that is the **last step** of its node's step list. Any other arrangement (splitter mid-chain, `feature_target_split`-only trunk → tuple NCA artifact activates ownership analysis) → bail.
2. **Payload** only via `_split_train_payload(artifact_store.load(F), target_col)` — F's stored SplitDataset is the single source of truth.
3. **Static screening**: reject branches containing splitters (`SPLITTER_STEP_TYPES`) or row-count-changing steps (`FeatureEngineer._ROW_DROP_TYPES ∪ _RESAMPLING_TYPES`) → bail.
4. **Branch order** from `self._merge_input_order(training_node)`; loader ids of all branch chains must match.
5. **Adapter merge** must reproduce the pure-strategy path of `_merge_frames_columnwise` **exactly, including first_wins reversed column insertion order** (train/test alignment is load-bearing).
6. **Routing**: merged resolver only attempted when `_upstream_fe_chain` returned `None` **and** the training node has >1 unique input (keeps `test_merged_branches_fall_back_with_warning` green).

Common-trunk detection: F = last node of the longest common node-id prefix of the branch chains. All-empty branches → silent `None`.

## Implementation (ordered phases, red→green each)

### Phase 1 — Core `MergedBranchFoldAdapter` (red → green)
- New tests: `skyulf-core/tests/unit/test_fold_merged_adapter.py`
  - `fit_transform`: re-runs each branch step list on the fold-train payload, merges per strategy, returns `(X, y)`.
  - `transform`: keeps all held-out rows (F-18).
  - last_wins and first_wins column-order/ownership parity with `_merge_frames_columnwise` pure path (incl. reversed insertion order for first_wins).
  - Deep-copy isolation per fold (originals unfitted), `sklearn.clone()` safe.
- Implement in `skyulf-core/skyulf/preprocessing/fold_adapter.py`:
  - `MergedBranchFoldAdapter(branch_step_lists, strategy, target_column, drop_columns)` implementing the `FoldPreprocessor` protocol.
  - Module-level `_merge_branch_frames_columnwise(frames, strategy)` mirroring the pure path of `_merge_frames_columnwise` (`backend/ml_pipeline/_execution/engine/_merge.py:206`).
  - Export from `skyulf-core/skyulf/preprocessing/__init__.py`.

### Phase 2 — Core integration honesty proof
- New test (in `skyulf-core/tests/integration/test_tuning_per_fold_refit.py`): two branches `[WOEEncoder(city)]` + `[StandardScaler(num)]` on a noise target.
  - Leaky control (merge once on full train) disc AUC > 0.75.
  - Adapter-wrapped CV disc AUC < 0.65.
  - Both merge strategies.

### Phase 3 — Backend resolver (red → green)
- `backend/ml_pipeline/_execution/engine/_feature_eng.py`:
  - New `_resolve_fold_preprocessing_merged(training_node, target_col)` implementing fork-point detection, trunk/loader validation, static screening, payload load via existing `_split_train_payload` (:325-329), and construction of `MergedBranchFoldAdapter`.
  - Routing edit at ~:344-349: only when chain is `None` **and** `len(set(training_node.inputs)) > 1`; otherwise existing fallback.
  - Mixin type stubs for `_merge_input_order`, `_get_merge_strategy`, `_upstream_dropped_columns` (pattern at `_feature_eng.py:37-41`).
- New backend tests in `tests/integration/test_fold_preprocessing_refit.py` (or sibling file):
  - Fork-join honesty end-to-end (scenario-06 shape: split → scaler_a/scaler_b → training, noise target; refit log line present, disc < 0.65).
  - first_wins variant honesty.
  - Fallback: nested merge (merge node feeding training) → warning fallback.
  - Fallback: branch containing a row-dropping step (e.g. `DropMissingRows`).
  - Fallback: learning step between splitter and fork (non-pure-strategy graph).

### Phase 4 — Regression sweep
- `uv run pytest` over: `skyulf-core/tests/unit/test_fold_merged_adapter.py`, `skyulf-core/tests/unit/test_fold_pipeline.py`, `skyulf-core/tests/integration/test_tuning_per_fold_refit.py`, `tests/integration/test_fold_preprocessing_refit.py`, `tests/integration/test_fold_preprocessing_stress.py`, `tests/integration/test_merge_scenarios_e2e.py`
- `ruff check` + `ruff format` + `ty check` on touched files.

### Phase 5 — Release hygiene
- Core version bump `skyulf-core/setup.py` 0.7.0 → 0.8.0.
- `changelog/0.8.x.md` entry: merged fork-join graphs now refit per fold; non-fork-join merges still fall back with warning.
- Amend `initiatives/dual-engine-correctness/2026-08-23-f15-per-fold-refit-design.md`.
- Fix stale `docs/examples/leakage_proof_pandas.md:485-492` (still claims halving/optuna uncovered and all merged graphs fall back).

## Critical files

| File | Change |
|---|---|
| `skyulf-core/skyulf/preprocessing/fold_adapter.py` | Add `MergedBranchFoldAdapter` + `_merge_branch_frames_columnwise` |
| `skyulf-core/skyulf/preprocessing/__init__.py` | Export |
| `skyulf-core/tests/unit/test_fold_merged_adapter.py` | New unit tests |
| `backend/ml_pipeline/_execution/engine/_feature_eng.py` | `_resolve_fold_preprocessing_merged` + routing |
| `tests/integration/test_fold_preprocessing_refit.py` | New backend fork-join + fallback tests |
| `skyulf-core/setup.py`, `changelog/0.8.x.md`, design note, `docs/examples/leakage_proof_pandas.md` | Hygiene |

## Reused helpers

- `_split_train_payload` (`_feature_eng.py:325-329`), `_resolve_fold_preprocessing` routing (:331-405)
- `SPLITTER_STEP_TYPES`, `FeatureEngineer._ROW_DROP_TYPES`/`_RESAMPLING_TYPES` (screening)
- `_merge_frames_columnwise` semantics (`_merge.py:206`) as the mirror spec
- `_merge_input_order` (`__init__.py:364`), `_get_merge_strategy` (`_merge.py:188`), `_upstream_dropped_columns`

## Verification

1. All Phase 1-3 tests pass red→green (each failing first without the implementation).
2. Measured honesty numbers printed in test output: leaky disc > 0.75 vs refit disc < 0.65.
3. Full regression sweep green (Phase 4), including unchanged merge-scenario e2e suite.
4. `ruff check`, `ruff format --check`, `ty check` clean on all touched files.
5. `test_merged_branches_fall_back_with_warning` still green (fallback path preserved).

## Known constraint (2026-08-26)

Post-split merges are pure-order — ownership is inert for `SplitDataset` baselines, so the last
connected branch wins every shared column. Documented in `docs/guides/multi_path_pipelines.md`
("After a Split: Order Decides Everything") and `docs/user_guide/troubleshooting.md`, and enforced
at training time by the non-numeric fail-fast guard in `_node_runners.py` (findings doc
`2026-08-26-holdout-validation-refit-and-merge-findings.md`, §2 Finding 1).
