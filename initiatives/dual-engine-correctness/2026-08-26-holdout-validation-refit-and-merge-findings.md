# Holdout + Validation-Split Refit (v0.8.4) — Findings & Two Product Realities

**Date:** 2026-08-26
**Status:** Shipped (core + app at **0.8.4**, no version bump), fully tested. Two product
realities surfaced while building the dedicated integration tests are logged below with
root-cause analysis and recommendations. **Neither is a correctness defect.**
**Companion:** `2026-08-23-f15-per-fold-refit-design.md` (Amendments 1–2),
`2026-08-23-task11-merged-branch-fold-refit-plan.md`, `2026-08-11-leakage-enforcement-plan.md`.

---

## 0. What shipped in this slice

The last documented gap in F-15 was **holdout tuning with a validation split**. Previously:

- The backend skipped per-fold preprocessing entirely when `data.validation` existed
  (`_node_runners.py` logged *"Per-fold preprocessing refit skipped…"*).
- `TuningCalculator.tune()` raised `ValueError` if `preprocessing` + `validation_data` were combined.

With SMOTE/WOE upstream, candidates then trained on the already-preprocessed full train frame —
the last remaining optimistic path. This slice closes it:

- Preprocessing refits on the **train rows only**; candidates score against the **untouched
  validation split**. Works for all five strategies:
  - **grid/random** — the custom loop (`engine.py:799-800`) slices concatenated train+val frames
    with a `PredefinedSplit` (`test_fold = [-1]*n_train + [0]*n_val`).
  - **halving/optuna** — `FoldAwareModelStep.fit` receives only the train rows (`test_fold == -1`
    mask) and scores validation rows in the original space via the fold-aware meta-estimator.
- Post-tuning CV (`_run_tuned_cv`) gained the adapter too (it previously scored raw train when a
  validation split was present).
- Final best-model refit unchanged: train-only, preprocessing applied once — the artifact serving uses.
- **Polars payloads are accepted** for both frame pairs; the engine and the fold-aware wrap convert
  them via `to_pandas()` with dtypes intact.

Key entry points:
- `holdout_refit` gate + `_build_predefined_split_cv_frames` — `skyulf-core/skyulf/modeling/_tuning/engine.py:~1350, 507-534`.
- `validation_frames` param threaded through `TuningCalculator.fit`/`tune()` and `base.py:fit_predict`.
- Backend payload reconstruction — `backend/ml_pipeline/_execution/engine/_feature_eng.py`
  (`_split_validation_payload`, `_resolve_fold_preprocessing` 3-tuple) and `_node_runners.py`
  (`preprocessing_validation` kwarg).

---

## 1. Dedicated integration tests & measured scores

New block in `tests/integration/test_fold_preprocessing_refit.py` ("Holdout tuning + validation
split + detailed preprocessing"), five scenarios, all passing. Scores measured live (ROC-AUC,
chance = 0.5):

| Scenario | best_score (holdout/val) | CV mean | val_auc |
|---|---|---|---|
| multi-step chain · **noise** · grid | 0.4718 | 0.4462 | 0.4718 |
| multi-step chain · **signal** · optuna | 0.9680 | 0.9380 | 0.9680 |
| fork-join · **noise** · grid | 0.4787 | 0.4527 | 0.4787 |
| fork-join · **signal** · grid | 0.9680 | 0.9380 | 0.9680 |
| row-isolation · signal · grid | 0.9458 | 0.9027 | 0.9458 |

In holdout mode `best_score` **is** the validation-split score, hence it equals `val_auc`;
`cv_mean` is the post-tuning cross-validation on the train split.

### What "noise" and "signal" mean

These are two deliberately-constructed synthetic datasets that bracket the leakage question. They
are the *targets*, not the features.

- **Noise target** (`_noise_csv_with_nan`): the label is `rng.integers(0, 2)` — a fair coin flip,
  **completely independent of every feature**. There is no learnable relationship. The trap is the
  `city` column (200 categories over 400 rows ≈ 1.6 rows/category): a target-aware encoder like WOE
  can "predict" the label only by **memorising** which label each rare category happened to carry.
  On a leakage-free pipeline such a dataset must score **at chance (~0.5)** — anything materially
  above 0.5 proves held-out rows leaked into preprocessing. Historically (fit-once-on-all-rows) this
  exact shape scored ~0.87; with per-fold refit it now sits at **0.44–0.48**. That is the honesty proof.

- **Signal target** (`_signal_csv`): the label **does** drive the features — `f1 = N(0,1) + 1.5·y`,
  `f2 = N(0,1) − 1.0·y`, and `cat` is informative (80% correlated with `y`), with 5% label noise.
  There is real structure to learn. The guard here is the opposite: a leakage fix must **not** destroy
  genuine predictive power, so these scores must stay high (**> 0.8**). They do (0.90–0.97).

So the pair is a two-sided test of the same guarantee: **noise stays at chance (nothing invented),
signal stays high (nothing destroyed)**. A correct leakage-free pipeline produces exactly this
pattern; a leaking pipeline inflates the noise rows, and an over-aggressive "fix" collapses the
signal rows.

The row-isolation test additionally asserts the structural invariant (see §3): no preprocessing fit
ever receives more than the train-split row count.

---

## 2. Finding 1 — `MergedBranchFoldAdapter` merges with pure `last_wins` (no ownership)

**Verdict: by-design and consistent, not a bug — but it carries a UX constraint worth documenting
and optionally hardening.**

### What happens

The eager engine merge resolves overlapping columns by **ownership** when it can: "a column carried
unchanged by one branch and rewritten by another is not a conflict — the branch that actually
modified it owns it" (`_merge.py:206` `_merge_frames_columnwise`, `_column_owners` `_merge.py:126`).

Ownership needs a **baseline frame** = the nearest-common-ancestor artifact (`_baseline_frame`,
`_merge.py:57`), compared column-by-column (`_column_modifiers`, `_merge.py:93`). In fork-join-after-
split graphs the NCA is the splitter, whose stored artifact is a `SplitDataset` — and
`_coerce_to_frame` returns `None` for a `SplitDataset` (`_merge.py:151`). So `_column_modifiers`
returns `{}`, ownership goes **inert**, and the eager path **falls back to the configured strategy**
(`last_wins`).

The refit adapter (`_merge_branch_frames_columnwise`,
`skyulf-core/skyulf/preprocessing/fold_adapter.py:37`) **mirrors exactly that fallback** — pure
strategy, no ownership. Its docstring says so: "ownership analysis is inert in this shape because
the fork's stored artifact is a SplitDataset."

### Why it matters (and why it's OK)

**Eager (serving) path == refit (tuning/CV) path == pure `last_wins`** for fork-join-after-split.
They produce the *same* merged frame, which is precisely what F-15 requires — tuning optimizes the
features that get served. **No divergence, no leakage.**

The consequence is a **design constraint**, not a defect: after a split, overlapping columns resolve
purely by merge order, so the **last branch fully decides every shared column**. Practical fallout I
hit while writing the tests:

- A last branch that still carries a raw string column breaks the model ("All trials failed"), because
  nothing "rescues" that column from an earlier branch that encoded it.
- The tests therefore had to make **every fork-join branch emit a fully-numeric frame** (e.g. both
  branches impute + WOE-encode + scale), rather than letting branches specialise.

### Recommendations (in order of value/effort)

| Option | What it does | Effort | Risk | Recommendation |
|---|---|---|---|---|
| **A. Document** | State in docs + FE guidance that post-split merges are **order-based**: the last branch wins every shared column, so branches should each emit a clean numeric frame or be disjoint. | Low | None | **Do this.** Cheapest, behaviour is already consistent. |
| **B. Fail-fast guard** | At training, if a merged branch frame still contains object/string columns heading into a numeric model, raise an actionable error instead of the cryptic "Hyperparameter tuning failed: All trials failed." | Low–Med | None | **Do this next.** Highest-value hardening. Turns a confusing failure into a fixable one. |
| **C. Ownership across `SplitDataset` baselines** | Use the SplitDataset's train slot as the comparison baseline so ownership works post-split, letting heterogeneous branches merge "by who changed it." | High | Medium — changes behaviour of existing graphs; must keep eager==refit parity on **both** code paths | **Defer.** Only if users repeatedly want specialised overlapping branches. Needs its own design note + parity tests. |

**Net: nothing is broken. Do A now, B soon, park C.**

---

## 3. Finding 2 — index-based row tracking is unreliable for isolation assertions

**Verdict: a test-instrumentation limitation, not a product defect. The leakage guarantee does not
depend on it and is proven another way.**

### What happens

Trying to prove "a validation row never enters a preprocessing fit" by tracking **row indexes** fails,
for two independent reasons:

1. **The splitter preserves scattered original indexes.** `train_test_split` does not reset the index,
   so the train split keeps a *scattered subset* of the original CSV row labels (`split.py:393`), not
   `0..n_train-1`. Assertions like `fit_rows ⊆ range(n_train)` are therefore wrong for the final refit
   and post-tuning CV.
2. **The wrapped halving/optuna path rebuilds indexes.** Searchers hand `FoldAwareModelStep` numpy
   slices; `_ensure_frames` rebuilds a fresh `RangeIndex` (`fold_pipeline.py:51`), discarding positional
   identity. Only the `grid` custom loop slices pandas with `.iloc` and preserves the concat positions
   (`engine.py:799-800`).

So absolute row identity is not a stable handle across all five strategies.

### Why it's OK

Correctness never depended on row *identity*. The holdout refit enforces isolation **structurally**:
the `PredefinedSplit` mask trains on `test_fold == -1` (train rows) only, and scores the `test_fold == 0`
(validation) rows untouched. The test therefore asserts the robust **row-count invariant** instead:

> Every `fit_transform` call receives **at most the train-split row count** — a leaked fit would
> receive `n_train + n_val` rows. Combined with the near-chance noise scores (§1), this proves
> validation rows never entered a fit, index-free.

### Recommendations

| Option | What it does | Effort | Risk | Recommendation |
|---|---|---|---|---|
| **A. Do nothing** | The guarantee is already proven and tested via the row-count invariant + noise scores. | None | None | **Default.** No correctness gap. |
| **B. Refit audit telemetry** | Have the adapter/engine record per-fold fit/transform **row counts** (optionally a content hash) into metrics/job-log so a run can be audited post-hoc. | Low–Med | None (observability-only) | **Optional.** Nice-to-have for enterprise trust; pairs with Finding 1 option B. |
| **C. Normalize indexes** | Force `reset_index(drop=True)` on payloads to make row identity stable. | Med | Medium — touches many paths, could mask real alignment bugs | **Don't.** Index normalisation for test convenience is the wrong trade; it would hide, not fix, misalignment. |

**Net: do nothing for correctness; B is a cheap optional audit feature.**

---

## 4. What needs to be done — consolidated

Both realities are **enhancements, not fixes**. The leakage-free guarantee already holds and is tested.
Prioritised:

1. **[Low, do now]** Document the post-split merge-order constraint (Finding 1 / A) — docs + FE helper
   text, and a one-line note in `2026-08-23-task11-merged-branch-fold-refit-plan.md`.
2. **[Low–Med, do soon]** Fail-fast guard for non-numeric merged frames before a numeric model
   (Finding 1 / B). Turns "All trials failed" into an actionable error.
3. **[Optional]** Per-fold refit audit telemetry (Finding 2 / B) — row counts in metrics/log for
   post-hoc audit. Pair with #2 for an enterprise-trust story.
4. **[Defer / park]** Ownership across `SplitDataset` baselines (Finding 1 / C). Needs a dedicated
   design note and eager==refit parity tests; only if demand materialises.

**Explicitly not doing:** index normalisation for test convenience (Finding 2 / C).

None of these require a version bump or are blocked by other initiatives; #1 and #2 can land in the
next routine core release when commits are permitted.

---

## 5. Reproduction

- Scores table: `C:\Users\Murat\AppData\Local\Temp\print_val_split_scores.py` reuses the test
  module's helpers to run each scenario and print `best_score` / `cv_roc_auc_mean` / `val_auc`.
- Legacy-score proof against literal old code: `C:\Users\Murat\AppData\Local\Temp\old_woe_leak.py`
  (noise dataset, WOE fitted on the full frame, then CV-grid tune of the transformed rows) was run
  in a git worktree checked out at tag **v0.7.9** (pre-F-15, `PYTHONPATH` pointed at that tree's
  `skyulf-core`): legacy best_score **0.8669**. The identical script on current code with
  `preprocessing=None` reproduces **0.8669** exactly; the per-fold refit on the same dataset scores
  **0.4944**. The documented "leaky ≈ 0.87" claim is therefore proven on the actual previous
  version, not just emulated.
- Full old-vs-new matrix (CV, holdout across all five strategies, SMOTE signal deltas) is now a
  **persisted benchmark**, not a throwaway script:
  `skyulf-core/benchmarks/bench_holdout_refit_leakage.py` — run from the repo root with
  `.venv/Scripts/python.exe skyulf-core/benchmarks/bench_holdout_refit_leakage.py`. Seeded and
  reproducible; its docstring records the v0.7.9 reference numbers above.
- Tests: `TESTING=1 PYTHONPATH=. uv run pytest tests/integration/test_fold_preprocessing_refit.py -q`
  (38 passed at time of writing).
