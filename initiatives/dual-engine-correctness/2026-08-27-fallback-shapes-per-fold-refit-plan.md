# Per-Fold Refit for Fallback Shapes — Investigation & Plan

**Date:** 2026-08-27 · **Status:** proposal (not started) · **Parent:** findings
`2026-08-26-holdout-validation-refit-and-merge-findings.md` §1 (noise-bias proof),
F-15 design `2026-08-23-f15-per-fold-refit-design.md`, task 11 plan
`2026-08-23-task11-merged-branch-fold-refit-plan.md`.

## 1. Current state — what falls back today, and where

Per-fold preprocessing refit (F-15) makes CV/tuning scores honest by re-fitting
preprocessing inside every fold. Two resolvers gate it:

- **Linear chains** — `_resolve_fold_preprocessing`
  (`backend/ml_pipeline/_execution/engine/_feature_eng.py:515`): walks the
  upstream FE chain, finds the last splitter, rebuilds the pre-transform
  payload, and hands `FeatureEngineerFoldAdapter` to CV/tuning.
- **Merged fork-join graphs** — `_try_fork_join_refit`
  (`backend/ml_pipeline/_execution/engine/_feature_eng.py:417`): matches exactly
  one shape — shared trunk ending in a `TrainTestSplitter`/`Split` (the fork
  point), N parallel **linear** transformer branches, column-wise merge straight
  into the training node — and builds a `MergedBranchFoldAdapter`
  (`skyulf-core/skyulf/preprocessing/fold_adapter.py`).

Anything that fails a gate keeps **pre-transformed scoring**: the pipeline runs
once eagerly and CV/tuning folds score the already-transformed table. Preprocessing
statistics were then learned from all train-split rows (including rows later scored
as validation), so the reported score is **mildly optimistic**. The model itself is
still trained correctly, and the fallback is **never silent** — every bail writes an
explicit reason to the job log.

### The exact bail-out reasons (code-grounded)

| # | Bail reason (verbatim from code) | Shape it rejects |
|---|---|---|
| S1 | `branch '{id}' is not a linear transformer chain from a loader` (`:441`) | **Nested merges** — a branch contains its own merge node |
| S2 | `fork node '{id}' must end with a TrainTestSplitter/Split step` (`:466`) | **Splitter mid-chain** — the branches' common fork point is not the splitter |
| S3 | `data-dependent trunk step(s) before the fork splitter ({names}) cannot be re-fit safely per fold` (`:481`) | **Learner before the split** in a merged graph |
| S4 | `data-dependent step(s) before the last splitter ({names}) cannot be re-fit safely per fold` (`:570`, linear path) | **Learner before the split** in a linear chain |
| S5 | `branch step '{transformer}' splits data or changes row counts` (`:494`, `UNSAFE_BRANCH_STEP_TYPES`) | Row-changing branch under grid/random CV |

Diagrams of the two structural families:

```
S1 — nested merge (merge feeding merge):
    Split -> A --+
          -> B --+-> M1 --+
          -> C ---------- +-> M2 -> Train

S2 — splitter mid-chain:
    Split -> A -> Split2 -> A1 --+
                          -> A2 --+-> M -> Train
    (folds are defined on Split, but the fan-in forks at Split2)

S3/S4 — learner before the last splitter:
    Dataset -> Imputer(mean) -> Split -> ... -> Train
              ^ mean learned from ALL rows, incl. future validation rows
```

### Why each bail is correct today

- **S1/S2:** the adapter replays **one** fan-in level (branch step lists →
  column-wise merge). Nested/mid-split DAGs would need recursive replay of an
  arbitrary sub-DAG per fold; there is no code for that, and faking it would
  silently diverge refit from the eager path — breaking the eager==refit parity
  invariant that F-15 depends on.
- **S3/S4:** per-fold honesty requires re-fitting the learner on each fold's
  train rows, but the fold is defined on the *post-split* frame. The pre-split
  frame (the only thing the learner may legally see per fold) is upstream of the
  splitter, so the adapter refuses rather than re-fit the learner on the full
  frame (which would bake held-out rows into the statistics — the exact leak
  F-15 eliminates).
- **S5:** row-count-changing steps (resampling, row drops) break the 1:1
  fold-row correspondence the adapter slices on. (Under `halving_*`/`optuna`
  these chains ARE covered via `FoldAwareModelStep`, which runs the whole chain
  inside `fit` per fold.)

## 2. Is this worth fixing? — assessment

**Honest answer: possible but not cheap; payoff is small until real graphs hit it.**

- **Severity when hit:** mild scoring optimism + a loud job-log warning. Not a
  broken model, not silent leakage, not data loss.
- **Frequency:** unknown. All verified suites and every documented example are
  fork-join or linear. No user graph has hit a fallback yet (this repo's own
  graphs are the only customers so far).
- **User-side fix exists for every shape:** chain instead of tangle (S1/S2);
  move data-dependent steps after the split (S3/S4 — already discouraged by the
  leakage structure gate); avoid row-changing branches under grid/random or use
  halving/optuna (S5).

**Recommendation:** do **Phase 0 (telemetry)** first; it is cheap, pairs with the
already-parked "per-fold refit audit telemetry" enhancement (findings item 3),
and produces the demand data that justifies (or kills) the expensive phases.

## 3. Fix designs (if demand materialises)

### Phase 0 — Bail-reason telemetry + louder surface (low effort, zero risk)

1. Count bail reasons: increment per-reason counters in the job metrics
   (`job.metrics["fold_refit_fallback"] = reason`) alongside the existing log
   line. This answers "do users actually hit S1–S5?" with data.
2. Optionally fold in the parked **audit telemetry** (findings §3 option B):
   record per-fold fit/transform **row counts** (+ optional content hash) into
   metrics/log, so covered runs are post-hoc auditable.
   **Done (branch 085):** `AuditedFoldPreprocessor` records the counts,
   `_run_training_tuned` logs the isolation verdict and persists
   `fold_refit_audit` in node metrics. Only the fallback *counters* remain.
3. Canvas lint (frontend, cheap): when a training node's upstream matches a
   fallback shape (detectable statically from node/edge structure — multi-input
   node whose branch contains another merge node, or a learner upstream of the
   last splitter), show an advisory *before* the run: "CV/tuning scores for this
   graph use pre-transformed scoring — chain the nodes or move X after the
   split." Reuses the `predictMergeConflict` static-analysis pattern from
   `PropertiesPanel.tsx`.

**Tests:** unit tests for the metric emission and the lint predicate; no engine
behaviour change.

### Phase 1 — Single source of truth for merge semantics (prereq for Phase 2)

Today merge logic exists twice: the engine's eager path
(`backend/ml_pipeline/_execution/engine/_merge.py` — `_merge_frames_columnwise`,
`_column_owners`, `_baseline_frame`) and the adapter's mirror
(`skyulf-core/skyulf/preprocessing/fold_adapter.py` —
`_merge_branch_frames_columnwise`, pure strategy since ownership is inert
post-split). Extract the column-wise merge + ownership resolution into one shared
core module both import, so a generalized replay cannot drift from the eager path.

**Tests:** property-style parity tests — random 2–4 branch frames, all strategies,
with/without baseline — asserting shared-core output == current eager output ==
current adapter output.

### Phase 2 — Recursive sub-DAG replay (fixes S1, S2)

Generalize `MergedBranchFoldAdapter` from "N step-lists → one merge" to "replay
the FE sub-DAG between the fold-defining splitter and the training node":

1. **Scope extraction:** from the training node, walk upstream to the
   fold-defining splitter (the artifact whose train payload feeds the folds).
   Everything in between is the replay sub-DAG.
2. **Topological scheduling per fold:** seed the fold's train rows into the
   splitter slot; execute each FE node in topological order, applying its steps
   through a per-node `FeatureEngineerFoldAdapter` (learning steps re-fit on the
   node's incoming fold rows only); at each merge node, merge incoming frames
   with the shared core from Phase 1.
3. **Baseline availability:** post-split, the baseline is a `SplitDataset` →
   ownership stays inert → pure strategy (identical to today, parity trivial).
   Pre-split merge nodes *inside* the sub-DAG have real frame artifacts as
   baselines — ownership must then run on per-fold frames, which is correct
   (owners computed on fold-train rows) but needs dedicated parity tests.
4. **Row-count changes:** keep the S5 bail for grid/random unless the replay
   re-derives fold membership per node (Phase 3 tech); halving/optuna already
   covered via `FoldAwareModelStep`.

**Effort:** high — this is the "recursive payload reconstruction" piece.
**Tests (red→green):**
- parity: for every replayed DAG, eager full-split frame == fold-replay frame on
  the full train split (one fold = all rows).
- honesty: noise-target WOE benchmark (0.867 → ~0.50) on an S1 nested-merge
  graph, previously impossible to express.
- regression: every existing bail test stays green until its shape is enabled;
  `test_merged_branches_fall_back_with_warning` flips to a covered assertion
  per shape.

### Phase 3 — Upstream fold-mask propagation (fixes S3, S4)

Let learners that sit *before* the last splitter re-fit per fold:

1. The `SplitDataset` artifact preserves **original row labels** for each slot
   (`train_test_split` keeps scattered original indexes — findings §3 fact 1,
   which becomes an asset here). Per fold, compute the fold-train membership on
   the **pre-split** frame: `mask = pre_split_frame.index.isin(fold_train_labels)`.
2. Replay pre-split steps on the masked subset per fold (learning steps re-fit
   on fold-train rows only), then re-apply the split assignment to route rows
   downstream — i.e., move the splitter *into* the replay instead of treating it
   as the seed.
3. **Bail subcases (keep warning):** pre-split steps that change row count or
   the target (resampling, row drops, target re-encoding) — membership mapping
   breaks. These stay fallback (or route through `FoldAwareModelStep` semantics).
4. Interaction with Phase 2: the fold-defining splitter becomes "the last
   splitter in topological order," and everything upstream of it joins the
   replay sub-DAG with the mask-seeded origin frame.

**Effort:** high. **Tests:** row-count invariant (findings §3 — every fit sees ≤
fold-train rows) must hold for pre-split learners; honesty benchmark on an
`Imputer → Split` chain that currently falls back; index-alignment tests for
pandas **and** polars (different index semantics).

### Phase 4 — Docs + frontend follow-through

Remove each "falls back" bullet from `docs/examples/leakage_proof_pandas.md` as
its shape becomes covered; drop the matching canvas lint warnings; extend
`multi_path_pipelines.md` guarantees table.

## 4. Effort & risk summary

| Phase | Fixes | Effort | Risk | Trigger |
|---|---|---|---|---|
| 0. Telemetry + lint | — (visibility) | Low | None | **Do when convenient** |
| 1. Shared merge core | — (prereq) | Med | Low (parity-tested) | Only if Phase 2 proceeds |
| 2. Sub-DAG replay | S1, S2 | High | Medium — parity must hold on both engine paths | Only if telemetry shows demand |
| 3. Upstream masks | S3, S4 | High | Medium — index semantics, dual engine | Only if telemetry shows demand |
| 4. Doc/FE cleanup | — | Low | None | With each covered shape |

## 5. Decision gate

Do not start Phases 1–3 on speculation. Ship Phase 0 (cheap, independently
useful), watch `fold_refit_fallback` counters for 1–2 release cycles:

- **Counts ≈ 0** → keep documented limitation + loud warning; close as
  won't-fix-by-default. The contract ("unsupported shapes score pre-transformed
  data and tell you so") is sound.
- **Counts > 0, concentrated in one shape** → fund that shape's phase only.
- **S5 demand under grid/random** → cheapest real fix is advising halving/optuna
  in the warning text (already covered there), not new machinery.

## 6. Explicit non-goals

- Index normalisation for row-identity tracking (findings §3 option C — rejected:
  would mask alignment bugs).
- Making fallback shapes *fail* the run. The fallback contract is warn-and-continue;
  the non-numeric training guard (findings §2 Finding 1/B, shipped on branch 085)
  is the only fail-fast, and it fires on broken frames, not unsupported shapes.
