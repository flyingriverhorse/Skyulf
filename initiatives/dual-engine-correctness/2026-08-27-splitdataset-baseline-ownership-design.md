# Ownership Across `SplitDataset` Baselines — Design Note (Option C)

**Date:** 2026-08-27 · **Status:** proposal — **parked**; implement only if the Phase 0
demand gate fires.
**Parent:** `2026-08-26-holdout-validation-refit-and-merge-findings.md` §2 Finding 1
(options table, option C) and §4 item 4 ("Defer / park … needs a dedicated design note
and eager==refit parity tests").
**Related:** `2026-08-23-task11-merged-branch-fold-refit-plan.md` (merge mirror + its
"Known constraint" note), `2026-08-23-f15-per-fold-refit-design.md`,
`2026-08-27-fallback-shapes-per-fold-refit-plan.md` (Phase 0 telemetry pairing),
`docs/guides/multi_path_pipelines.md`, `docs/user_guide/troubleshooting.md`.

---

## 1. Problem

When sibling branches fan into one node, the engine resolves overlapping columns by
**ownership** where it can: each branch frame is diffed against the
nearest-common-ancestor (NCA) artifact as a baseline, and a column changed by exactly
one branch belongs to that branch regardless of edge order
(`backend/ml_pipeline/_execution/engine/_merge.py:206` `_merge_frames_columnwise`,
`:126` `_column_owners`, `:93` `_column_modifiers`). The configured strategy
(`last_wins` default / `first_wins`, `_get_merge_strategy` `_merge.py:188`) only breaks
ties between columns two or more branches *each* rewrote.

Ownership needs a baseline frame. `_baseline_frame` (`_merge.py:57`) loads the NCA
artifact (`:63`) and coerces it via `_coerce_to_frame` (`_merge.py:151`). That coercion
handles `None`, polars frames, pandas frames, and `(X, y)` tuples — a `SplitDataset`
falls through every branch to `return None` (`_merge.py:165`). In fork-join-after-split
graphs the NCA **is** the splitter, whose stored artifact is a `SplitDataset`, so:

1. `_coerce_to_frame` returns `None` → `_baseline_frame` returns `None`;
2. `_column_modifiers` returns `{}` (`_merge.py:104`) → ownership goes **inert**;
3. `_merge_frames_columnwise` falls back to pure configured strategy: under `last_wins`
   the last-connected branch wins **every** shared column, including columns it merely
   carried through untouched.

The refit path mirrors exactly that fallback: `_merge_branch_frames_columnwise`
(`skyulf-core/skyulf/preprocessing/fold_adapter.py:37`) is ownership-free by design; its
docstring (`:40-46`) says ownership "is inert … because the fork's stored artifact is a
SplitDataset." So eager serving merge == per-fold refit merge == pure order. Consistent,
leakage-free — but constrained.

**User-facing consequence (documented today as a constraint):** post-split branches must
be disjoint or each emit a fully-numeric frame, otherwise the last branch's raw leftover
column discards an earlier branch's encoding and training fails. Three interim
mitigations exist precisely because of this:

- **Docs** — `docs/guides/multi_path_pipelines.md:38-57` ("After a Split: Order Decides
  Everything") and `docs/user_guide/troubleshooting.md:140-171`.
- **Frontend winner hint** — `frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx:298-302`
  warns "After a Split, ownership doesn't apply — the winning branch takes every
  overlapping column"; the merge-mode hint at `:192-198` says the same.
- **Fail-fast guard** — `_assert_numeric_training_frame`
  (`backend/ml_pipeline/_execution/engine/_node_runners.py:309-362`, called at `:929`;
  every supervised run routes through `_run_training_tuned` via `_run_training`
  `:670-706`, so one call site covers fixed and tuned modes). It converts the cryptic
  "All trials failed" into an actionable error naming the leftover string columns.

## 2. Why it is parked by default

- **Behaviour-change risk.** Outputs of existing fork-join-after-split graphs with
  overlapping columns *can* change (see §5). Today's semantics are a documented trap,
  not a feature — but "can change" still demands a compatibility strategy and a demand
  signal before touching the merge core.
- **The eager==refit parity invariant.** F-15 correctness rests on the serving merge and
  the per-fold refit mirror producing the *same* merged frame, or tuning optimizes
  features serving doesn't produce. This change must land on **both** code paths in
  lockstep (backend `_merge.py` + `skyulf-core` `fold_adapter.py`), with parity tests,
  and with the same semantics for train, test, and validation slot merges.
- **Demand is unknown.** Nobody has yet asked for specialised overlapping post-split
  branches; the constraint docs may be absorbing the demand silently. Hence Phase 0.

## 3. Design

### 3.1 Verified shape of the artifact

`SplitDataset` (`skyulf-core/skyulf/data/dataset.py:21-25`) is a dataclass:

| Field | Type | Notes |
|---|---|---|
| `train` | `SplitPayload` (required) | what the branches forked from |
| `test` | `SplitPayload` (required) | may be an empty frame (`_to_split_dataset`, `_node_runners.py:271-273`) |
| `validation` | `SplitPayload \| None` | `None` unless `validation_size > 0` |

with `SplitPayload = SkyulfDataFrame | pd.DataFrame | pl.DataFrame | tuple[frame, Any]`
(`dataset.py:13-18`) — each slot is a frame **or** an `(X, y)` tuple (tuples appear once
a `feature_target_split` ran, `skyulf-core/skyulf/preprocessing/split.py:472-480`, or the
splitter was given a `target_column`, `split.py:231-258`).

The splitter preserves **scattered original indexes**: `train_test_split` slices without
resetting (`split.py:240-247`, `:267-273`, second split for validation `:393-400`), so
the train slot carries a non-contiguous subset of the source CSV row labels (findings
doc §3.1). Polars slots have no index; rows are gathered by position (`split.py:296-354`).

### 3.2 Injection point: `_baseline_frame`, keyed on artifact type

Inject the SplitDataset→frame coercion in **`_baseline_frame` (`_merge.py:57`)** — not in
`_coerce_to_frame`. Extend it to take the merge part being computed and select the
matching slot:

```
_baseline_frame(node_id, part_label="")
    artifact = artifact_store.load(ancestor_id)
    frame = _coerce_to_frame(artifact)                 # plain frame / (X, y) NCA: today's path
    if frame is None and isinstance(artifact, SplitDataset):
        slot = artifact.test      if part_label == "test"
             else artifact.validation if part_label == "validation"
             else artifact.train  # "train" | "X" | "" (fallback/mixed merges)
        frame = _coerce_to_frame(slot)                 # (X, y) slot with target_col="" → X only
    return _to_pandas_frame(frame) if frame is not None else None
```

Details, each grounded in existing behaviour:

- **Per-slot baseline, not train-for-everything.** `_merge_split_datasets`
  (`_merge.py:554-585`) merges train/test/validation independently, and `part_label`
  ("train"/"test"/"validation") already reaches `_merge_frames` (`_merge.py:523, :535`,
  `:560-572`). Using the *train* slot as baseline for the test/validation merges would
  trip `_column_changed`'s row-count rule (`_merge.py:78`) for every column — all
  branches would register as "changed", ownership would degrade to strategy exactly for
  the held-out slots, and train and test could end up with different versions of the
  same column. Each slot's merge must be diffed against **the same slot the branches
  received**.
- **`(X, y)` slots.** `_coerce_to_frame(slot)` with the default empty `target_col`
  returns just X (`_coerce_tuple_to_frame`, `_merge.py:134-149`), which is exactly what
  branch frames contain post-`feature_target_split`.
- **Missing/empty train slot.** `train` is a required field but may coerce to `None`
  (empty frame) → baseline `None` → ownership inert → pure-strategy fallback, i.e.
  today's behaviour. Baseline loading is already wrapped in a non-fatal `except`
  (`_merge.py:65-67`). When *all* train slots are empty, `_merge_split_datasets` raises
  first (`_merge.py:563-566`), before ownership is consulted. `validation=None` means
  no validation merge happens at all (`_merge_split_dataset_part`, `_merge.py:546-548`).
- **pandas vs polars.** Comparison is already always pandas-vs-pandas: `_baseline_frame`
  converts the baseline (`_merge.py:64`) and `_merge_frames` converts every branch frame
  (`_merge.py:332`) via `_to_pandas_frame` (`_merge.py:295`); the result is converted
  back to polars only when inputs were polars and `SKYULF_ENGINE == "polars"`
  (`_merge.py:350-351`, setting at `backend/config/mixins/core.py:30`). Polars slots
  convert to a `RangeIndex`, so positional alignment below holds for both engines.
- **Index alignment is already solved.** `_column_changed` (`_merge.py:69-80`) compares
  `frame[col].reset_index(drop=True).equals(baseline[col].reset_index(drop=True))` —
  positional **value equality**, index labels ignored. That tolerates the splitter's
  scattered original indexes (§3.1) for free, because branch frames are order-preserving
  transformations of the same slot. No index normalisation (explicitly rejected in
  findings §2/C). A branch that reorders rows reads as "changed" positionally — correct,
  it did change the data.

### 3.3 What changes in `_merge_frames_columnwise` / `_column_owners` — and what does not

- **Threading only.** `_merge_frames` (`_merge.py:302`) passes its existing `part_label`
  down: `_merge_frames_columnwise(frames, node_id, strategy, prefix, part_label)` →
  `_column_owners(frames, node_id, part_label)` → `_column_modifiers(...)` →
  `_baseline_frame(node_id, part_label)`. The owner-skip loop itself
  (`_merge.py:229-231`) is untouched — once owners exist post-split it already does the
  right thing, and `first_wins` iteration order (`:222`) keeps meaning for contested
  columns.
- **`_coerce_to_frame` stays SplitDataset-unaware.** Its documented contract is
  shape-coercion returning `None` for empty/missing payloads (`_merge.py:152-156`);
  `_merge_fallback_frames` (`_merge.py:603-605`) relies on the `None` to route
  SplitDatasets through `_to_dataframe` (`_merge.py:167-186`, which already knows how to
  take the train slot). Folding merge semantics into the shape helper would be a larger,
  harder-to-audit change for zero behavioural gain.
- **Advisory precision.** `_sibling_fan_in_overlap_columns` (`_merge.py:373-391`)
  coerces each *input artifact*; SplitDataset artifacts coerce to `None` today, so it
  falls back to plain name overlap (`:387-391`) and the fan-in advisory reports every
  shared column post-split. Coerce SplitDataset artifacts to their **train slot** there
  (mirroring `_to_dataframe`) so `_column_modifiers` runs and only true conflicts are
  reported. Consequence: specialised post-split branches produce **no advisory**, the
  Merge Strategy dropdown hides (`PropertiesPanel.tsx:247`), and "wins merge" edge labels
  disappear (`FlowCanvas.tsx:173`); genuine two-branch edits still advise with
  `overlap_columns` naming exactly the contested columns
  (`_build_sibling_fan_in_advisory`, `_merge.py:404-439`).
- **Unaffected mechanics (verified):** redundant-edge suppression is topology-only
  (`_warn_sibling_fan_in` `_merge.py:459-462`, `_has_redundant_ancestor_edge` `:393`);
  upstream-drop reapplication runs after any merge and still strips resurrected columns
  (`_enforce_upstream_drops` `_merge.py:655-684`); row-wise merges (unequal branch row
  counts, gate at `_merge.py:340`) never consult ownership, as today.

### 3.4 Mirror change in `fold_adapter.py` (parity)

`MergedBranchFoldAdapter` re-runs the branches per fold and merges with
`_merge_branch_frames_columnwise` (`fold_adapter.py:37-55`). Mirror the ownership logic
there:

- **Baseline = the adapter's own input payload.** `fit_transform(X, y)` /
  `transform(X, y)` (`fold_adapter.py:112-125`) receive the fold rows *the branches are
  about to transform* — the per-fold analogue of the splitter slot the eager branches
  received. Pass `to_pandas(X)` as the baseline into `_finalize` (`:146`) and on to
  `_merge_branch_frames_columnwise(frames, strategy, baseline)`, replicating
  `_column_changed` / `_modifiers_agree` / owner selection. **No constructor change**:
  `_feature_eng.py:501-506` constructs the adapter with step lists, strategy, target,
  drops only, and the payload is already loaded from the fork's SplitDataset train slot
  via `_split_train_payload` (`_feature_eng.py:365-368, :498-499`).
- **Why this keeps eager==refit identical:** both paths now implement "compare each
  branch's output against what that branch received", eager over the full slot, refit
  over fold subsets. Positional value equality makes the verdict row-subset-independent.
  The adapter only ever merges fold-train payloads (the validation payload is scored
  untouched, findings §0), so one input-derived baseline suffices — no per-slot plumbing
  needed in core.
- **Screening unchanged:** branch steps are still rejected if they split or change row
  counts (`UNSAFE_BRANCH_STEP_TYPES`, `fold_adapter.py:28-34, :96-101`), which is what
  makes the equal-row-count column-wise merge — and hence ownership — applicable inside
  a fold at all.
- Because `skyulf-core` cannot import the backend, the comparison helpers are duplicated
  (the mirror already exists; this extends it). The parity test suite (§7) is the
  guard against the two copies drifting.

### 3.5 Edge cases

| Case | Behaviour |
|---|---|
| Branch passes a column through untouched; sibling edits it | Editor owns it, whatever the edge order — the motivating fix. |
| Both branches derive **identical** values (findings "case 2", e.g. same imputation twice) | `_modifiers_agree` (`_merge.py:82-91`) collapses them to one owner → no conflict, no advisory. Works post-split unchanged once a baseline exists. |
| Both branches edit the same column **differently** | True conflict → strategy decides, advisory emitted, `_merge_frames_columnwise` logs "broke ties on …" (`_merge.py:238-242`). Same as today. |
| Branch filtered/reordered rows after the splitter | If branch row counts diverge from each other → row-wise merge, ownership not involved (`_merge.py:340-348`). If both filtered to the same count, both read as "changed" vs baseline (`_column_changed` row-count rule `:78`) → identical results collapse, the rest fall to strategy. Safe degradation. |
| Rows filtered **upstream** of the splitter | No effect: the train slot already reflects them and is still exactly what branches received. |
| SplitDataset without a usable train slot (empty frame) | Baseline `None` → ownership inert → pure strategy (today's behaviour). All-empty trains raise earlier (`_merge.py:563-566`). |
| `validation=None` | No validation merge (`_merge.py:546-548`); nothing to baseline. |
| Polars slots / mixed-engine branches | All comparisons happen in pandas (§3.2); result type round-trips per `_merge.py:350-351`. |
| NCA is a plain frame (fork at a loader, no splitter) | `_coerce_to_frame` succeeds as today; the SplitDataset branch is never reached. |

## 4. Interactions summary

| Surface | Effect |
|---|---|
| `_merge_strategy` dropdown (`last_wins`/`first_wins`) | Unchanged; now only breaks genuine ties post-split. FE semantics text needs updating (§6 Phase 3). |
| `sibling_fan_in` advisory `overlap_columns` | Becomes precise post-split (true conflicts only); may disappear entirely for specialised branches. |
| Redundant-edge suppression | Unaffected (topology-only). |
| `upstream_drop_reapplied` advisory | Unaffected (post-merge strip). |
| `_assert_numeric_training_frame` guard | **Kept** — contested columns can still resolve to a raw string winner. Its message (`_node_runners.py:353-362`) currently blames merge order; reword to the residual causes (genuine conflict won by a raw branch, or unencoded winning branch). |
| FE `predictMergeConflict` (`predictMergeConflict.ts:76-116`) | Already ownership-shaped (predicts only columns written by 2+ branches) — no change needed. |

## 5. Rollout / compatibility

Outputs can change for one graph class: fork-join-after-split graphs where a pass-through
branch is connected **after** an editing branch. Today the pass-through wins (raw
column); with ownership the edit wins. Graphs following the documented constraint
(disjoint branches, or all-numeric identical work) are unaffected: disjoint → no overlap;
identical work → `_modifiers_agree`; different edits → still strategy-decided.

**Recommendation: default-on behind the Phase 0 gate, announced in the changelog — not
an opt-in flag.** Justification:

1. The displaced behaviour is the documented trap, not a contract — nobody wires a graph
   *relying* on a pass-through branch overriding an editing branch; the editor branch
   exists because the user wanted the edit.
2. A flag permanently doubles the semantic surface and the parity matrix
   (flag × eager/refit × slot) for a behaviour the docs call a failure mode.
3. One semantic keeps the eager==refit invariant trivially auditable.

Mitigations: changelog entry calling out the merge-semantics change with a before/after
example; updated constraint docs (§6 Phase 3); the numeric guard stays as the safety net.
If Phase 0 telemetry instead shows real graphs relying on pure order post-split, revisit
and ship as an opt-in per-node param (the `_merge_strategy` param plumbing,
`pipelineConverter.ts:565-579`, is the existing pattern to extend).

## 6. Phased plan

**Phase 0 — decision gate (do first, cheap).** Pair with the fold-refit audit telemetry
(findings §4 item 3, in flight) and the fallback-shapes plan's bail-reason logging.
Signals to count in job logs/metrics: (a) `_assert_numeric_training_frame` raises; (b)
`sibling_fan_in` advisories whose `common_ancestors` contain a splitter (log one marker
line when `_baseline_frame` returns `None` *because* the NCA artifact is a SplitDataset —
a two-line observability addition that pays for the decision); (c) post-split "broke ties
on" logs (`_merge.py:239`). **Gate:** implement Phases 1-3 only if users demonstrably hit
post-split overlap; otherwise keep parked and the constraint docs authoritative.

**Phase 1 — eager ownership (backend).**
1. Extend `_baseline_frame` with `part_label` + SplitDataset slot selection (§3.2).
2. Thread `part_label` through `_merge_frames_columnwise` → `_column_owners` →
   `_column_modifiers` (`_merge.py`).
3. Train-slot coercion in `_sibling_fan_in_overlap_columns` (§3.3).
4. Red→green tests per §7 items 1-4, 7.

**Phase 2 — refit mirror (skyulf-core + parity).**
1. Add `baseline` param to `_merge_branch_frames_columnwise`; replicate
   `_column_changed`/`_modifiers_agree`/owners (`fold_adapter.py`).
2. Derive the baseline from the payload in `fit_transform`/`transform` → `_finalize`.
3. Red→green §7 items 5-6 (adapter unit tests in
   `skyulf-core/tests/unit/test_fold_merged_adapter.py` + backend parity integration).

**Phase 3 — docs, FE, guard wording.**
1. Rewrite `docs/guides/multi_path_pipelines.md:38-57`: constraint becomes "ownership
   works post-split; strategy breaks genuine ties only"; update the error table row
   (`:73`). Same for `docs/user_guide/troubleshooting.md:140-171`.
2. Replace `PropertiesPanel.tsx:298-302` note and the `:192-198` hint.
3. Reword `_assert_numeric_training_frame`'s message (`_node_runners.py:353-362`).
4. Changelog entry + core version bump (release hygiene as in the task-11 plan Phase 5).

**Phase 4 — regression sweep.** Full suites of §7 plus `ruff check`, `ruff format`,
`ty check` on touched files.

## 7. Test plan (red→green)

New/changed, referencing existing patterns:

1. **Eager ownership post-split** — in `tests/integration/test_merge_scenarios_e2e.py`
   (helpers `_new_engine` `:98`, `_record_run` `:103`): loader → `TrainTestSplitter` →
   branch A WOE-encodes `city`, branch B scales `amount` and passes `city` raw → merge.
   Assert the merged train frame carries A's encoded `city` with B connected **last**,
   and again with the edges reversed (order-independence of owned columns).
2. **True conflict post-split** — both branches rewrite `city` differently → strategy
   decides; assert `merge_warnings` contains `sibling_fan_in` with
   `overlap_columns == ["city"]` and the correct `winner_input`.
3. **Identical edits (case 2)** — both branches emit the same derived column → merged
   once, no advisory (`_modifiers_agree`).
4. **Slot parity** — merged test/validation slots carry the editing branch's version too
   (same column set as merged train); guards the per-slot baseline decision (§3.2).
5. **eager==refit parity** — extend the fork-join pattern
   (`tests/integration/test_fold_preprocessing_refit.py:233`, `_fork_join_nodes` `:221`)
   with a pass-through-vs-editor pair: assert the refit-enabled run succeeds, stays
   near-chance on the noise target, and the eagerly merged train columns equal the
   adapter's fold-merge columns.
6. **Polars variants** — items 1-2 with `SKYULF_ENGINE=polars`; result type must
   round-trip (`_merge.py:350-351`).
7. **SplitDataset without usable train slot** — engineered empty-train baseline →
   strategy fallback, no crash.
8. **Deliberate updates (behaviour flips):**
   `test_fork_join_unencoded_last_branch_fails_fast` (`:1524`) and its tuned twin
   (`:1546`) currently assert the guard fires because branch B's raw `city` wins. With
   ownership, branch A's encoding wins → rewrite both as **ownership-rescue** tests
   (run succeeds; noise scores stay honest). Guard coverage moves to a new true-conflict
   variant where both branches rewrite `city` and the winning version stays non-numeric.
9. **Must stay green unchanged:** merge scenarios 01-09 (`test_merge_scenarios_e2e.py`;
   pre-split or single-input NCAs); fork-join honesty tests `:233/:248` (their two WOE
   branches use different regularization — already a genuine conflict, strategy still
   decides); validation-split fork-join `:1164/:1222` (fully-numeric branches); row
   isolation `:1263`; all fallback-shape tests (nested merge, row-changing branch,
   trunk learner) — none of them depend on ownership being inert.

## 8. Effort / risk

| Item | Effort | Risk | Mitigation |
|---|---|---|---|
| Phase 0 telemetry markers | Low | None | Observability-only |
| Phase 1 eager coercion + threading | Med | Behaviour change in merge core; slot misassignment would desync train/test | Per-slot baseline + §7 items 1-4, 7; `_coerce_to_frame` contract untouched |
| Phase 2 refit mirror | Med | Two ownership implementations drift → tuning optimizes unserved features | Parity tests (§7.5) as the contract; ship only with Phase 1 |
| Phase 3 docs/FE/guard wording | Low | Stale text confuses users if partial | Checklist in §6 Phase 3 |
| Compatibility | Med | Outputs change for pass-through-last graphs | Default-on + changelog, gated by Phase 0 evidence (§5) |

Total: roughly a task-11-sized slice (2-3 focused days with tests) once the gate fires.

**Non-goals:** ownership for row-wise (unequal row count) merges; nested-merge or
mid-chain-splitter replay (owned by `2026-08-27-fallback-shapes-per-fold-refit-plan.md`);
index normalisation for row identity (findings §2/C, rejected); removing or relaxing the
numeric fail-fast guard; new merge strategies or a changed default; making
`_coerce_to_frame` SplitDataset-aware (§3.3).
