# Leakage Enforcement — Strengthening Plan

**Date:** 2026-08-11
**Branch:** `078`
**Decision:** *strengthen enforcement so the claim becomes true* (rather than softening the docs)
**Companion document:** `dual-engine-audit-findings.md` (findings F-14, F-15, F-16, F-17)

---

## 1. What the claim currently is, and how true it is

`docs/examples/leakage_proof.md:459` states:

> This proves that **Skyulf pipelines are leakage-free by design**. The strict separation of
> `fit()` (Calculator) and `transform()` (Applier) ensures that no information from the Test set
> (or future data) can influence the model training.

**Verdict: the mechanism is real, the scope of the word "pipelines" is too broad.**

What is genuinely true and was verified adversarially (poisoned test set, flipped test labels):
the Calculator/Applier split is enforced per node, learned artifacts are invariant to test data and
test labels, and `TargetEncoder` correctly cross-fits (`target.py:323` `fit_transform_train`).
That is a stronger guarantee than most libraries offer, and it should keep being advertised.

What the sentence over-claims:

| Gap | Consequence |
|---|---|
| **G1** — Enforcement is keyed on finding a splitter node. `leakage.py:53-54` returns `[]` when no splitter is present. | A pipeline with no splitter — the common "prepare a dataset" and CV-only shapes — gets **zero** leakage checking, while the docs promise "by design". |
| **G2** — Two independently-maintained allow-lists. `skyulf-core/skyulf/leakage.py:_DATA_DEPENDENT_TRANSFORMERS` and `backend/ml_pipeline/_execution/_leakage_validation.py:DATA_DEPENDENT_FIT_STEP_TYPES` are separate frozensets with no shared source. | They can (and do) drift. A node added to one is not protected in the other. |
| **G3** — 10 stateful nodes are explicitly excluded as "stateless / rule-based". Two of them are not: `MissingIndicator` learns *which columns contain missing values*, and `DropMissingColumns` learns *which columns to drop* — both from the data it is fitted on. `HashEncoder`, `Deduplicate` and the resamplers also change behaviour with the data given. | Placing these before a split leaks the test set's missingness/cardinality structure, unflagged. |
| **G4** — `WOEEncoder` has no cross-fitting hook (`woe.py:218-231`), unlike `TargetEncoder`. | Measured: on a **pure-noise** target (true AUC 0.500), WOE reaches CV ROC-AUC **0.791** vs TargetEncoder's correct 0.503. This is real, quantified leakage inside a single node. |
| **G5** — CV/tuning never re-fit preprocessing per fold (`cross_validation.py`, `_tuning/engine.py:271-357, 663`). | Preprocessing is fitted once on the full training set, so every fold's validation rows influenced the statistics used to transform them. **All reported CV and tuning scores are optimistic.** |
| **G6** — `_TRAIN_TEST_SPLITTERS` matches only `{"TrainTestSplitter", "Split"}`. | Any future or aliased splitter silently disables the gate entirely (fails *open*). |
| **G7** — Warnings only. The gate returns a list of strings; nothing blocks execution. | A user can ignore every warning and still get a model the docs describe as leakage-free. |

**Blunt summary:** the *node-level* guarantee is solid and defensible. The *pipeline-level*
guarantee is enforced for one graph shape, in one code path, against a hand-maintained list, with
warnings that don't bind. That gap is what this plan closes.

---

## 2. Plan

### Phase 1 — Make the contract single-sourced and fail-closed *(ship with Tier 3)*

1. **One list, one home.** Move the data-dependent node set into `skyulf-core` as the single source
   of truth and have `backend/_leakage_validation.py` import it. Delete the duplicate frozenset.
   Closes **G2**.
2. **Derive it, don't hand-maintain it.** Every node already declares itself via `@node_meta`.
   Add an explicit `learns_from_data: bool` field to the decorator and derive both lists from the
   registry. A new node then *cannot* be silently omitted — and make the field **required**, so
   forgetting it is a registration error rather than a silent opt-out. Closes **G2**, **G6**.
3. **Fail closed on unknown nodes.** An unrecognised transformer must be treated as
   data-dependent until proven otherwise, and an unrecognised splitter must not disable the gate.
   Closes **G6**.
4. **Correct the exclusion list.** Reclassify `MissingIndicator`, `DropMissingColumns`,
   `HashEncoder`, `Deduplicate` and the over/under-samplers as data-dependent, and fix the
   misleading comment in `_leakage_validation.py:30-38` that calls the first two "stateless".
   Closes **G3**.

### Phase 2 — Cover the no-splitter case *(ship with Tier 3)*

5. **Replace "is there a splitter?" with "is every learned fit provably train-only?"** When no
   splitter exists, the correct verdict is not "no warnings" — it is either *"no split defined; the
   leakage guarantee does not apply to this pipeline"* or, if the pipeline feeds CV, *"the
   guarantee is delegated to the CV boundary"*. Emit an explicit, distinct diagnostic. Silence is
   the bug. Closes **G1**.
6. **Escalate to an error by default, with an explicit opt-out.** Add
   `on_leakage: "raise" | "warn" | "ignore"`, defaulting to `"raise"` for definite violations
   (learned fit before a split) and `"warn"` for advisory ones. Users who want the old behaviour
   set `"warn"`. Closes **G7**. *This is a breaking change — it belongs in a minor release with a
   changelog entry.*

### Phase 3 — Close the in-node leak *(ship with Tier 2)*

7. **Give `WOEEncoder` the `fit_transform_train` cross-fitting hook** that `TargetEncoder` already
   has at `target.py:323`. Closes **G4**.
8. **Audit every other target-aware encoder** for the same missing hook — do not assume WOE is the
   only one. Verify each with the pure-noise-target test below.
9. **Regression test, red-green.** Fit on a pure-noise target and assert CV ROC-AUC ≈ 0.5 (say,
   within 0.05) for *every* target-aware encoder. Confirm the test fails at 0.791 before the fix
   and passes after. This test would have caught G4 years ago.

### Phase 4 — The CV boundary *(separate initiative, do not bundle)*

10. **G5 is a design change, not a patch.** See §4 of the findings document. Deliver a design note
    first — refit contract, performance budget (`n_splits`× preprocessing cost), migration plan for
    users whose scores will drop, and whether to land it opt-in
    (`refit_preprocessing_per_fold=True`) before flipping the default.
11. **Until it lands, say so.** The docs must not imply CV scores are leakage-free while G5 is
    open. This is the one place where documentation must be qualified *now*, ahead of the fix.

---

## 3. Documentation changes

Enforcement and documentation must land together — strengthening the code while the docs continue
to over-claim just moves the problem.

| File | Change |
|---|---|
| `docs/examples/leakage_proof.md:459` | Keep the conclusion, bound it. State precisely what was proven: *learned preprocessing artifacts are invariant to test data and test labels for the steps demonstrated*. Replace the unqualified "Skyulf pipelines are leakage-free by design" with the guarantee actually enforced, and link to the enforcement rules. |
| `docs/examples/leakage_proof.md` (new section) | "What is and is not covered" — cover the node-level guarantee, name the no-splitter case, and state the CV caveat until Phase 4 lands. |
| `docs/index.md:99` | "Proof that Skyulf avoids leakage" → scope it to the demonstrated steps. |
| `docs/user_guide/validation_vs_sklearn.md:290` | Same qualification; it currently inherits the broad claim. |
| `docs/user_guide/overview.md:47` | Point at the new coverage section. |
| `changelog/0.7.x.md` (or the release in flight) | Record the `on_leakage="raise"` default change as **breaking**, and the WOE cross-fitting fix as a **scores-will-change** entry. |
| Frontend | Per repo policy, if `on_leakage` becomes a user-visible pipeline setting it needs a matching control in `frontend/ml-canvas/src/modules/nodes/`, and the warning/error surface must render the new distinct diagnostics. |

**Recommended replacement wording for the conclusion** (adjust to house style):

> Skyulf enforces a strict `fit()` / `transform()` separation: preprocessing statistics are learned
> only from data the pipeline has designated as training data, and are then applied unchanged to
> validation and test data. Under adversarial conditions — a poisoned test set and flipped test
> labels — the learned artifacts are provably invariant, demonstrating that no test-set information
> reaches the fitted parameters for the steps shown.
>
> This guarantee applies per preprocessing step, relative to the split defined in your pipeline.
> Skyulf additionally validates pipeline *structure* and will reject a configuration that fits a
> data-dependent step before the split. See "What is and is not covered" for the current limits.

---

## 4. Definition of done

- [ ] One registry-derived data-dependent node list; the duplicate frozenset deleted.
- [ ] `learns_from_data` is a **required** `@node_meta` field; omitting it is a registration error.
- [ ] Unknown transformer or unknown splitter → gate fails **closed**, with a test proving it.
- [ ] `MissingIndicator`, `DropMissingColumns`, `HashEncoder`, `Deduplicate`, resamplers
      reclassified; misleading comment corrected.
- [ ] No-splitter pipelines emit an explicit diagnostic instead of silence.
- [ ] `on_leakage` implemented, default `"raise"` for definite violations, changelog marks it breaking.
- [x] `WOEEncoder.fit_transform_train` implemented (done on `080`, 2026-08-21, red-green);
      every other target-aware encoder audited — `TargetEncoder` (hook pre-existing) and
      `WOEEncoder` are the only target-aware encoders in the registry.
- [ ] Pure-noise-target regression test covers all target-aware encoders, verified **red-green**.
- [ ] All doc sites in §3 updated in the same PR as the code.
- [ ] Phase 4 (CV refit) has a written design note; docs carry the CV caveat until it lands.
- [ ] `ruff check` / `ruff format --check` / `ty check` clean; full backend + core suites pass.
