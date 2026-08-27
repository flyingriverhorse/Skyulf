# Dual-Engine (Polars / Pandas) Correctness

Investigation into whether Skyulf's dual-engine design is actually correct
end-to-end — engine → preprocessing nodes → modeling → persistence →
inference → monitoring — and whether the documented leakage guarantee holds.

Driven by a direct question: *if a user starts with Polars, can they stay on
Polars the whole way through, converting to numpy or pandas only where that is
genuinely required? And can we honestly claim leakage-proof?*

## Headline result

The architecture is sound. Polars runs end-to-end — of the **100 registered
nodes**, **zero** are Polars-incapable and **zero** silently downgrade Polars
to pandas — and the Polars → numpy → sklearn handoff produces **bit-identical
predictions** to the pandas path. Pandas users stay 100% pandas.

## Closed work (archived out of this folder)

The investigation phase and all of its fix waves are complete; their documents
were removed from this folder once shipped:

- **Audit findings** (2026-08-11) — 49 findings (4 CRITICAL, 18 HIGH,
  17 MED, 10 LOW), every one reproduced with an executable probe. **All fixes
  applied**: T1 (0.7.9), T2 (core 0.6.0), T2b/T3/T4 on `080`/`081`.
- **Leakage enforcement** (2026-08-11) — F-16/F-17, `on_leakage="raise"`
  default, shipped on `081`.
- **Backend Polars migration** (2026-08-11) — the product gap (Skyulf
  presented as Polars-backed while the backend ran 100% pandas) closed on
  `080`, all five phases; benchmark published in `docs/performance.md`.
- **F-15 per-fold preprocessing refit** (2026-08-23 design) — shipped
  always-on in core 0.8.0: every CV/tuning fold refits preprocessing on
  fold-train rows only.
- **Task 11 merged-branch fold refit** (2026-08-23 plan) — fork-join merged
  graphs refit per fold too (`MergedBranchFoldAdapter`).
- **Holdout + validation-split refit and merge findings** (2026-08-26) —
  v0.8.4 closed the last F-15 gap (holdout tuning with a validation split
  refits on train rows only, all five strategies). Its two follow-up product
  realities were resolved on branch 085 (docs for the post-split merge-order
  constraint, the non-numeric fail-fast guard, and the fold-refit audit
  telemetry); the third was parked — see the ownership design note below.
- Experiments-subsystem audit (16 findings, 12 live at the time) — fixed.

Historical changelog entries still reference those documents by name.

## Open work (kept here)

| Document | What it covers | Status |
|---|---|---|
| [2026-08-27-fallback-shapes-per-fold-refit-plan.md](2026-08-27-fallback-shapes-per-fold-refit-plan.md) | Graphs that fall back to pre-transformed (mildly optimistic) scoring: the exact bail shapes (S1–S5), demand telemetry, and fix designs. | Phase 0 telemetry done (branch 085 — `fold_refit_fallback` reason codes + Score Advisory UI); remaining: Phase 0 canvas lint, Phases 1–4 gated on demand. |
| [2026-08-27-splitdataset-baseline-ownership-design.md](2026-08-27-splitdataset-baseline-ownership-design.md) | Ownership tracking across `SplitDataset` baselines so merged branches don't resolve overlapping columns by pure merge order. | **Parked** — implement only if the Phase 0 demand gate fires. |
