# Execution tasks and versioning — Stage 0

**Date:** 2026-08-12 · **Branch:** `078` · **Status:** planning (no task started)

**Sources:** this file sequences work decided in two places and adds nothing
new:

- `initiatives/growth/2026-08-11-growth-plan.md` — execution order (decided)
- `initiatives/dual-engine-correctness/2026-08-11-audit-findings.md` §5 — the
  fix-order and version ledger. Per the growth plan: *"use it rather than
  inventing a second one here."*

**Version baselines (verified 2026-08-12 on `078`):**

| Component | Version file | Current |
|---|---|---|
| backend | `pyproject.toml` | 0.7.8 |
| frontend (ml-canvas) | `frontend/ml-canvas/package.json` | 0.7.8 |
| skyulf-core | `skyulf-core/setup.py` | 0.5.8 |

The growth README's "no version numbers" rule applies to the plan's prose;
this file is the designated place for them, by request.

**Discipline (non-negotiable, from both source docs):** every fix is written
red-green — failing test first, confirm it fails, then fix. Parity tests must
use **float NaN** (not only nulls) and **wrapped** frames.

---

## Release targets — complete ledger

| Release | Component(s) | Type | Contents |
|---|---|---|---|
| **0.7.9** | backend + frontend | patch | F-02, F-03 (deployment blockers) · F-33, F-34, F-35, F-37 (experiments) · A2.2 (templates) · ride-alongs: `start.sh` chmod, A2.6 upload-size text |
| **core 0.6.0** | skyulf-core only (`core-v*` tag, PyPI) | **minor** — behaviour changes | F-01, F-04, F-05, F-06 · growth T1, T2, T3, T6 · T5 contract test + NaN/wrapped parity tests |
| demo redeploy | `deploy/demo-mode` | **no bump** (stays 0.7.6) | cherry-pick of A2.2 only (growth 1a) |
| **0.8.0** | backend + frontend | minor | F-36, F-38–F-44 (experiments batch 2, audit T2b) · F-07–F-14 backend-side effects if any · A2.1 sample-dataset entry point + rest of Stage 2, weighted by 1b traffic data |
| **core 0.6.0 cont. or next minor** | skyulf-core | minor | F-07–F-14 (audit T2 remainder) — into 0.6.0 if done before the tag, else next minor; ledger is explicit either way |
| **0.8.1 / core 0.6.1** | all three | patch | F-16–F-26 + leakage enforcement (audit T3). F-22 needs a frontend sync check (param rename) |
| **0.8.2 / core 0.6.2** | all three | patch | F-27–F-32, F-45–F-48 + dead-code cleanup (F-26 fallback, F-48 `refit()`) (audit T4) |
| **core 0.7.0** | skyulf-core | minor/major | F-15 per-fold preprocessing refit — **design note first**, separate initiative, changes every reported score |

**One deliberate deviation from the audit ledger, recorded:** the audit's T1
ships F-01 alone as core **0.5.9** (patch). The growth plan bundles F-01 with
F-04–F-06 for one PyPI ship, and those change behaviour — so **core skips
0.5.9 and the bundle ships as 0.6.0**. Backend/frontend 0.7.9 is unaffected
(F-02/F-03/F-33/F-34/F-35/F-37 are engine-independent and contain no core
change).

---

## Wave 1 — backend + frontend → **0.7.9** (patch)

Engine-independent, LIVE today. Nothing else competes with deployed models
returning different predictions for the same data.

| # | Task | Finds | Component | Notes |
|---|---|---|---|---|
| 1.1 | Deployment determinism + input-schema fix | F-02, F-03 | backend (`deployment/service.py`, `_node_runners.py`) + **frontend sync check** in `frontend/ml-canvas/src/modules/nodes/` | Blocks deployments; F-03 drives the UI input form, so node components must be verified against the corrected contract |
| 1.2 | Experiments correctness (the cheap four) | F-33, F-34, F-35, F-37 | backend + frontend | F-33 shows wrong job's evaluation; F-35 makes "Recall" tuning actively harmful; F-37 is a ~2-line SHAP fix restoring 6 model families |
| 1.3 | Fix the 4 blocked templates | growth A2.2 | frontend `pipelineTemplates.ts:112-127` | Move `TrainTestSplitter` upstream of imputation/encoding/scaling. Do **not** weaken the leakage guard |
| 1.4 | Ride-along: `start.sh` executable bit | growth A2.3 | repo metadata | `git update-index --chmod=+x start.sh` |
| 1.5 | Ride-along: upload-size message + formats | growth A2.6 | frontend `FileUpload.tsx:52-54` | Says 500MB; server accepts 10GB (`config/mixins/files.py:18`); `accept=` hides `.xls/.txt/.feather` |

**Ship rule:** one commit per row, tests first. Version bump in
`pyproject.toml` + `frontend/ml-canvas/package.json` in the final commit of
the wave. Core stays 0.5.8.

---

## Wave 2 — skyulf-core → **0.6.0** (minor, PyPI ship)

**Minor, not patch:** F-04–F-06 change behaviour (rows previously dropped are
now kept; imputers that previously no-opped now impute). Needs a release note.
`skyulf-core` releases on its own `core-v*` tag and never touches the demo
branch — this path is unblocked today. Backend/frontend stay 0.7.9.

| # | Task | Finds | Notes |
|---|---|---|---|
| 2.1 | Null/NaN semantics across dual engine | F-01, F-04, F-05, F-06 | The root bug class: Polars `is_null()` misses float NaN where pandas `isna()` matches both |
| 2.2 | Lag/Rolling stale misaligned `y` | growth T1 | Root cause: `@apply_method` (`preprocessing/base.py:52`) + `apply_dual_engine` (`dispatcher.py:81`) double-unpack makes `_y` None. Fix pattern exists at `deduplicate.py:52-58`; one line each at `lag.py:77`, `rolling.py:121` |
| 2.3 | FeatureSelection default `variance` unknown to dispatch | growth T2 | `feature_selection/facade.py` knows only `variance_threshold` |
| 2.4 | GeneralBinning default `uniform` produces no bins | growth T3 | Fit handles `equal_width`; align the default |
| 2.5 | FeatureMath datetime drop | growth T6 | `_pandas_ops.py:179,185-186,226-228` |
| 2.6 | Registry-wide contract test | growth T5 | 3 clauses: (1) a node's own `@node_meta` defaults produce no empty artifact/no-op/unknown-method, (2) `y` length+order matches `X`, (3) engine parity on float NaN + wrapped frames. Highest-value artifact in the plan |

**Ship rule:** red-green per row; wave lands as core `0.6.0` on a `core-v*`
tag with a release note covering the behaviour changes. Growth A2.4 (PyPI
polish) and A2.5 (notebook-export wording in README) ship with this tag —
packaging metadata and docs, no version content of their own.

---

## Wave 3 — demo redeploy + measurement (no code version)

| # | Task | Source | Version impact | Notes |
|---|---|---|---|---|
| 3.1 | Cherry-pick A2.2 template fix to `deploy/demo-mode`, redeploy | growth 1a | demo stays **0.7.6** — single-commit pick, no bump | Only after 1.3 lands on `078`. An hour, not a stage |
| 3.2 | Check GitHub traffic panel | growth 1b | none — no code | Free. Decides whether Stage 2 funnel work is justified at all |

---

## Later — scheduled by version, not by date

Promoted only after Wave 1–3 data, per the growth plan. Every item carries
its ledger version so nothing has to be re-derived:

| Scope | Finds | skyulf-core | backend + frontend |
|---|---|---|---|
| Audit Tier 2 remainder (engine-divergent transformers, drift NaN, WOE leakage, clustering deploy, pandas `Int64` crashes…) | F-07–F-14 | **0.6.0** if merged before the tag, else next minor | **0.8.0** |
| Experiments batch 2 (metric comparability, diff rendering, threshold validation) | F-36, F-38–F-44 | — | **0.8.0** (call out F-36 in changelog) |
| Stage 2 funnel: sample-dataset entry point, dashboard, install story, rest of activation | growth A2.1, A2.3 (remainder), A2.4/A2.5 (if not shipped with core 0.6.0) | — | **0.8.0** |
| Leakage enforcement + Tier 3 (gate bypasses, no-splitter pipelines, inference row-drops) | F-16–F-26 | **0.6.1** | **0.8.1** (F-22 frontend sync check) |
| Cosmetic + latent + dead code (F-26 fallback, F-48 `refit()`) | F-27–F-32, F-45–F-48 | **0.6.2** | **0.8.2** |
| Per-fold preprocessing refit — **design note first, separate initiative** | F-15 | **0.7.0** | ships alongside |
| Stage 3 retention candidates (shareable experiments URL, per-node preview, actionable errors…) | growth Stage 3 | — | **0.9.0** (provisional; scope chosen after Stage 1b data) |
