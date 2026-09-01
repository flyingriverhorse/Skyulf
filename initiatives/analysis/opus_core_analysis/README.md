# Skyulf Deep Audit (Opus) — Report Index

**Date:** 2026-08-31 → 2026-09-01
**Auditor:** GitHub Copilot CLI (Claude Opus 5) + 15 parallel read-only audit agents, plus direct hands-on verification
**Baseline commit:** `93d7719e` (master)
**Scope:** `skyulf-core/` (33,735 lines of `skyulf/`, every file, every line), extended on request to `backend/` (27,564 lines) and `frontend/ml-canvas/src` (71,666 lines)

**No repository source files were modified by this audit.** Every agent was
instructed read-only; all repro scripts were written to `/tmp` and deleted.

---

## Severity key

🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low

Finding IDs use the `OC-` prefix so they never collide with `F-01..F-31` in the
prior audit, [`../skyulf-core-findings.md`](../skyulf-core-findings.md).

---

## Result

**116 findings** — `OC-01 … OC-81` from the agent phase, plus `OC-90 … OC-91`,
`OC-101 … OC-102`, `OC-110 … OC-114`, `OC-120 … OC-122`, `OC-130 … OC-132` and
`OC-140 … OC-159` from the lead auditor's direct review of the modules no agent
covered:

| Severity | Count | |
|---|---|---|
| 🔴 Critical | **5** | [OC-12](./02-encoding-cleaning-imputation-scaling.md#oc-12) · [OC-58](./07-outliers-timeseries-geo.md#oc-58) · [OC-62](./06-core-engines-pipeline.md#oc-62) · [OC-75](./11-tests-packaging-ci.md#oc-75) · [OC-146](./04-evaluation-explainability.md#oc-146) — all five independently re-verified by execution |
| 🟠 High | 45 | |
| 🟡 Medium | 44 | |
| ⚪ Low | 22 | |

⚠️ **Three findings were corrected during verification.** Two agent-phase
findings — [OC-01](./00-validation-log.md#oc-01) (narrower than filed) and
[OC-46](./00-validation-log.md#oc-46) (downgraded 🟠 High → 🟡 Medium) — and one
of the lead auditor's own, [OC-100](./00-validation-log.md#oc-100), **retracted
outright as a false positive**. See the [validation log](./00-validation-log.md).

Plus one systemic recommendation — [**R1: generate the frontend contract from
`@node_meta`**](../opus_core_analysis.md#r1) — which retires eight of these
findings as a *class* rather than one at a time.

**And an equally important negative result:** the
[leakage audit is clean](./08-modeling-tuning.md#leakage-audit-table--all-clean)
(all 7 fit/apply boundaries), all 389 tested hyperparameter ranges are valid, 34/34
nodes are deterministic, coverage is 98.40%, and the backend's path-traversal,
SQLi, SSRF and deserialization defences all hold.

---

## Reports

| # | Report | Scope | Findings |
|---|---|---|---|
| — | [`../opus_core_analysis.md`](../opus_core_analysis.md) | **Master report** — headline, all summary tables, [R1](../opus_core_analysis.md#r1), [suggested fix order](../opus_core_analysis.md#suggested-fix-order), [what was checked and found sound](../opus_core_analysis.md#what-i-checked-and-found-sound) | all |
| 00 | [`00-validation-log.md`](./00-validation-log.md) | **Verification log** — 27 findings independently re-verified by execution; 2 corrections, 4 upgrades, and the calling-convention traps that produce false negatives | — |
| 01 | [`01-cross-cutting.md`](./01-cross-cutting.md) | Version/packaging integrity, schema prediction, engine parity harness, determinism, registry/UI coverage, lint debt | OC-01 … OC-11 |
| 02 | [`02-encoding-cleaning-imputation-scaling.md`](./02-encoding-cleaning-imputation-scaling.md) | Encoders, text/value cleaning, imputers, scalers, drop & missing-value nodes | OC-12 … OC-22 |
| 03 | [`03-feature-generation-selection-vectorization.md`](./03-feature-generation-selection-vectorization.md) | Feature generation & interaction, feature selection, text vectorizers, transformations | OC-23 … OC-34 |
| 04 | [`04-evaluation-explainability.md`](./04-evaluation-explainability.md) | Classification/regression/clustering metrics, decision thresholds, SHAP | OC-35 … OC-38, **OC-146**, OC-147, OC-149 |
| 05 | [`05-profiling.md`](./05-profiling.md) | EDA analyzer statistics, correlations, drift detection, expectations, visualisation | OC-39 … OC-52 |
| 06 | [`06-core-engines-pipeline.md`](./06-core-engines-pipeline.md) | Core abstractions, engine dispatch, schema, pipeline sealing, registry | OC-62 … OC-65, OC-74 |
| 07 | [`07-outliers-timeseries-geo.md`](./07-outliers-timeseries-geo.md) | Outlier detection, bucketing, casting, resampling, time series, geospatial | OC-58 … OC-61 |
| 08 | [`08-modeling-tuning.md`](./08-modeling-tuning.md) | Estimators, cross-validation, **leakage audit**, hyperparameter tuning | OC-66, OC-67 |
| 09 | [`09-backend.md`](./09-backend.md) | Pipeline execution engine, model registry, API routers, services, **security checklist**, config/CORS fail-open, **`ml_pipeline` merge/serving pass** | OC-68 … OC-73, OC-130 … OC-132, OC-150 … OC-159 |
| 10 | [`10-frontend.md`](./10-frontend.md) | **Node config parity matrix**, state, API client, canvas, type/lint health | OC-53 … OC-57 |
| 11 | [`11-tests-packaging-ci.md`](./11-tests-packaging-ci.md) | Test-suite quality, coverage, benchmarks, packaging, CI | OC-75 … OC-81 |
| 12 | [`12-splitters.md`](./12-splitters.md) | Train/test/validation splitting, stratification, fold adapters — **module verified sound** | OC-90 |
| 13 | [`13-core-internals.md`](./13-core-internals.md) | `skyulf/core/` seams, engine registry globals, deprecation policy, model registry | OC-91 |
| 14 | [`14-hyperparameters.md`](./14-hyperparameters.md) | Search spaces, sklearn param filtering, `random_state` propagation | OC-101 … OC-102 (OC-100 retracted) |
| 15 | [`15-profiling-analyzers.md`](./15-profiling-analyzers.md) | Column analyzer, semantic-type inference, task-type detection, recommendations | OC-110 … OC-114, OC-148 |
| 16 | [`16-dtype-coverage.md`](./16-dtype-coverage.md) | Shared preprocessing helpers, artifacts/schema, dual-engine dtype allow-lists | OC-120 … OC-122 |
| 17 | [`17-file-coverage.md`](./17-file-coverage.md) | **Per-file coverage audit of all 188 core files** + findings from reading previously-unexamined files; **gap now closed** | OC-140 … OC-144 |
| 18 | [`18-file-audit-matrix.md`](./18-file-audit-matrix.md) | **Per-file audit matrix** — every one of the 188 core files classified as bug-found / checked-clean / re-export shim; **0 unchecked** | — |

Reports **00** and **12–16** were written by the lead auditor from direct
investigation only, with no agent input — see [Method](#method).

**Conventions.** The master report is authoritative for the finding list; the
per-domain files hold the detail. Findings surfaced by more than one auditor are
**merged into the existing id** with a "*Merged, not re-filed*" note rather than
given a new number.

---

## Where to start

1. Read [The headline](#the-headline) below — it explains ~11 of the findings at once.
2. Read the [suggested fix order](../opus_core_analysis.md#suggested-fix-order) — 4 tiers, ordered by risk removed per unit of effort.
3. If you only fix four things, fix the four 🔴 Criticals above.

---

## The headline

The single largest source of real, shipped bugs in this repository is **not**
the core algorithms — those are largely sound and well tested. It is the
**hand-duplicated contract between `skyulf-core` and `frontend/ml-canvas`**.

`frontend/ml-canvas/src/core/utils/pipelineConverter.ts` forwards `node.data`
to the backend **byte-for-byte with no validation, renaming, or schema check**.
Every node's parameter names, enum values and defaults are retyped by hand in
TypeScript. This audit found **at least 11 distinct places** where the UI emits
a parameter name or enum value the Python side never reads — each one a silent
no-op or silent fallback, none raising an error, none caught by any test.

The symptom is always the same and always invisible: **the canvas says one
thing, the pipeline does another.**

This is one systemic architectural issue, not eleven independent bugs. The
structural fix is [**R1**](../opus_core_analysis.md#r1).

---

<a id="audit-coverage"></a>
## Audit coverage — and what is *not* covered

Measured, not estimated: every `.py` file under `skyulf/` (188 files) was checked
against the text of all 18 reports.

| Category | Files | Lines | Confidence |
|---|---|---|---|
| Named in a report | 121 | ~25,800 | **Read and reasoned about** |
| Not named, but a registered node | 21 | 3,711 | **Behaviourally exercised** — fit+apply on both engines by the parity harness, and re-run across `PYTHONHASHSEED` by the determinism harness |
| Not named, not a node | 46 | 4,209 | **See below** |

Of that last 4,209 lines, most is low-risk re-export surface (24 of the 46 files
are `__init__.py`, ~700 lines). The substantive remainder:

| Area | Lines | Status |
|---|---|---|
| `modeling/hyperparameters/` (`_tree`, `_linear`, `_svm`, `_bayes`, `_neighbors`, `_clustering`, `_field`) | 1,331 | **Behaviourally validated** — all 389 declared ranges were executed against their real estimators; not line-read |
| `profiling/_analyzer/` (`rules`, `temporal`, `decomposition`, `geo`, `causal`, `numeric`) | 1,216 | ✅ **Now read** — yielded [OC-113](./15-profiling-analyzers.md#oc-113) and [OC-114](./15-profiling-analyzers.md#oc-114); the rest [verified sound](./15-profiling-analyzers.md#final-core-sweep) |
| `modeling/_tuning/` (`optuna`, `halving`, `reporter`) | 425 | **Spot-checked** — sampler seeding verified correct; not line-read |
| `modeling/_evaluation/` (`metrics`, `classification`, `clustering`, `thresholds`, `regression`, `common`), `profiling/_analyzer/text.py`, `preprocessing/base.py`, `preprocessing/*/_common.py` | ~2,100 | ✅ **Now read** — yielded [OC-146](./04-evaluation-explainability.md#oc-146) 🔴, [OC-147](./04-evaluation-explainability.md#oc-147), [OC-148](./15-profiling-analyzers.md#oc-148), [OC-149](./04-evaluation-explainability.md#oc-149); see [17 — follow-up pass](./17-file-coverage.md#followup-pass) |
| `engines/protocol.py`, `core/protocols.py`, `profiling/distributions.py`, `preprocessing/transformations/_power_common.py`, `modeling/_evaluation/regression.py` | 289 | ✅ **Now read** — nothing filed; see [18 — matrix](./18-file-audit-matrix.md#last-five) |

**Both named gaps are now closed.** `profiling/_analyzer/` was the highest-yield
area in the whole audit (six findings: OC-110 – OC-114, OC-148), and the
`modeling/_evaluation/` gap flagged in report 17 produced the audit's **only
post-agent 🔴 Critical** ([OC-146](./04-evaluation-explainability.md#oc-146)).
Per-file coverage is now **188 / 188 files (100%)** — see the
[per-file audit matrix](./18-file-audit-matrix.md) for the file-by-file
classification. The last 289 lines of protocol/declaration surface
(`engines/protocol.py`, `core/protocols.py`, `modeling/_evaluation/regression.py`,
`profiling/distributions.py`, `preprocessing/transformations/_power_common.py`)
and `types.py` were read while compiling that matrix and yielded no findings.
Note that 100% *file* coverage is not 100% *line* coverage: confidence is
uneven across files, and "no finding" means nothing reproducible surfaced —
not proof of correctness.

---

## Method

The audit ran in two phases.

**Phase 1 — parallel survey.** **15 read-only agents** partitioned the source so
that no two agents reviewed the same file, each required to produce executed
repro output for every claim. In parallel, the lead auditor directly checked
everything no single agent could see: the package import graph,
registry-vs-frontend coverage, suppressed-lint debt, version/packaging
integrity, and a purpose-built **all-node cross-engine parity harness** (fit +
apply every registered node on both pandas and polars, diffing column sets,
order, shapes, dtypes and values) plus a **determinism harness** (re-running
every node across `PYTHONHASHSEED` values).

**Phase 2 — verification, by hand.** Agent output is a *lead*, not a finding.
Every claim was therefore re-checked by the lead auditor by execution against
the real code:

- **27 findings were independently re-verified**, logged with command-level
  evidence in [`00-validation-log.md`](./00-validation-log.md). Two were
  **wrong as filed and corrected**; four turned out **worse** than reported.
- The remaining domains — splitting, core internals, hyperparameters, the
  profiling analyzers, and the shared dtype helpers (reports **12–16**) — were
  audited **entirely by the lead auditor with no agent involvement**, after the
  verification pass showed agent output needed more checking than it saved.
- Verification is *not* complete: findings not listed in the validation log
  still rest on agent-supplied evidence and are marked as such.

**Findings without executed evidence were discarded.** Several suspected bugs
were investigated, found to be intentional and documented, and dropped rather
than padding the count — including binning's `NaN` for out-of-range values, the
xgboost `class_weight` parameter (traced end-to-end: it *is* honoured, via
`**kwargs` plus sample-weight translation), and two agent claims that did not
survive re-execution.
