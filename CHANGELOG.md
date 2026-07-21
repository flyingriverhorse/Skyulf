# Changelog

All notable changes to the Skyulf project are documented here, organized by release series.

Each major/minor series has its own file in the [`changelog/`](changelog/) folder.

---

## Release Series

| Series | File | Status | Description |
|--------|------|--------|-------------|
| **0.7.x** | [changelog/0.7.x.md](changelog/0.7.x.md) | Active | Model explainability — SHAP Explainability tab with Summary/Beeswarm/Dependence/Waterfall/Force/Interaction views + dark-mode chart readability fixes; Segmentation (clustering) as a first-class model type with 4 algorithms (K-Means, Mini-Batch K-Means, Gaussian Mixture, Birch) and auto-labeling (profile + reference-column crosstab) (v0.7.0); unifies Basic Training and Advanced Tuning into a single `training` step/`TrainingJob` flow end-to-end — orchestration, node UI, DB schema, legacy canvas nodes, and a fixed-mode ensemble training crash fix — plus a dedicated Ensemble category in the job-tracking UI (Job History, Jobs page, Experiments filters, Job Details badge) (v0.7.1) |
| **0.6.x** | [changelog/0.6.x.md](changelog/0.6.x.md) | Stable | Ensemble models — Voting / Stacking (v0.6.0); Ruff migration & config-parsing fixes (v0.6.1); Security & reliability hardening (v0.6.3 – v0.6.6); Training-button dataset guard & skyulf-core backlog cleanup (v0.6.7); Frontend reliability, refactoring, dark-mode UI polish & skyulf-core preprocessing audit (encoding/imputation/casting/cleaning/pipeline resampling-leak) sweep (v0.6.8); CodeQL path-injection fix, repo-wide ruff format, `ty` pin bump, Dependabot backlog consolidation & full skyulf-core `core`/`data`/`engines`/`modeling`/`profiling` audit sweep (analyzer/tuning/imbalance/LR/visualizer bugs, engine detection, schema/model-registry safety, drift categorical support, rule-tree readability) with matching frontend updates, plus a third audit pass fixing a critical Polars `DateFeatures` silent-null bug, pipeline registry-fallback bug, profiling `exclude_cols` leaks, drift small-int-dtype gap, tuning silent-failure fallback, `EllipticEnvelope` determinism, `H3Index` NaN handling, and more (v0.6.9); repo-wide cyclomatic-complexity cleanup — refactored ~200 over-CCN-8 functions across `backend`/`skyulf-core` via pure extract-method refactors, no behavior change (v0.6.10) |
| **0.5.x** | [changelog/0.5.x.md](changelog/0.5.x.md) | Stable | Promote Winner, Branch-Aware Preview & Architecture Improvements |
| **0.4.x** | [changelog/0.4.x.md](changelog/0.4.x.md) | Stable | Parallel Experiment Execution (v0.4.0+) |
| **0.3.x** | [changelog/0.3.x.md](changelog/0.3.x.md) | Stable | Multi-Path Canvas Execution (v0.3.0 — v0.3.1) |
| **0.2.x** | [changelog/0.2.x.md](changelog/0.2.x.md) | Stable | Data Drift Redesign & Monitoring (v0.2.0) |
| **0.1.x** | [changelog/0.1.x.md](changelog/0.1.x.md) | Stable | Foundation through Theme-Aware Charts (v0.1.0 — v0.1.19) |
