# Enterprise Readiness — Node & Canvas Flexibility

**Date:** 2026-08-11
**Status:** Investigation complete (subagent audit, spot-checked against real code)
**Scope:** Consumer-facing flexibility and power gaps — both new capabilities
and improvements to existing nodes/canvas. Companion to
[2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md), which
covers infrastructure/governance blockers for enterprise adoption.

## How this was produced

A background agent audited `frontend/ml-canvas/` and `skyulf-core/skyulf/`
against 7 flexibility areas (extensibility, reuse/composition, existing-node
rigidity, collaboration, debuggability, data connectivity, export/canvas
scale). One representative claim (a registered backend outlier-detection
node with zero frontend UI exposure) was independently spot-checked and
confirmed exactly as reported — a real instance of the frontend/backend
option-mismatch pattern this repo's own coding rules already warn about.

## 1. Node Extensibility — Large effort

**Current state:** `NodeRegistry` supports programmatic registration, but
every node registers via decorators at **import time**, and the package
deliberately disables auto-discovery in favor of explicit imports
(`skyulf-core/skyulf/registry.py:17-54`, `preprocessing/__init__.py:
239-244`, `modeling/__init__.py:97-101`). The frontend node registry is
similarly static/compile-time (`frontend/ml-canvas/src/core/registry/
init.ts:39-74`). **There is no way today for a customer to add a custom
node — custom transform or custom model — without forking and modifying
`skyulf-core` itself.**

**Recommendation:** A real plugin mechanism needs two tiers:
1. **Metadata-only plugins** (safe, no code execution): a signed manifest
   describing node params via JSON Schema, rendered through a generic
   settings-form component — no sandboxing needed since no arbitrary code
   runs.
2. **Code plugins** (custom Python transforms/models): must execute in an
   isolated worker (separate container/process, restricted filesystem and
   network, resource limits, and a dependency allowlist) — this is a
   significant security surface and should be scoped as its own project,
   not bundled casually into a "nice to have" node feature.

This is one of the highest-leverage capabilities for enterprise customers
specifically, since large orgs frequently have proprietary transforms/models
they cannot wait for upstream to add.

## 2. Pipeline Reuse & Composition — Large effort

**Current state:** Five curated, hardcoded starter templates exist
(`pipelineTemplates.ts:104-238`) that **replace the entire canvas** when
applied (`TemplatesGalleryModal.tsx:24-42`) — there's no merge/insert
option. Server-side pipeline versioning is solid: durable, per-dataset,
monotonically numbered graph snapshots with restore/rename/pin/delete APIs
(`backend/database/models.py:196-227`, `pipelines_io.py:182-252`). What's
missing: **parameterized, reusable user-authored templates**, **sub-pipeline
composition** (using one pipeline as a step inside another), and any
graph-level input/output contract that would make composition possible.

**Recommendation:** Add first-class template records with typed parameters,
defaults, and secret-reference bindings (not raw secrets baked into a
saved graph); support semver-style versioning and an explicit
"instantiate/update from template" flow, distinct from today's
whole-canvas-replace behavior.

## 3. Existing-Node Rigidity — Medium effort (concrete, high-value quick wins here)

Four nodes were inspected in detail against their backend counterparts:

- **Imputation** — the strongest node today: UI exposes simple/KNN/MICE,
  KNN weights, and multiple MICE estimators (`ImputationNode.tsx:16-31,
  205-276`), matching the core's KNN metadata (`knn.py:40-77`). Still
  missing: KNN distance metric choice, custom missing-value marker, MICE
  tolerance/iteration-order/skip-complete/bounds options — all real
  sklearn/core capabilities not surfaced.
- **Outliers — a confirmed, concrete gap:** the UI exposes only
  IQR/Z-score/Winsorize/Elliptic-Envelope
  (`OutlierNode.tsx:16, 309` — `method: 'iqr' | 'zscore' | 'winsorize' |
  'elliptic_envelope'`), but the backend has a **fully registered, working
  `ManualBounds` node** (`manual_bounds.py:85-103`,
  `@NodeRegistry.register("ManualBounds", ...)`) that **has no frontend UI
  path at all** — independently verified: it does not appear anywhere in
  `OutlierNode.tsx`. A user who wants to specify explicit min/max bounds for
  outlier clipping cannot do so through the canvas today, even though the
  backend fully supports it. Elliptic Envelope's `random_state` param is
  also silently unexposed in the UI (`elliptic.py:84-105`).
- **Training** — hyperparameters and several CV strategies are exposed
  (`TrainingSettings.tsx:21-40, 555-628`), and SHAP explainability artifacts
  already exist (`_artifacts.py:20, 114-127`). Gap: the UI's single
  `target_column: string` field (`TrainingSettings.tsx:21-24`) structurally
  prevents multi-output/multi-label target selection even if the backend
  could support it.
- **Train/Test Split** — random split, validation split, shuffle, and
  target stratification exist (`TrainTestSplitNode.tsx:10-15, 94-166`), but
  there's no group-aware split (e.g. split by customer ID to prevent leakage
  across a group), no time-series blocked/purge-embargo split beyond what
  exists, and no custom fold assignment.

**Recommendation, prioritized:**
1. **Fastest win:** add the `ManualBounds` option to `OutlierNode.tsx`'s
   method dropdown — the backend node already exists and works; this is
   purely a frontend gap.
2. Add an "Advanced options" expandable section per node exposing the
   sklearn parameters already computed/available but not surfaced (KNN
   metric, MICE tolerance, Elliptic `random_state`, etc.) — schema-driven
   from the backend's `@node_meta(params=...)` so it can't drift out of
   sync the way it has here.
3. Add multi-target and group-aware/time-blocked split nodes as new,
   scoped features (larger effort than #1/#2).

## 4. Collaboration & Governance — Large effort

**Current state:** Model/version lineage and deployment history exist
(`model_registry/schemas.py:7-30`, `deployment/schemas.py:11-30`). Pipeline
versions carry an optional user ID, but the save routes don't actually
supply an authenticated actor today (`pipelines_io.py:112-123, 201-210`) —
consistent with the auth gap in the backend-blockers doc. Deployment is a
direct, ungated action on a completed job (`deployment/api.py:26-42`) — no
approval workflow, required reviewers, policy gates, node comments, edit
locks, or real-time multi-user canvas collaboration.

**Recommendation:** This depends on the backend's auth/multi-tenancy work
landing first (see backend-blockers §1/§2) — collaboration features (node
comments, approval gates, deployment policy) need a real identity/org model
to attach to. Sequence this phase *after* backend-blockers §1/§2, not in
parallel.

## 5. Debuggability & Iteration Speed — Large effort (high day-to-day value)

**Current state:** The Data Preview node is a genuinely useful terminal
inspector — it submits the graph targeted at just that node
(`DataPreviewComponents.tsx:123-147`, `DataPreviewNode.ts:5-16`), and schema
prediction is debounced with broken-reference highlighting
(`useSchemaPreview.ts:1-74`). **However, preview results are never
persisted** — they're written to a temporary, discarded artifact store
(`preview.py:661-672`). This means there's no node-level result cache: after
any small config edit, re-inspecting a downstream node re-runs everything
from scratch, and there's no selective invalidation of only the affected
downstream nodes.

**Recommendation:** Persist content-addressed node results keyed by
`(dataset fingerprint, node config hash, upstream result hashes)`; on edit,
invalidate only descendant nodes whose hash inputs actually changed
(classic DAG memoization). This is a genuinely high-value change for daily
usability — iteration speed on non-trivial pipelines is probably the single
most-felt pain point for any active user, not just enterprise ones.

## 6. Data Connectivity Flexibility — Medium (batch SQL) to Large (streaming/CDC)

**Current state:** `BaseConnector`'s connect/schema/fetch/validate contract
(`connectors/base.py:6-42`) is a workable abstraction. Only local files
(CSV/Excel/Parquet/JSON, `file.py:11-24`) and S3 (CSV/Parquet,
`s3.py:130-179`) are implemented; the dataset UI is upload-or-select-only
(`DatasetNode.tsx:50-82`, `FileUpload.tsx:110-120`).

**Recommendation:** The existing `BaseConnector` abstraction genuinely makes
new batch SQL connectors (PostgreSQL/MySQL/SQL Server, Snowflake, BigQuery,
Redshift, Databricks) a **medium**-effort addition each — the contract
already generalizes. Streaming/CDC sources (Kafka/Kinesis, incremental
watermarks) are **large** effort because they break the current
run-to-completion pipeline execution model, which assumes a bounded input.
Prioritize batch SQL connectors first — they're the most commonly requested
enterprise data source and don't require an execution-model change.

## 7. Export & Interoperability, Canvas Scalability — Large effort

- **Export:** artifacts are joblib/pickle-oriented, local or S3
  (`artifacts/local.py:6-44`, `artifacts/factory.py:24-37`) — no ONNX,
  MLflow model packaging, or PMML export exists. Notebook export exists but
  produces re-runnable code, not a portable model package.
- **Canvas scale:** templates are static and the node registry is
  compile-time; there's no `parentId`/group/subflow/collapse mechanism for
  large graphs (50+ nodes stays a flat, unmanageable canvas).

**Recommendation:** Add a signed export bundle (model + preprocessing +
feature contract + environment lockfile + metrics + lineage), plus ONNX
export where the model type supports it and MLflow model packaging for
interoperability with existing enterprise MLOps tooling. Separately, add
group/frame nodes with collapse/expand, named ports, minimap navigation, and
subflow extraction — needed once pipeline composition (§2) exists anyway,
so consider scoping them together.

## Top 3 Highest-Value Improvements

1. **Enterprise data connectors with secure credential handling and
   incremental refresh.** The most commonly requested capability by
   real enterprise data teams, and the existing `BaseConnector` abstraction
   makes the batch-SQL subset a medium-effort win.
2. **Reusable, parameterized templates/subflows with versioning and
   promotion governance.** Directly enables the "build once, reuse across
   teams/projects" pattern enterprise customers expect, and is a natural
   pairing with the canvas-scalability (subflow) work in §7.
3. **Persistent node-level caching and inspection for fast debugging.**
   Not enterprise-specific, but the highest day-to-day usability lever for
   every user, and a prerequisite for confidently iterating on large
   pipelines once they become common (which composition/templates in #2
   will encourage).

## Cross-Cutting Note: the Sync-Rule Gap is Real, Not Hypothetical

The `ManualBounds` finding in §3 is a live instance of exactly the
frontend/backend option-mismatch failure mode this repository's own coding
rules already document (the `FeatureGenerationNode`/
`InvalidValueReplacementNode` history). Any future node-flexibility work
should audit every existing node's frontend dropdown against its backend
`@node_meta(params=...)` allow-list as a first pass — there may be more
gaps like this one beyond what this investigation's 4-node sample covered.
