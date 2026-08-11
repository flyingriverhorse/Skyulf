# Enterprise Readiness — Technical Debt & Quality Deep Dive

**Date:** 2026-08-11
**Status:** Investigation complete — two independent audits + rubber-duck
critical review, cross-checked against real code.
**Scope:** Correctness, resilience, testing, maintainability, performance,
and accessibility issues across backend/`skyulf-core` and the
`ml-canvas` frontend — **beyond** what's already covered in
[2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md) (auth,
tenancy, DB, secrets, audit, observability, deployment, licensing) and
[2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md) (node
extensibility, templates, rigidity, collaboration, caching, connectors,
export/canvas scale).

## How this was produced

Two `general-purpose` background agents independently audited backend/core
and frontend for issues neither prior audit covered. A `rubber-duck` agent
then (a) independently re-verified the most load-bearing claims from
*both* prior enterprise-readiness docs against the real files, and (b) ran
its own additional sweep, including cross-checking 3 more node types
frontend-vs-backend the same way the `ManualBounds` gap was found. **One
real error was caught and corrected**: the backend-blockers doc had
attributed a placeholder `has_permission()`/`is_admin` pair to `User` when
it actually belongs to `DataSource` — fixed in that doc. One other claim
(the orphaned-dataset-file "bug") was reframed as an intentional,
operator-visible tradeoff rather than a bug, per the rubber-duck's review.
Everything below reflects the corrected, validated set of findings.

---

## Part A — Backend & skyulf-core

### A1. Error handling & resilience

| Finding | Severity | Effort |
|---|---|---|
| **Cancelling a queued job doesn't stop it.** `job_manager_base.py:95-102` marks a queued job cancelled, but `pipeline_execution_service.py:185-188` unconditionally flips the same row to `running` when the worker picks it up — the cancellation flag is only checked *after* execution starts (`:218-223`). A user can cancel a job seconds before it starts and it trains anyway, burning compute and leaving misleading history. | **High** | Medium |
| **Ingestion cancellation is advisory only** — `service.py:142-150` writes `cancelled`, but the task unconditionally overwrites to `processing` and later `completed` (`tasks.py:104-121,160-167`); no Celery task ID is stored/revoked. | Medium | Medium |
| **In-process background jobs can strand forever if the API process dies mid-fit.** Celery is disabled by default (`config/mixins/celery.py:7`); stale-job repair only runs at API *startup* with a 2-hour default window (`main.py:174-223`, `celery.py:36-40`) — no periodic reaper independent of a restart. | Medium | Medium |
| **No Celery task time/resource limits; S3 calls have no retry.** `celery_app.py:12-28` sets only ack/prefetch/serializer options. `artifacts/s3.py:118-149` makes one attempt per call, and `:192-194` silently converts a storage failure into an *empty* artifact list — indistinguishable from "no artifacts exist." | Medium | Medium |
| Broad `except`-style suppression in evaluation metrics hides real bugs with no log (`skyulf-core/skyulf/modeling/_evaluation/metrics.py:225-226,278-279,292-303`). | Low | Small |

**Recommendation:** Atomically claim jobs (`UPDATE ... WHERE status='queued'`) and re-check cancellation immediately before expensive work; persist/revoke Celery task IDs for ingestion; add a heartbeat/lease-based reaper independent of API restarts; configure Celery soft/hard time limits + worker recycling; add exponential-backoff retries for S3/Redis and never silently coerce a storage error into an empty result.

### A2. Testing coverage & quality

- **The "integration" tests don't exercise the real production path.** `tests/test_pipeline_task.py:35-75` mocks out `PipelineEngine` and the DB session entirely; the API job test uses in-process `BackgroundTasks` with `time.sleep(0.5)` polling against shared local files (`tests/test_api_integration_job.py:16-68`), not real Celery/Redis/S3. There are 243 test files with solid unit coverage of node registry, tuning, and serialization — but the production boundaries that actually fail in the real world (worker death, duplicate delivery, cancellation races, S3 round-trips) are untested. **(Medium/Large)**
- The same test file's fixed shared directory + fixed sleep timings are flaky under parallel runs or slow CI (`test_api_integration_job.py:17-20,52-68`). **(Low/Small)**

**Recommendation:** Add a Docker-Compose-based integration test tier running real PostgreSQL + Redis + a worker + MinIO (S3-compatible), explicitly covering: worker killed mid-fit, duplicate task delivery, cancel-before-start, cancel-during-fit, retry exhaustion, and artifact round-trip. Move the flaky test to `tmp_path` + deterministic sync.

### A3. Maintainability hotspots

- **`backend/monitoring/router.py` is 1,970 lines, 21 endpoints**, mixing drift calculation, alert lifecycle, error tracking, slow-node analytics, pipeline logging, and node inspection in one file (`list_error_events` alone has 16 branches at `:904`). **(Medium/Large)**
- **`_node_runners.py` (959 lines)** and **`_tuning/engine.py` (1,249 lines)** concentrate loader/training/transform logic and CV/strategy/reporting/Optuna integration respectively — this is also the exact code the [deep-learning plan](../deep-learning/2026-08-11-architecture-design.md) needs to extend, so cleaning this up first would make the DL work safer too. **(Medium/Large)**

**Recommendation:** Split `monitoring/router.py` into `drift_router`/`error_events_router`/`pipeline_logs_router`/`node_inspector_router` backed by real services; extract strategy-specific search executors from the tuning engine before adding the DL tuning loop on top of it (sequencing note for the DL plan).

### A4. Dependency & version hygiene — Low severity, Small effort

- Production requirements leave `numpy`, `xgboost`, `lightgbm`, `sentence-transformers`, `boto3`, `s3fs`, `pyarrow` unbounded (`requirements-fastapi.txt:30-32,59-70`); `uv.lock` exists but a plain `pip install -r requirements-fastapi.txt` bypasses it.
- Root `pyproject.toml` requires `skyulf-core>=0.5.0` while `skyulf-core/setup.py` is at `0.5.7` — non-workspace installs can silently resolve a newer, untested core.

**Recommendation:** Make the lockfile the supported deployment path, or add tested upper bounds; pin backend/core compatibility as an explicit, tested pair.

### A5. Concurrency & race conditions

- **Pipeline version numbers can collide.** `PipelineVersionsService.create_version()` reads all versions, computes `max + 1`, then inserts (`pipeline_versions_service.py:54-79`) — `PipelineVersion.version_int` has no uniqueness constraint or atomic counter (`models.py:209-224`). Concurrent saves can assign the same version number, corrupting ordering. **(High/Medium)**
- **Job rows have no optimistic lock.** `TrainingJob.version` is a model-version field, not a row revision (`models.py:311-329`); ordinary load/mutate/commit means at-least-once delivery or concurrent actions can last-writer-win status/logs/metrics. **(Medium/Medium)**

**Recommendation:** Add a unique `(dataset_source_id, version_int)` constraint with retry-on-conflict; add a row revision/lease token to `TrainingJob` and make execution idempotent per job ID.

### A6. Data validation & input sanitization

- **Upload validation is filename/extension-only — no decompression-bomb protection.** `_validate_upload_filename` (`service.py:411-427`) blocks path traversal and checks extension, with a 10GB compressed-upload cap (`config/mixins/files.py:13-35`), but Excel/JSON are eagerly materialized in full (`connectors/file.py:103-120`) with no check on decompressed size, row/column count, or ZIP-member expansion ratio before parsing. A small compressed file can expand to exhaust worker memory. **(High/Medium)** — *Positive counterpoint the rubber-duck surfaced: filename handling itself is genuinely solid (traversal blocked, extension allow-listed, size-capped, server-generated storage names) — this is specifically a decompression/parse-resource-limit gap, not a broad "uploads are unsafe" problem.*
- Drift-check upload accepts any extension and parses synchronously in the async route, treating anything non-`.parquet` as CSV (`monitoring/router.py:239-259`). **(Medium/Small)**

**Recommendation:** Validate file signatures (magic bytes) in addition to extensions; inspect ZIP member count/expansion ratio before parsing XLSX; enforce decompressed-byte/row/column limits; move parsing into a bounded worker process for both ingestion and drift uploads.

### A7. Pipeline backward compatibility — High severity, Large effort

- **Saved pipeline snapshots have no schema version.** `params: dict[str, Any]` with no version field (`_internal/_schemas.py:28-41,82-99`); versions store the raw graph unchanged (`pipeline_versions_service.py:62-79`); execution assumes every node has current keys (`pipeline_execution_service.py:71-86`). If a node's param schema changes (exactly the kind of change the deep-learning and node-flexibility plans will make), an old saved pipeline either behaves differently or fails outright on load/run, with no migration path or warning.

**Recommendation:** Add `pipeline_schema_version` + per-node schema version metadata, a deterministic migration registry, and a "validate/migrate preview" step before running an old snapshot. **This should land before or alongside the deep-learning node additions**, since new DL nodes are exactly the kind of schema change that will start breaking old pipelines without it.

### A8. Performance under load

- Monitoring dashboards run full-table scans/global facet counts on unindexed columns (`PipelineRunLog.node_id`/`node_type`, `models.py:585-600`; router queries at `monitoring/router.py:891-991,1626-1673`) with non-sargable `%term%` search. **(Medium/Medium)**
- `TrainingJob` has single-column indexes only; active-job lookups filter 4+ columns together (`jobs.py:104-114`, `models.py:252-259,311-314`). **(Medium/Small)**
- **Blocking dataframe I/O inside async handlers**: `DataService.load_file()` is `async` but calls synchronous Polars/Pandas readers directly (`services/data_service.py:33-64,86-112`), same pattern in EDA (`eda/tasks.py:146-177`) and drift upload parsing — large files can stall the event loop for *all* requests, not just the one loading data. **(Medium/Medium)**

**Recommendation:** Add composite indexes matching real query patterns (e.g. `(dataset_source_id, node_id, status, created_at)`, `(run_mode, started_at DESC)`); move blocking dataframe reads to a bounded thread/process pool; paginate/cache monitoring facets and add retention/partitioning for log tables.

---

## Part B — Frontend (ml-canvas)

### B1. State management

- `useGraphStore` is becoming a "god store" — graph, preview output, job summaries, branch derivations, schema predictions, validation, and undo state all combined (`core/store/useGraphStore.ts:91-150`). **(Medium/Medium)**
- **Autosave/server-save can silently diverge.** Autosave is browser-local only (`useCanvasAutoSave.ts:6-19,30-42`); server persistence happens only on explicit Save (`usePipelineActions.ts:143-169`) — no dirty/sync indicator or conflict detection between the two. **(Medium/Medium)**
- Cross-component coordination uses untyped global `CustomEvent`s (`useKeyboardShortcuts.ts:12-78`). **(Low/Small)**

**Recommendation:** Split the graph store into execution/schema/derived-canvas slices; add explicit dirty/synced/conflict state surfaced in the UI (ties directly into the pipeline-version-race fix in A5 — a user should see when their local edits diverge from the last saved server version); replace ad-hoc events with a typed command bus.

### B2. Performance at scale

- Positive: node cards and edges are properly memoized already (`CustomNodeWrapper.tsx:544`, `FlowCanvas.tsx:40-63`).
- Dataset table renders every row with no virtualization, unlike job history which virtualizes above 50 items (`DataSources.tsx:350-386` vs `JobsDrawer.tsx:366-387`, `VirtualList.tsx`). **(Medium/Medium)**
- Every node card does an O(jobs) linear scan per render (`CustomNodeWrapper.tsx:37-41`) — at 50 nodes × frequent job updates this compounds. **(Low/Small)**

**Recommendation:** Reuse the existing `VirtualList` component for the dataset table; replace the linear job scan with an indexed active-job-by-node map.

### B3. Accessibility — the most consistently severe area found

- **Node palette entries are unlabeled, unfocusable, keyboard-inert `<div>`s** with an explicit lint-disable for the missing keyboard handler (`Sidebar.tsx:116-135`) — a keyboard-only user cannot add a node to the canvas at all. **(High/Medium)**
- **Canvas connections are mouse/drag-only**; the only keyboard handling is fit-view (`FlowCanvas.tsx:205-231`); ports have no accessible names (`CustomNodeWrapper.tsx:509-538`) — a keyboard/screen-reader user cannot wire two nodes together. **(High/Medium)**
- Several icon-only controls lack `aria-label`s (Properties panel expand/close, upload cancel). **(Medium/Small)**
- Upload progress bar exposes only visible text, no `role="progressbar"` semantics. **(Medium/Small)**
- The E2E accessibility test suite only *fails* on axe-`critical` violations — `serious` violations are logged but accepted, not blocked (`e2e/a11y.spec.ts:5-10,50-69`). **(Medium testing gap)**

**Recommendation:** This is the single most consequential fix area for genuinely broad ("everyone wants to use it") adoption — many enterprise procurement processes require WCAG 2.1 AA / VPAT compliance, and today the core "build a pipeline" flow is provably impossible via keyboard alone. Convert palette entries to real focusable buttons with drag as an enhancement, not the only path; add a keyboard "select source port → select target port" connection flow with labelled ports and live feedback; label every icon-only control; promote `serious` axe violations to blocking in CI.

### B4. Error handling & feedback

- Monitoring badge fetch failures are silently swallowed — an outage looks identical to "no alerts" (`components/Layout.tsx:61-87`). **(Medium/Small)**
- Job-store fetch/poll errors only `console.log`, with no error state surfaced to any consumer (`useJobStore.ts:130-173`) — job history can look empty/stale with no indication anything failed. **(Medium/Small)**
- Experiment run submission discards the real error message before showing a toast, unlike preview which does this correctly (`useRunControls.ts:99-107` vs `:154-155`). **(Low/Small)**

**Recommendation:** Add a visible "unavailable, retrying" state to the monitoring badge and job store instead of silent failure; make the experiment-submission error path consistent with the (already correct) preview error path.

### B5. Testing coverage

- The canvas E2E test seeds Zustand state directly rather than performing real drag/drop/connect interactions, because canvas interaction is "considered unreliable" per its own comments (`e2e/preview.spec.ts:11-17,61-109`) — **the core "build a pipeline" user journey has no real browser-level test.** **(Medium/Medium)**
- E2E only runs Desktop Chrome despite the app's own responsive/read-only tablet behavior (`playwright.config.ts:25-30`). **(Medium/Small)**
- Accessibility E2E covers only 4 routes, omitting operational/modal-heavy pages. **(Low/Small)**

**Recommendation:** Build stable `data-testid`-based interaction helpers for drag/connect and add a real end-to-end "place node → connect → configure → run" browser test; add a tablet viewport project to the Playwright config; expand a11y test route coverage.

### B6. Code organization & duplication

- Adding a new node today costs ~135 LOC for a simple node up to 500–1,171 LOC for complex ones, spread across a settings form, metadata, defaults, validation, ports, and a manual registry-import edit (`core/types/nodes.ts:34-70`, `core/registry/init.ts:2-37,39-75`, `DropRowsNode.tsx:16-135`). **(Medium/Large)** — **this directly affects the plugin-system feasibility discussed in the node-flexibility doc**: a schema-driven settings/metadata generator would lower both the plugin cost and the day-to-day cost of adding official new nodes (including the upcoming DL nodes).
- Node forms repeat schema lookup, upstream traversal, metrics feedback, and responsive layout boilerplate per-node rather than as shared hooks (`DropRowsNode.tsx:21-30,55-110`). **(Medium/Medium)**

**Recommendation:** Build a field-schema renderer on top of the existing `PropertiesPanel` host, plus shared hooks for upstream-schema lookup and result-metrics display — this is foundational both for the plugin system (node-flexibility doc §1) and for keeping the upcoming DL settings panel (deep-learning frontend design doc) from repeating the same boilerplate a 16th time.

### B7. Mobile/responsive

- Canvas authoring is intentionally read-only below 1024px (`useReadOnlyMode.ts:5-18`) — an honest constraint, but undocumented as a product decision; general app navigation elsewhere is properly responsive. **(Medium/Small)** — Recommendation: explicitly label this as a desktop-authoring product decision (or scope a compact tablet editor) rather than leaving it as an implicit limitation users discover by trial.

### B8. Design system consistency

- A real semantic token system exists (`index.css:20-80`) alongside a legacy variable file (`styles/variables.css:1-67`) and pages that bypass both with raw Tailwind colors (`DataSources.tsx:234-255`, `Dashboard.tsx:122-189`). **(Medium/Medium)**

**Recommendation:** Consolidate on one semantic token source; this is also called out as a prerequisite in the [redesign doc](2026-08-11-redesign-existing-pages.md)'s cross-page recommendations — sequence them together.

---

## Independent Rubber-Duck Sweep — Additional Findings

Beyond validating the above, the rubber-duck agent ran its own sweep and found:

1. **N2 — A second, smaller instance of the exact ManualBounds-style drift**: `one_hot.py`'s `_resolve_fit_options` reads `prefix_separator`/`drop_original` from config, but only `drop_original` is in the frontend and neither is in the backend's own `@node_meta(params=...)` allow-list (`one_hot.py:116-124,211-217`). **(Non-blocking, Small)** — fold into the same "audit every node's param allow-list" effort recommended in the node-flexibility doc.
2. **N3 — Forward-looking risk once multi-tenancy lands.** Artifacts load via `joblib.load`/`pickle.load` (`local.py:44`, `s3.py:144`, `pipeline.py:569`) — already-documented arbitrary-code-execution risk, currently low because artifacts are app-generated only. **Once per-tenant artifact storage (backend-blockers §2/§9) is built, this becomes a real cross-tenant RCE vector** if artifact paths/keys are ever influenceable — pair tenant-scoped storage with either signed/verified artifacts or a non-pickle serialization for anything crossing a trust boundary. **This is a cross-cutting risk neither prior audit connected — add it explicitly to the multi-tenancy implementation plan.**
3. **N5 — Silent coercion**: `one_hot.py`'s `handle_unknown` silently collapses any value other than `"ignore"` to `"error"`; the frontend type also allows `'use_encoded_value'`, which would silently misbehave if ever wired up. **(Suggestion, Small)**
4. **Positive finding worth keeping in mind**: 3 additional nodes sampled (Scaling, Encoding's core options) were fully synced frontend-to-backend — the ManualBounds/prefix_separator gaps appear to be isolated cases, not a systemic pattern, based on this sample.

## Cross-Reference: How This Feeds Other In-Flight Plans

- **A3/A7** (tuning engine size, no pipeline schema versioning) should be addressed **before or alongside** the [deep-learning implementation roadmap](../deep-learning/2026-08-11-implementation-roadmap.md), since DL adds new node types and a new tuning path to exactly this code.
- **B6** (node-authoring cost) is the same finding underpinning the node-flexibility doc's plugin-system section — a schema-driven renderer serves both efforts.
- **N3** (deserialization × multi-tenancy) must be added to the backend-blockers §2/§9 implementation plan when that work starts.
