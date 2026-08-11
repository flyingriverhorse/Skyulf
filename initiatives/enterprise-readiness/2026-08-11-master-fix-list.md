# Enterprise Readiness — Master Fix List

**Date:** 2026-08-11
**Status:** Consolidated, prioritized action list synthesized from 5
investigation docs in this folder, cross-validated by an independent
rubber-duck review. This is the single document to work from when planning
implementation; the other docs are the detailed evidence/design behind
each item.

## How to read this list

Each item has: **Doc** (source), **Severity**, **Effort**, and whether it's
a **hard blocker** (must fix before any multi-customer/enterprise
deployment), a **quality/resilience** issue (should fix, real risk today),
or a **flexibility/UX** improvement (makes the product better, not a
safety issue). Items are grouped into phases reflecting real dependencies
between them — not arbitrary priority buckets.

---

## Phase 0 — Foundation (nothing else is safe to build on top of this)

These three must land together; auth without tenancy just adds a login
screen in front of one shared workspace, and neither is safe on today's
DB/storage defaults.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Real authentication (JWT/OIDC, replace hardcoded `user_id=1`, wire up the already-declared but uninstalled `passlib`/`bcrypt`/`python-jose` deps) | backend-blockers §1 | Blocker | Large |
| Multi-tenant/organization data model (`Organization`/`Membership`/`Workspace`, `workspace_id` on every table) | backend-blockers §2 | Blocker | Large |
| PostgreSQL mandatory + Alembic migrations (replace SQLite default + exception-swallowing `ALTER` statements) | backend-blockers §3 | Blocker | Large |
| Encrypted, managed object storage (SSE-KMS, per-tenant prefixes) | backend-blockers §9 | Blocker | Medium |
| Fix `DataSource.has_permission()` placeholder as part of the real auth work (corrected attribution — not `User`, see technical-debt-deep-dive.md intro) | backend-blockers §1 | Blocker | (included above) |
| **New:** pair per-tenant artifact storage with signed/verified artifacts or non-pickle serialization for cross-trust-boundary loads — `joblib`/`pickle.load` becomes a real cross-tenant RCE vector the moment artifact storage is tenant-scoped | technical-debt-deep-dive.md, rubber-duck finding N3 | Blocker (for Phase 0 work specifically) | Medium |

## Phase 1 — Production Operating Model (parallel with Phase 0)

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Secrets-manager integration (Vault/AWS Secrets Manager), IAM-role S3 access instead of static keys | backend-blockers §4 | High | Medium |
| Remove `--reload` from production Dockerfile; ship real Compose/Helm reference deployment with health probes, HPA | backend-blockers §8 | High | Large |
| Structured JSON logging, Prometheus metrics, OpenTelemetry tracing; fix readiness probe to actually check DB/Redis | backend-blockers §6 | Medium | Medium |
| API versioning (`/v1`), Redis-backed rate limiting keyed on principal/org not just IP, actually mount security-headers middleware | backend-blockers §7 | Medium | Medium |
| Append-only audit-event table + data-retention/deletion workflows with DSAR support | backend-blockers §5 | High | Large |
| Usage-metering/entitlement service tied to the commercial license tier | backend-blockers §10 | Medium | Medium |

## Phase 2 — Resilience & Correctness Fixes (independent of Phase 0/1, fix any time)

These are real bugs/gaps found in the deeper technical audit — not
enterprise-specific, but they affect every deployment today.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Fix job-cancellation race (queued job can be resurrected and trained anyway by the worker) | technical-debt-deep-dive.md §A1 | High | Medium |
| Fix pipeline-version-number collision race (no uniqueness constraint on `version_int`) | technical-debt-deep-dive.md §A5 | High | Medium |
| Add decompression-bomb/resource-limit protection to XLSX/JSON upload parsing | technical-debt-deep-dive.md §A6 | High | Medium |
| Add `pipeline_schema_version` + migration registry for saved pipeline graphs — **do this before or alongside the deep-learning node additions**, since new node types are exactly the kind of change that breaks old saved pipelines without it | technical-debt-deep-dive.md §A7 | High | Large |
| Add heartbeat/lease-based job reaper independent of API restarts; Celery time limits; S3/Redis retry with backoff (never silently coerce a storage failure into "no artifacts") | technical-debt-deep-dive.md §A1 | Medium | Medium |
| Add optimistic locking/row revision to `TrainingJob` | technical-debt-deep-dive.md §A5 | Medium | Medium |
| Add composite DB indexes for job/log query patterns actually used | technical-debt-deep-dive.md §A8 | Medium | Small |
| Move blocking dataframe I/O (Polars/Pandas reads) out of async request handlers | technical-debt-deep-dive.md §A8 | Medium | Medium |
| Split `monitoring/router.py` (1,970 lines/21 endpoints) into focused routers | technical-debt-deep-dive.md §A3 | Medium | Large |
| Add Docker-Compose-backed integration tests (real Postgres/Redis/worker/MinIO) covering worker-death, duplicate-delivery, cancellation races | technical-debt-deep-dive.md §A2 | Medium | Large |

## Phase 3 — Accessibility (treat as its own priority tier, not folded into general polish)

Called out separately because it's the most consistently severe, concrete,
and license-blocking category found (many enterprise procurement processes
require WCAG 2.1 AA/VPAT compliance) — and today the core "build a
pipeline" flow is **provably impossible via keyboard alone**.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Convert node-palette entries from inert `<div>`s to real focusable/keyboard-activatable buttons (drag becomes an enhancement, not the only path) | technical-debt-deep-dive.md §B3 | High | Medium |
| Add keyboard-driven node-connection flow (select source port → select target port) with labelled ports | technical-debt-deep-dive.md §B3 | High | Medium |
| Add `aria-label`s to icon-only controls; proper `role="progressbar"` semantics on upload progress | technical-debt-deep-dive.md §B3 | Medium | Small |
| Promote axe `serious` violations to blocking in CI (currently only `critical` fails the build) | technical-debt-deep-dive.md §B3 | Medium | Small |

## Phase 4 — Shared Frontend Infrastructure (build once, unblocks every page redesign)

Every page redesign in Phase 5 depends on these; build them first so pages
aren't redone twice.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| One shared `DataTable` (sticky header, sort, density, skeleton rows, empty/filter state, row-action overflow, detail drawer) replacing divergent table implementations | redesign-existing-pages.md, new-enterprise-pages.md | Medium | Medium |
| Consolidate on the *existing* shared `StatusBadge`; delete page-local reimplementations (confirmed duplicate in `pages/Jobs.tsx`) | redesign-existing-pages.md, technical-debt-deep-dive.md §B8 | Medium | Small |
| One semantic design-token source (currently two parallel token systems: `index.css` semantic vars + legacy `styles/variables.css`, with pages also bypassing both via raw Tailwind colors) | technical-debt-deep-dive.md §B8, redesign-existing-pages.md | Medium | Medium |
| Standardize `EmptyState`/`LoadingState`/`ErrorState` variants (first-use, filtered-empty, permission-error, recoverable-failure) | redesign-existing-pages.md | Medium | Small |
| Split the "god" `useGraphStore` into execution/schema/derived-canvas slices; add explicit dirty/synced/conflict indicator for autosave-vs-server-save divergence | technical-debt-deep-dive.md §B1 | Medium | Medium |
| Reuse existing `VirtualList` for the Dataset table (currently renders every row unvirtualized) | technical-debt-deep-dive.md §B2, redesign-existing-pages.md §4 | Medium | Small |
| Build a schema-driven node settings-form renderer (currently 135–1,171 LOC per node, repeated boilerplate) — this also directly de-risks the plugin-system idea in node-flexibility.md §1 and the upcoming DL settings panel | technical-debt-deep-dive.md §B6, node-flexibility.md §1 | Medium | Large |

## Phase 5 — Page Redesigns (existing pages)

| Page | Doc | Effort |
|---|---|---|
| Pipeline Canvas (health strip, command bar, inspector tabs) | redesign-existing-pages.md §1 | Large |
| Experiments/Run Comparison (ranked table, decision rail) | redesign-existing-pages.md §2 | Large |
| Jobs Monitoring (unify drawer + routed page, one source of truth) | redesign-existing-pages.md §3 | Medium |
| Dataset/Data Management (catalog + asset detail view) | redesign-existing-pages.md §4 | Large |
| Drift Monitoring (Overview + Analysis split, triage table) | redesign-existing-pages.md §5 | Large |
| Model Registry & Deployments (unified lifecycle shell, preflight checks) | redesign-existing-pages.md §6 | Large |
| **Follow-up, not yet designed:** Dashboard, routed Jobs page, EDA, Error Log, Slow Nodes pages — enumerated but not covered by the redesign doc; do this as a fast follow-up using the same method | redesign-existing-pages.md (correction note) | TBD |

## Phase 6 — New Enterprise Pages (build in parallel as mocked UI, wire up once Phase 0 lands)

| Page | Doc | Effort | Note |
|---|---|---|---|
| Login/SSO | new-enterprise-pages.md §1 | Medium | Genuinely new |
| Organization & Workspace Settings | new-enterprise-pages.md §2 | Medium | Genuinely new |
| Member/Role Management (RBAC) | new-enterprise-pages.md §3 | Medium | Genuinely new |
| Audit Log Viewer — **redesign/extend, not build from scratch** | new-enterprise-pages.md §4 | Small (audit current page first) | `pages/AuditLogPage.tsx` already exists — confirmed via frontend audit |
| Usage/Billing/Quota Dashboard | new-enterprise-pages.md §5 | Medium | Genuinely new |
| API Keys/Service Accounts | new-enterprise-pages.md §6 | Medium | Genuinely new |
| Unified app shell (org switcher, settings nav, Build/Operate/Observe/Settings grouping) | new-enterprise-pages.md, redesign-existing-pages.md | Medium | Ties everything together |

## Phase 7 — Flexibility & Extensibility (from the earlier node-flexibility audit, unaffected by anything new found this round)

Already documented in [2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md)
— restated here only for completeness of the master list, not re-audited:

1. Fix `ManualBounds` outlier node missing from frontend UI (fast, isolated win)
2. Fix `one_hot.py`'s `prefix_separator`/`drop_original` allow-list gap (new, smaller instance of the same pattern — technical-debt-deep-dive.md rubber-duck finding N2)
3. Enterprise SQL data connectors (medium effort given existing `BaseConnector` abstraction)
4. Reusable parameterized templates/subflows with versioning
5. Persistent node-level result caching (highest day-to-day usability lever, not enterprise-specific)
6. Plugin system for custom nodes (two-tier: metadata-only vs sandboxed code plugins)
7. ONNX/MLflow export

## Phase 8 — Quick, High-Leverage Wins (do these early/in-parallel — cheap, evidence-backed, and reused by multiple other phases)

New from Round 3 (differentiation-strategy.md + smooth-experience-fixes.md).
These are called out separately because they're unusually cheap relative
to their impact — the underlying data/code/plumbing for each already
exists, only the missing piece is small.

| Item | Doc | Effort | Why it's high-leverage |
|---|---|---|---|
| Surface a "Load sample dataset" option + bind one starter template to it | smooth-experience-fixes.md Top 3 #1 | Small | Example CSVs already ship in the repo (`skyulf-core/examples/data/`) and templates already exist — only the UI entry point and one binding are missing |
| Show a "Live / Reconnecting" WebSocket indicator | smooth-experience-fixes.md §C | Small | The connection-state plumbing (`jobEventsSocket.onStatus`) already exists and is unused by any UI |
| Add missing success toasts (delete dataset, create data source) | smooth-experience-fixes.md §G | Small | One-line additions, closes an inconsistency users will notice fast |
| Debounce the Inference page's 3x-per-keystroke JSON parsing | smooth-experience-fixes.md §B | Small | Isolated, contained fix |
| Port `BestParamsModal` onto the shared `ModalShell` | smooth-experience-fixes.md §D | Small | Removes a keyboard-nav dead spot |
| Adopt or delete the unused `Skeleton` component | smooth-experience-fixes.md §H | Small | Overlaps with Phase 4's shared-state-component work — do together |
| Coalesce per-keystroke node-config undo entries | smooth-experience-fixes.md §E | Small-Medium | Prevents silent eviction of structural undo history |

## Phase 9 — Differentiation & Core Investment (the "why choose us" work)

New from Round 3 (differentiation-strategy.md). These are the bets that
make Skyulf competitively different, not just "at parity." Ranked by the
strategy doc; sequencing notes below reflect real dependencies.

| Item | Doc | Effort | Sequencing note |
|---|---|---|---|
| **Foundational, do first:** partitionable/stateless calculator contract in `skyulf-core` | differentiation-strategy.md Part 3, technical-debt-deep-dive.md §A3 | XL | Blocks the planned Ray migration from working smoothly; also blocks safe parallel execution generally — do before piling on more node types (including DL) |
| **Foundational, do first:** versioned artifact schema/migration path in `skyulf-core` | differentiation-strategy.md Part 3 | Large | Every new node type today creates artifacts that can silently break on a future core upgrade — same urgency as pipeline schema versioning (Phase 2) |
| Post-upload pipeline recommendation ("point at data, get a baseline") | differentiation-strategy.md Bet #2 | Medium | Reuses the existing `EDAAnalyzer`/`profiling/recommendations.py` almost entirely as-is — build alongside Phase 8's sample-dataset work |
| Enforced (not just heuristic) leakage/data-quality guardrails | differentiation-strategy.md Bet #1 | Large (incremental) | Start by surfacing the *already-computed* server-side leakage/correlation checks as real-time canvas warnings, then add new checks (train/test overlap, temporal leakage) |
| Two-way notebook export/import loop ("graduate to code, don't leave") | differentiation-strategy.md Bet #3 | Medium-Large | Builds on the already-shipped one-way notebook export |
| Deployment/serving DX overhaul (telemetry, performance-decay monitoring, canary/champion-challenger) | differentiation-strategy.md Bet #4 | Large | Overlaps with the MLOps-lifecycle gaps found — sequence after Phase 0 (multi-tenancy) since production monitoring needs a real org/workspace model to attach to |
| Forecasting model family (ARIMA/Prophet-style) | differentiation-strategy.md Bet #5 | Large | Named, verifiable gap vs. a specific competitor capability (Databricks AutoML ships this) |
| Declarative per-node config validation (replace 246 ad-hoc `config.get` call sites) | differentiation-strategy.md Part 3 | Large | Improves error-message quality across the board — ties to the "generic error messages" finding in smooth-experience-fixes.md |
| Universal calculator contract tests (every registered node, not a curated subset) | differentiation-strategy.md Part 3 | Medium | Cheap insurance once the artifact-versioning/partitionable-contract work above lands — do together |
| **New from Round 4:** per-node/per-step data preview ("click any node, see the data there, no full run required") | round4-synthesis.md, user-complaints-research.md #4 | Medium-Large | The single most externally-validated UX gap found this session — a real competitor tool was built by an ex-user of an incumbent specifically to solve this exact problem. Directly reinforces Bets #1 and #3 above; pairs naturally with the existing canvas node-config UI |
| **New from Round 5:** unify job logs, per-node execution ledger, and data-quality warnings into one canonical per-run diagnostic timeline (the pieces already exist — job logs, preview-node failure cards, notification history — they're just fragmented across disconnected UI surfaces) | round5-synthesis.md, user-observability-audit.md | Medium-Large | Reinforces Bet #1 (transparency/anti-black-box) directly; unlike most Phase 9 items this is largely UI/wiring work reusing existing backend data, not new capability |

## Phase 10 — Security & Scale Hardening (Round 4)

New from Round 4 (security-review.md, scale-load-audit.md). Unlike Phases
0/1 (foundational identity/tenancy work), these are scoped, mostly cheap
fixes that should not wait for the bigger foundational phases.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Fix SSRF via EDA's unsanitized S3 `endpoint_url` (SEC-01) | security-review.md | Medium, high confidence | Small |
| Fix SSRF via pipeline resolution's nested `client_kwargs.endpoint_url` (SEC-02) | security-review.md | Medium, high confidence | Small — same shared fix as SEC-01 |
| Add per-user/per-org resource quotas (queued/running jobs, stored bytes, CPU/GPU time) — the IP-only rate limiter is not a substitute | scale-load-audit.md, backend-blockers.md (rate-limiter finding) | Critical | Medium |
| Require explicit, documented concurrent Celery worker deployment (not default `solo`) with separate queues | scale-load-audit.md | Critical | Medium |
| Enforce input/RSS memory budgets or move to streaming/chunked processing for large datasets | scale-load-audit.md | Critical | Large |
| Migrate production deployments off SQLite + local disk to PostgreSQL + object storage (ties to Phase 0) | scale-load-audit.md, backend-blockers.md | High | Medium |
| Virtualize/paginate large result tables and dataset previews (10,000+ rows currently render every `<tr>`) | scale-load-audit.md | High | Small-Medium |
| Address the two data-governance Critical items: retention/DSAR workflow and encryption at rest | data-governance-audit.md | Critical (procurement blocker) | Large |
| Broaden PII detection beyond email/phone and add a masking/tokenization workflow, not just an advisory alert | data-governance-audit.md DG-01 | High | Medium |

## Phase 11 — Testing/CI Foundations (Round 4)

New from Round 4 (testing-ci-audit.md). These should land before or
alongside the DL/Ray work specifically, since that work will plug into
exactly the areas found weakest here.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Replace the skipped, machine-specific full-inference test with a required API → DB → broker/worker → artifact → inference integration test | testing-ci-audit.md | High | High |
| Add real canvas drag/connect Playwright E2E (current spec seeds graph state via a dev hook, bypassing the actual drag-and-drop interaction entirely) | testing-ci-audit.md | High | Medium-High |
| Add coverage gates/ratchets for backend and frontend (none exist today beyond core's 45% floor) | testing-ci-audit.md | High | Medium |
| Add direct tests for `job_service.py` and `pipeline_versions_service.py` (retry/cancel/ownership/failure-state), plus an auth/authz endpoint matrix | testing-ci-audit.md | High | Medium |
| Build a Ray-local failure/retry/serialization/artifact test suite and one real node-contract E2E **before** enabling DL/Ray nodes | testing-ci-audit.md, deep-learning/README.md | High | Medium |

## Phase 12 — Confirmed Bugs (Round 5 — fix independent of any other phase, ASAP)

New from Round 5 (bug-hunt.md). Unlike every other phase, these are not
architecture/design decisions — they are **verified, reproducible logic
errors** with exact repro steps already written down. Treat items 1-3 as
release-blocking regardless of what else is being worked on; they can
silently corrupt model training results.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Fix cross-process duplicate pipeline-job creation (idempotency guard doesn't hold across processes/workers) | bug-hunt.md #1 | High | Medium |
| **Fix Lag Features node: `X` is sorted/dropped but `y` is returned unsorted/unfiltered** — silent train/label misalignment on any unsorted input | bug-hunt.md #2 | High | Small-Medium |
| **Fix Rolling Aggregate node: identical `X`/`y` misalignment bug as Lag Features** — fix the shared root cause once, reuse for both nodes | bug-hunt.md #3 | High | Small-Medium |
| Fix out-of-order job-list API responses reverting newer job state in the UI (add request sequencing/abort to the polling client) | bug-hunt.md #4 | Medium | Small |
| Reject cyclic pipeline graphs at validation time instead of failing confusingly at execution time | bug-hunt.md #5 | Medium | Small-Medium |
| Fix upload UI's incorrect 500MB rejection message when the server default is 10GB | bug-hunt.md #6 | Medium | Small |
| Fix Feature Selection node: advertised/documented default method silently no-ops instead of executing | bug-hunt.md #7 | Medium | Small |
| Fix General Binning node: `uniform` metadata default emitted but the calculator can't execute it, silently no-ops | bug-hunt.md #8 | Medium | Small |
| Fix FeatureMath silent no-op on mixed-timezone datetime extraction (normalize to UTC or raise an actionable error) | bug-hunt.md #9 | Medium | Small |

## Phase 13 — API Contract Hardening (Round 5)

New from Round 5 (api-contract-drift-audit.md). Addresses drift risk
beyond the already-documented node-param duplication pattern (see the
repo-wide Backend/Core ↔ Frontend Sync Rule) — this is about the general
API layer, not individual node configs.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Generate frontend TypeScript types/clients from the backend's OpenAPI spec in CI; fail CI on schema/generated-type drift | api-contract-drift-audit.md | High | Medium |
| Fix confirmed `JobInfo` drift (`created_at` nullability, missing `preview`, `output_artifact_id` vs `output` naming) | api-contract-drift-audit.md | Medium | Small |
| Type and normalize EDA job-status values instead of force-casting (`as JobStatus`); add exhaustive status-mapping tests | api-contract-drift-audit.md | High | Small-Medium |
| Runtime-validate WebSocket message envelopes on the frontend using the already-installed Zod dependency, with versioned fixture tests | api-contract-drift-audit.md | Medium | Small-Medium |
| Introduce `/api/v1` (or equivalent) versioning and a breaking-change/deprecation policy before any public API or independent frontend deployment | api-contract-drift-audit.md | High | Medium |

## Phase 14 — Internationalization, Mobile & Cross-Browser Reach (Round 5)

New from Round 5 (i18n-mobile-crossbrowser-audit.md). Distinct from Phase
3 (Accessibility) — these gaps don't block any current customer but
become hard blockers the moment there's a non-English enterprise
customer, a Middle-East expansion, or a sales demo on a tablet.

| Item | Doc | Severity | Effort |
|---|---|---|---|
| Adopt an i18n architecture (message catalog/provider, locale persistence, `Intl`-based date/number formatting, string-extraction workflow) before any international sales push | i18n-mobile-crossbrowser-audit.md | High (conditional on international GTM) | Large |
| Make an explicit, documented device-support decision for the canvas (desktop-only + tested tablet *inspection* mode, or fund real touch/pointer authoring) | i18n-mobile-crossbrowser-audit.md | Medium-High | Small (decision) to Large (implementation) |
| Add RTL as a deliberate workstream once i18n lands (logical CSS properties, real RTL visual testing) | i18n-mobile-crossbrowser-audit.md | High (conditional on Middle-East expansion) | Large |
| Declare and enforce a browser support matrix; add Firefox/WebKit Playwright projects and a tablet viewport to E2E | i18n-mobile-crossbrowser-audit.md | High | Medium |
| Centralize numeric/metric rendering (consistent significant-digit and p-value/scientific-notation rules, `Intl.NumberFormat` for counts) | i18n-mobile-crossbrowser-audit.md | Medium | Medium |

## What NOT to do

- Don't build Phase 5/6 UI against real backend endpoints before Phase 0
  lands — build as mocked/static UI in parallel, wire up once auth +
  tenancy are real, per new-enterprise-pages.md's explicit dependency note.
- Don't treat the orphaned-dataset-file behavior (backend-blockers §5) as a
  bug to "fix" by changing its logic — it's a deliberate, log-visible
  tradeoff; only add an automated reconciliation sweep on top of it.
- Don't skip Phase 4 (shared frontend infra) and jump straight to Phase 5
  page-by-page — every redesign proposal explicitly depends on the shared
  `DataTable`/`StatusBadge`/token work landing first.
- Don't assume all frontend/backend node param mismatches are as isolated
  as the sample suggests — the rubber-duck's 3-node spot-check came back
  clean plus the one new `one_hot.py` gap found, so the "ManualBounds is
  isolated" read is provisional, not proven across all ~15+ nodes.

## Cross-References

- [2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md) — Phase 0/1 detail
- [2026-08-11-technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md) — Phase 2/3/4 detail
- [2026-08-11-redesign-existing-pages.md](2026-08-11-redesign-existing-pages.md) — Phase 5 detail
- [2026-08-11-new-enterprise-pages.md](2026-08-11-new-enterprise-pages.md) — Phase 6 detail
- [2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md) — Phase 7 detail
- [2026-08-11-smooth-experience-fixes.md](2026-08-11-smooth-experience-fixes.md) — Phase 8 detail
- [2026-08-11-differentiation-strategy.md](2026-08-11-differentiation-strategy.md) — Phase 9 detail, plus the competitive-positioning rationale behind it
- [../deep-learning/README.md](../deep-learning/README.md) — orthogonal, in-flight plan; note the sequencing interactions called out in technical-debt-deep-dive.md (tuning-engine size, pipeline schema versioning) AND Phase 9's `skyulf-core` foundational items (partitionable calculators, artifact versioning) before starting DL implementation — DL adds exactly the kind of new node types that make both gaps more costly to fix later
- [2026-08-11-round4-synthesis.md](2026-08-11-round4-synthesis.md) — Round 4 overview and cross-links; Phase 10/11 detail
- [2026-08-11-security-review.md](2026-08-11-security-review.md) — Phase 10 SSRF findings detail
- [2026-08-11-scale-load-audit.md](2026-08-11-scale-load-audit.md) — Phase 10 scale/load findings detail
- [2026-08-11-data-governance-audit.md](2026-08-11-data-governance-audit.md) — Phase 10 compliance findings detail
- [2026-08-11-testing-ci-audit.md](2026-08-11-testing-ci-audit.md) — Phase 11 detail
- [2026-08-11-user-complaints-research.md](2026-08-11-user-complaints-research.md) — evidence behind the Phase 9 per-node-preview addition and cross-validation of Bets #1/#3
- [2026-08-11-round5-synthesis.md](2026-08-11-round5-synthesis.md) — Round 5 overview and cross-links; Phase 12/13/14 detail
- [2026-08-11-bug-hunt.md](2026-08-11-bug-hunt.md) — Phase 12 detail; 9 confirmed, reproducible bugs
- [2026-08-11-api-contract-drift-audit.md](2026-08-11-api-contract-drift-audit.md) — Phase 13 detail
- [2026-08-11-i18n-mobile-crossbrowser-audit.md](2026-08-11-i18n-mobile-crossbrowser-audit.md) — Phase 14 detail
- [2026-08-11-user-observability-audit.md](2026-08-11-user-observability-audit.md) — evidence behind the Phase 9 diagnostic-timeline addition
