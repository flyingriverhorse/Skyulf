# Enterprise Redesign — Existing Pages

**Date:** 2026-08-11
**Status:** Design proposal (subagent-produced, text-based design brief —
not pixel mockups). Companion to
[2026-08-11-new-enterprise-pages.md](2026-08-11-new-enterprise-pages.md)
(brand-new pages needed for enterprise/multi-tenancy) and
[2026-08-11-technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md)
(the accessibility findings in §B3 there directly inform the "keyboard/a11y"
notes throughout this doc — any redesign work should fix both cosmetics and
a11y together, not sequentially).

## Goal & note on current page inventory

Goal: redesign toward best-in-class UX comparable to Databricks/Datadog/
Retool/Linear — clean, information-dense but uncluttered, fast, and
trustworthy-feeling for an enterprise evaluator running a trial.

**Correction to scope:** the redesign subagent that produced this doc
worked from a partial page list. A separate, independent frontend audit
(see technical-debt-deep-dive.md) enumerated the *actual* full route list
from `src/App.tsx:31-92`, which includes **several pages this document's
per-area proposals don't yet cover**: `Dashboard`, `Jobs` (routed, not just
the canvas drawer), `EDA`, `ErrorLogPage`, `SlowNodesPage`, and — notably —
**an `AuditLogPage.tsx` that already exists**, which changes the "new
pages" doc's Audit Log section from a from-scratch build to a redesign of
an existing page. This is flagged here rather than silently reconciled, so
follow-up work confirms current `AuditLogPage.tsx` capability before either
document is treated as final. The six areas below are exactly as designed
by the redesign subagent, unedited; treat `Dashboard`, `Jobs` (routed
version), `EDA`, `ErrorLogPage`, `SlowNodesPage`, and `AuditLogPage` as
**follow-up redesign work using the same method**, not yet covered here.

---

## 1. Pipeline Canvas — Effort: Large

**Current state.** React Flow editor with palette, properties, results,
toolbar, keyboard shortcuts, templates, autosave, and read-only mode
(`components/canvas/FlowCanvas.tsx`, `components/layout/MainLayout.tsx`).
Nodes surface execution/validation/schema/merge/performance signals
(`CustomNodeWrapper.tsx`). The toolbar is feature-rich but floats many
actions over the canvas (`components/layout/Toolbar.tsx`).

**Problems.** Controls split across sidebar, editor navbar, palette,
toolbar clusters, properties panel, and a jobs modal — no single place
answers "is this pipeline healthy and ready to run?" Status is fragmented
into tiny per-node chips. Empty state offers templates but no clear
"start from data" path. *(Also see technical-debt-deep-dive.md §B3: palette
entries are currently keyboard-inert `<div>`s — this must be fixed as part
of any redesign, not treated as separate work.)*

**Redesign.**
- Stable editor shell: top command bar (name, saved/dirty state,
  environment, Run) → left collapsible node library → center canvas →
  right inspector.
- **Pipeline health strip** below the command bar: validation issue count,
  last run outcome/duration, output dataset/model link, "View results."
- Run is the single primary action; export/history/templates/perf-overlay/
  destructive actions move to an overflow menu.
- Inspector tabs: Configure, Input/Output Schema, Run History, Diagnostics.
- Empty state: "Start with a dataset" / "Use template" / "Import
  pipeline." Loading = skeleton node cards. Failed run opens a persistent
  diagnostic panel with retry and jump-to-failing-node.
- **Accessibility requirement (non-negotiable per technical-debt-deep-dive.md
  §B3):** node-library entries must be real focusable buttons (drag as an
  enhancement, not the only path), with a keyboard "select source port →
  select target port" connection flow and labelled ports.

**Design system.** Reuse `Button`, `Input`, `Tooltip`, `FormField`; add
`AppCommandBar`, `EntityHealthStrip`, `SplitPane`, `StatusBadge`,
`InspectorTabs`, `EmptyState` variants.

## 2. Experiments / Run Comparison — Effort: Large

**Current state.** Lives inside the canvas sub-navigation rather than a
routed page; selectable-runs sidebar, comparison tabs, charts, evaluation,
pipeline diff, feature importance, SHAP, segmentation
(`components/pages/ExperimentsPage/`, `HeaderAndTabs.tsx`,
`JobListSidebar.tsx`).

**Problems.** Selection-based comparison hides the primary workflow
(identify the best run → understand why → promote/deploy). Tabs appear
conditionally, shifting navigation. Filters and selected runs can diverge.
"Deploy to production" is available with no explicit readiness checklist.

**Redesign.**
- Three-region "experiment workspace": header with dataset/task/time
  filters → left ranked run table with sticky selection → main comparison
  → optional right "decision" rail.
- Default view: ranked run table + winner recommendation (score, metric
  direction, dataset/version, duration, artifacts, status, promotion
  state).
- Stable analysis tabs (Overview, Compare, Evaluation, Explainability,
  Pipeline Diff) — unavailable tabs shown disabled with an explanation,
  never removed (avoids the current layout-shift problem).
- Decision rail: selected-run count, metric comparability, best candidate,
  missing artifacts, promotion/deployment actions.
- Empty states distinguish "no runs" / "no matching filters" / "no
  selection." Independent skeletons for run table vs charts; preserve
  prior comparison while refreshing.

**Design system.** Reuse `MetricDirectionBadge`, shared states,
`StatusBadge`; add `DataTable`, `FilterBar`, `SelectionSummary`,
`RankedMetricCell`, `ArtifactAvailability`, `DecisionPanel`.

## 3. Jobs Monitoring — Effort: Medium

**Current state.** Two divergent experiences today: a modal-style canvas
drawer with virtualized cards (`components/panels/JobsDrawer.tsx`,
`jobs/JobCard.tsx`, `JobDetailsView.tsx`) and a separate routed table
(`pages/Jobs.tsx`) that duplicates much of the same behavior — including a
**duplicated `StatusBadge` implementation local to `pages/Jobs.tsx`**
instead of the shared one (this exact duplication is also flagged in
technical-debt-deep-dive.md's design-system section — fix once, not
twice).

**Problems.** Same operational entity, two materially different UIs. A
centered 85vh modal is a poor always-available monitoring surface. Queue
position, retry eligibility, owner, and related pipeline/model context
aren't uniformly visible.

**Redesign.**
- Make routed `/jobs` the authoritative operational center; convert the
  canvas drawer into a right-side, non-blocking "recent activity" panel
  that links out to `/jobs`.
- `/jobs` layout: KPI row → persistent filter/search bar → dense sortable
  table → details side panel (not a page takeover).
- Surface status, submitted/started time, duration, progress, dataset,
  model, score, owner, source pipeline, retry/cancel actions in the row.
- Saved views: Active, Needs attention, Failed, Completed, Ingestion, EDA.
- Table skeleton rows for loading; explicit "No active jobs" empty state;
  inline retry-failure explanation; visible reconnect/polling status.

**Design system.** Consolidate on the *shared* `StatusBadge` (delete the
`pages/Jobs.tsx` local copy); add `JobProgress`, `OperationalTable`,
`DetailsDrawer`, `SavedViewTabs`, `EntityLink`.

## 4. Dataset / Data Management — Effort: Large

**Current state.** Routed table with upload, S3 source, ingestion
activity, preview, export, EDA, canvas, and pipeline-version actions
(`pages/DataSources.tsx`). Preview has sample/statistics tabs with separate
recovery states (`components/data/DatasetPreviewModal.tsx`); ingestion
activity is a separate modal (`IngestionJobsModal.tsx`). *(Also see
technical-debt-deep-dive.md §B2: this table has no virtualization, unlike
job history — fold that fix into this redesign rather than doing it
twice.)*

**Problems.** Five-plus equally-weighted row actions (Canvas, EDA, CSV,
Versions, Preview, Delete). Data treated as a flat list, not an asset with
ownership/lineage/quality/freshness. Preview and ingestion are disconnected
modals instead of a unified dataset detail context.

**Redesign.**
- Dataset catalog: searchable, **virtualized** table — name, source,
  freshness, schema, rows/columns, quality, ingestion status, last used,
  owner.
- Row click → detail page/drawer: Overview, Preview, Profile, Lineage,
  Pipeline Versions, Ingestion History.
- Header primary action "Add data" → source chooser (Upload, S3, future
  connectors — ties into the node-flexibility doc's connector
  recommendations); credentials in an advanced, security-explained
  section.
- Row overflow holds export/delete; prominent CTAs are "Explore," "Build
  pipeline," "View quality."
- Ingestion progress and actionable failures shown directly in row/detail,
  not a separate modal only.

**Design system.** Reuse `ModalShell`, shared states, `FormField`; add
`EntityHeader`, `DataQualitySummary`, `SourceTypeBadge`, `ActionMenu`,
`LineagePanel`, consistent dataset status badges, and reuse the existing
`VirtualList` component for the table.

## 5. Drift Monitoring — Effort: Large

**Current state.** Analysis-first flow: select reference job, upload
current data, tune thresholds, run analysis, then inspect summary
cards/schema drift/sortable feature rows/history/alert investigations
(`pages/DataDriftPage.tsx`, `pages/drift/DriftTable.tsx`,
`DriftAlertsHistoryTable.tsx`).

**Problems.** Reads like an ad hoc upload tool, not production monitoring
— no overview, alert queue, monitored-model inventory, or SLA/ownership
concept. "Run Analysis" requires a manual local upload every time. Status
severity, feature-risk, and disposition are scattered across different
surfaces.

**Redesign.**
- Two stable tabs: **Monitoring Overview** (monitored deployments, open
  alerts, drift trend, checks due, high-risk features) and **Analysis**
  (today's manual flow).
- Analysis header establishes scope explicitly: model/deployment, reference
  version, current-data source/window, check time, threshold policy, owner.
- Alert severity/unresolved count/affected features/business-risk features
  above the charts, not below.
- Feature table becomes a triage table: severity, feature, drift score,
  trend, importance, recommendation, assignee/disposition; row detail in a
  side panel, not inline expansion.
- Empty state offers "Set up monitoring" and "Run one-time analysis"
  distinctly. Loading preserves prior results with a "refreshing"
  indicator; error states distinguish missing baseline / upload failure /
  calculation failure.

**Design system.** Reuse charts/table primitives and shared states;
replace page-local drift badges with the shared `StatusBadge`'s semantic
variants; add `AlertSeverityBadge`, `MonitoringScopeHeader`,
`AlertTriageTable`, `ThresholdPolicyCard`, `TrendCard`.

## 6. Model Registry & Deployments — Effort: Large

**Current state.** Registry is a paginated model/version table with
client-side filters and modals (`pages/ModelRegistry.tsx`); Deployments is
a separate page with one active-deployment card + history table
(`components/pages/DeploymentsPage.tsx`).

**Problems.** Registry and deployment are separate IAs despite users
evaluating both together. Registry metrics collapse to a single inferred
value, losing metric context/split/direction. Deployment status is binary
active/inactive with no health/traffic/environment/rollback/approval/
monitoring/drift posture.

**Redesign.**
- Keep separate routes under one Model Lifecycle shell: Registry,
  Deployments, Governance/Lineage.
- Registry table: model, latest candidate, primary metric+split,
  dataset/version, stage, approval, active environments, drift/health,
  last updated — row detail drawer for versions/artifacts/lineage/compare/
  deploy.
- Deployment page: environment selector, active deployment health card,
  traffic/latency/error/drift summary, version lineage, rollback/redeploy
  controls, immutable history.
- Deployment action is a confirmation drawer with **preflight checks**:
  completed run, artifacts present, metric available, dataset lineage
  intact, approval granted, replacement impact, monitoring configured.
- Empty state links to completed experiments; independent loading
  skeletons for active-deployment card vs history table.

**Design system.** Reuse `RecordLink`, `ModalShell`, shared states; add
`LifecycleStageBadge`, `DeploymentHealthBadge`, `MetricSummary`,
`PreflightChecklist`, `VersionTimeline`, `LineageBreadcrumb`.

---

## Cross-Page Design System Recommendations

- **Navigation shell:** a persistent global sidebar exists
  (`components/Layout.tsx`), but Canvas introduces a second local
  navbar/shell (`MainLayout.tsx`, `Navbar.tsx`) that makes it feel like a
  separate app. Preserve the global shell everywhere; Canvas may add an
  editor-local command bar but should not replace the global one. Group
  navigation by **Build** (Data, Canvas, Experiments), **Operate** (Jobs,
  Registry, Deployments), **Observe** (Drift, Errors, Audit) — this
  grouping should also host the new admin/settings area from
  [2026-08-11-new-enterprise-pages.md](2026-08-11-new-enterprise-pages.md)
  under a fourth **Settings** group so the whole app feels like one product.
- **Shared states:** standardize existing `EmptyState`/`LoadingState`/
  `ErrorState` (`components/shared/`) with variants for first-use,
  filtered-empty, permission-error, and recoverable-failure; prefer
  skeletons over spinners and retain stale data during refresh everywhere.
- **Status semantics:** expand the *existing* shared `StatusBadge`
  (`components/shared/StatusBadge.tsx`) into a typed system (neutral,
  running, success, warning, danger, paused, deployed, draft, approved)
  and **delete every page-local badge reimplementation** — at minimum the
  one in `pages/Jobs.tsx` confirmed above, plus the drift-page badges.
- **Data density:** build one reusable `DataTable` (sticky header, sortable
  columns, density modes, overflow affordance, skeleton rows, empty/filter
  state, row-action overflow, optional detail drawer) to replace the
  divergent table implementations across Jobs, Registry, Data, Drift, and
  Deployments.
- **Theme:** dark mode exists globally, but components mix `gray`/`slate`/
  raw indigo-purple/gradient styles inconsistently — define one semantic
  token source (see technical-debt-deep-dive.md §B8, which found the exact
  same two-token-system problem independently) and require both themes for
  every shared component going forward.

## Sequencing Note

Do the **cross-page design-system items first** (shared `StatusBadge`,
`DataTable`, `EmptyState`/`LoadingState`/`ErrorState` variants, one token
source) — every one of the 6 page redesigns above depends on them, and
building page-by-page first would mean re-doing each page once the shared
components land. This mirrors the same "fix the shared thing once" pattern
used for the a11y and duplication fixes noted throughout this doc.
