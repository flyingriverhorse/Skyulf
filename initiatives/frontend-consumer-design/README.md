# Frontend Consumer Design Update

**Date:** 2026-08-26 · **Status:** Investigation complete, prioritized change list ready, **no fixes applied**

**Companion reports in this folder:**

- [2026-08-26-canvas-node-journey.md](2026-08-26-canvas-node-journey.md) —
  deep dive on nodes, connections, and configuration (dataset → trained
  model difficulty table, settings-panel placement analysis, changes
  N1–N23).
- [2026-08-26-experiments-inference-ia.md](2026-08-26-experiments-inference-ia.md) —
  should Experiments/Inference be standalone pages? Architecture facts,
  options, and the recommended hybrid design.
- [2026-08-26-beyond-pages-opportunities.md](2026-08-26-beyond-pages-opportunities.md) —
  what the other reports didn't cover: missing consumer features
  (onboarding tour, pipeline lint, export), unaudited surfaces (mobile,
  perf, a11y, testing, realtime), and codebase/approach items.
- [2026-08-26-charts-gaps-and-additions.md](2026-08-26-charts-gaps-and-additions.md) —
  chart inventory, five targeted additions (registry sparklines,
  dashboard outcomes, drift verdicts, node-body training progress,
  experiments trend), where NOT to add charts, and anti-patterns.

## What this is

A consumer-perspective UX/design review of the entire `frontend/ml-canvas`
app: every page, the canvas shell, and the shared UI layer. Method: four
parallel full-file reviews (canvas experience, data/model lifecycle pages,
ops/monitoring pages, shared design-system audit) plus a direct read of
Layout, Dashboard, and CanvasPage.

**One-line verdict:** the engineering is strong (deep links, focus traps,
reduced-motion support, consistent toasts) — the gap is that the product
speaks **operator language to consumers**: statuses without causes, metrics
without meaning, and several silent dead-ends.

**Methodology caveat:** findings are from code review, not from using the
running app in a browser. Visual/behavioral spot-checks are recommended
before executing the bigger items.

All file references are relative to `frontend/ml-canvas/src/` unless
otherwise noted.

---

## A. The guided journey (biggest wins, P0)

### 1. Every flow needs a visible "next step"

The journey breaks at three handoffs:

| Where | What happens | Evidence | Fix |
|---|---|---|---|
| Upload finishes | Panel just closes, no guidance | `pages/DataSources.tsx:151` | Success toast with "Open in Canvas" / "Explore in EDA" actions |
| Dataset preview modal | Ends in a lone "Close" button | `components/data/DatasetPreviewModal.tsx:145` | Add "Start in Canvas" CTA |
| Deployments empty state | Says "Deploy from the Experiments page" with no link | `pages/DeploymentsPage.tsx:263` | Link directly to `/canvas?view=experiments` |

Related: new rows show "Queued for ingestion" with all actions hidden until
complete (`pages/DataSources.tsx:183,446`) — the user is left waiting with
no explanation of what's happening or what comes after.

### 2. Fix the EDA dead-end (HIGH)

Visiting `/eda` with no dataset selected shows "No analysis found for this
dataset." (`pages/EDAPage.tsx:281`) and a Run Analysis button that silently
no-ops because `runAnalysis` returns early when no dataset is selected
(`pages/EDAPage.tsx:182`). No prompt to pick a dataset, no link to Data
Sources.

**Fix:** dataset picker first ("Pick a dataset to explore"), with a link to
upload if none exist. Also unify the target picker — the setup form uses a
free-text input (`pages/EDAPage.tsx:286`) while the toolbar uses a column
dropdown (`pages/EDAPage.tsx:545`).

### 3. Bundle a sample dataset + auto-open templates

No sample data exists anywhere, so a new user can't try the product without
their own CSV (the empty state "Upload a dataset to get started"
(`pages/DataSources.tsx:377`) is fine but cold-start is hard). Template
cards show only name/blurb/"N nodes · category" — a blind choice
(`components/canvas/TemplatesGalleryModal.tsx:64`).

**Fix:** ship 1–2 sample datasets; add mini pipeline diagrams to template
cards; for first-timers, open the gallery automatically instead of an empty
canvas. (Note: the growth plan's A2.1 "sample-dataset entry point" is the
same item — coordinate rather than duplicate.)

### 4. Brand basics (HIGH)

- Favicon 404s: `index.html:5` points to `/vite.svg` but no `public/`
  directory exists — still Vite's default logo.
- Tab title is "ML Canvas" (`index.html:7`) while the sidebar brand is
  "Skyulf ML" (`components/Layout.tsx:135`). Three names coexist.

**Fix:** one name, one favicon, one title.

---

## B. Canvas run experience (P0–P1)

### 5. Never hide the Run button (HIGH)

"Run Preview" only renders when all preconditions pass
(`components/canvas/Toolbar.tsx:595`, gated in
`useRunControls.ts:36-43`: bound dataset, outgoing edge, zero validation
issues). Mid-build, a novice sees no run button at all and no hint about
what's missing.

**Fix:** always render it, disabled, with an inline checklist of what's
missing (bind dataset, connect output, fix N warnings).

### 6. Kill "Check console for details" (HIGH)

The failure toast sends users to the browser console
(`useRunControls.ts:104`) even though the real error message is already
stored (`setLastRunError`).

**Fix:** put the actual error in the toast/panel.

### 7. Live progress during runs (MED)

Preview runs are synchronous: the only feedback is the button label
"Running..." (`Toolbar.tsx:611`). Nodes get status chips only after results
exist (`CustomNodeWrapper.tsx:269`) — no per-node running state, so long
runs look frozen. Meanwhile all edges animate constantly
(`FlowCanvas.tsx:357`), implying flow is "running" when idle.

**Fix:** per-node running state, stop idle edge animation, show elapsed
time.

### 8. Distinguish "wrong settings" from "run failed" (MED)

Both conditions render near-identical red: failed run = `border-red-500`;
invalid config = `border-red-500/40` plus a near-identical red circle chip
(`CustomNodeWrapper.tsx:239-294`).

**Fix:** amber for misconfigured, red only for failed runs.

### 9. Explain the dual run model (MED)

"Run Preview" vs "Run All Experiments" (`Toolbar.tsx:591,599`) is never
explained; preview silently excludes Data Preview nodes
(`useRunControls.ts:88-94`), and Run All opens Job History automatically.

**Fix:** rename to "Quick check" / "Run pipeline" with one-line tooltips.

### 10. Small canvas cuts (MED/LOW)

- Edge × deletes without confirmation, while nodes get a confirm dialog
  (`CustomEdge.tsx:182` vs `FlowCanvas.tsx:106`) — easy misclick while
  panning.
- Edge hover tooltip exposes raw IDs `{source} → {target}`
  (`CustomEdge.tsx:125`).
- Failure chip points to a nonexistent "Error Page"
  (`CustomNodeWrapper.tsx:275`).
- Unexplained jargon in node settings: "Random State" (default 42),
  "Stratify by Target" (`TrainTestSplitNode.tsx:127,158`), sklearn params
  in UI ("Center Data (with_mean)", "Scale Variance (with_std)",
  `ScalingNode.tsx:158,168`), math notation "Target: μ ≈ 0, σ ≈ 1"
  (`ScalingNode.tsx:64`).
- Properties panel shows raw node IDs (`PropertiesPanel.tsx:98`); Dataset
  schema note references an expansion that doesn't exist
  (`DatasetNode.tsx:127`).
- Keyboard shortcuts discoverable only via a toolbar icon or pressing `?`
  (`Toolbar.tsx:245`); empty state never mentions them
  (`FlowCanvas.tsx:401`).

---

## C. Results, failures & monitoring (P0)

### 11. Model Registry can't answer "which is best?" (HIGH)

Main table has no metrics column (`pages/ModelRegistry.tsx:281`); metrics
appear only inside the versions modal as a bare number — `formatMetrics`
prints "0.9312" without naming accuracy vs RMSE
(`pages/ModelRegistry.tsx:155`). No comparison, sorting, or "best"
indicator, despite the subtitle "deploy your best models"
(`pages/ModelRegistry.tsx:202`). Also: "Model Types" stat card actually
shows entry count (`:228`), and the empty state shows filter text even when
no models exist at all (`:297`).

**Fix:** best-metric column per model with named metric, sorting + a "Best"
badge; empty state points to training on the canvas.

### 12. Error Log reads like raw logs (HIGH)

Columns are Severity/Code/Type/Message/Target/When with HTTP status codes
and mono `error_type` (`pages/ErrorLogPage.tsx:257,966`); detail is a raw
Python traceback modal. Subtitle jargon: "In-house tracker — all unhandled
5xx and pipeline failures" (`:721`). Destructive "Clear pipeline" / "Clear
HTTP" delete buttons sit in the header (`:737,:747`).

Bright spot: `ErrorResourceLink` already jumps to the responsible
job/node/pipeline (`:122-161`).

**Fix:** lead with a plain-language cause + suggested fix per error; keep
the traceback behind an "Advanced" toggle; hide the header delete buttons
from consumers.

### 13. Jobs: surface failures (MED)

Failed rows show no reason and no in-list retry — the user must open
details (`pages/Jobs.tsx:500,544`). The status filter is hidden behind a
"Filters" toggle (`:393`), so "show me my failures" takes extra clicks.
Rows lead with truncated mono job IDs (`:480`).

**Fix:** failure summary + Retry on the row; quick "Failed only" chip;
promote dataset/pipeline name over the raw ID.

### 14. Data Drift: verdict first (HIGH)

- The concept is never explained anywhere — no subtitle on the title
  (`pages/DataDriftPage.tsx:98`); the empty state explains the mechanics
  but not what drift means or why to care (`pages/drift/EmptyState.tsx:10`).
- Expert-gated workflow: pick a reference job, upload CSV/Parquet, then
  face raw thresholds `{ psi: 0.2, ks: 0.05, wasserstein: 0.1, kl: 0.1 }`
  (`pages/DataDriftPage.tsx:25`).
- Results lead with 4-decimal raw numbers under "Wasserstein/PSI/KL
  Div/KS P-Value" (`pages/drift/DriftTable.tsx:327`).

Bright spot: per-column suggestions (`pages/drift/DriftTable.tsx:379`) are
the one real "what to do next" affordance — but hidden behind row
expansion.

**Fix:** one-sentence explainer ("Your live data is starting to look
different from what the model learned on"), default thresholds, a
green/amber/red verdict banner before the numbers, and promote the
suggestions out of row expansion.

### 15. Audit Log trust fix (HIGH)

Footer says "Filters are applied across the full history"
(`pages/AuditLogPage.tsx:326`) while elsewhere it says "Filters apply only
to the loaded page" (`:495`). Contradictory copy on an *audit* page is
trust-breaking. Actors render as `user #12` (`:167`); changed nodes are
mono ID chips with no canvas link (`:99`).

**Fix:** make one statement true, resolve real user names, link node IDs to
the canvas.

### 16. Deployments for consumers (HIGH)

No endpoint URL, health, latency, or test-prediction affordance — only a
mono "Artifact URI" (`pages/DeploymentsPage.tsx:250`) and lineage links
("Job abc12345 / Version n / Replaced deployment #n", `:24-57`).

**Fix:** show endpoint + status + a "Try a prediction" affordance.

### 17. Slow Nodes framing (MED)

Header is expert framing ("...spot the cheapest optimisation wins",
`pages/SlowNodesPage.tsx:198`); a stat card literally displays "Unit"
(`:270`); the empty state leaks internals ("Pre-existing jobs are silently
skipped — re-run a pipeline to seed this view.", `:471`).

---

## D. Information architecture (P1)

### 18. Split the sidebar: Build vs Monitor

11 flat nav items mix creation (Data, Canvas, Registry, Deployments) with
ops (Jobs, Error Log, Slow Nodes, Audit Log, Drift, EDA)
(`components/Layout.tsx:150-184`).

**Fix:** two groups with section headers; the ops group can visually
recede.

### 19. Dashboard = outcomes, not counts

"Total Jobs" / "Data Sources" / "Active Deployments" are infra stats
(`pages/Dashboard.tsx:140-164`). The consumer question is "what's my best
model, and is it still healthy?"

**Fix:** lead with best model + metric + drift health; demote counters.

### 20. Jargon sweep

- "Ingestion" → "import/upload" (`pages/DataSources.tsx:239,121`); retry
  only re-opens the form, it isn't a real retry (`:144`).
- "Add Source" is S3+AWS-keys only — engineer-facing
  (`components/data/AddSourceModal.tsx:14`).
- Dataset stats tab uses "Std Dev", "Missing Cells"
  (`DatasetPreviewModal.tsx:250`).
- EDA sidebar groups "Univariate/Multivariate/Structure & Causal" with
  "PCA & Clusters", "Causal Graph" tabs (`components/eda/EDASidebar.tsx:141`);
  "Draft Filters (n)" model assumes expert mental models (`:187`).
- Formatting supports CSV/Excel/Parquet is shown only after clicking Upload
  (`components/data/FileUpload.tsx:139`); max-size appears only via error.

---

## E. Design system foundations (P1–P2, do early — these unblock everything)

### 21. One primary button (HIGH)

Four competing styles today:

1. shadcn `Button` default `bg-primary` — resolves to near-**black**
   (`--primary: 222.2 47.4% 11.2%`, `index.css:30`), used in only 2 files.
2. `.btn-primary` sky→indigo→purple gradient (`styles/components.css:27`).
3. Inline copy of that gradient
   (`modules/nodes/inspection/DataPreviewComponents.tsx:219`).
4. Plain `bg-blue-600` (`components/shared/ConfirmDialog.tsx:120`).
5. Plus a blue→purple gradient on active nav (`components/Layout.tsx:234`).

**Fix:** pick one token, point `--primary` at the indigo, delete the rest.

### 22. One focus ring (HIGH)

Three dialects: `.focus-ring` = blue-500 (`index.css:112`); shadcn
primitives use `ring-ring` (near-black in light mode, `index.css:47`); and
global raw-CSS input/select focus is **teal** `#2dd4bf`
(`styles/components.css:141,176`) — matching nothing else in the palette.

**Fix:** single `.focus-ring` convention everywhere.

### 23. One gray, one radius (MED)

- `gray-*`: 1372 uses / 100 files vs `slate-*`: 494 / 55 files, mixed
  within single components (`pages/drift/EmptyState.tsx:7`,
  `components/shared/RecordLink.tsx:90`).
- Radii scattered: `rounded-md` 242×, `rounded-lg` 243× (and `rounded-lg`
  remapped to 0.5rem in `tailwind.config.js:55`), `.75rem` `.btn`,
  `rounded-xl` modals.
- Heading scale varies: `text-3xl` (Jobs) vs `text-2xl` (most) vs
  `text-xl` (Error Log).

**Fix:** standardize on slate + two radii + one heading scale. Mechanical
codemod, low risk.

### 24. Stop the double-styling war (MED)

Global element selectors restyle every raw input/select — including shadcn
`Input` — causing double borders/focus (`styles/components.css:123-182`).

**Fix:** scope to a `.legacy` class or remove.

### 25. State components (MED)

- Duplicate `EmptyState` at `pages/drift/EmptyState.tsx` (card-styled,
  mixes gray/slate) instead of the shared one.
- Ad-hoc spinners: border spinners in `pages/ModelRegistry.tsx:396,549`,
  green/orange/red spinners in `pages/DataSources.tsx:479,505,529`,
  oversized Loader2 in `pages/EDAPage.tsx:343`.
- Missing retry on some `ErrorState` call sites, stranding users:
  `pages/Jobs.tsx:340`, `pages/ModelRegistry.tsx:238`.

**Fix:** enforce shared LoadingState/EmptyState/ErrorState + retry.

### 26. Typography (LOW)

No font stack defined anywhere — no `fontFamily` in tailwind config, no
`font-family` in CSS. Consistent only by accident.

**Fix:** set one family + explicit heading scale.

### 27. Status colors & a11y nits (LOW)

- Status semantics mostly good via `StatusBadge`; deviations: solid-500
  duplicate `StatusDot.tsx:8`, emerald alongside green in
  `MetricDirectionBadge.tsx:11`.
- `InfoTooltip` trigger has no accessible name (`InfoTooltip.tsx:22`).
- `StatusBadge` relies on `title` only (`StatusBadge.tsx:59`).
- Modal `aria-labelledby` falls back to undefined for non-string titles
  (`ModalShell.tsx:79`).
- Minor dark-mode gaps: `LoadingState.tsx:12` spinner, `ErrorState.tsx:42`
  icon.
- Canvas uses theme tokens (`bg-card`, `hsl(var(--primary))`) but
  JobsDrawer/RecommendationsPanel hardcode `bg-white`/blue-500/gray-800
  (`JobsDrawer.tsx:179`, `RecommendationsPanel.tsx:21`) — two color
  systems on one screen.

---

## Execution order

### Wave 1 — quick wins (~days, low risk)

| # | Item |
|---|---|
| 4 | Brand: favicon + title/name unification |
| 6 | Replace "Check console for details" with the real error |
| 5 | Always-visible (disabled) Run button with missing-steps checklist |
| 1 | Next-step CTAs after upload / preview / deployments empty state |
| 15 | Audit Log filter-copy contradiction + actor names |
| 21–22 | Primary-button + focus-ring token consolidation |
| 23 | gray→slate codemod + radius standardization |

### Wave 2 — consumer clarity (~1–2 weeks)

| # | Item |
|---|---|
| 11 | Registry metrics column + Best badge |
| 12 | Plain-language Error Log (cause + fix, traceback behind toggle) |
| 13 | Jobs: failure reasons + Retry + "Failed only" chip |
| 8 | Amber-vs-red status semantics |
| 9 | Run-model rename/copy |
| 18 | Nav grouping (Build vs Monitor) |
| 20 | Jargon sweep (ingestion, Random State, with_mean…) |

### Wave 3 — experience upgrades (bigger, verify in browser first)

| # | Item |
|---|---|
| 3 | Sample datasets + template diagrams + auto-open gallery |
| 7 | Live run progress (per-node states, elapsed time) |
| 14 | Drift verdict-first UX + explainer |
| 16 | Deployments endpoint/health/test-prediction |
| 19 | Outcome-first dashboard |
| 2 | EDA dataset-picker entry |

## Relation to other initiatives

- **growth/** A2.1 (sample-dataset entry point) and A2.6 (upload-size text)
  overlap items 3 and part of 20 — coordinate, don't duplicate.
- **enterprise-readiness/** master fix list covers page redesigns at a
  phase level; this document is the consumer-framed, file-referenced
  version of the frontend slice.
