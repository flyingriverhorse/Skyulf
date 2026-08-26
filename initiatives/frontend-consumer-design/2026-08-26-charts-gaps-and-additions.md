# Charts — Gaps, Additions, and Where Not to Add

**Date:** 2026-08-26 · **Status:** Spec ready, **nothing implemented**

## Context — where charts exist today

A grep for recharts/plotly usage (2026-08-26) shows the app is
**chart-rich but chart-misplaced**:

| Surface | Charts today |
|---|---|
| EDA | 8 components: DistributionChart, ThreeDScatterPlot, VariableCard/Row, DashboardTab, RuleDiscoveryTab, TargetAnalysisTab, TimeSeriesTab |
| Experiments | 12: Classification/Regression charts per split, MetricsComparisonChart, FeatureImportanceView, 5 SHAP views, SegmentationView |
| Jobs drawer | TuningTrialsChart (live boosting/trials, `components/panels/jobs/`) |
| Dashboard | Weekly Activity bar + Job Status pie (`pages/Dashboard.tsx`) |
| Drift | DriftHistoryChart + DriftTable |
| Slow Nodes | Runtime chart (log scale) |
| Error Log | Has charting |
| Inference | Some charting in the playground |
| **Model Registry** | **None** |
| **Canvas node bodies** | **None** (text bodyPreview only) |

## Principle

> A chart earns its place when it answers a **decision question at the
> point where the decision is made**. Count charts on dashboards and pie
> charts with many categories are decoration.

Each addition below closes a finding from the other reports in this folder.

---

## C1. Model Registry — per-model metric sparkline (HIGH, client-only)

**Decision it answers:** "Which model is best, and are we improving?"
(README.md §C.11 — the registry can't answer this today: no metrics
column, bare "0.9312" with no metric name, `ModelRegistry.tsx:155`).

- **Data:** already available — the versions modal receives versions with
  metrics per model (`ModelRegistry.tsx:437-442`); order by version
  number gives the time axis.
- **Design:** a small recharts `AreaChart`/`LineChart` (~120×32px) in
  each registry row, next to the model name: metric across versions,
  with the best point marked. Tooltip: "v3 · Accuracy 0.9312".
- **Must pair with:** named metric + direction indicator (reuse
  `MetricDirectionBadge` and `format.ts` descriptions — node-journey
  §3.4/N12). A sparkline of an unnamed number is worse than none.
- **Fallback:** models with a single version show the number, no chart;
  mixed metric families across versions (accuracy→F1) show a gap, not a
  fake line.
- **Effort:** ~1–2 days. No backend work.

## C2. Training progress on the node body (MED, medium)

**Decision it answers:** "Is my run alive, and how is it doing?"
(node-journey §7 — during runs the canvas looks frozen; status chips
appear only after completion).

- **Data:** `useJobStore` already receives job lifecycle events
  (`core/store/useJobStore.ts`); the live trials chart precedent proves
  per-iteration metrics arrive for tuning jobs
  (`components/panels/jobs/TuningTrialsChart.tsx`).
- **Design:** while a trainer node's job is running, its body shows a
  mini progress strip: elapsed time + iteration sparkline (when the
  backend emits per-iteration metrics) or a pulse with "epoch/iteration
  n" otherwise. On completion, the strip becomes the final score chip
  (pairs with N5 — score on the node).
- **Scope gate:** start with boosting/tuning jobs (events already exist);
  plain training gets "running · 12s" until the backend emits iterations.
- **Failure:** red strip + one-line reason (pairs with README.md §B.6 —
  never "check console").
- **Effort:** ~3–5 days frontend; iteration events for plain trainers is
  backend follow-up.

## C3. Dashboard — outcome charts replacing count charts (HIGH, client-only)

**Decision it answers:** "Is my ML work getting better, and is anything
unhealthy?" (README.md §C.19 — Dashboard shows infra counts, not
outcomes).

- **Data:** the Dashboard **already fetches and extracts per-job metrics**
  (`Dashboard.tsx:56-72`) — no new endpoint. Drift health is in
  `monitoringApi` (`core/api/monitoring.ts`: `has_drift`,
  `drifted_columns_count`, severities).
- **Add:**
  1. **Best metric over time** — line chart of best completed-job metric
     per day (last 30 days), one line per metric family present.
  2. **Drift health strip** — green/amber/red card with drifted-column
     count, linking to `/drift`.
- **Demote:** the Job Status pie is the weakest chart (few categories,
  counts) — replace with compact status counts inside the Weekly
  Activity bar (stacked bars) or keep as a small donut; do not add more
  pies.
- **Effort:** ~1–2 days.

## C4. Drift — verdict-first visuals (HIGH, two-phase)

**Decision it answers:** "Is my data changing, and how bad is it?"
(README.md §C.14 — results lead with 4-decimal PSI/KS numbers; the
per-column verdicts exist but hide behind row expansion).

**Phase A (client-only, ~2–3 days):** the drift API summary carries
scalars (`psi`, `wasserstein`, `ks_p_value`, `drifted` —
`core/api/monitoring.ts:52`) and thresholds are known
(`DataDriftPage.tsx:25`: psi 0.2, ks 0.05…). Render:
  1. A **verdict banner** (green/amber/red) above the table.
  2. A **horizontal PSI bar per column with a threshold marker** at 0.2
     instead of a bare 4-decimal number (bar = number made visual).
  3. Promote the existing per-column suggestions
     (`DriftTable.tsx:379-393`) into the visible row.

**Phase B (backend work much smaller than expected):** true distribution
overlays (reference vs current histogram per flagged column). The
backend-and-core review (2026-08-26) verified that **skyulf-core already
computes binned ref/current distributions**
(`skyulf-core/skyulf/profiling/drift.py:20-36,280-312`) and
`/monitoring/drift/calculate` + `/drift/alerts/{id}` already return them
(`monitoring/router.py:568,674`). Remaining gaps: categorical columns
return `distribution=None` (`drift.py:411` — add top-N category counts)
and the history-list `summary` exposes scalars only. Reuse EDA's
`DistributionChart` styling for consistency.

## C5. Experiments — metric-vs-time scatter for many runs (LOW, client-only)

**Decision it answers:** "Across all these runs, are we actually
improving?" (metrics comparison today is pairwise —
`MetricsComparisonChart`).

- **Data:** jobs already carry `created_at` + metrics in the Experiments
  store (`useJobStore` family).
- **Design:** optional view on the Experiments list header: scatter of
  metric vs time, one dot per completed run, colored by model family;
  hover = run details, click = select run. Only shown for ≥5 runs in the
  current filter (below that it's noise).
- **Effort:** ~1–2 days.

---

## Where NOT to add charts

| Surface | Why not |
|---|---|
| Jobs list page | Table + filters is the right tool; charts here duplicate Dashboard |
| Audit Log | Chronological text is the audit record; charting it adds noise |
| Data Sources | Small lists; a "rows over time" chart would be decoration |
| Error Log | Already charted; don't add more |
| Node settings panels | Settings are forms; charts inside forms distract |

Anti-patterns to enforce while implementing:

1. No new pie charts; max one donut app-wide.
2. Every chart gets a `useChartTheme` theme (dark/light), a
   `ResponsiveContainer`, and respects the global reduced-motion
   kill-switch (`index.css:96-105`).
3. Every numeric chart axis/tooltip names its metric and unit; scores get
   direction indicators (N12).
4. Empty data → EmptyState, never a chart of zeros.
5. Use recharts (existing convention); plotly stays confined to EDA's
   heavy visuals. No new chart libraries.

## Priority & effort summary

| # | Chart | Phase | Effort* | Backend |
|---|---|---|---|---|
| C1 | Registry sparklines | A | ~1–2 days | No |
| C3 | Dashboard outcome charts | A | ~1–2 days | No |
| C4-A | Drift verdict banner + PSI bars | A | ~2–3 days | No |
| C2 | Node-body training progress | B | ~3–5 days FE | Later |
| C5 | Experiments metric-vs-time | B | ~1–2 days | No |
| C4-B | Drift distribution overlays | C | ~2–3 days FE | Small (categorical bins + summary exposure) |

*\*Judgement estimates for sequencing only, per repo convention.*

**Suggested order:** C1 + C3 (both close HIGH findings with client-only
work) → C4-A → C2 → C5 → C4-B once the backend shape is agreed.

## Relation to other docs

- C1 realizes [README.md](README.md) §C.11; C3 §C.19; C4 §C.14;
  C2 realizes [2026-08-26-canvas-node-journey.md](2026-08-26-canvas-node-journey.md)
  §7 and pairs with N5 (score on node) and N12 (metric direction).
- C4-B's API shape should be agreed with the backend before the drift
  endpoint changes land; nothing else here touches the backend.
