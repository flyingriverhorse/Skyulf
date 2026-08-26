# Frontend — Beyond the Pages: Remaining Opportunities

**Date:** 2026-08-26 · **Status:** Opportunity scan complete, prioritized, **nothing started**

## What this is

The other three reports in this folder cover pages, nodes, and the
Experiments/Inference IA. This one scans what they **did not touch**:
missing consumer features, unaudited surfaces, and codebase/approach
improvements. Quick-check facts gathered directly (grep/config reads) on
2026-08-26 are marked ✔.

---

## 1. Consumer features that don't exist yet (highest value)

### F1. Onboarding tour / coach marks

Every report in this folder found dead-ends and unexplained steps. A
5-step guided first run (add dataset → pick target → train → see score →
compare) fixes more activation problems than any individual page change.
Prereq: sample datasets (README.md §A.3 / growth A2.1).

### F2. Pipeline "lint" / pre-run checklist

Validation issues, leakage warnings ("Move X after the split so it only
fits on training data", `useGraphStore.ts:287`), and stale-column chips
already exist but are scattered across red dots, amber chips, and a
Results-panel Issues tab that never auto-opens (canvas-node-journey §1.5).

**Change:** one pre-run checklist panel — "3 things to fix before you
run" — each item links to the offending node. This is the natural home
for the always-visible disabled Run button (README.md §B.5).

### F3. Time/cost estimates before Run

`/slow-nodes` already aggregates per-step runtimes from completed jobs.
Surface "this pipeline took ~40s last time" on the Run button and in the
job drawer before submission.

### F4. Natural-language assist

"Why did my run fail?" and "suggest a next node" — the Recommendations
Panel is rule-based today (`RecommendationsPanel.tsx`). An LLM layer over
job logs + schema would be a differentiator for non-experts. Larger
investment; park until F1/F2 prove activation lifts.

### F5. Report export

Consumers need to show stakeholders results: one-click PDF/HTML export of
Dashboard/Experiments ("model 93% accurate, drift stable"). Backend work
required; frontend mostly composition.

### F6. Make Undo visible ✔

`useGraphStore` already has undo support (store + tests reference it),
but nothing surfaces it. **Change:** Undo button in the toolbar +
action toasts with an "Undo" action ("Node deleted — Undo").

---

## 2. Unaudited surfaces (candidate deep-dives)

### S1. Mobile canvas (needs a decision)

Layout has a mobile drawer (`Layout.tsx`), but the canvas components have
**zero touch handling** (grep for touch handlers in
`components/canvas/` returned nothing) ✔. react-flow on touch is where
mobile users bounce. Decision needed: **mobile = monitoring-only**
(Dashboard/Jobs/Deployments responsive, canvas desktop-first) or invest
in touch pan/zoom/connect.

### S2. Performance at scale (needs measurement)

`App.tsx` uses 6 lazy() imports and vite `manualChunks` exists
(`vite.config.ts:23`) ✔, but nothing has been measured: bundle size,
canvas render with 50+ nodes, Experiments chart mount (three requests +
recharts), plotly load (heavy dependency, `core/plotly.ts`). A profiling
pass turns guesses into a backlog. `core/perf/perfThresholds.ts` exists ✔
— check whether it's enforced or aspirational.

### S3. Accessibility beyond shared components

The shared-layer audit found strong modal focus handling but small gaps
(InfoTooltip name, StatusBadge title-only). Page-level keyboard flows
(tables, canvas, nested modals) were never audited. An a11y Playwright
spec exists ✔ (`e2e/a11y.spec.ts`) — extend rather than start from zero.

### S4. Testing depth (needs the contract-test habit)

e2e coverage is 4 specs (a11y, preview, routes, smoke) ✔. The
`error:` vs `message:` validation bug class
(canvas-node-journey §2.5) is exactly what contract tests catch: every
node's `validate()` output shape vs what `CustomNodeWrapper` reads.

### S5. Realtime consistency ✔

`core/realtime/jobEventsSocket.ts` exists (websocket for job events), yet
`Layout.tsx:75-87` still polls the error count every 5 minutes and
Drift status once. Pick one pattern: socket for job lifecycle, polling
only where sockets don't reach.

---

## 3. Codebase / approach

### C1. Component gallery / Storybook

The design-system report found 4 button styles and 3 focus-ring dialects
(README.md §E.21-22). Without a visual source of truth these regenerate.
Storybook for `components/ui` + `components/shared` would make the
consolidation stick.

### C2. Funnel instrumentation

The growth initiative measures clones vs views; activation can't be
measured without in-app events: first dataset uploaded, first run
attempted, first trained model, first comparison. Small, high-leverage.

### C3. One API error convention

Pages each invent error copy ("Failed to load dashboard data. Please try
again.", `Dashboard.tsx:100`). A single `useApi` hook (or apiClient
interceptor) with retry + plain-language mapping kills a whole bug class
and pairs with README.md §C.12 (plain-language errors).

---

## Priority ranking

| # | Item | Why first |
|---|---|---|
| 1 | F1 Onboarding tour (+ sample data) | Converts directly to activation; all four reports point at the same dead-ends |
| 2 | F2 Pipeline lint checklist | Same activation lever; mostly composition of existing signals |
| 3 | S1 Mobile decision | Blocks or unblocks a whole audience; decision first, work second |
| 4 | S2 Perf measurement | De-risks everything else; produces a backlog from numbers |
| 5 | F6 Undo visibility, C3 API convention | Cheap, kill recurring paper cuts |
| park | F4 NL assist, F5 export | High cost; revisit after activation data exists |

## Relation to other docs

- F2 realizes README.md §B.5; F1 pairs with README.md §A.3 and growth
  A2.1; S4 implements the lesson from
  [2026-08-26-canvas-node-journey.md](2026-08-26-canvas-node-journey.md)
  §2.5; C2 feeds the growth initiative's measurement stage.
