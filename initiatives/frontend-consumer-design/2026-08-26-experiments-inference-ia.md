# Experiments & Inference — IA Design Report

**Date:** 2026-08-26 · **Status:** Analysis complete, design proposed, **no code changed**

## The question

Should Experiments and Inference stay as views inside the canvas shell, or
become separate top-level pages? (Today they exist only inside the canvas:
an internal navbar in `components/layout/Navbar.tsx` switches
Canvas/Experiments/Inference, synced to `/canvas?view=...` in
`pages/CanvasPage.tsx:78-114`.)

## Part 1 — Facts about the current architecture

Two independent deep analyses produced these findings.

### 1.1 Both views are already global and loosely coupled

- **Experiments** fetches `GET /pipeline/jobs?limit=50&skip=0` with **no
  pipeline_id** (`components/pages/ExperimentsPage.tsx:136`,
  `core/api/jobs.ts:125`). Its only filters are client-side: task type,
  `job.dataset_id === selectedDatasetId`, `status === 'completed'`
  (`ExperimentsPage.tsx:291-292`).
- **Inference** reads the single globally active deployment
  (`deploymentApi.getActive()`, `InferencePage.tsx:868`). No pipeline
  reference anywhere.
- **Zero canvas coupling:** neither page (nor any of their components)
  imports `useGraphStore` or any node module. The only shell tie is
  `useViewStore` for mounting/visibility in `MainLayout`.
- **Verdict:** both could become standalone sidebar routes with **no
  data-fetching changes**. Extraction cost is state-preservation, not
  rewiring.

### 1.2 State is not shareable today

- Shell view is deep-linkable (`?view=`) and survives refresh/Back.
- Experiments' inner state (selected runs, filters, tabs, eval job) is
  memory-only — **lost on refresh** (`ExperimentsPage.tsx:42-100`).
- Inference persists to localStorage (inputs, results view, run history)
  — refresh-safe but not shareable (`InferencePage.tsx:46-50,795-864`).
- MainLayout keeps visited views mounted forever via display-toggling to
  preserve that memory state (`MainLayout.tsx:32-37,90-109`).

### 1.3 Nobody links to them

Repo-wide search found **zero** inbound links to `?view=experiments` or
`?view=inference`. Other pages link only to `/canvas` or
`/canvas?source_id=`. The Deployments empty state even says "Deploy a
model from the Experiments page" as plain text with no link
(`pages/DeploymentsPage.tsx:263`).

### 1.4 The surfaces around them heavily overlap

| Ranking | Overlap | Evidence |
|---|---|---|
| Worst | **Three surfaces render the same `/pipeline/jobs` pool** with copy-pasted task tabs/filters: Jobs page (`pages/Jobs.tsx`), canvas JobsDrawer (`components/layout/JobsDrawer.tsx`), Experiments sidebar (`ExperimentsPage.tsx`) | same `useJobStore` family |
| High | **Deploy action exists in 3 places**: Registry (`ModelRegistry.tsx:122`), DeploymentsPage, ExperimentsPage (`:181`) | same endpoint `/deployment/deploy/{jobId}` |
| Med | DeploymentsPage ↔ Registry versions modal (version/deploy status duplicated) | |
| Med | **Deactivate/Undeploy duplicated** against the same endpoint: DeploymentsPage (`:131`) vs InferencePage (`:1110`) | |
| Low | Registry ↔ Jobs (complementary: versions vs runs) | |

### 1.5 Consumer-journey scorecard (today)

| Question | Answered by | Experience |
|---|---|---|
| "All my runs, incl. failures" | Jobs page, drawer, Experiments sidebar — 3× | none shows all task types at once; must click tabs |
| "Compare two runs" | Experiments only | no link from Jobs/drawer — manual hop into canvas shell |
| "What model is serving?" | 3 sources of truth: DeploymentsPage card, Registry Active col, Inference header | |
| "Test a prediction" | Inference only | **no inbound link from Registry or Deployments** — you must already know it exists |
| "Re-run a failed job" | JobDetailsView Retry | fine (reachable from Jobs page + drawer) |

---

## Part 2 — Options

### Option A: Keep canvas-internal (status quo)

- Pros: zero work; flow continuity; keep-alive state preservation works.
- Cons: undiscoverable (evidence: zero inbound links anywhere, empty-state
  text references it by name only); cross-pipeline comparison feels
  accidental; Deployments/Registry have no path to "test a prediction".
- Verdict: the facts say this design already *behaves* global but is
  *hidden* local. Worst of both worlds.

### Option B: Fully standalone, remove from canvas

- Pros: one entry point, clean IA.
- Cons: loses in-flow access ("I just trained this — show me the runs")
  which is the canvas's strongest argument; breaks existing muscle memory
  and the `?view=` deep links that were just built (FND-006).
- Verdict: over-correction.

### Option C (recommended): Standalone pages + canvas as a scoped shortcut

Promote Experiments and Inference to first-class sidebar routes. Keep the
canvas tabs, but make them **deep links into the standalone pages with a
scope** (dataset filter) rather than internal views.

Why C, based on the facts:

1. **The code is already there and loosely coupled** — this is a routing
   + nav change, not a rewrite (§1.1).
2. **Their natural scope is global** — comparison across datasets/runs is
   the whole point of Experiments; Inference is deployment-global by
   definition (§1.1). Canvas-scoping them was always a framing accident.
3. **Discoverability is the measured gap** — zero inbound links, plus the
   plain-text reference in DeploymentsPage (§1.3).
4. **It fixes the journey gaps directly** — Jobs→Experiments compare link,
   Registry/Deployments→Inference "test a prediction" link (§1.5).
5. **Consistent with the app's own IA decision** — Jobs already chose
   standalone over canvas-internal; Experiments is Jobs' analysis twin.

Costs accepted: two entry points per surface (mitigated by the canvas tab
becoming a *link*, not a second implementation), and the keep-alive state
must migrate to the URL (§3.3).

---

## Part 3 — Design

### 3.1 Routes & navigation

| Route | Content | Sidebar group |
|---|---|---|
| `/experiments` | ExperimentsPage, global by default, filterable | **Build**: Data → Canvas → **Experiments** → Registry → Deployments |
| `/inference` (label: "Predictions") | InferencePage playground | Build (next to Deployments) |

- Add sidebar entries in `components/Layout.tsx` (also realizes the
  Build-vs-Monitor grouping from [README.md §18](README.md)).
- Keep `/canvas?view=...` deep links working (redirect or keep rendering)
  so existing bookmarks/history don't break.

### 3.2 Canvas tabs become scoped deep links

- Canvas navbar "Experiments" tab → navigates to
  `/experiments?dataset_id=<current dataset>` — the scoped view is just a
  pre-applied filter, which is exactly what Experiments already supports
  (`ExperimentsPage.tsx:291`).
- Canvas navbar "Inference" tab → navigates to `/inference` (there is no
  meaningful pipeline scope; deployment is global).
- Remove the `visitedViews` keep-alive machinery once the views move out
  (`MainLayout.tsx:32-37`).

### 3.3 State migration (the real work)

- **Experiments:** move selected-run IDs, task-type filter, dataset
  filter, and active tab into query params
  (`?runs=a,b&task=classification&dataset=x&tab=evaluation`). Refresh and
  sharing then work; keep-alive is no longer needed.
- **Inference:** keep localStorage (already refresh-safe); add a small
  URL hook for the deployment/job context if sharing becomes needed.
- `useJobStore` stays a global singleton — already shared by all three
  job surfaces, unaffected.

### 3.4 Dedup program (do alongside, since we're touching these files)

| # | Change | Evidence |
|---|---|---|
| D1 | Extract one shared **JobListPanel** (tabs, filters, auto-load) used by Jobs page, JobsDrawer, and Experiments sidebar | §1.4 worst overlap |
| D2 | One **DeployButton** component + single cache-invalidation path; deploy action stays in 3 places but shares implementation | §1.4 |
| D3 | Deactivate lives **only** on DeploymentsPage; Inference shows status + link | §1.4 |
| D4 | Add cross-links: Jobs→"Compare in Experiments", Registry/Deployments→"Test a prediction" (`/inference`), deploy toasts with links | §1.5 gaps |

### 3.5 Phasing

| Phase | Scope | Effort* |
|---|---|---|
| 1 | Add `/experiments` + `/inference` routes rendering the existing components; sidebar entries; canvas tabs still render internally (both entry points live). Zero logic changes. | ~1–2 days |
| 2 | URL-state migration for Experiments; canvas tabs become scoped deep links; remove keep-alive; D3 + D4 links | ~1 week |
| 3 | D1 shared JobListPanel + D2 DeployButton; sidebar Build/Monitor grouping polish | ~1 week |

*\*Judgement estimates for sequencing only, per repo convention.*

### 3.6 Risks

- **State loss on refresh mid-migration** (Phase 1–2 window) — acceptable,
  matches today's behavior.
- **Nav bloat** — sidebar grows by 2; offset by grouping (Build/Monitor)
  rather than hiding.
- **Bookmark breakage** — keep `/canvas?view=` handling as redirects for
  at least one release.
- **Dual entry confusion** — the canvas tab must visibly be the *same*
  page (same header, plus a "Filtered to this dataset" chip that can be
  cleared), not a second feature.

---

## Relation to other docs

- [README.md](README.md) §18 (nav grouping) and §19 (outcome dashboard)
  are realized here; §16 (Deployments for consumers) gains the
  "Try a prediction" link via D4.
- [2026-08-26-canvas-node-journey.md](2026-08-26-canvas-node-journey.md)
  N5 (score on the trained node) pairs with the new
  Jobs→Experiments compare link: the node shows the score, the link shows
  the comparison.
