# Frontend UX Roadmap

## Executive Summary

This synthesis groups the reviewed Foundations, Canvas, Data/EDA,
Experiments/Inference, and Operations evidence by the user outcome it changes.
It does not treat an inferred risk as a reproduced defect, and it does not let
one well-instrumented journey stand in for another. The **2026-08-07 audit
rerun** (see `## 2026-08-07 Audit Rerun`) repeated the engineering baseline and
refreshed every journey's evidence, reconciling all `37` current findings:
`5` New (`FND-007`, `DAT-008`, `DAT-009`, `EXP-008`, `OPS-008`), `6` Changed,
`26` Confirmed, and `0` Resolved. Much of the prior work moved from `Inferred`
to directly `Observed` against a live frontend and, for several Experiments and
Data/EDA findings, a real backend.

> **This document mixes two timelines.** Everything below records the
> `2026-08-07` audit as it was taken. A separate implementation pass has since
> resolved `28` of the `37` findings; that record lives in
> `## Historically Resolved Findings` at the end of this file. Audit sections
> are deliberately not edited in place so the audit stays reproducible.

The highest-value opportunities are:

1. **Let people complete core work on every supported screen size.**
   Eliminate clipped global, Data/EDA, and Canvas controls rather than asking
   users to discover hidden actions (`FND-001`, `DAT-004`, `CAN-005`).
2. **Keep the current task and its context visible across navigation.**
   Restore shell-view history, make retained experiment selections explicit,
   and establish the Operations context/link primitive before consumer pages
   adopt it (`FND-006`, `EXP-001`, `OPS-007`).
3. **Make failure, waiting, and recovery understandable at the point of work.**
   Give async changes usable semantics and turn Canvas, dataset-preview,
   ingestion, EDA, and inference failures into recoverable states
   (`FND-003`, `CAN-002`, `DAT-002`, `DAT-003`, `DAT-005`, `EXP-007`).
4. **Prevent invalid ML decisions before they are submitted.** Make source,
   Canvas, EDA, and inference inputs identify their purpose — including
   disambiguating look-alike dataset choices — and make typed inference
   validation and repairs reviewable (`DAT-001`, `FND-005`, `DAT-006`,
   `DAT-008`, `EXP-006`).
5. **Make the Canvas reliably operable while building a pipeline.** A newly
   added node should be selectable and configurable, and every visible toolbar
   action should receive its own click (`CAN-001`, `CAN-005`).
6. **Make analysis results attributable and interpretable.** Preserve analysis
   inputs through history and then add explicit applied context, accessible
   chart interpretation, and honest chart-export gating (`DAT-005`, `DAT-006`,
   `DAT-007`, `DAT-009`).
7. **Make model comparison and threshold choices defensible.** Expose selected
   runs, metric direction and missingness, keep run identifiers consistent
   across every comparison tab, then carry threshold provenance to prediction
   (`EXP-001`, `EXP-002`, `EXP-008`, `EXP-005`).
8. **Let operators investigate a record instead of copying an identifier, and
   trust the records they see.** Build the shared context contract first, then
   connect jobs, model lineage, drift, errors, performance, and audit history
   using only API-supported correlations, and fix the duplicate-row/colliding-
   key defect that undermines the Jobs table (`OPS-007`, `OPS-001`–`OPS-006`,
   `OPS-008`).

## 2026-08-07 Audit Rerun

### Delta Summary

This rerun repeats the complete engineering baseline and refreshes the route
and navigation inventory recorded in `## Method and Evidence` below. Task 1
only reestablishes the baseline; it does not yet reconcile individual
journey findings against this new evidence. Compared with the original
baseline captured in `## Method and Evidence`:

- `npm run lint`, `npx tsc --project tsconfig.json --noEmit`, `npm run build`,
  `npm run test -- --reporter=dot`, and `npm run size-check` all still exit
  `0` with results consistent with the original baseline (same warnings, same
  budget headroom, same `335` passing unit tests across `40` files).
- `npm run test:e2e -- --project=chromium` still exits `1` for the same
  root cause as the original baseline: the Chromium
  `chrome-headless-shell` executable is not installed in this environment
  (`npx playwright install` was not run, per audit scope). All `12` Chromium
  E2E tests fail before route interaction. Unlike the original baseline run,
  no `.superpowers/sdd/playwright.chrome.config.ts` fallback config or
  system-Chrome installation is present in this environment, so no
  alternate-browser measurement could be taken this rerun; this gap is
  recorded as a rerun limitation rather than fixed.
- The route and navigation baseline (lazy-loading, sidebar-collapse, alert
  badges, and `e2e/routes.spec.ts` coverage) is unchanged from the original
  baseline: the same 5 of 11 top-level routes (`/`, `/canvas`, `/jobs`,
  `/data`, `/eda`) are covered by the routes smoke spec, and `/drift`,
  `/registry`, `/deployments`, `/errors`, `/slow-nodes`, and `/audit` remain
  uncovered.
- No product source code, dependencies, or test files were modified while
  collecting this evidence.

### Current Engineering Baseline

- **Measured:** `npm run lint` exited `0` with no warnings or errors.
- **Measured:** `npx tsc --project tsconfig.json --noEmit` exited `0` with no
  reported diagnostics.
- **Measured:** `npm run build` exited `0` after Vite transformed `2996`
  modules and completed in `9.15s`. The output repeated the same two
  pre-existing warnings as the original baseline:
  `Circular chunk: vendor-flow -> vendor-charts -> vendor-flow` and
  `Generated an empty chunk: "vendor-react"`.

| Build chunk | Raw size | Gzip size |
|-------------|----------|-----------|
| `index.html` | `1.43 kB` | `0.71 kB` |
| `EDAPage-Dgihpmma.css` | `15.04 kB` | `6.38 kB` |
| `index-BbfmQ8_Q.css` | `133.01 kB` | `21.49 kB` |
| `vendor-react-l0sNRNKZ.js` | `0.00 kB` | `0.02 kB` |
| `DeploymentsPage-Cbf33fBc.js` | `7.36 kB` | `2.11 kB` |
| `AuditLogPage-D7Ib8q8L.js` | `9.66 kB` | `3.07 kB` |
| `SlowNodesPage-D9XNrHug.js` | `9.92 kB` | `3.16 kB` |
| `ModelRegistry-Dxe1jfMI.js` | `20.60 kB` | `5.17 kB` |
| `DataDriftPage-GYap9hgX.js` | `35.30 kB` | `9.91 kB` |
| `vendor-flow-CnABvqEr.js` | `167.70 kB` | `53.47 kB` |
| `vendor-utils-BhHEl1zz.js` | `215.21 kB` | `68.32 kB` |
| `EDAPage-B2QWV4Sw.js` | `295.82 kB` | `79.61 kB` |
| `vendor-charts--VuOzEp7.js` | `700.94 kB` | `216.58 kB` |
| `index-DEWG0WD0.js` | `1,005.54 kB` | `254.97 kB` |
| `vendor-plotly-B9LYHcu8.js` | `1,686.94 kB` | `537.39 kB` |

- **Measured:** `npm run test -- --reporter=dot` exited `0` with
  `Test Files 40 passed (40)` and `Tests 335 passed (335)` in `5.07s`. The
  output repeated the same pre-existing console noise as the original
  baseline: jsdom `localStorage` `ExperimentalWarning`s, `Unknown node type:
  StandardScaler` in `pipelineConverter.snapshot.test.ts`, and Recharts
  zero-size container warnings.
- **Measured:** `npm run test:e2e -- --project=chromium` exited `1`.
  Playwright attempted `12` Chromium tests across `e2e/a11y.spec.ts`,
  `e2e/preview.spec.ts`, `e2e/routes.spec.ts`, and `e2e/smoke.spec.ts`, and
  all `12` failed before route interaction because the Chromium
  `chrome-headless-shell` executable is missing at
  `/Users/BH7043/Library/Caches/ms-playwright/chromium_headless_shell-1217/chrome-headless-shell-mac-arm64/chrome-headless-shell`.
  The runner advised `npx playwright install`. Per audit scope, this failure
  is recorded as current baseline evidence and was not fixed. Unlike the
  original baseline, no `.superpowers/sdd/playwright.chrome.config.ts`
  fallback configuration or system Google Chrome installation exists in this
  environment, so no alternate-browser E2E measurement was possible this
  rerun.
- **Measured:** `npm run size-check` exited `0`; all checked chunks were
  within budget.

| Size-check target | Raw size | Gzip size | Budget | Result |
|-------------------|----------|-----------|--------|--------|
| `vendor-plotly` | `1647.4 KB` | `524.8 KB` | `750.0 KB` | `OK (70%)` |
| `vendor-charts` | `684.5 KB` | `211.5 KB` | `220.0 KB` | `OK (96%)` |
| `vendor-flow` | `163.8 KB` | `52.2 KB` | `80.0 KB` | `OK (65%)` |
| `vendor-react` | `0.0 KB` | `0.0 KB` | `70.0 KB` | `OK (0%)` |
| `vendor-utils` | `210.2 KB` | `66.7 KB` | `90.0 KB` | `OK (74%)` |
| `index (main)` | `982.0 KB` | `249.0 KB` | `260.0 KB` | `OK (96%)` |
| `route:EDA` | `288.9 KB` | `77.7 KB` | `140.0 KB` | `OK (56%)` |
| `route:DataDrift` | `34.5 KB` | `9.7 KB` | `20.0 KB` | `OK (48%)` |
| `route:ModelRegistry` | `20.1 KB` | `5.1 KB` | `15.0 KB` | `OK (34%)` |
| `route:Deployments` | `7.2 KB` | `2.1 KB` | `10.0 KB` | `OK (21%)` |

### Current Route and Navigation Baseline

| Route | Surface | Lazy-loaded | Sidebar collapsed | Alert badge | Covered by `e2e/routes.spec.ts` | Evidence |
|-------|---------|-------------|-------------------|-------------|----------------------------------|----------|
| `/` | Dashboard | No | No | No | Yes | `Inferred` |
| `/jobs` | Jobs | No | No | No | Yes | `Inferred` |
| `/data` | Data Sources | No | No | No | Yes | `Inferred` |
| `/eda` | EDA | Yes | Yes | No | Yes | `Inferred` |
| `/drift` | Data Drift | Yes | No | Yes | No | `Inferred` |
| `/canvas` | ML Canvas | No | Yes | No | Yes | `Inferred` |
| `/registry` | Model Registry | Yes | No | No | No | `Inferred` |
| `/deployments` | Deployments | Yes | No | No | No | `Inferred` |
| `/errors` | Error Log | No | No | Yes | No | `Inferred` |
| `/slow-nodes` | Slow Nodes | Yes | No | No | No | `Inferred` |
| `/audit` | Audit Log | Yes | No | No | No | `Inferred` |

- **Inferred:** `src/App.tsx` still lazy-loads `EDAPage`, `DataDriftPage`,
  `ModelRegistry`, `DeploymentsPage`, `SlowNodesPage`, and `AuditLogPage`
  behind per-route `Suspense` + `ErrorBoundary` wrappers (`LazyRoute`); `/`,
  `/jobs`, `/data`, and `/canvas` remain eagerly imported.
- **Inferred:** `src/components/Layout.tsx` still collapses the sidebar only
  on `/canvas` and `/eda` (`isCollapsed`), and still exposes alert badges
  only on `/drift` (`driftAlert`, from `monitoringApi.getDriftStatus()`) and
  `/errors` (`errorAlert`, polled every 5 minutes from
  `monitoringApi.getUnresolvedCount()` while the tab is visible and the user
  is not already on `/errors`).
- **Inferred:** `e2e/routes.spec.ts` still covers only `/`, `/canvas`,
  `/jobs`, `/data`, and `/eda`; `/drift`, `/registry`, `/deployments`,
  `/errors`, `/slow-nodes`, and `/audit` remain outside that smoke coverage
  file.
- **Inferred:** `playwright.config.ts` still targets a single `chromium`
  project against a Vite dev server on `http://127.0.0.1:5173`, with every
  backend call stubbed via `page.route()` rather than a live API.

### Finding Status Summary

| Status | Count | Meaning |
|--------|-------|---------|
| New | 5 | Not present in the previous roadmap. |
| Changed | 6 | Evidence, scope, priority, or proposed behavior materially changed. |
| Confirmed | 26 | Current evidence still supports the finding without material change. |
| Resolved | 0 | Current evidence demonstrates that the prior user problem no longer occurs. |

> **As-audited counts.** The table records the state of the `2026-08-07` audit
> and is intentionally frozen. A later implementation pass has since fixed `28`
> of these findings — see `## Historically Resolved Findings` for the list and
> `changelog/0.7.x.md` for per-fix evidence. The next audit rerun should recount
> from live evidence rather than adjusting these numbers in place.

These counts are the reconciled totals across all rerun tasks (Task 2 shared
foundations, Task 3 Canvas, Task 4 Data/EDA, Task 5 Experiments/Inference, and
Task 6 Operations) and cover all `37` current findings (`5 + 6 + 26 + 0 = 37`).
They match the required
`grep -c "**2026-08-07 status:** <Status>"` count for each status:

- **New (5):** `FND-007`, `DAT-008`, `DAT-009`, `EXP-008`, `OPS-008`.
- **Changed (6):** `FND-002`, `CAN-001`, `CAN-005`, `DAT-004`, `DAT-005`,
  `DAT-007`.
- **Confirmed (26):** `FND-001`, `FND-003`, `FND-004`, `FND-005`, `FND-006`,
  `CAN-002`–`CAN-004`, `DAT-001`–`DAT-003`, `DAT-006`, `EXP-001`–`EXP-007`, and
  `OPS-001`–`OPS-007`.
- **Resolved (0):** none this rerun. No prior finding's user problem was
  demonstrated resolved by current evidence; `## Historically Resolved
  Findings` below records this explicitly so a future rerun preserves any
  resolution rather than silently deleting a finding.

**Synthesis reconciliation (Task 7):** the Task 5 Experiments/Inference pass
originally labeled `EXP-001`, `EXP-005`, `EXP-006`, and `EXP-007` with a
non-vocabulary status string (`Observed (upgrade from Inferred)` /
`Observed (reconfirmed)`), which mixed the evidence-label vocabulary
(`Observed`/`Measured`/`Inferred`) with the rerun-status vocabulary and left
those four findings uncounted by the required status grep. This synthesis
normalized each to exactly one of the required `New`/`Changed`/`Confirmed`/
`Resolved` values, choosing `Confirmed` for all four — consistent with how the
sibling `EXP-002`/`EXP-003`/`EXP-004` evidence-upgrades and cross-references were
classified, since in every case the finding's user problem, scope, and proposed
behavior are unchanged and only the evidence strength moved from inferred to
observed. The evidence/delta prose for each still records the upgrade nuance.

### Final Validation (2026-08-07)

Task 8 re-ran the complete brief-required validation against the corrected
roadmap (post Task 7 review-correction commit) from `frontend/ml-canvas/`.
A review of that Task 8 pass found that its recorded baseline numbers had
been carried over from an earlier, uncommitted evidence pass rather than
physically re-executed at commit time. This subsection now reflects a
genuinely fresh baseline, with every command below re-executed after the
prior final-validation commit (`docs: finalize frontend UX audit rerun`,
`6f8d594a`) and its review, not reused from that earlier pass:

- **Measured:** the brief's placeholder/vague-language scan (checking for
  incomplete-language markers and hedging phrases) against
  `docs/ux/frontend-ux-roadmap.md` — **pass, exit `0`, zero matches**.
- **Measured:** finding-ID coverage grep
  (`FND-[0-9]{3}|CAN-[0-9]{3}|DAT-[0-9]{3}|EXP-[0-9]{3}|OPS-[0-9]{3}`) —
  **37 distinct IDs, complete coverage**: every ID has a detailed finding
  block, a `## Prioritized Findings Inventory` row, a `## Now / Next / Later
  Roadmap` bullet, and a `## Validation Matrix` row (37/37 each, 0
  milestone/dependency-ordering mismatches).
- **Measured:** full frontend baseline rerun — `npm run lint` **pass**
  (exit `0`, no warnings); `npx tsc --project tsconfig.json --noEmit`
  **pass** (exit `0`, no diagnostics); `npm run build` **pass** (exit `0`,
  `2996` modules transformed, `14` emitted asset chunks, repeating the same
  `Circular chunk: vendor-flow -> vendor-charts -> vendor-flow` and
  `Generated an empty chunk: "vendor-react"` warnings) — this fresh build's
  own chunk output was diffed against the Task 1 baseline table recorded
  above and every filename (content hash), raw size, and gzip size matched
  exactly, so the manifest is confirmed byte-identical by this rerun itself,
  not merely asserted from a prior pass; `npm run test -- --reporter=dot`
  **pass** (exit `0`, `335/335` tests across `40/40` files); `npm run
  size-check` **pass** (exit `0`, all chunks within budget, sizes matching
  the Task 1 table above exactly, likewise freshly diffed rather than
  assumed).
- **Measured:** `npm run test:e2e -- --project=chromium` — **all 12 tests
  fail** (exit `1`), every failure for the identical cause already recorded
  in the Task 1 baseline: the Chromium `chrome-headless-shell` executable for
  revision `1217` is missing from
  `~/Library/Caches/ms-playwright/` in this environment, and Playwright's own
  output advises running `npx playwright install`. This is an
  environment/tooling gap (missing browser binary), not a product or roadmap
  defect, and is **not** claimed as passing; it is recorded as an
  unresolved, out-of-scope environment limitation per audit rules (no
  product/environment fix applied).
- **Measured:** zero baseline drift — this fresh rerun produced the same
  exit code, counts, and warnings as the Task 1 baseline for every command
  above, and the build/size-check outputs were directly compared
  line-for-line against the Task 1 table and found byte-for-byte identical;
  no regression or improvement occurred between Task 1 and this fresh final
  validation pass.
- **Measured:** repository scope — `git diff --check` exits `0` (no
  whitespace/conflict-marker errors); `git status --short` shows only the
  pre-existing, controller-owned `.superpowers/sdd/progress.md`
  modification; every audit/design/plan commit in `merge-base..HEAD`
  modifies only its own document, and the roadmap-only commit scope is
  preserved. No product code, dependency, completed-plan cleanup, or
  progress-ledger change was made or is claimed by this validation.

**Conclusion:** this fresh rerun is validated as internally consistent and
implementation-ready, and corrects the prior report's gap by physically
re-executing every baseline command rather than reusing earlier numbers.
The only outstanding item is the pre-existing, environment-caused Chromium
E2E gap above, which remains explicitly unresolved and undisguised in this
record.

### Task 2 — Shared Foundations Rerun

#### Shared-state usage reinspection

- **Measured:** From `frontend/ml-canvas/`,
  `grep -RInE "LoadingState|EmptyState|ErrorState|PageSkeleton|toast\.|disabled=" src --include="*.ts" --include="*.tsx"`
  returned `256` matches. `LoadingState`/`EmptyState`/`ErrorState` are used
  consistently for first-load, empty, and failure states in Dashboard, Data
  Sources, Jobs, Model Registry, Deployments, EDA, Error Log, Evaluation, and
  Segmentation, with the same neither-`role`-nor-`aria-live` composition as the
  original baseline: `LoadingState.tsx`, `EmptyState.tsx`, and `ErrorState.tsx`
  still contain no `role=` or `aria-live` attribute. `PageSkeleton` is still
  used only once, as the route-level `Suspense` fallback in `App.tsx`
  (`RouteFallback`). `toast.` call sites (`success`/`error`/`info`/`warning`)
  now number `80` across the app (`grep -RIn "toast\.\(success\|error\|info\|warning\)"`),
  consistent with `toast.ts`'s own comment estimating "~40 call sites" as an
  approximation, not a hard count; this is not a material change to `FND-003`
  or `FND-004`, which are about shared-semantics/retry-consistency gaps, not
  call-site volume. `disabled=` appears `85` times, unchanged in kind from the
  original evidence (used for in-flight/unavailable actions across the same
  surfaces).
- **Inferred:** Re-reading `Dashboard.tsx`, `EDAPage.tsx`, and
  `DeploymentsPage.tsx` confirms they still pass `onRetry` to `ErrorState`,
  while `ModelRegistry.tsx` (`ErrorState error={error}` with no `onRetry`,
  line 158) and the Experiments `EvaluationView.tsx`/`SegmentationView.tsx`
  (`ErrorState error={evalError}` with no `onRetry`) still render it without
  one. This is the same asymmetry as the original `FND-004` evidence, with no
  material change.

#### Shared live walkthroughs (1440 / 1024 / 768 / 390)

- **Method:** Started the Vite dev server used by the project's own Playwright
  config (`npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`) and
  drove a live Chromium browser against it with the Playwright MCP browser
  tool (`playwright-browser_navigate`, `_resize`, `_snapshot`, `_evaluate`,
  `_click`, `_press_key`, `_take_screenshot`). This is a real, interactive
  browser session against the actual running frontend, not a static reading
  of source — every measurement below is **Observed** unless explicitly
  marked otherwise. Widths exercised: `1440×900`, `1024×900`, `768×1024`, and
  `390×844` (portrait), matching the required breakpoints.
- **Observed — 1440/1024/768/390, `/canvas`:** The Canvas view switcher
  (`[data-testid="navbar-views"]`) measured `352.56px` wide at every width.
  At 390 px, the switcher spans `x=50.72–403.28` while the main content pane
  spans `x=64–390` (`326px` wide) and the viewport is `390px` wide — the
  switcher's right edge (`403.28`) extends `13.28px` beyond both the main
  pane and the viewport. In the same layout, the Inference button
  (`x=289.14–399.28`) overlaps the Read-only toggle (`x=298–334`) and the
  Notifications bell (`x=342–378`), both of which sit inside the switcher's
  overrun span. This reproduces and strengthens the existing `FND-001`
  evidence for Canvas's own view-switcher clipping risk (the original
  finding cites a `353px`-wide switcher inside a `326px` pane); the
  Dashboard shell clipping below reproduces the same underlying
  compact-shell defect on a different top-level route, so `FND-001` remains
  Confirmed with no material change.
- **Observed — 390 px, `/` (Dashboard):** The persistent sidebar (`<aside>`)
  measured `256px` wide with the main content pane immediately following at
  `x=256`, `width=134px`, and no document horizontal overflow
  (`scrollWidth === 390`). This reproduces the original `FND-001` evidence
  exactly: the fixed 256 px sidebar leaves only 134 px for Dashboard's main
  content at 390 px.
- **Observed — `FND-002` overlay focus containment, 1440 px, `/canvas`:**
  Opened the Shortcuts overlay (`?`/click); the next Tab moved focus to the
  covered "Open command palette" navbar button, outside the dialog
  (`role="dialog" aria-label="Keyboard shortcuts"`), reproducing the original
  evidence exactly. Opened Command Palette (Ctrl/Cmd+K) and focused its last
  in-dialog focusable element; the next Tab moved focus to the "Open Tanstack
  query devtools" button behind the (still-open) dialog — this is new,
  concrete **Observed** evidence for Command Palette, which the original
  audit only **Inferred**. Seeded one notification via `localStorage`
  (`skyulf-notifications`) and opened its detail modal
  (`role="presentation"` backdrop, no `role="dialog"`); focus was never
  moved into the modal on open (`document.activeElement` remained on
  `<body>`), and the next Tab moved focus to the sidebar's "Collapse
  sidebar" button, escaping the modal entirely — this is new, concrete
  **Observed** evidence for the notification detail modal, which the
  original audit only **Inferred**.
- **Observed — `FND-006` shell-view history, 1440 px, `/canvas`:** Clicking
  "Experiments" in the shared view switcher left the URL at `/canvas` (no
  navigation, no history entry) and none of the three switcher buttons
  exposed `aria-current`, `aria-selected`, or `aria-pressed`. Navigating back
  (`browser_navigate_back`) returned to `/` (the route visited before
  `/canvas`), leaving the shell entirely rather than returning to the prior
  shell view. This reproduces the original `Observed`/`Inferred` evidence
  exactly, now confirmed for the Canvas→Experiments transition with an
  explicit `aria-*` state check.
- **New — Observed: `NotificationCenter` nests an interactive Dismiss button
  inside another interactive row button.** With a seeded notification, the
  browser logged a live React `validateDOMNesting(...)` warning
  ("`<button>` cannot appear as a descendant of `<button>`") at
  `NotificationCenter.tsx:274`. The accessibility snapshot's computed
  accessible name for the row button concatenated the nested Dismiss
  button's text into the row's own name (`"error Encoding 12:02 PM Test
  warning message node-1 Click to see full details Dismiss"`). Tabbing from
  the row button moved focus onto the nested `aria-label="Dismiss"` button,
  confirming the invalid button-in-button nesting is live and
  keyboard-reachable, not just a static markup smell. Recorded as `FND-007`
  below.
- **Inferred (not reproduced this rerun):** Inference's own shell-view
  focus/history/selected-state behavior, EDA/Data/Operations route-level
  compact-navigation behavior at 390 px beyond the already-Observed
  Dashboard/Canvas cases, and full keyboard-only completion of the Canvas
  node forms, Add Source modal, EDA setup/filter controls, and the Inference
  editor were not independently re-driven this rerun; their standing
  evidence is the source-level review recorded in `FND-005` and the
  Reconciliation notes below, unchanged from the original audit.

#### Accessibility automation rerun

- **Measured:** `cd frontend/ml-canvas && npm run test:e2e -- e2e/a11y.spec.ts --project=chromium`
  exited `1`. All `4` tests (`dashboard (/)`, `canvas (/canvas)`, `data
  (/data)`, `eda (/eda)`) failed identically before any route loaded or axe
  ran: `browserType.launch: Executable doesn't exist at
  /Users/BH7043/Library/Caches/ms-playwright/chromium_headless_shell-1217/chrome-headless-shell-mac-arm64/chrome-headless-shell`,
  the same missing-Chromium root cause recorded in the Task 1 rerun baseline.
  Because no browser launched, axe-core never ran and produced zero
  critical or non-blocking (`serious`) findings this rerun — there is no
  current automated evidence, positive or negative, for the dashboard
  `color-contrast` or canvas `scrollable-region-focusable` findings the
  original audit recorded from a working system-Chrome run. This gap is
  recorded as-is, per audit scope; the executable was not installed and no
  product or test code was modified to make it pass.
- **Inferred (environment note, not a code finding):** `node_modules/playwright-core/browsers.json`
  in this checkout still pins Chromium headless-shell revision `1217`, while
  `~/Library/Caches/ms-playwright/` on this machine only has revision `1228`
  installed; the two do not match, so no locally cached browser satisfies
  this project's pinned revision without running `npx playwright install`.
  A system Google Chrome installation exists at `/Applications/Google
  Chrome.app` (`151.0.7922.76`), but the project's `playwright.config.ts`
  targets the default Chromium channel, not `channel: 'chrome'`, and no
  `.superpowers/sdd/playwright.chrome.config.ts` fallback (used by the
  original audit) exists in this checkout, matching the Task 1 rerun's
  documented limitation.

### Task 3 — Canvas Rerun

#### Method

- **Method:** Started the same project Vite dev server used by Task 1/2
  (`npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`) and drove a
  live Chromium session against `/canvas` with the Playwright MCP browser
  tool (`playwright-browser_navigate`, `_resize`, `_snapshot`, `_evaluate`,
  `_click`, `_press_key`, `_find`, `_take_screenshot`, plus
  `_run_code_unsafe` for CDP-level `page.mouse` sequences). Every measurement
  below is **Observed** in this live session unless marked otherwise. Widths
  exercised: `1440×900`, `1024×900`, `768×1024`, and `390×844` (portrait),
  matching the required breakpoints. Scenarios exercised: click-to-add
  pipeline creation (Dataset, Encoding, Feature Generation), the click-to-add
  overlap/pointer-interception reproduction, the toolbar-cluster collision at
  1440 px, keyboard Undo/Redo (`⌘Z`/`⌘⇧Z`) against a real click-added graph,
  the autosave Restore/Discard banner, the Keyboard Shortcuts sheet at all
  four widths, per-width toolbar/control-panel geometry, and Feature
  Generation's validation chip and Apply-recommendation surface. Two
  screenshots were captured to `.superpowers/sdd/` (git-ignored, not
  committed): the Dataset/Encoding overlap and the toolbar collision.
- **Evidence limit (unchanged from the original audit):** The local
  environment's only usable dataset source still returns
  `500 Internal Server Error` on `GET /api/pipeline/datasets/{id}/schema`
  ("No schema available."), so a real column schema, a completed
  Dataset → Encoding drag/handle connection, a real invalid-run submission,
  and a real Feature Generation recommendation payload remain unavailable
  live, exactly as the original audit's "Canvas audit evidence and limits"
  section already recorded. This rerun did not claim a completed edge
  connection for the same reason the original audit did not: React Flow
  handle-to-handle drag requires a continuous native pointer gesture that
  this environment's automation could not reliably reproduce (both raw
  `PointerEvent` dispatch and CDP-level `page.mouse` down/move/up sequences
  on the exact handle coordinates left `.react-flow__edge` count at `0`);
  this is recorded as a test-automation limitation, not a product regression,
  since the underlying `onConnect` wiring in `FlowCanvas.tsx` is unchanged
  from the original evidence and library defaults (`connectOnClick: true`)
  are unchanged in `node_modules/@xyflow/react`.
- **Discarded test artifact (not reported as a finding):** An earlier attempt
  to build the pipeline using synthetically dispatched HTML5 `DragEvent`s
  (bypassing Playwright's actionability checks) left the undo/redo history
  in an inconsistent state (a single `⌘Z` removed two nodes at once, and
  `⌘⇧Z` did not restore them). Repeating the same sequence with only real
  palette clicks — the method the original audit used — showed correct
  one-step Undo/Redo behavior (`1 node → 0 → 1`), confirming the earlier
  anomaly was an artifact of the synthetic-dispatch method bypassing the
  store's normal history middleware, not a live product defect. No finding
  was created for it.

#### Pipeline creation walkthrough (1440 px)

- **Observed:** Clicking the Dataset palette card inserted a new node that
  was **not** selected (no Properties panel opened); a second click on the
  same card selected it and opened Properties, exactly reproducing the
  original `CAN-001` evidence. Source re-reading confirms `addNode()` in
  `useGraphStore.ts` never sets `selected: true` on a newly inserted node —
  unlike `duplicateSelectedNodes`, which does — and this applies to both the
  click-to-add and native drag-and-drop insertion paths, since both call the
  same `addNode()` action. This slightly broadens the original finding's
  described root cause (click-to-add only) without changing its user-facing
  problem or proposed fix.
  - Selecting a local "test source" from the Dataset dropdown reproduced the
    documented `500` schema error and "No schema available." message — the
    known, unchanged evidence limit above.
  - Clicking the Encoding palette card placed a new card only ~60 px from the
    Dataset card, mostly underneath it, reproducing `CAN-005`'s companion
    evidence for `CAN-001`. Attempting to click the Encoding card's heading
    directly (`playwright-browser_click`) failed after a 5 s actionability
    timeout with an explicit Playwright log confirming the Dataset node's
    subtree "intercepts pointer events" over the Encoding node at the same
    screen coordinates — a live, reproducible pointer-interception failure
    (screenshot: `.superpowers/sdd/task-3-1440-overlap.png`), not just the
    original's static geometry inference. `CAN-001` remains **Changed**
    below with this stronger, directly observed evidence.
  - Adding a Feature Generation node the same way reproduced the same
    unselected-on-first-click behavior and rendered a
    `Configuration issue: Add at least one operation.` chip; its Properties
    panel showed empty `Add:` operation-type buttons and no `Apply`
    recommendation control, matching the original "recommendations are
    conditional" / local-schema-limited evidence for `CAN-004` — no material
    change.
- **Observed — toolbar collision, 1440 px:** Clicking "Clear canvas" in the
  Toolbar (`playwright-browser_click`) failed after a 5 s actionability
  timeout because the right-cluster "Templates" button intercepted the
  pointer event, not just "Undo" as the original finding stated.
  `getBoundingClientRect()` measured Templates at `x=492.3–610.5`; Undo
  (`x=528–568`) sits fully inside that span, and Clear canvas
  (`x=576–616`) overlaps it by ~34 px (`x=576–610`) — both left-cluster
  destructive/undo actions are occluded, not one
  (screenshot: `.superpowers/sdd/task-3-1440-toolbar-collision.png`).
  `CAN-005` remains **Changed** below with this broadened affected-surface
  evidence.
- **Observed — keyboard recovery still works despite the pointer occlusion:**
  With a single real click-added Dataset node, `⌘Z` removed it (`1 → 0`) and
  `⌘⇧Z` restored it (`0 → 1`), confirming Undo/Redo history is intact and
  keyboard-reachable even though the equivalent pointer controls are
  occluded at 1440 px — unchanged from the original evidence.

#### Diagnosis and recovery

- **Observed — autosave/restore banner:** Reloading `/canvas` with a
  previously saved local snapshot surfaced "Restore previous session?" with
  Restore, Discard, a relative timestamp, and a node count; Discard cleared
  the prompt. This is unchanged from the original evidence for `CAN-003`.
- **Observed — node removal confirmation:** Clicking a node's "Remove node"
  control opened a "Delete node?" confirmation dialog ("The selected node
  and any connected edges will be removed. You can undo with Ctrl+Z.") with
  Cancel/Delete actions; this safeguard is consistent, unrelated to any
  existing `CAN-*` finding, and is not reported as a new finding.
  Run Preview could not be exercised against a real invalid graph live
  because `canRunPreview` in `useRunControls.ts` (unchanged, confirmed by
  source re-reading) requires a Dataset with an outgoing edge, which the
  connection-gesture evidence limit above prevented from being created live;
  `CAN-002` therefore keeps its original evidence level (source-confirmed,
  not independently re-observed end-to-end) below.

#### Responsive and keyboard verification (1024 / 768 / 390 px)

- **Observed — 1024 px:** The right toolbar cluster collapsed into a single
  "More canvas tools" button, leaving only Undo (`528,72 40×40`) and Clear
  canvas (`576,72 40×40`) in the left cluster with no overlap between them
  and no other visible control at that row — the `CAN-005` collision is
  specific to the 1440 px two-cluster layout, unchanged from the original
  table.
- **Observed — 768 px:** Both side panels collapsed; "More canvas tools"
  (`710,72 42×34`) remained reachable and in-bounds; document
  `scrollWidth` equalled the 768 px viewport width. Unchanged from the
  original table.
- **Observed — 390 px:** Both panels stayed collapsed; the shell's
  "Inference" switcher button spans `x=289.14–399.28`, ending `9.28px`
  beyond the 390 px viewport — reproducing the exact `FND-001` shell-clipping
  evidence already reconciled under Task 2, cited here only because it
  appears on the Canvas route. The Keyboard Shortcuts sheet still listed
  Undo, Redo, Command palette, Run Preview, and Escape. No Canvas-owned
  clipping was observed at this width, unchanged from the original table.

#### Representative node-form recomparison

- **Observed:** Dataset (Select Dataset control, `500` schema error) and
  Feature Generation (empty-operations validation chip, no Apply button
  under the current evidence limit) were re-opened live and match the
  existing "Representative node-form comparison" table with no material
  change. The remaining six representatives (Encoding, Feature Selection,
  Training, Ensemble, Segmentation, Data Preview) were re-confirmed by
  source re-reading only, matching this rerun's evidence level for those
  forms; no row in that table required an update.

### Task 4 — Data and EDA Rerun

#### Method

- **Method:** Drove a live Chromium browser with the Playwright MCP browser
  tools (`browser_navigate`, `_resize`, `_evaluate`, `_click`, `_find`,
  `_take_screenshot`, `_tabs`) against an **already-running** local dev
  stack rather than a fresh Playwright-launched server: the project's own
  Vite dev server (`localhost:5173`) proxying to an already-running FastAPI
  backend (`localhost:8000`) via `vite.config.ts`'s `/api` and `/data/api`
  rules. Unlike the Task 2/3 reruns, which drove a Playwright-launched dev
  server with `page.route()` request mocks, every response below (including
  the reproduced failures) is a real backend response against real,
  accumulated seed/test data (dozens of pre-existing "test source"/"s3
  source" datasets, real jobs history). Findings are labelled **Observed
  (live, real backend)** where this applies, versus **Observed (source)**
  for claims confirmed by reading code only. Widths exercised: `1440×900`,
  `1024×768/900`, `768×1024`, and `390×844`, matching the required
  breakpoints. Source files re-read: `DataSources.tsx`, `EDAPage.tsx`,
  `useEDAStore.ts`, `core/api/datasets.ts`, `core/api/eda.ts`,
  `AddSourceModal.tsx`, `DatasetPreviewModal.tsx`, `IngestionJobsModal.tsx`,
  `PipelineVersionsModal.tsx`, `EDASidebar.tsx`, `CorrelationHeatmap.tsx`,
  `CanvasScatterPlot.tsx`, `ThreeDScatterPlot.tsx`, `VariableRow.tsx`,
  `chartUtils.ts`, and the 14 files under `components/eda/tabs/*.tsx`.
- **Real backend interactions performed:** created a real S3 source pointed
  at a nonexistent bucket (reached `status: failed`), opened Dataset Preview
  on a broken dataset (real `400`), opened the Ingestion Jobs modal, ran a
  real EDA analysis end-to-end on a 150-row/5-column Iris-like dataset via
  `POST /api/eda/9/analyze` → `GET /api/eda/9/latest`, inspected
  Dashboard/Correlations/Variables/Bivariate tabs, toggled dark mode, added a
  sidebar filter, and tested return navigation (`/data` → `/eda`) and
  deep-link handoffs (`/canvas?source_id=`, `/eda?dataset_id=`).
- **Evidence-collection limitations (weighed in the reconciliation below):**
  1. **No Chromium/axe automation available**, the same limitation already
     recorded in the Task 2 rerun (`#### Accessibility automation rerun`
     above): no `.superpowers/sdd/playwright.chrome.config.ts` exists in
     this checkout, so every accessibility claim in this task is manual DOM
     inspection (`querySelectorAll`, attribute checks), not an automated
     axe-core scan.
  2. **Shared-browser anomaly.** Partway through the EDA tab/filter
     investigation, the controlled browser began spontaneously navigating to
     unrelated routes (`/jobs`, `/registry`, `/deployments`, `/drift`,
     `/canvas`, `/errors`, `/slow-nodes`) and spawning extra tabs with no
     corresponding tool call causing it; a grep of the frontend source found
     no `window.open`/`target="_blank"` that could explain this, and `lsof`
     showed a VS Code "Code Helper" process also connected to port 5173 —
     strongly suggesting a second, externally-controlled client shared the
     same Chrome/CDP session. Every measurement below was taken immediately
     after an explicit `browser_navigate`/`browser_tabs close` sequence to
     re-establish a known single-tab state, so nothing here was reported from
     a stale or ambiguous page. This is recorded as an **evidence-collection
     risk for this rerun**, not a product finding: this session was less
     isolated than Task 2/3's own Playwright-launched-server sessions, and a
     future rerun should insist on a dedicated, unshared browser profile.
  3. Some deeper per-tab interactions (Expand/Collapse All in Variables,
     Bivariate X/Y custom dropdowns, Outliers tab detail) were cut short by
     the anomaly above and were confirmed via source reading rather than a
     full live click-through; this is noted per finding below.

#### Data source onboarding walkthrough

- **Observed (live, real backend):** Add Source at 1440 px offers only S3;
  its Name and S3 Path inputs have no `id`, `aria-label`, or
  `aria-labelledby`, and submitting the empty form is blocked purely by
  native HTML5 `required` validation. Submitting a real S3 path to a
  nonexistent bucket created a real source that reached `status: failed`;
  the Data Sources row rendered only a bare "Failed" badge with a single
  "Delete dataset" action and no retry or error text, while the real backend
  error is visible only inside the Ingestion Jobs modal.
- **Observed (live, real backend), positive:** `DataSources.tsx` correctly
  passes `?source_id=` to `/canvas` and `?dataset_id=` to `/eda`, and both
  `CanvasPage.tsx` and `EDAPage.tsx` consume these correctly on first mount —
  the onboarding→handoff context is preserved for the very first navigation.
- **Observed (live, real backend):** Preview on a broken dataset showed
  "0 rows • 0 columns • 0 Bytes" with the generic "Failed to load dataset
  preview." text; the underlying request returned a real `400` with
  `{"error":"HTTP 400","message":"Invalid file path"}`, a specific message
  the UI discards. The error paragraph has no `role`/`aria-live` attribute.
- **Observed (source + live):** `IngestionJobsModal.tsx` builds its job list
  from every dataset (`// For now, we show all as 'jobs'`), but it does
  surface the real backend `job.message` for failed/cancelled jobs — the
  modal is the one place error detail surfaces; only the Data Sources row is
  silent. Only pending/processing jobs expose a Cancel action; there is no
  Retry action anywhere in the flow.

#### EDA workflow walkthrough

- **Observed (live, real backend):** Selected dataset id 9 (a real completed
  report), navigated `/data` → `/eda` via the sidebar's EDA link (confirmed
  client-side SPA navigation, not a reload) — the Dataset dropdown silently
  reset to the first dataset (id 3) every time, discarding the dataset-9
  selection and its report view. The `/data → EDA button → /eda?dataset_id=X`
  deep-link path itself was unaffected.
- **Observed (source):** `EDAPage.tsx` seeds `selectedDataset` from
  `?dataset_id=` only once on mount and only if the store's
  `selectedDataset` is `null` (lines 54–61), while a separate effect
  auto-selects the first dataset whenever `!selectedDataset && datasets.length
  > 0` (lines 121–125); `useEDAStore.ts` has no `persist` middleware (plain
  `create<EDAState>(...)`), so `selectedDataset` is wiped to `null` on every
  `EDAPage` unmount. On a bare `/eda` navigation with no `dataset_id` param,
  the seeding effect is a no-op and the auto-select effect resets to the
  first dataset.
- **Observed (live + source):** Opened "Add Filter" in `EDASidebar` — the
  Column select, Operator select, and Value input all have no
  `id`/`aria-label`/`aria-labelledby`. `handleAddFilter` (lines 178–183) and
  `handleRemoveFilter` (lines 185–189) both immediately call `runAnalysis`
  with no confirmation/apply step or undo affordance, while
  `handleApplyExcluded` (lines 195–199) gates its own re-analysis behind an
  explicit Apply button — filters are the only control on this page without
  that pattern.
- **Observed (live, real backend):** At 1440 px, the EDA Dataset `<select>`
  (`EDAPage.tsx` line 469, `datasets.map((ds) => <option key={ds.id}
  value={ds.id}>{ds.name}</option>)`) contains dozens of options with
  identical visible text (e.g. "test source" appears roughly 50+ times),
  differentiated only by the underlying numeric `value` (dataset id), which
  is never shown.

#### Visualization usability recheck

- **Observed (live, real data, Iris dataset id 9):** In both light and dark
  mode, the Correlations tab renders no persistent −1/0/+1 legend, and
  column headers are truncated (e.g. "sepal.le…") relying solely on the
  native `title` attribute for the full name on hover. The correlation
  **values** themselves (e.g. "0.87") are printed as visible text inside
  each cell, so cell color is reinforced by text for the magnitude — only
  the axis/column **labels** are hover-only. Dark-mode toggling is correct
  across the app; the permanently near-black left icon rail is confirmed
  (source + DOM) to be an intentional fixed-dark nav-rail choice, not a
  theming bug.
- **Observed (source):** `BivariateTab.tsx` wraps both the actual scatter
  chart and its "Select X and Y variables to generate scatter plot."
  empty-state placeholder inside the same `id="bivariate-chart"` container
  (lines 54–61 render the always-enabled Download button; lines 147–174
  wrap chart-or-placeholder), and `downloadChart` (`chartUtils.ts`) calls
  `document.getElementById(elementId)` unconditionally and rasterizes
  whatever is inside — clicking "Download Chart" before selecting X/Y
  silently produces a PNG of the placeholder text with no warning. The same
  `id`-wraps-empty-state pattern, with the same always-enabled Download
  button, is confirmed in `PCATab.tsx` (`id="pca-chart"`).
- **Observed (source):** `SampleDataTab.tsx` is table-only; no CSV/tabular
  export exists anywhere in the EDA surface, only the chart tabs' PNG
  Download button — noted as a related but separate gap, not raised as its
  own finding (see "Not recommended as new findings" reasoning folded into
  DAT-007 below).

### Task 5 — Experiments and Inference Rerun

#### Method

- **Method:** Reused the already-running `vite` dev server (`localhost:5173`)
  and FastAPI backend (`localhost:8000`) for this checkout. Live walkthrough
  used the Playwright MCP browser tools (`browser_run_code_unsafe` against a
  `Page` object, plus navigate/click/snapshot/screenshot) at four widths —
  **1440×900**, **1024×900**, **768×1024**, **390×844** — against the real
  active deployment (`random_forest_classifier`, job
  `7c1ec203-dadc-4cb2-8e04-fd4a52c11813`) and real completed jobs. Source
  re-reading covered every file in the task brief plus files it referenced
  transitively (`jobMeta.ts`, `EvaluationView.tsx`, `PipelineDiffView.tsx`,
  `FeatureImportanceView.tsx`, `ShapSummaryView.tsx`). `git diff --stat` for
  every Experiments/Inference file named in the brief against the original
  audit commit is **empty** — no source changed since `EXP-001`–`EXP-007`
  were first written, so absent new live findings, priors read as
  *Confirmed*, not *Changed*.
- **Environment limitation (shared browser):** this session ran concurrently
  with sibling background agents doing the same kind of rerun for other
  journeys, sharing one Playwright MCP browser instance/context. Standard
  navigation calls repeatedly landed on another agent's page instead of the
  intended one, and tabs were intermittently closed or had state mutated
  mid-script by a sibling. Workaround: every scripted step re-resolved the
  target `Page` by exact URL inside one atomic `run_code_unsafe` call. Two
  attempts still failed outright because a sibling closed/repurposed the
  page mid-script; those are called out per finding below rather than
  silently retried into a misleading result. Console error/warning counts
  during navigation are **not attributed to the app**, since they correlate
  with sibling-agent navigation timing, not with any action this session
  took. This session also clicked **Undeploy** on the shared active
  deployment to test the confirmation dialog, then clicked **Cancel** — the
  deployment was left **Active** and unchanged (confirmed via follow-up
  screenshot); no destructive backend mutation was left behind
  intentionally. Several unrelated working-tree changes (`.superpowers/plans/
  *.md`, `.superpowers/sdd/progress.md`) were observed via `git status` but
  not made by this session — flagged for transparency, out of scope.
- **Incomplete-view limitation:** at 390 px, a second scripted attempt to
  open Inference failed because the notification-bell button intercepted
  the click target used by Playwright's strict-mode locator; the same
  "Inference" tab was clickable earlier in the session at 1440/1024/768, so
  this looks like tight header spacing at narrow widths rather than a
  functional block, but it was not re-verified with a different click
  strategy before time ran out — the 390 px Inference view is treated as
  **not captured**, not as broken. The SHAP Beeswarm/Dependence/Waterfall/
  Force/Interaction sub-tabs beyond Summary, Segmentation's metric cards for
  a genuinely-selected clustering pair, the CSV/drag-drop input path, a
  genuinely long-running/timeout prediction, and Save/Toggle/Clear threshold
  mutations were not exercised live this rerun (time-boxing and/or shared-
  browser contention); each is called out inline below as **not
  independently re-verified**, code-evidence-only where cited, rather than
  silently assumed equivalent to a live reproduction.

#### Experiment comparison walkthrough

- **Observed (live):** selecting 2 classification jobs, then switching the
  task-type filter to Segmentation, hides both selected rows from the
  sidebar with no visible selected-state indicator anywhere in the list, yet
  the header still reads `SELECT RUNS (2)` and Visual Comparison keeps
  rendering both hidden jobs' bars (`ExperimentsPage.tsx` lines 329–338: the
  effect that resolves `evalJobId` only re-picks `selectedJobIds[0]` when
  the current `evalJobId` drops out of the visible/selected set, never
  preferring a job compatible with the active tab). Selecting 2 additional,
  *visible* clustering jobs on top of the hidden pair (`SELECT RUNS (4)`)
  still left Model Evaluation showing the confusion-matrix UI for the
  hidden classification job, and the newly-revealed Segmentation tab
  defaulted to "The selected run is not a Segmentation (clustering) job"
  even though 2 of the 4 selected runs are valid clustering jobs — see
  **EXP-001** and **EXP-003** below.
- **Observed (live):** Visual Comparison's per-metric bars carry no
  direction/unit indicator, and Detailed Metrics & Params renders the same
  `—` glyph for "this run's pipeline has fewer/different steps than the
  compared run" as for "value not reported," with no visual distinction —
  confirms **EXP-002** unchanged.
- **Observed (live):** Feature Importance and SHAP Summary rendered 4
  features normalized 0–1 per run with a "values normalised per-run (max =
  1.0)" legend note but no explanation of what a 0 bar means — confirms
  **EXP-003**'s explainability-availability claim; the Segmentation
  present-but-wrong-default behavior above is new corroborating evidence for
  the same finding.
- **Observed (live):** Pipeline Diff assigned Baseline/Candidate by
  selection order exactly as documented, with no swap control and no
  dataset/timestamp/scoring context in the header — confirms **EXP-004**.
  **New this rerun:** the two short IDs shown in Pipeline Diff's header
  (`27b2bf2b`, `e58ea66c`) are **not** the same identifiers the sidebar,
  Detailed Metrics, and Visual Comparison show for the identical two jobs
  (`f245bcf3`, `6cdfb46e`) — see **EXP-008** below.
- **Observed (live):** Model Evaluation's Threshold Slider/Tuning tabs
  produced per-class confusion matrices and an F1-best-threshold badge for
  Train/Test; Threshold Tuning Preview showed the caption "Computed from
  test split (no validation split available — using test split)," directly
  confirming the validation→test fallback **EXP-005** describes. Save was
  not exercised this run to avoid mutating the shared active deployment's
  threshold state for sibling agents relying on the same fixture.

#### Inference walkthrough

- **Observed (live), against the real active deployment** (schema
  `sepal.length`/`sepal.width`/`petal.length`/`petal.width`, all
  `unknown`-typed): entering `[{"sepal.length": "wrong"}]` (1 wrong-type
  field, 3 missing) showed "3 missing" plus Fix while Run Prediction stayed
  enabled. Running it produced a clean backend rejection: `"Missing
  required column(s) for prediction: ['sepal.width', 'petal.length',
  'petal.width']. Expected columns: [...]"`. Clicking Fix zero-filled only
  the 3 *missing* fields, leaving the existing wrong-type field untouched
  (`checkSchema`'s name-only comparison cannot see it); running again
  crashed the backend with a raw, verbatim exception —
  `"Feature engineering failed: unsupported operand type(s) for -: 'str'
  and 'float'"` — shown at the same time as a green "✓ Added 3 missing
  field(s)" success toast. This is a materially stronger reproduction of
  **EXP-006** than the original audit's single-mechanism claim.
- **Observed (live):** both failure reproductions above rendered as raw,
  unstyled strings in the Prediction Results pane with no structured cause,
  scoped retry, or next-action guidance, directly confirming **EXP-007**'s
  raw-error-string and no-explicit-retry claims. A successful run (5
  sampled rows) produced List/Table toggle, Copy/JSON/CSV export, and a
  "RECENT RUNS" entry, confirming the export/recent-runs affordances exist.
  **Undeploy** required an explicit confirm dialog (Cancel/Undeploy) —
  a positive detail not previously called out: destructive reset is not
  one-click.
- **Observed (live):** the Inference page's "Advanced: override thresholds"
  panel stated verbatim that saved, enabled thresholds from Evaluation
  apply automatically to every real prediction; a subsequent real
  prediction run displayed a "THRESHOLDS APPLIED 0:1 1:1 2:1" banner above
  the results list. This is a live, end-to-end confirmation that saved
  thresholds affect real predictions and are surfaced back to the user,
  upgrading the tuning-affects-inference half of **EXP-005** from inferred
  to directly observed; the "no durable decision record / no provenance"
  half of the problem statement was not contradicted.

#### Component-boundary reassessment

- `useJobPolling.ts` and `useNodeJobSummaries.ts` are **not used** by
  `ExperimentsPage.tsx` or `InferencePage.tsx` (grep-confirmed; both hooks
  are consumed only by Canvas/Jobs-panel components). No user-facing risk
  currently traces through these hooks for the Experiments/Inference
  journeys, so no boundary recommendation is added for them.
- Experiments and Inference remain non-route views toggled under `/canvas`
  by `MainLayout.tsx` (`display:contents`/`none`, lazily mounted once and
  kept mounted) — this is unchanged and already the mechanism behind
  `FND-006` and the local-state-retention evidence in `EXP-001`/`EXP-007`;
  it does not warrant a new or revised boundary entry.
- The existing `InferencePage.tsx` boundary recommendation (see
  Component-Boundary Recommendations below) is reconfirmed unchanged: its
  cited risk (`EXP-005`–`EXP-007`) still matches current behavior, and this
  rerun found no new risk inside `InferencePage.tsx` requiring a revision.
- The `EXP-008` cross-tab identifier mismatch is a genuine duplicated-logic
  risk (6 components independently deciding how to label "a job" instead of
  sharing `shortRunId`), but its fix is a same-file, low-effort call-site
  swap with no shared mutable state or lifecycle coordination at stake —
  it does not meet this roadmap's bar for a component-boundary
  recommendation (no measured reliability failure or independently testable
  user-state risk beyond the label itself), so it is captured only as
  `EXP-008`'s own finding, not as a new boundary section.

### Task 6 — Operations Rerun

#### Method

- **Method:** Reused an already-running local dev stack rather than starting a
  fresh one: FastAPI backend (`run_skyulf.py`, port 8000, SQLite-backed, real
  seeded data — 56 registered model versions, 1 active deployment, dozens of
  tuning-job history rows, 30 HTTP error events) and Vite dev server (port
  5173); neither process was started or stopped by this task. Live
  walkthroughs used the Playwright MCP browser tools at **1440, 1024, 768, and
  390 px** against routes confirmed directly from `App.tsx`: `/jobs`,
  `/registry`, `/deployments`, `/drift` (there is no literal
  `/monitoring/drift` route), `/errors`, `/slow-nodes`, `/audit`, plus the
  Canvas-mounted Job History drawer (`JobsDrawer`/`JobDetailsView`, opened via
  the toolbar's "Job runs history" button, not a route). Source re-reading
  covered `Jobs.tsx`, `components/panels/jobs/` (`JobsDrawer.tsx`,
  `JobDetailsView.tsx`), `ModelRegistry.tsx`, `DeploymentsPage.tsx`,
  `DataDriftPage.tsx`, `ErrorLogPage.tsx`, `SlowNodesPage.tsx`,
  `AuditLogPage.tsx`, and `core/api/monitoring.ts`/`core/api/deployment.ts`,
  plus a dedicated `explore` sub-agent line-by-line re-check of every OPS-00N
  claim against current code. `git diff --stat` for these files against the
  original audit commit is empty — no source changed since `OPS-001`–`OPS-007`
  were first written, so absent new live findings, priors read as *Confirmed*,
  not *Changed*. **Correction carried over from the task brief, not silently
  fixed:** the brief names `core/api/registry.ts` as the model-registry API
  file; that file is actually the pipeline **node catalog**
  (`GET /pipeline/registry`) used by `Jobs.tsx`/`JobsDrawer` to resolve a
  job's `model_type` to a task tab. The real model-registry client backing
  `ModelRegistry.tsx` is `core/hooks/useModelRegistry.ts`
  (`useRegistryStats`/`useRegistryModels`/`useArtifacts`/`useDeployModel`,
  hitting `/registry/stats`, `/registry/models`, `/registry/artifacts/{jobId}`,
  `POST /deployment/deploy/{jobId}`); all OPS-002 evidence below cites the
  correct file.
- **No destructive mutation was submitted.** No deploy, redeploy, deactivate,
  resolve, or drift-analysis run was executed to completion against the live
  system; every affordance for those actions was confirmed to exist and be
  reachable, not exercised to completion.
- **Shared-browser limitation.** This session's Playwright browser instance
  was shared with sibling background agents concurrently doing related Task
  4/5/6 UX-audit work from the same root session. Several actions landed on a
  tab that had been silently navigated by another agent between a snapshot
  and the next action, producing stale-`ref` errors. Mitigation: every page
  visit in the second half of this pass used a fresh `browser_navigate` call
  and an immediate fresh snapshot before acting. Where an interaction could
  not be completed live before the tab was reclaimed — the Drift Thresholds
  dialog contents, the Errors Traceback dialog contents, a populated Audit Log
  entry, and a full second 390 px pass on Deployments/Drift/Errors/Slow
  Nodes/Audit — the corresponding evidence below is marked **Inferred (source
  only)** for that specific control rather than **Observed (live)**, and this
  distinction is preserved per finding rather than collapsed.
- **No fixture seeding**, as in the original audit: no populated Drift report,
  representative-sample Slow Nodes node, or populated Audit Log fixture was
  constructed. Live behavior is **Observed** only for the empty/current-data
  state actually present, and **Inferred (source)** for other lifecycle states
  (e.g., populated drift alert, acknowledged/resolved disposition).

#### Jobs and job-detail walkthrough

- **Observed (live, 1440 px, `/jobs`):** a fresh table load shows
  completed/failed rows with status, truncated ID, model type, one metric
  value, duration, and created time; clicking any row/cell produces no
  navigation, modal, or visible affordance. The only interactive control
  besides tabs is a "Filters" button revealing a single Status facet (All /
  Completed / Failed) — no task/model/date facets.
- **Observed (live, Canvas Job History drawer):** the drawer lists the same
  job pool as cards; clicking a completed job card opened a
  `JobDetailsView` dialog with Overview/Live Logs tabs, Status/Dataset/
  Duration, Execution Results, full Tuning Configuration, Best Score, and an
  Evaluation Metrics table. No Retry action was present for the terminal job
  (consistent with cancel-only framing); no cross-links to the source
  dataset, registry version, or deployment were rendered — the dataset
  filename shown as plain text only.
- **Observed (live, new this rerun):** on two separate fresh, uncached
  `/jobs` page loads, the table intermittently rendered **duplicate rows for
  the same `job_id`**, accompanied by a live React console warning
  (`Warning: Encountered two children with the same key, '<job_id>'...`).
  Root-caused in `Jobs.tsx` — see **OPS-008** below.
- **Observed (live, responsive, 390 px):** the fixed 256 px sidebar leaves
  ~134 px for the main pane; the Jobs table, Refresh button, and tabs are
  clipped/overflowed — this reconfirms `FND-001`, not a new Operations
  finding, but the OPS-001 investigation gap is present at every width
  tested and compounds with the layout defect at 390 px. At 768 px the tab
  row and table header text are also clipped at the viewport edge.

#### Registry and deployment walkthrough

- **Observed (live, 1440 px, `/registry`):** "View Versions" opens a
  version-history dialog (version, date, `best_score`, status, per-row
  Deploy/View Artifacts). "View Artifacts" lists raw artifact/pipeline-step
  file names as plain text with no link back to the originating job or
  dataset. At least one live registry entry shows `model_type: "unknown"`
  and `dataset_id: "unknown"` — a real backend data gap, not a UI bug.
- **Observed (live, 1440 px, `/deployments`):** the Active Deployment card
  shows model_type, the full Job ID as plain unlinked text, deployed
  timestamp, artifact URI, and Deactivate; Deployment History shows a
  truncated Job ID, model_type, Active status, deployed-at, and an empty
  Actions column for the sole entry. No link from either surface back to
  Jobs or a Registry version detail was found.
- **New this rerun — client-side-only "manual deployment" tracker
  (source-confirmed, `ModelRegistry.tsx` lines 53–68, 261–318):**
  `ModelRegistry.tsx` maintains a `localStorage` key
  `skyulf_manual_deployments` that lets a user mark a registry row's model
  family as "Manual"-deployed via a checkbox, independent of any real backend
  deployment record. The checkbox is `disabled` when the backend already
  reports `deployment_count > 0` for that row (so it cannot override a real
  active deployment), but for any model family the backend has not deployed,
  a user can locally flip its Registry-displayed state to "Manual"/deployed
  with no corresponding record in Deployments' active/history data, and this
  state is scoped to one browser's `localStorage` only — invisible to any
  other user or session. This is folded into **OPS-002** as an addendum below
  (same lineage-consistency problem space), not treated as a new ID.
- **Responsive (390 px, `/registry`):** reconfirms `FND-001`-style sidebar
  clipping — not new Operations content, but confirms OPS-002's journey is
  also blocked by the layout defect at narrow widths.

#### Monitoring investigations walkthrough

- **Observed (live, 1440 px, `/drift`):** live data has no drift report yet;
  the "No Drift Report Yet" empty state reproduces exactly, with a
  reference-job selector, Upload CSV/Parquet control, a disabled "Run
  Analysis" button, a "Refresh jobs" control, and a "Drift thresholds"
  button.
- **Inferred (source only — Drift Thresholds dialog contents):** not
  confirmed via a completed live open this pass (tab reclaimed mid-
  interaction, per the shared-browser limitation above). Confirmed instead
  via `core/api/monitoring.ts`: `DriftThresholds` (PSI/KS/Wasserstein/KL
  fields) is a page-state object passed per-request to the analysis call;
  `DriftHistoryEntry` has no threshold-snapshot field, confirming thresholds
  are not versioned against history, exactly as `OPS-003` states.
- **Observed (live, 1440 px, `/errors`):** Events(30)/Issues(6) tabs, stat
  cards (HTTP events: 30, Server errors: 30, Pipeline failures: 0), an hourly
  bar chart, a generic Search box, time-range buttons (1h/6h/24h/7d/All), a
  "Show resolved" toggle, and a populated table of 500-level exception events
  with per-row "✓ Resolve" and "Traceback" buttons. The Node/Route column
  renders plain-text values (e.g. `/api/pipeline/datasets/273/schema`,
  `celery/pipeline`) — text, not links.
- **Inferred (source only — Traceback dialog contents, Resolve mutation
  outcome):** not re-opened to completion this pass; corroborated instead by
  `ErrorLogPage.tsx`'s generic search matching an HTTP event's `job_id` and a
  pipeline log's `node_id` as substrings of a combined searchable text field,
  while the actual API request only accepts time-range and resolved-state
  parameters — confirming no typed severity/resource facet exists
  server-side to expose. No Resolve mutation was submitted.
- **Observed (live, 1440 px, `/slow-nodes`):** lookback controls
  (24h/7d/30d/90d), Top-10/25/50 controls, a Refresh button, summary stats
  (Step types: 8, Node runs: 65, Jobs scanned: 12, Window: 7 days), a "Total
  time by step type" bar chart, and a sortable table. Each step-type row's
  `sample_node_id` renders as literal "e.g. `<uuid-like string>`" text with no
  click handler or drill-down — confirmed via `SlowNodesResponse`'s type
  shape, which supplies only aggregate statistics plus an optional
  `sample_node_id` string, no run-ID list or dataset/pipeline/deployment
  linkage field.
- **Limitation:** the responsive (1024/768/390 px) pass for Slow Nodes
  specifically was not completed live this session; given `FND-001`'s
  confirmed uniform sidebar-clipping pattern at 390 px across every other
  Operations page tested, the same clipping is expected here but is
  **Inferred, not directly Observed**, for this page.
- **Observed (live, 1440 px, `/audit`):** a Dataset combobox populated with
  dozens of real dataset entries and a Limit control (25/50/100/200), a
  Refresh button, and — for the default-selected dataset — the exact
  empty-state copy "No saves recorded for this dataset yet." No time-range,
  actor, or action-type filter exists; the only two facets are Dataset and
  Limit.
- **Inferred (source only — populated-entry rendering):** not re-confirmed
  against a populated dataset this pass (interrupted by a tab reclaim before
  a populated entry could be captured); confirmed instead via
  `AuditLogPage.tsx` calling `pipelineVersionsApi.audit(datasetId, limit)`
  and rendering actor, timestamp, save action kind, version, and
  added/removed/modified node diffs when entries exist — matching
  `OPS-006`'s existing Inferred framing exactly.
- **Source-verified (`OPS-007`, absence claim):** grepped Jobs, Registry,
  Deployments, Drift, Errors, Slow Nodes, and Audit Log route/page sources
  for `useSearchParams`, `<Link`, or any shared record-link/query-
  serialization helper — none exists; every page's filter/search/tab/
  selection state is local `useState`. Live corroboration: every page
  visited reset its filter/tab/search state on reload or route re-entry
  (e.g., returning to `/jobs` after opening the Canvas Job History drawer did
  not preserve prior Jobs-table filter state across the two decoupled
  surfaces).

## Synthesis, Deduplication, and Ranking

### Root-cause decisions

- **Shared responsive geometry:** `FND-001` is the one cross-journey root
  cause. `DAT-004` remains a Data/EDA consumer slice because its responsive
  header/table hierarchy needs route-specific behavior; `CAN-005` remains a
  separate measured Canvas-pane collision. Neither claims a second shell root.
- **Shared semantics, not duplicate lifecycle findings:** `FND-003` owns
  live-region semantics and `FND-005` owns reusable field naming/error
  semantics. `CAN-002`, `DAT-001`–`DAT-006`, and `EXP-005`–`EXP-007` remain
  distinct where their evidence concerns domain state, validation, provenance,
  or recovery beyond those primitives. Their scoped Now work may use local
  semantics without waiting for the broader `FND-005` normalization.
- **Shared shell interaction:** `FND-002` owns the common overlay contract and
  `FND-006` owns shell-view history/selected state. Canvas evidence is retained
  under those shared IDs; no duplicate CAN item is created.
- **Operational continuity boundary:** `OPS-007` owns only typed
  serialization/parsing and record-link construction. `OPS-001`–`OPS-006`
  remain separate consumers because jobs, lineage, drift, errors, performance,
  and audit records require different server evidence and acceptance criteria.
- **No false merge:** ingestion (`DAT-003`), EDA analysis (`DAT-005`), and
  inference (`EXP-007`) are all async lifecycles, but have different resource
  contracts and recovery actions. Experiment comparison, Canvas diagnosis, and
  Operations investigation likewise remain journey-specific. This preserves
  equal representation instead of collapsing evidence from less-observed
  journeys into a generic finding.
- **New rerun findings kept distinct (not merged):** the five findings added
  this rerun each describe a defect separable from its nearest neighbor, with
  its own fix, so none is folded into an existing ID:
  - `FND-007` (button-in-button DOM nesting in `NotificationCenter`) is a
    shared-shell accessibility defect distinct from `FND-002`'s focus-
    containment contract; its fix is a markup change, not a focus-management
    change.
  - `DAT-008` (indistinguishable duplicate Dataset-dropdown option labels) is
    about making an informed selection in the first place, upstream of and
    separable from `DAT-005`'s loss of a selection after navigation, and
    unrelated to `DAT-007`'s chart-interpretation scope from which it was spun
    off.
  - `DAT-009` (chart Download never disabled for empty/unconfigured charts) is
    an export-gating defect in `chartUtils.ts`/EDA tabs, separable from
    `DAT-007`'s visualization-interpretation contract.
  - `EXP-008` (cross-tab `job_id`-vs-`pipeline_id` label mismatch) compounds
    `EXP-001` and `EXP-004` but is a single-file label-consistency swap, not a
    selection-retention or diff-role change; it does not meet the component-
    boundary bar (see Component-Boundary Recommendations) and stays its own
    finding.
  - `OPS-008` (duplicate Jobs rows / colliding React keys from a `poolSkip`
    closure race) is an active rendering/data-integrity bug distinct from
    `OPS-001`'s missing-investigation-affordance framing for the same table.

### Normalized ranking

Ranks use the required order: impact/severity, frequency, journeys improved,
accessibility or data-loss risk, effort, then regression risk/dependencies.
The inventory retains only the normalized values `High`/`Medium`/`Low`,
`Frequent`/`Occasional`/`Rare`, `S`/`M`/`L`, `Low`/`Medium`/`High`, and
`Now`/`Next`/`Later`. A later milestone can still rank highly when its
dependency-complete slice is intentionally broader than the smallest Now
change.

| Rank | Finding IDs in priority order | Sequencing rationale |
|------|-------------------------------|----------------------|| 1–6 | FND-001, FND-006, CAN-001, CAN-005, EXP-006, DAT-001 | Frequent high-impact access, task-completion, or data-validity risks with observed evidence and dependency-light fixes; the shared shell and Canvas blockers lead. |
| 7–12 | OPS-007, DAT-004, DAT-005, EXP-001, EXP-002, FND-005 | Frequent journey blockers and the cross-page Operations foundation, plus shared form semantics that rank highly even though its normalization slice lands in Next. |
| 13–18 | FND-003, FND-002, CAN-002, DAT-002, DAT-003, EXP-005 | High-impact recovery, overlay-accessibility, and decision-provenance work whose dependencies fit within Now. |
| 19–24 | EXP-007, OPS-001, OPS-004, DAT-006, DAT-007, EXP-003 | High-impact run lifecycle and Frequent Operations investigations, then broad Data/EDA and explainability interpretation sequenced after their foundations. |
| 25–30 | OPS-002, OPS-003, EXP-004, OPS-008, DAT-008, EXP-008 | High-impact Operations lineage/drift redesigns, then the nominally Medium-impact Jobs defect `OPS-008` — its live data-integrity/duplicate-key defect (colliding React keys causing reordering, incorrect DOM reuse, or dropped updates) raises its effective severity under ranking criterion 1 (impact/severity) above a plain label-consistency gap, so it is sequenced ahead of the Medium label-consistency fixes `DAT-008`/`EXP-008` while its normalized `Impact` value and `Now` milestone remain unchanged. |
| 31–37 | FND-004, FND-007, OPS-005, OPS-006, CAN-003, CAN-004, DAT-009 | Lower-frequency normalization, a contained accessibility DOM-nesting fix, larger historical-context work, and optional polish after the preceding outcomes. |

**Implementation status (2026-08-08):** ranks `1–18` are complete, as are
`DAT-006`, `EXP-004`, `EXP-008`, `OPS-006`, `OPS-008`, `DAT-008`, `FND-004`,
`FND-007`, `CAN-004`, and `DAT-009` out of order. `9` findings remain — see
`## Historically Resolved Findings` for the full breakdown, including which
resolutions shipped a scoped slice rather than the full proposed behavior.

## Method and Evidence

This roadmap combines required engineering baseline checks with source
inspection of the shared application shell and route coverage before later
tasks add live walkthrough evidence.

### Evidence Labels
- **Observed:** Reproduced in the running interface.
- **Measured:** Verified through CLI, build, or automated test output.
- **Inferred:** Identified from code or test structure as a UX regression risk.

### Baseline

#### Engineering baseline

- **Measured:** `npm run lint` exited `0`.
- **Measured:** `npx tsc --project tsconfig.json --noEmit` exited `0`.
- **Measured:** `npm run build` exited `0` after Vite transformed `2939`
  modules and completed in `9.48s`. The output also reported
  `Circular chunk: vendor-flow -> vendor-charts -> vendor-flow` and
  `Generated an empty chunk: "vendor-react"`.

| Build chunk | Raw size | Gzip size |
|-------------|----------|-----------|
| `index.html` | `1.43 kB` | `0.71 kB` |
| `EDAPage-Dgihpmma.css` | `15.04 kB` | `6.38 kB` |
| `index-oa1aIZJq.css` | `133.03 kB` | `21.50 kB` |
| `vendor-react-l0sNRNKZ.js` | `0.00 kB` | `0.02 kB` |
| `DeploymentsPage-BPJNZTuK.js` | `7.36 kB` | `2.11 kB` |
| `AuditLogPage-nFTvZ0kZ.js` | `9.66 kB` | `3.07 kB` |
| `SlowNodesPage-BwFllXZg.js` | `9.92 kB` | `3.16 kB` |
| `ModelRegistry-BW1Qr2Dh.js` | `20.60 kB` | `5.17 kB` |
| `DataDriftPage-CzhK1_YF.js` | `35.30 kB` | `9.91 kB` |
| `vendor-flow-DKdhqECu.js` | `171.04 kB` | `54.51 kB` |
| `vendor-utils-BhHEl1zz.js` | `215.21 kB` | `68.32 kB` |
| `EDAPage-0oTaJpB1.js` | `294.04 kB` | `78.92 kB` |
| `vendor-charts-BZV-Qcd2.js` | `695.79 kB` | `215.01 kB` |
| `index-Bfinbtv3.js` | `1,015.86 kB` | `258.24 kB` |
| `vendor-plotly-J_-jSK3N.js` | `1,704.42 kB` | `541.11 kB` |

- **Measured:** `npm run test -- --reporter=dot` exited `0` with
  `Test Files 40 passed (40)` and `Tests 335 passed (335)` in `4.89s`.
  The output also included existing test-run console noise from jsdom
  `localStorage` warnings, `useConfirm must be used inside <ConfirmProvider>`
  coverage in `ConfirmDialog.test.tsx`, repeated `404 not found` polling logs
  in `useJobPolling.test.ts`, and Recharts zero-size warnings in
  `EvaluationView.test.tsx`.
- **Measured:** `npm run test:e2e -- --project=chromium` exited `1`.
  Playwright attempted `12` Chromium tests and all `12` failed before route
  interaction because the Chromium executable was missing at
  `/Users/BH7043/Library/Caches/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-mac-arm64/chrome-headless-shell`.
  The runner advised `npx playwright install`. Per audit scope, this is
  recorded as pre-existing baseline evidence and was not fixed.
- **Measured:** Fallback E2E invocation:
  `cd frontend/ml-canvas && NODE_PATH="$PWD/node_modules" npx playwright test --config=/Users/BH7043/Skyulf/.worktrees/frontend-ux-audit/.superpowers/sdd/playwright.chrome.config.ts`.
  The audit config is ignored by the normal Chromium runner and instead uses
  the installed Google Chrome binary. `11` tests passed and
  `e2e/preview.spec.ts:24` failed after `30s` because the Run Preview button
  repeatedly became unstable/detached from the DOM while Playwright tried to
  click it (`preview.spec.ts:113`). Keep this as historical measured evidence
  of a one-off preview-button instability. The later targeted preview reruns
  and the final system-Chrome validation superseded it: Task 8 ended with
  `12/12` system-Chrome E2E tests passing, so the authoritative baseline
  status is full pass.
- **Measured:** The later targeted preview reruns and the final Task 8
  system-Chrome validation passed `12/12` E2E tests. That final run is the
  baseline status to trust; the earlier `11/12` measurement above remains only
  as historical evidence. The accessibility suite passed its critical-only gate
  while logging two non-blocking serious findings: dashboard `color-contrast`
  and canvas `scrollable-region-focusable`. These remain pre-existing UX/test
  evidence, not fixes.
- **Measured:** `npm run size-check` exited `0`; all checked chunks were within
  budget.

| Size-check target | Raw size | Gzip size | Budget | Result |
|-------------------|----------|-----------|--------|--------|
| `vendor-plotly` | `1664.5 KB` | `528.4 KB` | `750.0 KB` | `OK (70%)` |
| `vendor-charts` | `679.5 KB` | `210.0 KB` | `220.0 KB` | `OK (95%)` |
| `vendor-flow` | `167.0 KB` | `53.2 KB` | `80.0 KB` | `OK (67%)` |
| `vendor-react` | `0.0 KB` | `0.0 KB` | `70.0 KB` | `OK (0%)` |
| `vendor-utils` | `210.2 KB` | `66.7 KB` | `90.0 KB` | `OK (74%)` |
| `index (main)` | `992.0 KB` | `252.2 KB` | `260.0 KB` | `OK (97%)` |
| `route:EDA` | `287.1 KB` | `77.1 KB` | `140.0 KB` | `OK (55%)` |
| `route:DataDrift` | `34.5 KB` | `9.7 KB` | `20.0 KB` | `OK (48%)` |
| `route:ModelRegistry` | `20.1 KB` | `5.0 KB` | `15.0 KB` | `OK (34%)` |
| `route:Deployments` | `7.2 KB` | `2.1 KB` | `10.0 KB` | `OK (21%)` |

#### Route and navigation baseline

| Route | Surface | Lazy-loaded | Sidebar collapsed | Alert badge | Covered by `e2e/routes.spec.ts` | Evidence |
|-------|---------|-------------|-------------------|-------------|----------------------------------|----------|
| `/` | Dashboard | No | No | No | Yes | `Inferred` |
| `/jobs` | Jobs | No | No | No | Yes | `Inferred` |
| `/data` | Data Sources | No | No | No | Yes | `Inferred` |
| `/eda` | EDA | Yes | Yes | No | Yes | `Inferred` |
| `/drift` | Data Drift | Yes | No | Yes | No | `Inferred` |
| `/canvas` | ML Canvas | No | Yes | No | Yes | `Inferred` |
| `/registry` | Model Registry | Yes | No | No | No | `Inferred` |
| `/deployments` | Deployments | Yes | No | No | No | `Inferred` |
| `/errors` | Error Log | No | No | Yes | No | `Inferred` |
| `/slow-nodes` | Slow Nodes | Yes | No | No | No | `Inferred` |
| `/audit` | Audit Log | Yes | No | No | No | `Inferred` |

- **Inferred:** `src/App.tsx` lazy-loads `EDAPage`, `DataDriftPage`,
  `ModelRegistry`, `DeploymentsPage`, `SlowNodesPage`, and `AuditLogPage`
  behind per-route `Suspense` + `ErrorBoundary` wrappers.
- **Inferred:** `src/components/Layout.tsx` collapses the sidebar only on
  `/canvas` and `/eda`, and exposes alert badges only on `/drift` and
  `/errors`.
- **Inferred:** `e2e/routes.spec.ts` currently covers only `/`, `/canvas`,
  `/jobs`, `/data`, and `/eda`; the remaining top-level operations routes are
  not included in that smoke coverage file.
- **Inferred:** The shared shell exposes Canvas, Data/EDA, and Operations at
  the top level; Experiments and Inference are shell views, not App routes.
  They are opened from the `Navbar` buttons and lazily mounted by
  `MainLayout`, which keeps each page alive in `visitedViews` so its local
  state survives switches.

## Shared Foundations

### Navigation and Orientation

- **FND-006 — Inferred: shared shell view selection is neither history-backed
  nor programmatically selected.**
  - **Evidence:** **Observed** once while switching from Canvas to
    Experiments: the URL remained `/canvas`, and Back left Canvas for the
    prior route. **Inferred** across the three shell-view journeys:
    `Navbar.tsx` sends Canvas, Experiments, and Inference through the same
    `setView` calls; `useViewStore.ts` stores only `activeView` in memory; and
    `MainLayout.tsx` displays the same retained views from that state. The
    three buttons have no selected/current ARIA state. Inference was not
    separately reproduced, so this inventory row is Inferred.
  - **User problem:** A user moving among Canvas, Experiments, and Inference
    cannot restore the selected shell view with Back/Forward, and assistive
    technology receives three ordinary, unselected buttons instead of the
    current view.
  - **Affected surfaces:** Canvas, Experiments, and Inference; `Navbar.tsx`,
    `MainLayout.tsx`, `useViewStore.ts`, and `Breadcrumb.tsx`.
  - **Proposed behavior:** Represent the selected shell view in
    navigation state that participates in Back/Forward (for example, a
    query parameter), and expose the control group with the appropriate
    selected state and accessible name.
  - **Acceptance criteria:** Switching views updates a restorable URL/state;
    Back/Forward returns to the prior shell view without resetting its
    retained local state; the active control is programmatically identified
    without depending on its color.
  - **Validation method:** Playwright navigates Canvas → Experiments →
    Inference → Back/Forward at 1440, 1024, 768, and 390 px; an accessibility
    snapshot asserts one selected/current subview for each shell view.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** `useViewStore` and retained-view behavior.
    **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed** again at 1440 px on `/canvas`:
    clicking "Experiments" left the URL at `/canvas` with no new history
    entry, none of the three switcher buttons exposed `aria-current`,
    `aria-selected`, or `aria-pressed`, and Back returned to `/` (the route
    visited before Canvas) rather than to a prior shell view. Source
    re-reading confirms `Navbar.tsx`, `useViewStore.ts`, and `MainLayout.tsx`
    are unchanged from the original evidence. Inference's own switch was not
    independently re-driven this rerun, so that inventory row remains
    Inferred, as in the original audit.
  - **Delta:** No material change.

### Async and Feedback States

- **FND-003 — Inferred: Shared loading and error states do not announce
  async changes.**
  - **Evidence:** Source review found that `LoadingState.tsx`, `EmptyState.tsx`,
    `ErrorState.tsx`, and `toast.ts` do not provide a shared live
    status/alert contract for equivalent state transitions across the affected
    journeys.
  - **User problem:** A spinner, empty result, or request error can appear
    without a status or alert announcement, leaving screen-reader users
    unaware that a load or failure has completed.
  - **Affected surfaces:** Dashboard, Data Sources, EDA, Jobs, Model
    Registry, Deployments, Canvas execution feedback, Experiments, and
    Inference; `LoadingState.tsx`, `EmptyState.tsx`, `ErrorState.tsx`, and
    `toast.ts`.
  - **Proposed behavior:** Give shared loading feedback a concise polite
    status, errors an assertive alert linked to their retry action, and keep
    empty states descriptive rather than announcing decorative icons. Use the
    shared semantics wherever equivalent page-local states are rendered.
  - **Acceptance criteria:** Each first-load, empty, error, retry, success,
    and unavailable-action transition supplies one intelligible message and
    the appropriate live-region semantics; retries retain the user's current
    filters, dataset, and view.
  - **Validation method:** Component tests assert status/alert roles and
    retry behavior; Playwright exercises mocked success, empty, and failure
    paths for Canvas, Data/EDA, Experiments/Inference, and Operations; run
    axe afterward.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** S. **Risk:**
    Low. **Dependencies:** None. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** Re-reading `LoadingState.tsx`, `EmptyState.tsx`,
    and `ErrorState.tsx` confirms none contain a `role=` or `aria-live`
    attribute; `grep -RInE "LoadingState|EmptyState|ErrorState|PageSkeleton|toast\.|disabled=" src`
    found `256` matches across the same affected journeys, with no shared
    live-region contract introduced. Inferred: source review, not a live
    screen-reader reproduction.
  - **Delta:** No material change.
  - **RESOLVED (v0.7.5).** `LoadingState` and `EmptyState` render a polite
    `role="status"`; `ErrorState` renders an assertive `role="alert"` whose
    retry button is bound to the message via `aria-describedby`. Every
    decorative icon (including `RefreshCw` inside the Retry button) is
    `aria-hidden="true"`. Verified live in Chrome against a running app, not
    only in jsdom: filtering Data Sources to an empty result exposed a single
    `role="status"` reading "No datasets match your search or filters" with
    zero unhidden icons; injecting an API failure on the Dashboard exposed
    `role="alert"` whose Retry button's accessible description resolved to the
    error text. Covered by `states.test.tsx` (11 tests).
    **`toast.ts` intentionally unchanged:** the finding's premise was wrong.
    `sonner`'s `<Toaster>` already renders `aria-live="polite"
    aria-relevant="additions text"`, so toasts were always announced; grepping
    `toast.ts` for `role=` missed it because the live region lives in the
    `<Toaster>` component, not the helper. An added second announcer was
    measured reading each toast twice, violating the "one intelligible
    message" criterion, and was removed. `toastAnnouncement.test.tsx` renders
    the real `Toaster` and pins each toast to exactly one live region.
    **Still open:** the Playwright mocked success/empty/failure runs across
    Canvas, Data/EDA, Experiments/Inference, and Operations, and the axe pass,
    were not added — this repo has no seeded E2E fixtures for those paths.
    Announcements were verified by direct live DOM inspection instead. The
    "retries retain the user's current filters, dataset, and view" clause is
    owned by FND-004 (retry affordance coverage) and is not claimed here.

- **FND-004 — Inferred: Retry affordances are inconsistent for equivalent
  request failures.**
  - **Evidence:** `Dashboard.tsx`, `EDAPage.tsx`, and
    `DeploymentsPage.tsx` pass `onRetry` to the shared `ErrorState`, while
    `ModelRegistry.tsx` and the Experiments Evaluation/Segmentation views
    render it without one. This is concrete code evidence across the
    Dashboard, Data/EDA, Operations, and Experiments journeys. Canvas is
    intentionally not included: `CanvasPage.tsx` performs URL/restore
    handling and reports its scoped failures with `toast`, rather than
    rendering this page-fetch `ErrorState` pattern.
  - **User problem:** Users can retry failed dashboard, EDA, and deployment
    loads in place, but Model Registry and evaluation error uses can render
    `ErrorState` without a retry, forcing a reload or a route change.
  - **Affected surfaces:** Dashboard, EDA, Model Registry, Deployments, and
    Experiments evaluation/segmentation; shared `ErrorState`. Canvas is out
    of scope for this route-fetch finding.
  - **Proposed behavior:** Pass a safe, idempotent retry action to every
    recoverable request error, retain the prior selection/filter context, and
    distinguish unavailable actions from a failed request.
  - **Acceptance criteria:** Every recoverable top-level and subview fetch
    error presents a Retry action; activation performs one new request,
    disables duplicate submission while pending, and restores the same
    context on success or a useful error on failure.
  - **Validation method:** Add focused page tests for each current
    `ErrorState` use and Playwright request-failure/retry checks for the four
    affected journeys.
  - **Impact:** Medium. **Frequency:** Occasional. **Effort:** S. **Risk:**
    Low. **Dependencies:** Existing page fetch functions. **Milestone:** Next.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** Re-grepped and re-read all five call sites:
    `Dashboard.tsx`, `EDAPage.tsx`, and `DeploymentsPage.tsx` still pass
    `onRetry`; `ModelRegistry.tsx` (line 158, `<ErrorState error={error} />`)
    and `EvaluationView.tsx`/`SegmentationView.tsx` (`<ErrorState
    error={evalError} />`) still render `ErrorState` with no `onRetry`.
    Inferred: source review, not a live reproduction of each failure path.
  - **Delta:** No material change.


### Forms and Validation

- **FND-005 — Inferred: Canvas, Data/EDA, and Inference forms lack consistent
  programmatic field semantics.**
  - **Evidence:** In Canvas, `EncodingNode.tsx` places the visible “Encoding
    Method” `span` beside a `select` without `htmlFor`, `id`, or ARIA
    association; the representative node settings use the same visual-label
    convention. In Data Sources, the observed Add Source modal rendered zero
    associated `label` elements for the required Name and S3 Path inputs, and
    `AddSourceModal.tsx` uses visible `span` labels for those fields and the
    optional credential inputs without `id`/`htmlFor` or ARIA association. In
    EDA, `EDASidebar.tsx` renders filter/exclusion column/operator/value
    controls with placeholder-only prompts and no programmatic label
    association, and `EDAPage.tsx` renders the no-analysis Target Column input
    and Task Type `select` without a label. In Inference, `InferencePage.tsx`
    renders the JSON `textarea` without a `label`, `aria-label`, or
    `aria-labelledby`, and displays parse status without associating it to the
    editor or setting its invalid state. This is concrete evidence across four
    journeys, not a claim about uninspected Experiments or Operations forms.
  - **User problem:** Keyboard and assistive-technology users can reach Canvas
    node controls, Data source fields, EDA analysis/filter inputs, or the
    Inference prediction editor without a programmatically conveyed purpose or
    invalid/error relationship.
  - **Affected surfaces:** Canvas node settings, including Encoding; Data
    Sources Add Source; EDA analysis setup plus filter/exclusion controls;
    Inference prediction input; shared `Input` and `Button` primitives where
    adopted.
  - **Proposed behavior:** Require an explicit label association (or
    `aria-labelledby`) for the affected Canvas, Data, EDA, and Inference
    controls, expose required and invalid states programmatically, and place a
    persistent error beside the field while preserving helpful defaults.
  - **Acceptance criteria:** Every interactive control in the affected Canvas
    node panels, Add Source modal, EDA setup/sidebar controls, and Inference
    prediction editor has a unique accessible name; required fields announce as
    required before submission; invalid fields expose `aria-invalid` and an
    associated error; Enter submits only forms or actions with a defined,
    announced submit action.
  - **Validation method:** Render representative forms in component tests,
    assert accessible names/required/error relationships, and complete
    keyboard-only configuration for Canvas node panels, Add Source, EDA
    setup/filter flows, and the Inference editor at desktop and 390 px; run
    axe on each representative surface.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** shared form primitives, node configuration
    metadata, and existing source/EDA/inference validation rules.
    **Milestone:** Next.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** Re-read all four cited files. `EncodingNode.tsx`
    still places `<span className="block text-sm font-medium">Encoding
    Method</span>` beside its `<select>` with no `htmlFor`/`id`/ARIA
    association. `AddSourceModal.tsx` still uses `<span
    className="...">Name</span>` and `<span className="...">S3 Path</span>`
    beside their inputs with no `id`/`htmlFor`. `EDASidebar.tsx` still has no
    `htmlFor`/`aria-label`/`<label>` for its filter controls (placeholder
    `"Value"` only). `InferencePage.tsx` still renders the prediction
    `<textarea>` with no `label`, `aria-label`, or `aria-labelledby`.
    Inferred: source review, not a live keyboard-only reproduction across
    all four surfaces this rerun.
  - **Delta:** No material change.
  - **2026-08-07 resolution (partial):**
    - **The roadmap's `AddSourceModal.tsx` evidence was stale.** Re-reading the
      file found it already fully corrected: `Name` and `S3 Path` use
      `<label htmlFor>`, carry `required`, flip `aria-invalid`, and link
      `aria-describedby` to a `role="alert"` error; the credential inputs are
      labelled too. Live-verified at 390 px — submitting the empty form sets
      `aria-invalid="true"` on both required fields with "Name is required." /
      "S3 Path is required." as their accessible descriptions. No change was
      needed there, and none was made.
    - Fixed the genuinely unlabelled controls: `EncodingNode`'s Encoding Method
      select (`span` → `<label htmlFor>` + `useId`), EDA's toolbar
      Dataset/Target Column/Task Type selects, EDA's no-analysis setup Target
      Column input and Task Type select, `EDASidebar`'s filter
      column/operator/value and exclusion-column controls, and Data Sources'
      dataset search input.
    - Inference's prediction `<textarea>` now uses `aria-labelledby` to its
      "Input Data (JSON)" heading, `aria-describedby` to the parse-status line
      (given `role="status"`), and `aria-invalid` driven by `inputStatus.valid`.
      Live-verified: typing `[{broken` flips `aria-invalid` to `true` and the
      accessible description becomes the parser's message.
    - Added `components/ui/FormField.tsx` (11 tests, TDD) as the shared
      primitive so future forms get label/required/error association by
      construction.
    - Added accessible-name regression tests for `EDASidebar` (2) and
      `EncodingNode` (1). The `EncodingNode` test was written after the fix, so
      it was **verified by reverting the fix and watching it fail** with
      "Unable to find an accessible element with the role combobox and name
      Encoding Method".
    - Live sweep of every rendered control: `/eda` (3 controls at 1440 px and
      390 px, plus 4 more with the filter/exclusion forms open) and `/data`
      with the Add Source modal and credentials expanded (6 controls) — **zero
      unnamed controls** on either surface.
  - **Still open (deliberately not claimed as done):**
    - Only the four cited surfaces were swept. Experiments and Operations forms
      were **not** audited, matching the finding's own scope note.
    - The `FormField` primitive is built and tested but not yet adopted by the
      existing forms — they were fixed in place with native `label`/`aria-*`.
      Migrating them is follow-up work.
    - "Enter submits only forms with a defined, announced submit action" was
      not systematically audited.
    - **axe was not run** on these surfaces. `@axe-core/playwright` is a
      dependency but the finding's validation method needs an E2E lane; the
      verification above is a DOM-level accessible-name computation, not a full
      axe pass.

### Accessibility and Keyboard UX

- **FND-002 — Observed: shell overlays lack a shared focus-containment and
  focus-return contract.**
  - **Evidence:** **Observed** at 1440 px on `/canvas` for all three shell
    overlays: Shortcuts (Tab reached the covered “More canvas tools” control
    instead of a dialog control), Command Palette (Tab from its last
    in-dialog element reached a covered devtools button behind the still-open
    dialog), and the notification detail modal (focus never moved into the
    dialog on open, and the next Tab escaped to the sidebar's "Collapse
    sidebar" button). **Inferred** that this generalizes to the Experiments
    and Inference shell views: `MainLayout.tsx` mounts Shortcuts and Command
    Palette alongside Canvas, Experiments, and Inference, and `Navbar.tsx`
    renders NotificationCenter for the same three views, but these overlays
    were only independently re-driven from `/canvas`, not from Experiments or
    Inference, so that cross-view scope remains source-inferred.
  - **User problem:** With the shortcuts overlay open, the next Tab focused
    the covered “More canvas tools” button instead of an element in the
    dialog. Other shell overlays have the same missing shared focus-management
    contract, creating a cross-view regression risk rather than an observed
    duplicate failure.
  - **Affected surfaces:** Shortcuts and Command Palette in Canvas,
    Experiments, and Inference; Notification detail from the shared Navbar;
    `ModalShell.tsx`.
  - **Proposed behavior:** Apply the existing modal focus containment and
    focus-return behavior to these custom overlays, with an initial focus on
    the first useful dialog control and Escape/backdrop close where allowed.
  - **Acceptance criteria:** Tab and Shift+Tab remain within each open
    overlay; Escape closes it; focus returns to the invoker; no interactive
    element remains reachable behind a modal. Command Palette retains arrow
    navigation and search focus.
  - **Validation method:** Playwright keyboard tests open/close Shortcuts,
    Command Palette, and Notification detail from Canvas, Experiments, and
    Inference and assert the active element throughout; run existing
    overlay/component tests.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** S. **Risk:**
    Low. **Dependencies:** `ModalShell` focus helpers. **Milestone:** Now.
  - **2026-08-07 status:** Changed.
  - **Current evidence:** **Observed** again at 1440 px on `/canvas` for
    Shortcuts: the next Tab after opening it moved focus to the covered
    "Open command palette" navbar button, outside the dialog — reproducing
    the original evidence exactly. **Newly Observed** (previously Inferred)
    for the other two surfaces: opening Command Palette (Ctrl/Cmd+K) and
    focusing its last in-dialog element, the next Tab moved focus to the
    "Open Tanstack query devtools" button behind the still-open dialog.
    Seeding one notification and opening its detail modal, focus was never
    moved into the dialog on open, and the next Tab moved focus to the
    sidebar's "Collapse sidebar" button, escaping the modal entirely. All
    three shell overlays now have direct, live confirmation of the same
    missing focus-containment/return contract.
  - **Delta:** Evidence for Command Palette and the notification detail
    modal upgraded from Inferred to Observed; the user problem, affected
    surfaces, proposed behavior, and priority are unchanged.
  - **RESOLVED (v0.7.5).** The focus-trap and focus-return logic already
    embedded in `ModalShell.tsx` was extracted into a shared `useModalFocus`
    hook and applied to `ShortcutsOverlay`, `CommandPalette`, and the
    NotificationCenter detail modal; `ModalShell` now consumes the same hook,
    losing ~90 duplicated lines with no behavior change. The hook also skips
    `tabindex="-1"` elements, which the original selector matched but should
    never have trapped, and both backdrops were made non-tabbable.
    Re-driven live in Chrome at the same viewport as the original evidence,
    all three overlays now pass the full contract: focus moves into the dialog
    on open (Shortcuts → "Close shortcuts overlay", Palette → its search
    input, Notification detail → "Close detail"), Tab and Shift+Tab stay
    inside, Escape closes, and focus returns to the invoker ("Keyboard
    shortcuts", "Keyboard shortcuts", "Notifications (1)" respectively). The
    Command Palette's arrow-key navigation still moves the selection
    (Dataset → Drop Columns) and its search input still takes initial focus.
    **Still open:** the Playwright keyboard tests driving these overlays from
    Experiments and Inference were not added; only Canvas was re-driven live.
    `MainLayout` mounts the same three overlays for all three views, so the
    fix is shared, but the cross-view assertion remains source-inferred.
    jsdom cannot traverse real tab order, so the committed component tests
    assert initial focus, Escape handling, focus return, and dialog
    semantics — the Tab containment evidence is the live Chrome run above.

- **FND-007 — Observed: `NotificationCenter` nests an interactive Dismiss
  button inside another interactive row button.**
  - **Evidence:** With a seeded notification (`localStorage`
    `skyulf-notifications`), opening the panel triggered a live React
    `validateDOMNesting(...)` console warning ("`<button>` cannot appear as
    a descendant of `<button>`") pointing at
    `NotificationCenter.tsx:274`, where each `<li>` row (lines 247–284)
    renders `<button onClick={() => openDetail(it)}>...<button
    aria-label="Dismiss" onClick={(e) => { e.stopPropagation();
    dismiss(it.id); }}>...</button></button>` — an interactive Dismiss
    button nested inside the interactive row button, which is invalid per
    the HTML interactive-content model. The accessibility tree's computed
    accessible name for the outer row concatenated the nested button's own
    text into the row's name (`"error Encoding 12:02 PM Test warning
    message node-1 Click to see full details Dismiss"`), and Tab from the
    row moved focus onto the nested `aria-label="Dismiss"` button,
    confirming the invalid nesting is live in the rendered DOM and
    keyboard-reachable, not only a static markup smell.
  - **User problem:** A screen-reader user tabbing to a notification row
    hears an accessible name that folds in the unrelated "Dismiss" action
    text, making it unclear whether activating the row opens details or
    dismisses the notification; the underlying invalid DOM structure is
    also fragile across browsers/AT that may reflow or drop the nested
    control.
  - **Affected surfaces:** `NotificationCenter.tsx` notification list rows,
    reachable from the shared Navbar in Canvas, Experiments, and Inference.
  - **Proposed behavior:** Restructure each row so the open-detail action
    and the Dismiss action are sibling interactive elements (for example, a
    non-interactive row container plus two independent buttons, or an
    interactive row using `role="button"` with the Dismiss control placed
    outside it), so no button is a descendant of another button.
  - **Acceptance criteria:** No interactive element is nested inside
    another interactive element in the notification list; each row and its
    Dismiss control expose distinct, correctly scoped accessible names; Tab
    order and activation semantics remain unchanged for sighted mouse and
    keyboard users.
  - **Validation method:** A component test renders a populated
    notification list and asserts no `button` has a `button` ancestor
    within the list, plus that each row's and Dismiss control's accessible
    name does not include the other's text; Playwright keyboard test tabs
    through a populated list and asserts focus order and the two controls'
    accessible names.
  - **Impact:** Medium. **Frequency:** Occasional (only reachable once at
    least one pipeline warning has been buffered). **Effort:** S. **Risk:**
    Low. **Dependencies:** None. **Milestone:** Next.
  - **2026-08-07 status:** New.
  - **Current evidence:** Observed live via a seeded `localStorage`
    notification and a real browser session (console warning, accessible
    name concatenation, and Tab focus landing on the nested button); not
    present in the previous roadmap because the notification panel was
    previously exercised only in its empty state.
  - **Delta:** New finding; no prior entry to compare against.

### Responsive Behavior

- **FND-001 — Observed: the shared shell is not usable at 390 px.**
  - **Evidence:** A 390 px Chrome walkthrough measured the Dashboard's fixed
    256 px sidebar, leaving a 134 px main pane, and Canvas's 353 px view
    switcher inside a 326 px pane, where it overlapped adjacent controls.
  - **User problem:** On Dashboard, the fixed 256 px sidebar leaves only
    134 px for the main content; it begins beyond the visible content area.
    On Canvas, the 353 px view switcher overflows its 326 px main pane and
    collides with read-only and notification controls. This hides content and
    prevents reliable touch or keyboard operation on narrow screens.
  - **Affected surfaces:** Shared `Layout` across Dashboard, Data/EDA, and
    Operations; Canvas, Experiments, and Inference `Navbar`/view switcher.
  - **Proposed behavior:** At a defined compact breakpoint, replace the
    persistent sidebar with an accessible disclosure/drawer and make the
    Canvas view navigation wrap, scroll intentionally, or use a compact
    control so all actions remain visible without overlap.
  - **Acceptance criteria:** At 390 px every top-level route has a usable
    content width, a discoverable current page, and no clipped or overlapping
    global controls; interactive global targets are at least 44 by 44 CSS px
    where touch is expected; 768 px and above preserve the efficient desktop
    layout.
  - **Validation method:** Playwright screenshots and bounding-box/overflow
    assertions for all top-level routes plus Canvas, Experiments, and
    Inference at 1440, 1024, 768, and 390 px; keyboard-open/close the compact
    navigation.
  - **Impact:** High. **Frequency:** Frequent on narrow screens. **Effort:**
    M. **Risk:** Medium. **Dependencies:** Shared Layout and Canvas
    read-only breakpoint behavior. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed** again at 390 px with a live
    Playwright MCP browser session against the project's own Vite dev
    server: Dashboard's `<aside>` measured `256px` wide with the main pane
    starting at `x=256` and measuring `134px` wide (no document horizontal
    overflow). Canvas's view switcher measured `352.56px` wide at
    `x=50.72–403.28` against a `326px` main pane (`x=64–390`) and a `390px`
    viewport: the switcher's right edge extends `13.28px` beyond both,
    and the Inference button (`x=289.14–399.28`) overlaps the Read-only
    toggle (`x=298–334`) and the Notifications bell (`x=342–378`). This
    strengthens the finding's existing evidence of clipping/overlap at
    390 px; the finding is not resolved.
  - **Delta:** No material change.
### Terminology and Visual Hierarchy

### Perceived Performance

## Journey Findings

### Canvas

- **Inferred:** `Navbar.tsx` and `MainLayout.tsx` keep Canvas as the default
  shell view (`activeView === 'canvas'`), with Sidebar, Toolbar,
  `FlowCanvas`, `RestoreSessionBanner`, `ResultsPanel`, and `PropertiesPanel`
  rendered inside the canvas branch. The read-only chip only appears while the
  canvas view is active, and the canvas branch is hidden with `display:
  contents` rather than unmounted.

#### Canvas audit evidence and limits

- **Observed:** In the local Chrome walkthrough, the empty canvas offered a
  sidebar palette, an empty-state template CTA, a command-palette control,
  disabled undo/redo/clear controls, and no Run Preview action. Clicking the
  Dataset palette card inserted an invalid Dataset node but did not select it;
  selecting the card a second time opened its Properties panel. After choosing
  a local `test source`, the panel reported “No schema available.” Clicking
  Encoding then placed its card only 30 px from the Dataset card, underneath
  it; the Dataset card intercepted the attempted pointer click on Encoding.
- **Measured:** The Chrome fallback command ran
  `e2e/preview.spec.ts` successfully (1 passed). It seeds a Dataset → Drop
  Columns graph and mocks `/api/pipeline/preview`, so it verifies the real
  toolbar gate, converter, request, and Results panel rendering, but not
  palette creation, real source schema, connection gestures, or a backend
  completion.
- **Evidence limit:** The local source list was available, but its selected
  `test source` did not return a schema and logged four browser console errors.
  A complete real-data pipeline, connection, training, server-version restore,
  and job-failure walkthrough was therefore not claimed. The findings below
  distinguish this limit from source-supported behavior.

- **Observed — responsive and keyboard verification:** A Chrome DOM-geometry
  walkthrough covered the requested widths. All widths had a document
  `scrollWidth` equal to the viewport width; “reachable” below means a visible
  non-disabled control's rectangle stayed within that viewport. The snapshots
  and measurement are Canvas-specific unless marked `FND-001` or `FND-002`.

  | Width | Toolbar, panels, and canvas result | Keyboard/focus and shortcut result | Limitation owner |
  |-------|------------------------------------|------------------------------------|------------------|
  | 1440 px | Components (256 px), Flow canvas (800 px), and Properties (320 px) were present and the document did not overflow. The two absolute toolbar clusters collided: Undo at `528,72 40×40` was covered by Templates at `492,72 118×38`; a real pointer click was intercepted by Templates. | Tab exposed visible focus for normal toolbar controls; `⌘Z` changed the restored graph from two nodes to zero and `⌘⇧Z` restored two. `⌘K` opened the Command palette and focused its search input. | **Skyulf-controlled (`CAN-005`)**: `Toolbar` positioning responds to its Canvas pane, not React Flow. |
  | 1024 px | Components (256 px) and a 703 px Flow canvas remained reachable; Properties was collapsed. No measured off-screen enabled control or document overflow. | Toolbar and Canvas control-panel buttons remained tabbable with visible focus. | **Skyulf-controlled responsive layout; no observed limitation.** |
  | 768 px | Both side panels collapsed, leaving a 704 px Flow canvas. Keyboard Shortcuts stayed visible; More canvas tools was in bounds at `710,72 42×34`. No document overflow or measured off-screen enabled control. | The More menu exposed Jobs, performance overlay, and disabled export controls. `⌘K` remains the insertion route after its icon control hides. | **Skyulf-controlled responsive layout; no observed limitation.** |
  | 390 px | Both panels remained collapsed and the 326 px Flow canvas, shortcut control (`128,72 40×40`), More control (`332,72 42×34`), and Restore/Discard banner were reachable without document overflow. The shell Inference switcher ended at x=399, outside the 390 px viewport. | The shortcut sheet fit at `16,219 359×406` and listed Undo/Redo, Command palette, Run Preview, and Escape. Opening it left focus on its invoker; Tab reached covered More tools, Flow canvas, then Zoom In/Out. | **Skyulf-controlled:** shell clipping remains `FND-001`; overlay focus containment remains `FND-002`. Canvas panel collapse itself had no observed clipping. |

- **Observed — Canvas versus React Flow boundary:** React Flow's focusable
  `application` canvas and its Zoom In/Out, Fit View, and interactivity
  controls remained reachable at all four widths; no library-owned viewport
  clipping was observed. React Flow owns graph panning, node focus/movement,
  handles, and the control panel. Skyulf owns the Sidebar palette, Toolbar,
  panel breakpoints, shortcut overlay, and custom node cards. In the 1440 px
  Tab sequence, Sidebar Search was followed by one focusable palette scroll
  container rather than an individual Dataset/Encoding card; those cards are
  pointer click/drag affordances. The Skyulf command-palette alternative is
  keyboard-operable (`⌘K`, focused search, Arrow navigation, Enter insertion,
  Escape close), but its discoverability is only the icon's title and the
  shortcut sheet. This keyboard insertion limitation is retained under
  `CAN-001`, not attributed to React Flow.

- **Observed/Measured — diagnosis and recovery exercised within mock limits:**
  an empty Canvas exposed disabled Undo/Redo/Clear and no Run Preview; the
  palette-added Dataset displayed “Configuration issue: Dataset is required.”
  The autosave banner showed “Restore previous session?” with Restore, Discard,
  timestamp, and node count; selecting Restore recreated the two stored cards.
  Keyboard Undo/Redo changed the graph `2 → 0 → 2`, while the overlapping
  1440 px toolbar prevented the equivalent Undo pointer click. The Chrome
  mock in `e2e/preview.spec.ts` seeded Dataset → Drop Columns, intercepted
  `POST /api/pipeline/preview`, and rendered `setosa`/`virginica` in
  `ResultsPanel` (1 passing test). The selected local source still returned no
  schema, so missing-config warning navigation, leakage selection, real
  backend failure, and server-version loading were not represented as live
  outcomes. The earlier Baseline note about a one-off preview-button
  instability is superseded by the later targeted preview reruns and the final
  Task 8 system-Chrome `12/12` pass, so it stays historical rather than
  becoming a Canvas finding.

- **Inferred — unavailable diagnosis/recovery paths, with exact support:**
  `useRunControls.ts:35-82` enables Preview from a dataset plus outgoing edge,
  calls the leakage toast guard, and on backend failure only toasts “Check
  console for details”; it does not call `useGraphStore.ts:454-490`
  `validateGraph`, which only warns to the console and returns `false`.
  `RestoreSessionBanner.tsx:26-50` probes one local snapshot once for an empty
  graph; `canvasPersistence.ts:24-61` silently absorbs storage errors and
  returns `null` for corrupt/version-mismatched data. The passing targeted
  tests are `core/hooks/useCanvasAutoSave.test.ts` (four cases: empty,
  delayed/coalesced, unmount flush) and `core/utils/canvasPersistence.test.ts`
  (six cases: round-trip, missing, corrupt, mismatched, malformed, clear).
  They support the persistence mechanics, not an explainable UI for unavailable
  recovery. Backend-dependent run failure and server-version restore therefore
  remain **Inferred** only.

#### Representative node-form comparison

All eight required representatives were inspected at their settings render
paths. The local schema failure prevented a representative live column
configuration, so this is **Inferred** source evidence, except the Dataset
form's observed no-schema state above. Their visible `span` labels beside
unassociated native controls remain the shared `FND-005` issue; no additional
Canvas finding is warranted solely for this repeated form root.

| Representative | Labels, defaults, help, and validation | Column/advanced behavior | Apply/reset outcome |
|----------------|-----------------------------------------|--------------------------|---------------------|
| DatasetNode | Visible Select Dataset text; default `datasetId: ''`; validation says Dataset is required. New Upload and schema/loading/no-schema feedback are present. | Dataset dropdown and schema table; no advanced section. | Selection writes immediately through `onChange`; no Apply or reset. |
| EncodingNode | Default is one-hot with empty columns; conditional method controls include inline help/default text. Validator requires a column except Label/Ordinal and requires a binary target for WOE. | `ColumnMultiSelect` is labelled Columns to Encode; method-specific options appear conditionally. | Immediate `onChange`; no Apply or reset. |
| FeatureGenerationNode | Starts with no operations and validates Add at least one operation; operation cards expose contextual help and defaults. | Per-operation compact column selectors, date-feature checkboxes, and output name; recommendations are conditional. | Immediate edits; no reset. Its exposed recommendation Apply handler is empty (`CAN-004`). |
| FeatureSelectionNode | Defaults to Select K Best, `k: 10`; target-required validation is conditional; explanatory help is present for parameters. | Auto-detected/selectable target and conditional scoring/estimator/threshold controls; responsive two-column parameters are an advanced layout, not a reset path. | Immediate `onChange`; no Apply or reset. |
| TrainingSettings | Basic/Advanced (Tuning) mode, model/target guidance, scaling notice, dynamic hyperparameter/search-space defaults, and conditional CV help. | Upstream columns/target and registry/API definitions control selectors and advanced sections. | Changes are immediate; Start Training submits; no form Apply/reset. |
| EnsembleSettings | Classification/Regression, Voting/Stacking, Basic/Advanced mode, estimator count, scaling advice, CV, and tuning help are conditional. | Base estimators, auto-synced upstream model/target/CV state, weights, parallelism, calibration, and advanced tuning are available. | Changes are immediate; Start Ensemble Training/Modeling is the execution action, not Apply/reset. |
| SegmentationSettings | Model configuration, optional reference column help, scaling notice, model/hyperparameter loading states, and two responsive tabs are present. | Upstream dataset/schema columns and backend clustering/hyperparameter definitions gate selectors and advanced parameters. | Changes are immediate; Start Segmentation is disabled without upstream data; no Apply/reset. |
| DataPreviewNode | Empty default config and always-valid validator; it is an inspection sink rather than a config form. | Result/branch/tab controls derive from execution output and incoming branches, not editable source columns. | No form Apply/reset; local backend-less walkthrough did not yield preview content. |

- **CAN-001 — Observed: click-to-add places full-size cards on top of each
  other and does not take the user to configuration.**
  - **Evidence:** `Sidebar.tsx` increments click-to-add placement by only
    30 px from `(100, 100)` while `CustomNodeWrapper.tsx` renders a
    `min-w-[200px]` card. In the running Canvas, click-adding Dataset then
    Encoding visibly overlapped the two cards; the Dataset card intercepted
    the attempt to select Encoding. The first click-added Dataset was not
    selected, so its required setting was unavailable until a second click on
    the new card opened Properties. The palette's individual cards were not
    individual Tab stops; `⌘K` is the keyboard insertion fallback.
  - **User problem:** A user who uses the advertised click alternative to
    drag-and-drop can immediately lose the newly added node under an existing
    card and must discover a second selection step before configuration. This
    turns the first pipeline into canvas rearrangement rather than a
    Dataset → transform workflow.
  - **Affected surfaces:** Canvas Components sidebar; `useGraphStore.addNode`;
    `FlowCanvas`; `CustomNodeWrapper`; Properties panel.
  - **Why Canvas-specific:** This is the Canvas graph placement and selection
    contract, not a shared-shell issue.
  - **Proposed behavior:** Place click-added nodes at a visible non-overlapping
    position (or intelligently cascade from existing bounds), select and bring
    the new node into view, and open its configuration when required fields
    are incomplete.
  - **Acceptance criteria:** Two consecutive click-added cards have
    non-overlapping hit targets; each new node is visible, selected, and
    keyboard-reachable; an invalid new node exposes its required setting
    without a second pointer action; drag-and-drop and command-palette
    insertion retain predictable placement.
  - **Validation method:** Playwright adds Dataset, Encoding,
    Feature Selection, Split, Training, and Data Preview by palette click,
    drag/drop, and command palette; assert non-overlap, selection, Properties
    visibility, and keyboard focus at 1440, 1024, 768, and 390 px. Complete
    the same sequence with a mocked usable dataset.
  - **Impact:** High. **Frequency:** Frequent for click-to-add workflows.
    **Effort:** S. **Risk:** Low. **Dependencies:** custom-node bounds,
    Sidebar placement, and Properties panel selection. **Milestone:** Now.
  - **2026-08-07 status:** Changed.
  - **Current evidence:** **Observed** again live at 1440 px: click-adding
    Dataset left it unselected until a second click, and click-adding
    Encoding 60 px from Dataset reproduced the overlap; attempting to click
    the Encoding card directly failed after a 5 s Playwright actionability
    timeout with an explicit log confirming the Dataset node's subtree
    "intercepts pointer events" over Encoding — a directly reproduced live
    failure, not only the original's static geometry inference (screenshot:
    `.superpowers/sdd/task-3-1440-overlap.png`, not committed). Source
    re-reading confirms `Sidebar.tsx`'s 30 px cascading placement,
    `CustomNodeWrapper.tsx`'s `min-w-[200px]` card, and `useGraphStore.ts`'s
    `addNode()` are unchanged. I did not separately repeat Split or Training
    click-adds after Dataset, Encoding, and Feature Generation independently
    reproduced the shared `addNode()` selection failure; those nodes use the
    same insertion path, so this is not separate successful coverage.
  - **Delta:** Broadened root-cause scope: `addNode()` never sets
    `selected: true` regardless of insertion method (click-to-add **or**
    native drag-and-drop both call `addNode()`), not only for click-to-add
    as originally described. User problem, proposed behavior, and
    acceptance criteria are unchanged since both insertion paths already
    fell under this finding's "drag-and-drop and command-palette insertion
    retain predictable placement" acceptance criterion.

- **CAN-005 — Observed: Canvas toolbar clusters overlap and intercept actions
  when Properties narrows the 1440 px Flow pane.**
  - **Evidence:** At 1440 px, Components (256 px) + Flow canvas (800 px) +
    Properties (320 px) fit the viewport without document overflow, but the
    Toolbar's two absolute clusters do not share their occupied width. The
    Undo button was measured at `528,72 40×40`; the right-cluster Templates
    control was measured at `492,72 118×38`. Chrome's real pointer click on
    Undo timed out because Templates intercepted it. `Toolbar.tsx:202-268`
    independently positions the left cluster from `left-4` and the right
    cluster from `right-4`, only constraining the latter to
    `max-w-[calc(100%-13rem)]`; `xl` retains both clusters. The same restored
    graph did respond to `⌘Z` and `⌘⇧Z`, proving history exists but the visible
    pointer control is occluded.
  - **User problem:** Opening a node's Properties panel can make visible,
    enabled actions such as Undo appear available while another toolbar action
    receives the click. Users cannot reliably recover a graph mistake by
    pointer at a common desktop width.
  - **Affected surfaces:** Canvas Toolbar, Flow-canvas viewport, Properties
    panel, Undo/Redo/Clear, and secondary toolbar actions.
  - **Why Canvas-specific:** This is Skyulf's Canvas toolbar/panel width
    allocation, not a React Flow viewport/control limitation and not the
    shared-shell clipping in `FND-001`.
  - **Proposed behavior:** Reserve non-overlapping space for both toolbar
    clusters based on the live Canvas pane width; collapse secondary actions
    before collision and preserve a labelled keyboard-operable overflow menu.
  - **Acceptance criteria:** At 1440, 1024, 768, and 390 px with each panel
    state, every visible enabled toolbar target has a non-overlapping hit
    rectangle; pointer activation calls its own action exactly once; secondary
    actions remain reachable through an accessible overflow menu; Undo and
    Redo continue to work by pointer and keyboard.
  - **Validation method:** Playwright opens/closes Components and Properties,
    measures toolbar hit-target intersection, clicks Undo/Redo/Load/Save and
    overflow actions, and checks `⌘Z`/`⌘⇧Z` graph state at all four widths.
    Include a visual screenshot assertion only for the two-cluster desktop
    state.
  - **Impact:** High. **Frequency:** Frequent when configuring an existing
    graph. **Effort:** S. **Risk:** Low. **Dependencies:** `Toolbar`,
    responsive panel state, and z-index/positioning rules. **Milestone:** Now.
  - **2026-08-07 status:** Changed.
  - **Current evidence:** **Observed** again live at 1440 px: clicking
    "Clear canvas" failed after a 5 s Playwright actionability timeout
    because "Templates" intercepted the pointer event. Measured rectangles:
    Templates `x=492.3–610.5`; Undo `x=528–568` (fully inside); Clear canvas
    `x=576–616` (overlaps by ~34 px, `x=576–610`) — screenshot:
    `.superpowers/sdd/task-3-1440-toolbar-collision.png` (not committed).
    `⌘Z`/`⌘⇧Z` against a real click-added node still worked (`1 → 0 → 1`),
    confirming history remains intact while the pointer path stays occluded.
    At 1024 px the right cluster collapses into one "More canvas tools"
    button, removing the collision entirely; 768 px and 390 px are unchanged
    from the original table. Source re-reading confirms `Toolbar.tsx`'s
    independent `left-4`/`right-4` absolute clusters are unchanged.
  - **Delta:** Broadened affected surface: **both** Undo and Clear canvas
    are occluded by Templates at 1440 px, not only Undo as originally
    described. User problem, proposed behavior, and acceptance criteria are
    unchanged, since the acceptance criteria already require every visible
    enabled toolbar target (not just Undo) to have a non-overlapping hit
    rectangle.

- **CAN-002 — Inferred: run readiness and diagnosis do not form an actionable
  validation loop.**
  - **Evidence:** `useRunControls.ts` enables Run Preview when one Dataset has
    an ID and an outgoing edge; it does not call the store's
    `validateGraph`. `useGraphStore.ts` returns only `false` and logs missing
    configuration or disconnected-node messages to the console. The node
    wrapper renders validation and failed-run chips as non-actionable status
    spans, while preview failures toast only “Check console for details.”
    Leakage is more specific—it blocks before submission with a toast—but does
    not select or navigate to the cited nodes. This is code evidence; the
    unavailable local schema prevented a real end-to-end invalid run.
  - **User problem:** A user can see a disabled or failed outcome without a
    reliable path from Run Preview to every affected node and setting. Console
    text and tooltip-only chips are especially poor recovery paths in a
    crowded graph.
  - **Affected surfaces:** Run Preview and Run All controls; node validation
    chips; leakage guard; `ResultsPanel`; graph store; Properties panel.
  - **Why Canvas-specific:** The missing handoff is between graph validation,
    node location, and Canvas execution controls. It should consume
    **FND-003** for any shared toast/status announcement semantics rather than
    duplicate that finding.
  - **Proposed behavior:** Before a run, produce one structured Canvas
    validation summary that names each node by label, classifies
    configuration/connection/leakage errors, focuses the first issue, and
    lets the user step through all issues. Preserve a failed run's node-level
    error and link it to the same recovery surface.
  - **Acceptance criteria:** Run Preview never submits an invalid graph;
    every detected issue names a node and a next action; selecting an issue
    selects, pans to, and opens the node's Properties panel; leakage messages
    identify both the preprocessing and splitter nodes; a backend failure
    remains inspectable after its toast disappears.
  - **Validation method:** Component tests cover missing Dataset settings,
    disconnected transform/training nodes, invalid column settings, and
    leakage. Playwright creates each invalid graph, invokes run by button and
    Ctrl/Cmd+Enter, verifies no request is sent, then fixes it through the
    issue list. Mock a node-specific backend failure and assert durable
    Results/Canvas recovery navigation; run axe on the summary.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** Node registry validators, pipeline converter,
    leakage validator, Results panel, and **FND-003**. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** Source re-reading confirms `useRunControls.ts`'s
    `canRunPreview` still requires a Dataset node with a `datasetId` and an
    outgoing edge, still does not call `useGraphStore.ts`'s `validateGraph`,
    and a preview failure still only toasts "Check console for details";
    `validateGraph` still only `console.warn`s and returns `false`. A real
    invalid Run Preview submission could not be independently re-observed
    live this rerun: the local dataset source still returns a `500` on
    `GET /api/pipeline/datasets/{id}/schema` (unchanged evidence limit), and
    creating a real Dataset→transform edge to reach the enabled Run Preview
    state was not achievable through this session's browser automation (see
    the Task 3 Method notes above) — the same connection-gesture limit the
    original audit already recorded. Re-reading
    `frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts`, its four
    call sites, and `pipelineLeakageValidation.test.ts` shows no leakage-
    validation source change; that unchanged source evidence still supports
    CAN-002 staying Confirmed / No material change even without a fresh live
    invalid-run reproduction. This keeps the evidence at its original level
    (source-confirmed, not live-observed end to end).
  - **Delta:** No material change.

  - **2026-08-07 RESOLVED (frontend slice).**
    - `useGraphStore.validateGraph` now returns `GraphValidationIssue[]`
      (`nodeId`, `nodeLabel`, `category`, `message`) via the exported
      `collectGraphValidationIssues`, replacing the `console.warn`-and-return-
      `false` behavior. Both call sites in `useRunControls.ts` (`handleRun`,
      `handleRunAll`) now block submission when issues exist, so an invalid
      graph never reaches the backend.
    - `ResultsPanel` renders the issue list; each entry names the node and a
      next action, and selecting one selects that node — which opens its
      Properties panel automatically, since `PropertiesPanel` keys off
      `nodes.find(n => n.selected)`. Leakage issues name both the
      preprocessing node and the splitter.
    - `lastRunError` keeps a failed preview inspectable after the toast
      disappears.
    - Accessibility: the issue list is deliberately **outside** any live
      region — it recomputes on every graph edit, so an `aria-live` wrapper
      would re-read every issue on each keystroke. A polite `role="status"`
      announces only the count. Verified live in Chrome: the a11y tree shows
      `region "Validation issues"` containing a `status` with
      "2 validation issues blocking preview" and two issue buttons outside it,
      with 0 unhidden icons in any live region.
    - Verified live end to end at 1440 px against a seeded invalid graph
      (Dataset + Encoding, no edge): clicking the "Label Encoder" issue
      selected that node and opened Properties headed "Label Encoder".
    - **Still open:** selecting an issue does **not** pan/zoom the viewport to
      the node (only selects it), so an off-screen node stays off-screen; no
      Playwright coverage for creating each invalid graph and fixing it
      through the list; run-by-Ctrl/Cmd+Enter path not separately tested; no
      axe run on the summary; node-level chips on the canvas itself are still
      non-actionable status spans; the preview-failure toast still reads
      "Check console for details." even though the error is now durably in
      Results.

- **CAN-003 — Inferred: recovery sources are not explainable when Canvas
  autosave cannot be restored.**
  - **Evidence:** `useCanvasAutoSave.ts` writes a single local snapshot every
    second, and `canvasPersistence.ts` silently swallows storage/quota errors
    and returns `null` for corrupt or version-mismatched payloads.
    `RestoreSessionBanner.tsx` probes only once and only with an empty graph.
    Separately, the Toolbar offers server versions, a per-browser Recent
    fallback, and the route accepts a Data Sources version payload; their
    overwrite confirmations are source-specific. There is no Canvas state
    explaining why an expected autosave is unavailable or which recovery
    source is current.
  - **User problem:** After a refresh, storage failure, or incompatible saved
    shape, a user cannot distinguish “nothing was saved,” “the current graph
    suppressed restore,” and “the snapshot cannot be used.” They may start
    over or load the wrong source without understanding whether it is local or
    server-backed.
  - **Affected surfaces:** Autosave/Restore banner; Recent pipelines; server
    version load and Data Sources restore; Canvas toolbar.
  - **Why Canvas-specific:** This is the Canvas graph's local/server recovery
    model. It depends on **FND-003** only for shared status/error announcement
    behavior.
  - **Proposed behavior:** Present one recovery entry point that labels each
    candidate as autosave, local recent, or server version; explains
    availability/expiry/compatibility; and reports recoverable local-storage
    failure without exposing implementation details. Keep the existing
    overwrite protection and never replace a nonempty graph silently.
  - **Acceptance criteria:** A fresh Canvas can identify all available
    recovery sources and their timestamps; a corrupt, stale-schema, disabled,
    or quota-limited autosave yields an understandable non-blocking message;
    loading any source names what will be replaced and leaves the prior graph
    recoverable until confirmation; restore success focuses the restored
    graph.
  - **Validation method:** Unit-test valid, corrupt, mismatched-version, and
    storage-throwing snapshot cases. Playwright seeds local storage and mocked
    server versions, verifies source labels and overwrite/cancel behavior,
    reloads empty and nonempty canvases, and checks keyboard and live-region
    behavior.
  - **Impact:** Medium. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** canvas persistence, recent-pipeline utilities,
    pipeline versions API, and **FND-003**. **Milestone:** Later.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed** live: reloading `/canvas` with a
    previously saved local snapshot surfaced "Restore previous session?"
    with Restore, Discard, a relative timestamp, and a node count; Discard
    cleared the prompt as documented. Source re-reading confirms
    `useCanvasAutoSave.ts`'s one-second single snapshot,
    `canvasPersistence.ts`'s silent error/corrupt/mismatch handling, and
    `RestoreSessionBanner.tsx`'s one-shot empty-graph-only probe are all
    unchanged. Server-version load, Recent-pipeline fallback, and a
    storage-failure/corrupt-payload live reproduction remain **Inferred**
    only, unchanged from the original evidence level.
  - **Delta:** No material change.

- **CAN-004 — Inferred: Feature Generation presents an Apply action that
  silently does nothing.**
  - **Evidence:** `FeatureGenerationNode.tsx` passes
    `handleApplyRecommendation` to `RecommendationsPanel`, but that handler
    is an empty function. `RecommendationsPanel.tsx` consequently renders an
    “Apply Recommendation” button whenever recommendations exist. By
    comparison, the inspected Imputation, Resampling, and Drop Columns nodes
    implement nonempty recommendation-apply handlers. Recommendation data was
    not available from the local source, so this is source evidence rather
    than a claimed live click.
  - **User problem:** A user can choose an explicitly offered action in
    Feature Generation and receive no configuration change, confirmation, or
    explanation. This breaks the otherwise immediate configuration model and
    makes it unsafe to trust recommendations.
  - **Affected surfaces:** Feature Generation settings; Recommendations panel;
    preprocessing recommendation API.
  - **Why Canvas-specific:** This is a node-configuration behavior mismatch.
    It should use **FND-005** for the shared field semantic contract, not
    restate FND-005 as a duplicate finding.
  - **Proposed behavior:** Either apply a recommendation deterministically to
    the appropriate operation/configuration and confirm the change, or mark
    the recommendation informational and omit the Apply action until it is
    supported.
  - **Acceptance criteria:** Every visible Apply action changes the documented
    configuration once, makes the change inspectable and undoable, and
    announces success; unsupported recommendations expose no actionable Apply
    control; applying preserves valid user-entered operation data.
  - **Validation method:** Render Feature Generation with representative
    column recommendations and assert state changes, undo, and accessible
    feedback. Compare the same contract with Encoding, Feature Selection,
    Training, Ensemble, Segmentation, Dataset, and Data Preview configuration
    panels; run keyboard-only and axe checks.
  - **Impact:** Medium. **Frequency:** Occasional when recommendations are
    available. **Effort:** S. **Risk:** Low. **Dependencies:** recommendation
    payload schema, Feature Generation config shape, and **FND-005**.
    **Milestone:** Later.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** Source re-reading confirms
    `FeatureGenerationNode.tsx`'s `handleApplyRecommendation` is still an
    empty function, and Imputation/Resampling/Drop Columns still implement
    non-empty handlers. **Observed** live: a freshly added Feature
    Generation node showed a `Configuration issue: Add at least one
    operation.` chip and empty `Add:` operation-type buttons with no
    `Apply` recommendation control rendered — consistent with
    `RecommendationsPanel` only rendering Apply when recommendations exist,
    which the unchanged local schema limit still prevents from being
    populated live. This matches the original evidence level exactly
    (source-confirmed empty handler; recommendation payload still
    unavailable locally).
  - **Delta:** No material change.

### Data and EDA

- **Baseline entry-point mapping:** `/data` mounts the eager `DataSources`
  page, while `/eda` mounts `EDAPage` through `LazyRoute` / `React.lazy`.
  `EDAPage` owns the `EDASidebar` and analysis tabs, and the shared
  `Layout.tsx` also collapses the shell sidebar on `/eda` (same compact shell
  treatment used for `/canvas`).
- **Baseline entry-point mapping:** The only current route/a11y smoke
  coverage for these entry points is `e2e/routes.spec.ts` and
  `e2e/a11y.spec.ts`, both of which exercise `/data` and `/eda`.

#### Data and EDA audit evidence and limits

- **Observed:** A local Chrome walkthrough at `/data` exposed uploaded, S3,
  pending, preview, Canvas, EDA, CSV, version, and delete controls. The
  `Data Sources` introduction says datasets are used “in experiments,” but
  the completed-row actions use the more specific Canvas and EDA destinations.
  Opening Add Source exposed only the S3 choice and two required inputs. Its
  visible `Name` and `S3 Path` text is not associated with either input in the
  rendered accessibility tree. Opening the first preview produced the generic
  “Failed to load dataset preview.” retry state while its header still
  announced `0 rows`, `0 columns`, and `0 Bytes`.
- **Observed:** The local source endpoints returned a populated, synthetic
  source list, including pending rows. The preview sample/profile requests
  failed and `/api/eda/3/latest` returned `404`; EDA therefore correctly
  showed “No analysis found for this dataset.” The walkthrough did not create
  a source, upload a file, cancel a job, execute an analysis, load a report,
  or download output, so findings about those outcomes remain **Inferred**.
  In particular, these mocks cannot prove real source creation/ingestion
  completion or the depth, correctness, and chart interaction of an EDA
  result.
- **Observed:** At 390 px, Data Sources retained the 256 px shell sidebar,
  leaving a 134 px main pane; its content had a 518 px scroll width, and the
  table/actions extended from x=289 to x=1449. EDA uses the collapsed 64 px
  shell sidebar, but its 326 px header had a 962 px scroll width: Dataset,
  Target Column, Task Type, Analyze, and History ended at x=1026. This is
  journey-specific evidence of the shared compact-shell root cause
  **FND-001**, rather than a second shell finding.
- **Measured:** The ignored audit Chrome configuration
  (`.superpowers/sdd/playwright.chrome.config.ts`) ran
  `e2e/routes.spec.ts` and `e2e/a11y.spec.ts`: 9 passed. The tests mock
  `/api/**` and `/data/api/**`, establish route mounting and no critical axe
  violations, but do not seed a usable source, report, pending/failed EDA job,
  or chart payload. Existing non-blocking serious axe output concerned
  Dashboard color contrast and Canvas scroll-region focusability, not these
  two mocked routes.

- **DAT-001 — Observed: source onboarding does not establish a consistent,
  accessible next step from import to analysis.**
  - **Evidence:** The live introduction promises use “in experiments,” while
    completed rows offer Canvas and EDA; the `handleUseInCanvas` and EDA
    actions navigate to `/canvas?source_id=…` and `/eda?dataset_id=…`.
    Add Source offered only S3 and rendered zero associated `label` elements
    for the required Name and S3 Path inputs. Source creation itself was not
    attempted because the local walkthrough uses seeded/mock data rather than
    a safe real credential/source.
  - **User problem:** A first-time user cannot tell which source types are
    supported, what a successful import enables, or reliably identify the
    required connection fields. “Experiments” also conflicts with the actual
    Canvas/EDA choices.
  - **Affected surfaces:** Data Sources introduction, Add Source, file upload,
    completed dataset row, Canvas handoff, and EDA handoff.
  - **Proposed behavior:** Make the source-type decision explicit (including
    unavailable types), associate every Add Source field with a route-specific
    label and inline validation now, and end successful ingestion with
    contextual Canvas and EDA next-step cards that name the selected dataset.
    `FND-005` can later normalize the shared primitive layer without blocking
    this Data Sources slice.
  - **Acceptance criteria:** Every source form control has an accessible name,
    required/invalid state, and linked error; supported source types and
    credential expectations are explained before submission; a completed
    source has unambiguous Canvas and EDA actions whose destination preserves
    that source; copy does not name a destination that is not offered.
  - **Validation method:** Component accessibility tests cover required,
    malformed, API-failure, and successful S3/file source paths. Playwright
    mocks each outcome, follows Canvas and EDA handoffs, and checks selected
    source context at 1440 and 390 px; run axe on both modals.
  - **Impact:** High. **Frequency:** Frequent for new sources. **Effort:** M.
    **Risk:** Medium. **Dependencies:** source-type capability contract,
    ingestion API errors, and router handoff state. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live, real backend)** again at 1440 px:
    Add Source still offers only S3, and the Name/S3 Path `<input>`s still
    have no `id`, `aria-label`, or `aria-labelledby`
    (`document.querySelectorAll('[role="dialog"] input')` — all three
    attributes null on both fields); submitting the empty form is blocked
    purely by native HTML5 `required`/`valueMissing` validation. **New:**
    submitted a real (nonexistent) S3 path and it created an actual backend
    source (`POST /data/api/sources` succeeded) that asynchronously reached
    `status: failed`; the Data Sources row (`getStatusBadge`,
    `DataSources.tsx` lines 132–155) rendered only a bare "Failed" badge with
    a single "Delete dataset" action — no retry, no error text, no way to see
    *why* it failed from the table itself. The real backend error
    ("Failed to connect to S3 path s3://nonexistent-bucket-uxaudit/data.csv")
    is visible only inside the Ingestion Jobs modal (see `DAT-003`).
    Confirmed, positive: `DataSources.tsx` correctly passes `?source_id=` to
    `/canvas` (line 120) and `?dataset_id=` to `/eda` (line 422), and both
    `CanvasPage.tsx` and `EDAPage.tsx` (lines 54–61) consume these on first
    mount — the onboarding→handoff context is preserved for the very first
    navigation.
  - **Delta:** No contradiction. Adds a real, backend-verified reproduction
    of the failed-creation-with-no-feedback path, and confirms the
    Canvas/EDA deep-link handoff itself works correctly on first navigation —
    narrowing (not resolving) the acceptance-criteria gap specifically to
    in-app return navigation without a fresh query param, cross-referenced
    under `DAT-005`.

- **DAT-002 — Observed: failed dataset preview presents fabricated-looking
  zero metadata and insufficient recovery context.**
  - **Evidence:** In local Chrome, opening Test Dataset's preview failed both
    sample/profile calls and displayed “Failed to load dataset preview.” with
    Retry, while the modal header continued to show `0 rows`, `0 columns`, and
    `0 Bytes`. `DatasetPreviewModal` falls back to `0` for missing profile and
    dataset metadata and reduces non-404 errors to that generic text.
  - **User problem:** Users cannot distinguish an empty dataset from metadata
    that was never loaded, nor determine whether retrying, waiting for
    ingestion, returning to the source, or contacting an owner is appropriate.
  - **Affected surfaces:** Data Sources Preview action; DatasetPreviewModal
    sample/statistics tabs; profile and sample endpoints.
  - **Proposed behavior:** Keep metadata unknown until confirmed, state which
    preview part failed, show the source/job status and last successful
    metadata where available, and offer the appropriate retry or return-to-job
    action.
  - **Acceptance criteria:** A failed preview never labels unavailable counts
    as zero; empty sample, empty schema, deleted source, and transient
    sample/profile failures have distinct user-facing explanations; Retry
    refreshes both resources once and preserves the selected tab; the modal
    identifies the dataset throughout recovery.
  - **Validation method:** Mock successful, empty, 404, sample-only failure,
    profile-only failure, and retry success responses; assert header values,
    actions, focus, and status/alert semantics at 1440 and 390 px. Exercise
    horizontal table overflow with a wide schema.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** dataset sample/profile endpoint error shape and
    **FND-003**. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live, real backend)** again at
    1440 px: clicking "Preview" on a broken dataset showed "0 rows • 0
    columns • 0 Bytes" in the modal header and "Failed to load dataset
    preview." with Retry in the body. Network inspection of the underlying
    request showed the real backend response was a `400 Bad Request` with
    body `{"error":"HTTP 400","message":"Invalid file path"}` — a specific,
    actionable message the UI discards in favor of the generic string. The
    error paragraph still has no `role`/`aria-live` attribute.
  - **Delta:** No material change to the problem, surfaces, or acceptance
    criteria; this rerun reproduces the behavior against a real backend
    `400` with a real message payload (upgrading the citation from a mocked
    to a real reproduction) and reconfirms the missing `role`/`aria-live`
    observation.

  - **2026-08-07 RESOLVED (frontend slice).**
    - `DatasetPreviewModal` no longer coerces missing metadata to `0`. Row,
      column, and size each render an `aria-label="Unknown"` em-dash when the
      value is genuinely unavailable, while a real `0` from a successful
      profile still renders as `0`. Backend JSON `null` is normalized so the
      distinction survives without crashing `toLocaleString`.
    - `DatasetService.getSample`/`getProfile` now parse the error body and
      propagate the real backend `message`/`detail`/`error` into
      `DatasetApiError`, so "Invalid file path" reaches the user instead of
      the generic string.
    - The two requests use `Promise.allSettled`, so sample and profile
      failures are reported independently and a 404 reads as a
      deleted/missing source rather than a transient error. Retry refreshes
      both and preserves the selected tab.
    - Accessibility: the modal now uses the shared `ErrorState`/`LoadingState`/
      `EmptyState` primitives, so the error is a `role="alert"` with the retry
      button associated via `aria-describedby` — closing the missing
      `role`/`aria-live` observation.
    - **Still open:** the modal does **not** show source/job status or
      last-successful metadata alongside the failure; wide-schema horizontal
      table overflow was not exercised; 390 px behavior was not measured; no
      Playwright coverage. Verified by component tests only — not re-observed
      live against the real `400`.

- **DAT-003 — Inferred: ingestion states expose activity but not a complete,
  recoverable lifecycle.**
  - **Evidence:** `DataSources.tsx` polls pending/processing rows every five
    seconds and renders only “Pending...” or “Processing...” plus a spinner;
    the row and Ingestion Jobs modal offer cancellation. `IngestionJobsModal`
    derives “jobs” from every dataset and only shows its generic completion
    message for failed/cancelled cards. `DatasetService.uploadWithProgress`
    can report transfer percentage, but that progress is not represented in
    this page's ingestion list; failed sources have no retry/reconfigure
    route.
  - **User problem:** A user cannot see whether a delay is upload,
    queueing, parsing, profiling, or a stalled job; after a failure, the
    available next step is unclear and past job information is indistinguishable
    from the source inventory.
  - **Affected surfaces:** file upload, Add Source creation, dataset status
    badges, Ingestion Jobs modal, cancel confirmation, and failed source rows.
  - **Proposed behavior:** Use one job model that names lifecycle phase,
    elapsed/updated time, determinate progress when available, source context,
    error detail, and safe Retry/Reconfigure actions; retain completed jobs as
    history rather than presenting all sources as jobs.
  - **Acceptance criteria:** Every pending, processing, succeeded, failed, and
    cancelled ingestion has a named state and next action; progress does not
    imply completion before parsing/profile work finishes; failure explains
    whether credentials, format, connection, or a transient service caused it;
    cancel/retry are idempotent and update the originating row and history.
  - **Validation method:** Mock upload transfer events and each ingestion
    status/error transition; verify polling start/stop, cancel, retry,
    reconfiguration, persistence after navigation, duplicate-submit
    prevention, and live-region announcements at 1440 and 390 px.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** ingestion status/progress/error contract,
    upload mutation, job-history API, and **FND-003**. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (source):** `IngestionJobsModal.tsx`
    lines 41–50 still contain the literal comment `// For now, we show all
    as 'jobs'` and build the job list via `datasets.map(...)` — every
    dataset ever created (including long-completed ones) appears in this
    list, not just active/queued ingestions. **New nuance (source + live):**
    lines 88–96 show the modal DOES surface the real backend `job.message`
    string for failed/cancelled jobs; opening the modal after the real
    S3-failure reproduction under `DAT-001` confirmed the full error text
    ("Failed to connect to S3 path s3://nonexistent-bucket-uxaudit/data.csv")
    is indeed shown there — this modal is actually **more** informative than
    the Data Sources row, which shows nothing. Only pending/processing jobs
    get a Cancel action (icon-only, `title="Cancel Ingestion"`, no
    `aria-label`), and there is no Retry action anywhere in this modal or on
    the Data Sources row for failed jobs.
  - **Delta:** Adds the positive nuance that the modal shows real error text
    (only the Data Sources row itself is silent) and confirms there is
    genuinely no Retry action anywhere. User problem, proposed behavior, and
    acceptance criteria are unchanged; the eventual fix direction should
    surface `job.message` on the Data Sources row badge too, and add Retry.

  - **2026-08-07 RESOLVED (partial — frontend slice only).**
    - `IngestionJobsModal` no longer maps the whole dataset inventory to
      "jobs"; active ingestions and completed history are separated, and the
      `// For now, we show all as 'jobs'` comment is gone with the behavior.
    - The Data Sources row now surfaces `job.message` for failed/cancelled
      sources — previously only the modal had this text — and the status badge
      names the lifecycle phase ("Processing ingestion" / "Queued for
      ingestion") instead of "Processing..." / "Pending...".
    - Failed sources get a Retry action, and the icon-only Cancel button now
      has a real `aria-label`; decorative icons are `aria-hidden`.
    - Accessibility: the persistent row failure text is deliberately **not** a
      live region. The page polls every 5 s and there can be many failed rows,
      so `role="alert"` there would fire a burst of assertive announcements on
      arrival; the status badge already names the phase.
    - **Still open — needs backend work:** there is **no** retry/re-ingest
      endpoint, so Retry only routes the user back to the upload/add-source
      form rather than genuinely re-running ingestion. There is **no**
      job-history API, so "history" is still derived from the dataset list
      rather than being real job records. Determinate upload progress exists
      in `FileUpload.tsx` (XHR) but is still not represented in this page's
      ingestion list. Phase granularity (upload vs queue vs parse vs profile)
      is not available from the current status contract, so failures still do
      not distinguish credentials/format/connection/transient causes. No 390 px
      measurement and no Playwright coverage.

- **DAT-004 — Observed: Data and EDA controls become off-screen at narrow
  widths, blocking the journey before a source can be used or analyzed.**
  - **Evidence:** The 390 px Chrome measurements above placed Data Sources'
    header actions, filters, status chips, and table actions outside its 134 px
    main pane. EDA collapsed the global sidebar but retained a 962 px header
    in a 326 px pane, leaving all analysis controls and History to the right
    of the viewport. The document itself did not horizontally scroll because
    the overflow is clipped within route containers.
  - **User problem:** A narrow-width user can arrive at Data or EDA but cannot
    discover or activate essential import, selection, configuration, analysis,
    or history controls without an unavailable horizontal route scroll.
  - **Affected surfaces:** Data Sources header/filter/table actions; EDA
    dataset/target/task/analyze/history header; compact shared Layout.
  - **Proposed behavior:** Consume the compact shell contract in **FND-001**:
    replace Data's persistent shell/sidebar footprint with a drawer and give
    Data/EDA page controls an intentional compact hierarchy (stacked primary
    action, labelled overflow, and responsive table/card alternative).
  - **Acceptance criteria:** At 390 px, every visible enabled Data/EDA
    control has an in-viewport hit target; import, source filtering, preview,
    Canvas/EDA handoff, dataset selection, Analyze, and History remain
    reachable without clipped content; desktop data density is retained at
    768 px and above.
  - **Validation method:** Playwright measures route-container and control
    rectangles at 1440, 1024, 768, and 390 px with populated Data rows and EDA
    report/history fixtures; click every compact overflow action and complete
    keyboard traversal. Include light and dark screenshots.
  - **Impact:** High. **Frequency:** Frequent on narrow screens. **Effort:**
    M. **Risk:** Medium. **Dependencies:** **FND-001**, Data table/card
    presentation, EDA header, and responsive action menu. **Milestone:** Now.
  - **2026-08-07 status:** Changed.
  - **Current evidence:** **Observed** again at 390 px: Data Sources' `aside`
    measured 256 px with `main` at `x=256`/`width=134px` (no page-level
    overflow), and its table's own `.overflow-x-auto` wrapper measured a
    `clientWidth` well below that 134 px pane against a `scrollWidth` above
    1100 px — a nested-container squeeze beyond the single outer-container
    width previously documented. EDA's `aside`=64 px (collapsed) with
    `header` `clientWidth=326px` vs `scrollWidth=962px` (no page-level
    overflow), matching the original evidence almost exactly. **New,
    width-boundary-correcting measurements not in the original evidence:** at
    1024 px, Data Sources shows no clipping, and EDA's header narrows to a
    2 px overflow (`clientWidth=960px` vs `scrollWidth=962px`) that is
    effectively resolved; at 768 px, Data Sources shows no page-level
    overflow (its table's own self-contained horizontal-scroll region is a
    normal responsive-table pattern, not a clipping regression), but EDA's
    header overflow **reappears** at a real 258 px (`clientWidth=704px` vs
    `scrollWidth=962px`) — the EDA header clipping is not confined to
    390 px; it persists at 768 px and only clears at ≥1024 px.
  - **Delta:** Strengthens the 390 px evidence with a nested-container
    detail, and corrects the width framing: this rerun's per-width
    measurements show Data Sources and EDA now have measurably different
    breakpoint behavior — Data Sources fails materially only at 390 px,
    while EDA's header regression spans 390–768 px and clears only at
    ≥1024 px, rather than a single "narrow screens" descriptor for both
    surfaces. The Validation Matrix should record pass/fail per surface at
    each of 1440/1024/768/390 px rather than one shared per-width verdict.
  - **2026-08-07 resolution:** Fixed and re-measured per surface. Data Sources'
    header action row (`DataSources.tsx`) was a non-wrapping
    `flex justify-between`, pushing `Ingestion Jobs`/`Add Source`/`Upload File`
    to `right=508px` inside a 382 px pane; it now stacks (`flex-col` →
    `sm:flex-row`) with a wrapping button row, bringing `main` `scrollWidth`
    from 508 px to 382 px (= `clientWidth`). EDA's header (`EDAPage.tsx`) was a
    fixed `h-16` three-column flex with `overflow-x: visible`
    (`clientWidth=390` vs `scrollWidth=932`); it now wraps and grows below
    `lg`, keeping the single-row 64 px layout at ≥1024 px. Verified in Chrome
    at all four widths with populated rows and an existing EDA report: zero
    unreachable controls at 390 px and 768 px on both surfaces, header height
    back to 64 px at 1024 px and 1440 px. The table and filter-chip rows retain
    their own horizontal scroll, which the finding already classified as a
    normal responsive pattern rather than a regression.
  - **Measurement caveat:** the long-running dev server on port 5173 was
    serving a stale transformed module and initially masked the EDA fix; these
    measurements were taken against a freshly started server after clearing
    `node_modules/.vite`.

- **DAT-005 — Inferred: EDA analysis selection, job progress, failure, and
  history do not form a durable, contextual analysis loop.**
  - **Evidence:** `EDAPage.tsx` auto-selects the first usable dataset, treats
    `404` latest-report responses as the generic no-analysis state, polls only
    a `PENDING` report every three seconds, and renders “Analysis in
    progress...” without phase, job identity, target/task/filter summary, or
    main-screen cancel action. The `FAILED` state exposes a retry, while
    cancellation and a fuller target/task record live separately in
    `JobsHistoryModal`; `analyzeMutation` has no page-level error rendering.
    The local `404` was reproduced, but a real EDA run/result/failure was not,
    so this lifecycle finding is Inferred.
  - **User problem:** Users can lose track of which dataset and analytical
    choices produced a pending, failed, saved, or loaded report, and cannot
    confidently compare a rerun with history or recover from a submission
    failure.
  - **Affected surfaces:** EDA dataset selector, target/task controls,
    no-analysis form, Analyze/Re-Run, pending/failed panels, recent targets,
    Analysis History, and Data Sources → EDA handoff.
  - **Proposed behavior:** Treat an analysis as a named job with a persistent
    input summary (dataset, target, task, filters, exclusions), phase/progress,
    cancel/retry, completed timestamp, and explicit “currently viewing”
    context that carries through history load and return navigation.
  - **Acceptance criteria:** EDA never silently changes the selected dataset;
    every submission either creates a visible job or shows a recoverable
    request error; pending/failed/cancelled/completed states identify their
    inputs and next action; loading history visibly changes the current-report
    context without overwriting draft choices; retry uses the stated inputs.
  - **Validation method:** Mock no-report, submit rejection, pending,
    completion, failure, cancellation, stale history, and history-load
    responses. Test Data Sources handoff, dataset switch, target/task/filter/
    exclusion changes, Back navigation, polling, cancel/retry, and screen
    reader status/alert output at desktop and 390 px.
  - **Impact:** High. **Frequency:** Frequent for analysis work. **Effort:**
    M. **Risk:** Medium. **Dependencies:** EDA job/report/history API
    contract, React Query invalidation, `useEDAStore`, and **FND-003**.
    **Milestone:** Now.
  - **2026-08-07 status:** Changed.
  - **Current evidence:** **Observed (live, real backend):** selected
    dataset id 9 (a real, completed report from a real end-to-end EDA run),
    navigated `/data` → `/eda` via the sidebar's EDA link (confirmed
    client-side SPA navigation, not a full reload) — the Dataset dropdown
    silently **reset to the first dataset** (id 3) every time, discarding
    the dataset-9 selection and its report view. **Root cause fully
    confirmed (source):** `EDAPage.tsx` seeds `selectedDataset` from
    `?dataset_id=` only once on mount and only if the store's
    `selectedDataset` is `null` (lines 54–61); a separate effect
    auto-selects the first dataset whenever `!selectedDataset &&
    datasets.length > 0` (lines 121–125); `useEDAStore.ts` uses a plain
    `create<EDAState>(...)` with no `persist` middleware, so
    `selectedDataset` is wiped to `null` on every `EDAPage` unmount. On a
    bare `/eda` navigation with no `dataset_id` param, the seeding effect is
    a no-op and the auto-select effect resets to the first dataset. **New,
    scope-narrowing finding:** the `/data → EDA button → /eda?dataset_id=X`
    deep-link path is **not** affected — the seeding effect successfully
    seeds the correct dataset on that path (confirmed via `DataSources.tsx`
    line 422 and the seeding effect). The regression is specific to any
    subsequent in-app return navigation to a bare `/eda` route with no query
    parameter (sidebar link, browser back/forward, or any other internal
    link that doesn't carry `dataset_id`).
  - **2026-08-07 resolution (partial — dataset context only):**
    **Correction to the recorded evidence.** Re-reproduced live against the
    real backend: bare `/eda` return navigation does **not** reset the
    selection. `useEDAStore` is created with `create<EDAState>(...)` at module
    scope, so it is a singleton that *survives* `EDAPage` unmount — the
    previously recorded root cause ("wiped to `null` on every unmount") is
    wrong, and the scope narrowing derived from it was inverted. Selecting
    dataset 68, navigating to `/data`, and returning via the sidebar kept 68.
    The **actual** defect is the opposite path: because the seeding effect
    ran only on mount *and* only when `selectedDataset == null`, a retained
    selection made every subsequent deep link a no-op. Clicking "EDA" on a
    Data Sources row navigated to `/eda?dataset_id=212` while the page kept
    showing dataset 68 — URL, dropdown, and the report/history queries all
    disagreeing. A second, independent defect surfaced during the fix: when
    the requested id is absent from the usable-datasets list, the controlled
    `<select>` silently falls back to its first enabled option, so the page
    displayed `customer_churn_dataset.csv` while querying id 212.
    **Fixed:** dataset resolution moved to pure, tested helpers in
    `core/utils/edaDatasetSelection.ts` (`resolveEdaDatasetSelection`,
    `shouldSyncDatasetParam`, `isSelectionMissingFromDatasets`, 32 tests,
    written test-first). The URL is now authoritative when it names a usable
    id — deliberately never rewritten from a not-yet-reconciled selection,
    which a live run showed would otherwise clobber the incoming deep link —
    the dropdown writes the URL, a bare `/eda` persists its selection back
    into the URL, an unavailable selection is disclosed via a `role="alert"`
    notice naming the id, and a rejected `analyzeMutation` now renders a
    recoverable error banner with "Try again" (same inputs) and "Dismiss".
    **Verified live:** deep link 206 over a retained 68 → URL/dropdown/label
    all 206; deep link 212 (unavailable) → placeholder + alert, no
    misrepresentation; back/forward restored 68; a forced analyze failure
    rendered and dismissed the error banner. tsc/lint clean, 402/402 tests,
    build clean.
    **Still open:** the rest of this finding is unchanged — analyses are
    still not presented as named jobs with a persistent input summary,
    phase/progress, main-screen cancel, or completed timestamp, and history
    load still does not visibly re-label the current-report context. Those
    depend on the EDA job/report/history API contract and **FND-003**, and
    are not addressable from the frontend alone.
  - **Delta:** Upgrades this behavior's core reproduction from Inferred to
    directly **Observed** (a real dataset with a real report was lost on a
    real return navigation, not only a `404` no-analysis case), and fully
    confirms the previously-inferred root cause via source. Also narrows
    scope: the first Data Sources → EDA handoff via the dedicated button is
    unaffected; only sidebar/back-nav-style return visits regress. Suggested
    fix direction: persist `selectedDataset` (Zustand `persist` middleware
    or sessionStorage), or have the sidebar EDA link carry the last-viewed
    `dataset_id`.

- **DAT-006 — Inferred: EDA filter and exclusion controls hide consequential
  reanalysis behind unlabelled, immediately applied configuration.**
  - **Evidence:** `EDASidebar` renders visible `span` labels beside native
    selects/inputs for column, operator, and value without label association.
    Adding/removing/clearing a filter immediately calls `runAnalysis`, while
    exclusions alone use a draft/applied split and an “Apply changes” button.
    The sidebar shows filters only as compact operator strings and does not
    state the report's applied filter/exclusion summary beside the visualized
    result.
  - **User problem:** Users can make an expensive, result-changing filter edit
    without a clear accessible input purpose, a consistent apply model, or a
    durable way to verify which filters/exclusions explain the report and its
    charts/tables.
  - **Affected surfaces:** EDA sidebar Active Filters and Excluded controls,
    Variables/Sample Data exclusion actions, report header, analysis request,
    and all analysis tabs.
  - **Proposed behavior:** Use labelled controls and one explicit draft/apply
    model for filters and exclusions; display a concise applied-context banner
    with filter count/details, excluded columns, target, task type, and report
    timestamp above every result and in downloads/history.
  - **Acceptance criteria:** Every filter/exclusion control has an accessible
    name and linked validation; no edit silently re-runs analysis; Apply and
    Reset name the changed context and prevent duplicate requests; each tab,
    table, chart export, and loaded history report identifies the exact
    applied analysis context.
  - **Validation method:** Component tests assert labels, keyboard flow,
    invalid operators/values, draft/reset/apply, and request payloads.
    Playwright adds/removes filters and exclusions, switches tabs/datasets,
    loads history, and downloads a chart/table fixture; inspect the context in
    light/dark desktop and narrow layouts and run axe.
  - **Impact:** High. **Frequency:** Frequent when refining analysis. **Effort:**
    M. **Risk:** Medium. **Dependencies:** `useEDAStore`, EDA request/report
    schema, download utilities, and **FND-005**. **Milestone:** Next.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live):** opened "Add Filter" in
    `EDASidebar` and inspected the DOM — the Column select, Operator select,
    and Value input (`placeholder="Value"`) all still have no
    `id`/`aria-label`/`aria-labelledby`. **Observed (source, unambiguous):**
    `handleAddFilter` (lines 178–183) and `handleRemoveFilter` (lines
    185–189) both update the filter list and immediately call `runAnalysis`,
    firing a full `POST /api/eda/{id}/analyze` re-run the instant a filter is
    added or removed, with no explicit apply/confirmation step, no debounce,
    and no undo affordance beyond manually removing the filter again (which
    itself immediately re-runs analysis). By contrast, `handleApplyExcluded`
    (lines 195–199) gates its own re-analysis behind an explicit Apply
    button — filters are the only control on this page without that pattern.
  - **Delta:** No contradiction; both claims (unlabelled controls, immediate
    re-run) are now each independently confirmed via a different method
    (live DOM vs. unambiguous source) rather than either alone, a stronger
    evidence basis with no change to problem, surfaces, or acceptance
    criteria. Worth noting the asymmetry with the Excluded-columns
    Apply-button pattern as a concrete, low-effort consistency fix reference.

- **DAT-007 — Inferred: EDA visualizations do not consistently preserve color
  meaning, interpretation, and accessible alternatives across result density,
  dark mode, and narrow widths.**
  - **Evidence:** `CorrelationHeatmap` encodes negative/positive strength in
    blue/red with opacity but provides no persistent -1/0/+1 legend; it
    truncates to 20 columns and depends on hover `title` for full labels.
    `CanvasScatterPlot` hides its legend at 20 groups, while
    `ThreeDScatterPlot` always shows a Plotly legend but supplies no theme
    layout/background. Distribution charts rely on angled abbreviated axes and
    hover tooltips. Several tabs provide useful nearby interpretation and
    no-data copy (PCA, time series, correlations, outliers, geospatial), but
    that coverage is uneven; download functions export images rather than a
    corresponding data-table alternative. No live report fixture was
    available, so this is source-supported rather than a claim of a reproduced
    chart failure.
  - **User problem:** At high category/column counts or narrow widths, users
    can lose the mapping from color to group/correlation, cannot reliably read
    truncated axes or hover-only detail, and may receive an image download
    without enough context to interpret or reproduce it.
  - **Affected surfaces:** Dashboard pies/missing-value bars; Variables
    distributions; Bivariate/PCA 2D and 3D scatter; correlations heatmaps;
    Target/Time Series/Outliers/Geospatial/Decomposition charts; chart and
    matrix downloads.
  - **Proposed behavior:** Define a visualization contract: every color
    encoding has a persistent legend/scale and text alternative, every
    truncation names its omitted scope and offers a table/download of values,
    tooltips augment rather than carry essential meaning, and chart theme,
    overflow, no-data, and nearby interpretation are tested together.
  - **Acceptance criteria:** At desktop and 390 px, legends, axes, titles,
    values, and controls remain readable or intentionally scrollable; color
    meanings and correlation direction/strength are conveyed without color or
    hover alone; dark charts meet contrast requirements; empty/unsupported
    analyses explain why and offer a next step; every visual download includes
    title, selected variables, applied context, and a CSV/table alternative.
  - **Validation method:** Render deterministic low/high-cardinality fixtures
    for each chart family at 1440 and 390 px in light and dark themes. Assert
    legend/axis visibility, tooltip text, no-data copy, heatmap truncation,
    table alternatives, download metadata, horizontal/vertical overflow, and
    contrast with axe plus visual review.
  - **Impact:** High. **Frequency:** Frequent for EDA result interpretation.
    **Effort:** L. **Risk:** Medium. **Dependencies:** chart adapters,
    Plotly/Recharts/Chart.js theme tokens, report payloads, export utilities,
    and visualization data-table design. **Milestone:** Next.
  - **2026-08-07 status:** Changed.
  - **Current evidence:** **Observed (live, real Iris dataset id 9),
    Correlations tab, light and dark mode:** confirmed no persistent
    −1/0/+1 (or equivalent) legend is rendered anywhere on the page, and
    column headers are truncated to fixed-width strings (e.g. "sepal.le…")
    relying solely on the native `title` attribute for the full name on
    hover, with no rotated label, wrap, or on-focus expansion alternative.
    **New, partially mitigating nuance not in the original finding:** the
    actual correlation **values** (e.g. "0.87", "−0.12") ARE printed as
    visible text directly inside each colored cell — cell color is
    reinforced by text for the correlation magnitude itself; only the
    axis/column **labels** are hover-only. This softens (but does not
    invalidate) the finding's implied "relies solely on color" framing.
    Dark-mode toggling is confirmed correct app-wide; the permanently
    near-black left icon rail (`bg-slate-900 dark:bg-slate-950` in both
    themes) is confirmed via source/DOM to be an intentional fixed-dark
    nav-rail design choice, not a theming bug, and is not reported as a
    separate finding. **New, related observations spun off as `DAT-008` and
    `DAT-009` below:** the EDA Dataset dropdown's duplicate/ambiguous option
    labels, and both `BivariateTab.tsx`/`PCATab.tsx`'s Download buttons
    remaining enabled with no chart rendered. Also confirmed
    `SampleDataTab.tsx` has no CSV/tabular export anywhere in the EDA
    surface (only the chart tabs' PNG Download button) — a related but
    separate completeness gap, not severe/novel enough on its own to warrant
    a new ID.
  - **Delta:** Confirms the legend/label-truncation claims exactly as
    **Observed** rather than **Inferred**, and adds the "cell values ARE
    legible" nuance, which materially refines the finding's framing:
    write-ups should distinguish "value legibility: fine" from "column
    identity legibility: hover-only, a real gap," since these are different
    severities. This rerun also surfaced two closely related but distinct
    usability gaps while re-checking these visualizations: the chart
    Download-when-empty behavior is spun off as its own new finding,
    `DAT-009`, rather than folded into this finding's scope; the EDA Dataset
    dropdown's duplicate-option-label gap is unrelated to color/legend
    interpretation and is recorded separately as `DAT-008`.

- **DAT-008 — Observed: the EDA Dataset dropdown renders many indistinguishable
  duplicate-text options with no disambiguating detail.**
  - **Evidence:** At 1440 px, on a real, unfiltered dataset list, the EDA
    Dataset `<select>` (`EDAPage.tsx` line 469,
    `datasets.map((ds) => <option key={ds.id} value={ds.id}>{ds.name}</option>)`)
    auto-selects the first dataset on load and, on inspection
    (`document.querySelector('select').options`), contains dozens of entries
    with **identical visible text** (e.g. "test source" appears roughly 50+
    times), differentiated only by the underlying numeric `value` (dataset
    id), which is never shown to the user. There is no id, creation date,
    size, row/column count, or any other disambiguating detail in the option
    label.
  - **User problem:** A user with more than a handful of similarly-named
    sources has no way to tell which literal dataset they are selecting
    without trial-and-error (select → wait for the report/"no analysis"
    state → check the dataset id elsewhere, e.g. via the URL or Data
    Sources). This is upstream of `DAT-005`: `DAT-005` is about *losing
    track* of a selection after navigating away, while `DAT-008` is about
    being unable to *make an informed selection in the first place* even on
    a first, uninterrupted visit. They compound but are separable defects
    with separable fixes (option-label content versus state persistence).
  - **Affected surfaces:** EDA Dataset selector (`EDAPage.tsx`); dataset
    listing consumed from `DatasetService.getUsable()`.
  - **Proposed behavior:** Give each dataset option a disambiguating label
    (for example, name plus a short id fragment, creation date, or row/column
    count), and consider grouping or de-duplicating visually identical
    source names in the selector.
  - **Acceptance criteria:** No two entries in the Dataset selector render
    identical visible text without an additional disambiguating detail
    (id fragment, date, or size); the selected option's label remains
    sufficient on its own to confirm which dataset is active without
    cross-referencing another surface.
  - **Validation method:** Component test seeds a dataset list containing
    duplicate names and asserts each rendered option has a unique,
    disambiguated label; Playwright opens the selector with a realistic
    duplicate-heavy fixture and asserts no two option texts are identical at
    1440 and 390 px; run axe on the control.
  - **Impact:** Medium. **Frequency:** Frequent once a workspace accumulates
    several similarly-named sources. **Effort:** S. **Risk:** Low.
    **Dependencies:** dataset list/labeling logic in `EDAPage.tsx` and the
    dataset service response shape. **Milestone:** Next.
  - **2026-08-07 status:** New.
  - **Current evidence:** Observed live against a real, accumulated backend
    dataset list (not previously present in the roadmap because prior audits
    did not inspect the rendered `<option>` text against a workspace with
    dozens of similarly-named sources).
  - **Delta:** New finding; no prior entry to compare against.

- **DAT-009 — Observed: chart "Download" buttons are never disabled for
  empty/unconfigured charts and silently export the placeholder text.**
  - **Evidence:** `BivariateTab.tsx` wraps both the actual scatter chart and
    its "Select X and Y variables to generate scatter plot." empty-state
    placeholder inside the same `id="bivariate-chart"` container (lines
    147–174), and its Download button (lines 54–61) is never disabled based
    on whether X/Y are selected. `chartUtils.ts`'s `downloadChart` calls
    `document.getElementById(elementId)` unconditionally and — if present —
    rasterizes whatever is inside that container via `html-to-image`. Net
    effect: clicking "Download Chart" before selecting X/Y axes silently
    produces a PNG of the placeholder text with no error or warning that the
    download is empty/meaningless. The same `id`-wraps-empty-state pattern,
    with the same always-enabled Download button, is confirmed in
    `PCATab.tsx` (`id="pca-chart"`).
  - **User problem:** A user who clicks Download before configuring a chart
    receives a file that appears successfully downloaded but contains only
    placeholder text, with no indication that nothing useful was exported —
    the user has no signal to configure the chart first or to know the
    download failed to capture anything meaningful.
  - **Affected surfaces:** `BivariateTab.tsx`, `PCATab.tsx`, and any other
    tab sharing this download-container pattern; `chartUtils.ts`'s
    `downloadChart` helper.
  - **Proposed behavior:** Disable (or hide) each Download button until its
    chart's required inputs are selected and a chart is actually rendered,
    or give `downloadChart` a defensive check that refuses to export a
    container that only holds empty-state content.
  - **Acceptance criteria:** The Download button in Bivariate and PCA (and
    any tab sharing this pattern) is disabled or hidden whenever the
    underlying container renders only empty-state content; when enabled,
    activating it always produces a file containing the rendered chart, never
    placeholder text.
  - **Validation method:** Component test renders Bivariate/PCA with no
    X/Y or component selection and asserts the Download control is
    disabled/hidden; a second test selects valid inputs and asserts the
    control becomes enabled and its `downloadChart` call receives a
    container that contains the chart, not the empty-state text.
  - **Impact:** Low. **Frequency:** Occasional (only when a user reaches for
    Download before configuring the chart). **Effort:** S. **Risk:** Low.
    **Dependencies:** `chartUtils.ts` `downloadChart`, `BivariateTab.tsx`,
    `PCATab.tsx`. **Milestone:** Later.
  - **2026-08-07 status:** New.
  - **Current evidence:** Source-confirmed (`BivariateTab.tsx` lines 54–61,
    147–174; `PCATab.tsx` same pattern; `chartUtils.ts`'s unconditional
    `getElementById`); observed indirectly via DOM/empty-state inspection in
    the live session, not by actually clicking the button and inspecting the
    resulting file.
  - **Delta:** New finding; no prior entry to compare against.

### Experiments and Inference

- **Inferred:** `Navbar.tsx` exposes the Experiments and Inference entry
  points as shell tabs (`setView('experiments')` / `setView('inference')`),
  not top-level app routes. `MainLayout.tsx` mounts `ExperimentsPage` and
  `InferencePage` lazily on first visit and keeps them mounted thereafter so
  their local state survives navigation.
- **Inferred:** `ExperimentsPage.tsx` starts with dataset and model-type
  filters, a collapsible job list sidebar, and a tab strip for Visual
  Comparison, Detailed Metrics & Params, Model Evaluation, Pipeline Diff,
  Feature Importance, SHAP Explainability, and Segmentation. The Evaluation
  tab carries its own slider/tuning sub-tabs plus split visibility toggles and
  threshold state.
- **Inferred:** `InferencePage.tsx` centers the audit on a JSON input editor,
  sample-size segmented control, CSV upload/reload actions, a run button, a
  list/table results toggle, advanced threshold overrides, and a recent-runs
  restore strip. Its state is persisted in local storage and is restored when
  the shell view is revisited.

#### Experiments and Inference audit evidence and limits

- **Observed:** In local system Chrome at 390 px, the Canvas shell switcher
  showed the Experiments button, but the visible Inference button was covered
  by Notifications and Playwright reported that interception. This is
  Experiments/Inference evidence for the already-recorded shared compact-shell
  defect **FND-001**, not a duplicate EXP finding. At desktop width,
  Experiments loaded one synthetic `test_job` with an unknown model/dataset;
  it could be selected and its Visual Comparison, Detailed Metrics & Params,
  Model Evaluation, and Pipeline Diff tabs could be reached. The table
  reported no upstream steps/default parameters and the evaluation request
  rendered “Failed to fetch evaluation data.”
- **Observed:** The locally available Inference payload rendered an artifact
  schema with four fields whose types were `unknown`. A malformed JSON array
  named the parser position and disabled Run Prediction. A syntactically valid
  row with one string-valued field and three missing schema fields displayed
  “3 missing” plus Fix, while Run Prediction remained enabled. This proves
  only the rendered client-side name check, not backend acceptance or a real
  prediction result.
- **Inferred:** The shared Playwright API stub returns `{}` for `/api/**`;
  `routes.spec.ts` covers only `/`, `/canvas`, `/jobs`, `/data`, and `/eda`,
  while `a11y.spec.ts` covers only `/`, `/canvas`, `/data`, and `/eda`.
  The focused unit tests cover threshold-tab visibility, threshold math, and
  per-class matrix calculation, not an Experiments or Inference journey.
  Consequently, mocks do **not** expose realistic multi-run classification,
  regression, segmentation, feature-importance, SHAP, pipeline-diff,
  deployment, long-running prediction, failure/retry, export, or result
  fixtures. Claims about those unrendered states below are **Inferred** from
  code/test evidence, not reproduced live behavior.

- **EXP-001 — Inferred: filters can hide selected runs while comparison keeps
  using them.**
  - **Evidence:** `filteredJobs` applies dataset/task/status filters, but
    `selectedJobs` independently resolves `selectedJobIds` against all jobs.
    Selecting a run, then changing either header filter, can therefore leave a
    comparison active with no corresponding visible selected row. `MainLayout`
    intentionally retains this local state across shell switches.
  - **Problem:** Context retention becomes ambiguous: filtering appears to
    change the comparison population, while hidden prior selections can still
    determine charts, table rows, evaluation, and deployment actions.
  - **Surfaces:** Experiments dataset/task filters; run sidebar; Visual
    Comparison; Detailed Metrics & Params; Evaluation; Pipeline Diff.
  - **Proposed behavior:** Keep selections deliberately, but show a persistent
    selected-run summary with visible/hidden counts and run/dataset/model
    identity; offer “clear hidden” and “show selected” without silently
    dropping comparison context.
  - **Acceptance criteria:** Every selected run is either visible in the
    sidebar or named in a persistent summary; filter changes state whether
    hidden selections remain; all tabs identify their exact run set; clearing
    or restoring a hidden selection updates every view consistently.
  - **Validation method:** Playwright seeds mixed dataset/task completed jobs,
    selects runs, applies/removes filters, switches every tab and shell view,
    reloads where retained state is intended, and asserts selected identities,
    chart/table input, evaluation target, and keyboard operation.
  - **Impact:** High. **Frequency:** Frequent for cross-run comparison.
    **Effort:** M. **Risk:** Medium. **Dependencies:** `useJobStore`,
    `ExperimentsPage` selection state, and job fixture contract.
    **Milestone:** Now.
  - **2026-08-07 status:** Confirmed (evidence upgraded from Inferred to a
    live, twice-reproduced failure; the finding's user problem, scope, and
    proposed behavior are unchanged, so it is not classified Changed).
  - **Current evidence:** Live reproduction at 1440 px: selecting 2
    classification jobs, then switching the task-type filter to
    Segmentation, hid both selected sidebar rows with no selected-state
    indicator, yet the header still read `SELECT RUNS (2)` and Visual
    Comparison kept rendering both hidden jobs' bars. A stronger
    reproduction, same session, selected 2 additional *visible* clustering
    jobs (`SELECT RUNS (4)`); Model Evaluation still defaulted to the
    hidden classification job rather than either newly-selected visible
    run, and the newly-revealed Segmentation tab defaulted to "The selected
    run is not a Segmentation (clustering) job" even though 2 of the 4
    selected runs are valid clustering jobs. Source re-confirmed:
    `ExperimentsPage.tsx` lines 329–338 — the effect resolving `evalJobId`
    only re-picks `selectedJobIds[0]` (the first, i.e. oldest, selected job)
    when the current `evalJobId` drops out of `selectedJobIds`; it never
    prefers a job compatible with the active tab.
  - **Delta:** Upgrades from code-level ("Inferred") to a live, twice-
    reproduced failure, including a previously undocumented consequence:
    newly-added, valid, *visible* selections do not become the active
    target for Model Evaluation/Segmentation — the stale, hidden selection
    wins by array order. This "wrong-tab-default" behavior should be folded
    into the Evidence/Acceptance-criteria language; it intersects with the
    identifier-mismatch problem in **EXP-008** below.
  - **2026-08-07 resolution:** Selection bookkeeping extracted into pure,
    test-first helpers in `ExperimentsPage/utils/runSelection.ts`
    (`partitionSelection`, `resolveEvaluationTarget`, `selectRunsForView`; 17
    tests). `ExperimentsPage.tsx` now renders a persistent `role="status"`
    summary whenever the active filters hide a selected run — visible/total
    counts, the hidden runs named by short id, plus "Show all selected"
    (resets both filters, keeps the selection) and "Clear hidden" (reduces the
    selection to the visible runs). The evaluation/segmentation effect no
    longer re-picks `selectedJobIds[0]`: it keeps a still-selected, renderable
    target, otherwise prefers the first *visible* compatible run, and clears
    the view when no selected run suits the tab. `EvaluationView`'s run picker
    receives only renderable runs (prop renamed `selectedJobIds` →
    `eligibleJobIds`), so a clustering run can no longer be clicked into the
    Model Evaluation tab.
    **Verified live (1440 px, real backend):** selecting a classification run
    and switching the task filter to Segmentation/Regression produced
    "0 of 1 selected runs visible / Still comparing 1 run hidden by the current
    filters: 9ad0c0c6"; "Show all selected" reset both filters to `all` and
    kept `SELECT RUNS (1)`; "Clear hidden" dropped it to `SELECT RUNS (0)`. The
    rewritten effect showed no render loop or React error in the console.
    tsc/lint clean, 419/419 tests, build clean.
    **Verification caveat:** the backend had only **one** completed job at the
    time, so the multi-run, mixed-task path (this finding's original
    classification-plus-clustering reproduction) could not be re-exercised
    live. That behavior is covered by unit tests over the pure resolver, not by
    a live run — worth re-checking against a seeded fixture set, which is also
    what this finding's stated Playwright validation method calls for.
    **Still open:** per-tab "this view is showing run X" labelling across the
    non-evaluation tabs (Visual Comparison, Detailed Metrics, Pipeline Diff)
    is not implemented; those tabs still render the whole selected set without
    naming it beyond the chart series. The identifier-mismatch overlap with
    **EXP-008** is untouched.

- **EXP-002 — Inferred: metric comparison does not make decision direction,
  comparability, or missingness durable.**
  - **Evidence:** `MetricsComparisonChart` groups numeric keys and renders
    generic bars/axis values; `ComparisonTableView` renders unavailable values
    as `-`. Descriptions are tooltip-only when `getMetricDescription` knows a
    key. `BranchComparisonCard` locally guesses lower-is-better only for a
    small string list. Neither surface establishes a selected primary metric,
    units/scale, direction, or why a value is absent across the compared runs.
  - **Problem:** Users can compare incompatible train/test/CV values or choose
    the visually tallest bar when lower is better, while a dash can mean
    unreported, unsupported, or unavailable.
  - **Surfaces:** Visual Comparison; Detailed Metrics & Params; parallel
    branch comparison; metric/split toggles; run-selection summary.
  - **Proposed behavior:** Attach a comparison contract to each displayed
    metric: evaluation population/split, direction, unit/scale, availability
    reason, and explicit primary-metric/winner rule. Keep tooltips as detail,
    not the only carrier of that context.
  - **Acceptance criteria:** Each metric has visible higher/lower-is-better
    and split context; unavailable values distinguish absent artifact,
    unsupported metric, and filter exclusion; winner highlighting never
    contradicts metric direction; mixed task types or incompatible scales
    require an explicit user choice rather than a silent aggregate.
  - **Validation method:** Render deterministic classification, regression,
    tuned-CV, partial-metric, and parallel-branch fixtures. Assert direction,
    units, missing reasons, winner calculation, accessible names, and
    screenshot/geometry behavior at 1440 and 390 px in both themes.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** job metrics schema, metric metadata, chart/table
    adapters, and experiment fixtures. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed (no material delta), evidence
    strengthened with live screenshots.
  - **Current evidence:** Observed (live) at 1440 px: Visual Comparison's
    `ACCURACY` chart renders three bars per job group (test/train/val) with
    no direction indicator, unit, or explanation of why a metric was chosen;
    Detailed Metrics & Params shows `—` for one run's pipeline steps 6–11
    and the other run's steps 1–5 — the same dash glyph representing "this
    run's pipeline has fewer/different steps than the compared run" with no
    visual distinction from "value not reported." Source unchanged since the
    original audit (`git diff` empty for `ComparisonTableView.tsx`,
    `MetricsComparisonChart`/chart adapters).
  - **Delta:** None in substance. Live evidence directly confirms the two
    example mechanisms (`MetricsComparisonChart` generic bars,
    `ComparisonTableView`'s `-` placeholder) named in the original finding.
  - **2026-08-07 resolution (partial — direction and split shipped):**
    - Added `frontend/ml-canvas/src/core/utils/metricMeta.ts` (34 tests,
      TDD) exposing `getMetricDirection`, `getMetricSplitLabel`, and
      `pickBestIndex` as machine-readable metric metadata, plus a shared
      `components/ui/MetricDirectionBadge.tsx`.
    - Direction and split are now rendered **visibly** (badge + split chip) in
      `ComparisonTableView` metric rows and above the `MetricsComparisonChart`
      bar chart, rather than only inside an `InfoTooltip` the user must hover.
      The chart legend now names the split ("Test (held-out)") or the metric,
      instead of the previous meaningless `other` label.
    - **Winner highlighting no longer contradicts direction.** `pickBestIndex`
      returns `null` for unknown-direction metrics, ties, and rows where fewer
      than two runs reported a value, so Skyulf never asserts a "best" it
      cannot justify. `BranchComparisonCard`'s local substring heuristic
      (`includes('loss'|'error'|'mse'|'mae')`) was deleted — it mis-ranked
      `mape`, every `cv_*_std`, and `davies_bouldin`, and silently treated all
      unrecognised metrics as higher-is-better.
    - Live-verified at 1440 px against a real run: `Fit Time` and
      `Peak Memory Bytes` display "Lower is better"; `Rows In`/`Rows Out`
      display "Direction unknown — not ranked" and the chart states "Skyulf
      cannot rank this metric — read the bars against your own objective."
      All five comparison tabs render with no console error.
    - Missing values now render a muted `—` titled "was not reported by this
      run", separating that case from a genuine zero.
  - **Still open (deliberately not claimed as done):**
    - **Units/scale** are not yet rendered — this needs an authoritative
      per-metric unit contract from the job metrics schema (backend), so it is
      out of scope for a frontend-only change.
    - **Three-way missing-reason distinction** is only two-way today ("not
      reported by this run" vs. a value). Separating "unsupported for this
      task type" from "excluded by the current split filter" requires a
      metric↔task compatibility contract that does not exist yet.
    - **Mixed task types / incompatible scales** still do not force an
      explicit user choice.
    - **Validation caveat:** the acceptance criteria call for deterministic
      classification/regression/tuned-CV/partial-metric/parallel-branch
      fixtures. Only **one** completed job existed in the local backend, so
      the multi-run winner-highlighting and parallel-branch paths are covered
      by unit tests but were **not** exercised live. Seeded E2E fixtures
      remain the right way to close this.

- **EXP-003 — Inferred: conditional explanation and segmentation views can
  conceal availability and overstate comparability.**
  - **Evidence:** Feature Importance and SHAP tabs render only when at least
    one selected job supplies an artifact. Feature-importance rows substitute
    `0` for a feature not reported by a run after per-run normalization.
    SHAP's detailed views use a single selected run, while the summary
    aggregates selected runs. `SegmentationView` prints silhouette,
    Calinski-Harabasz, and Davies-Bouldin values without per-metric
    directionality beside the cards.
  - **Problem:** A missing tab does not explain whether an artifact is
    unsupported, pending, or failed; zero-like comparison bars can be read as
    measured zero; and cluster-quality numbers do not tell a user which
    direction supports a decision.
  - **Surfaces:** Feature Importance; SHAP Summary/Beeswarm/Dependence/
    Waterfall/Force/Interaction; Segmentation; artifact-empty states and
    exports.
  - **Proposed behavior:** Always expose an explainability/segmentation
    availability state for selected runs, annotate artifact coverage and
    normalization, render missing values distinctly from zero, and place
    metric direction/limits next to cluster-quality values.
  - **Acceptance criteria:** Every selected run names explainability artifact
    status and reason; comparisons distinguish zero from not reported; SHAP
    single-run views identify the active model/run/sample; segmentation marks
    silhouette/Calinski-Harabasz as higher-is-better and Davies-Bouldin as
    lower-is-better, with a non-ground-truth caveat; exports retain run,
    split, and normalization context.
  - **Validation method:** Use tree/non-tree, artifact-present/absent/pending/
    failed, overlapping/non-overlapping feature, multi-run SHAP, and
    clustering fixtures. Assert visible copy, data-table/export metadata,
    keyboard selection, and light/dark/narrow rendering.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** training artifact schema, explanation services,
    chart adapters, exports, and deterministic fixtures. **Milestone:** Next.
  - **2026-08-07 status:** Confirmed, with one refinement.
  - **Current evidence:** Observed (live) at 1440 px: Feature Importance and
    SHAP Summary rendered without incident for 2 selected classification
    jobs, each showing 4 features normalized 0–1 per run with a "values
    normalised per-run (max = 1.0)" legend note; no copy explains what a bar
    of 0 means (unreported vs. genuinely negligible importance), consistent
    with the original finding. **Refinement:** with only classification jobs
    selected, the Segmentation tab is present but showed "The selected run
    is not a Segmentation (clustering) job" by default (see **EXP-001**),
    rather than being hidden/disabled or auto-selecting a compatible run —
    a slightly different and arguably worse manifestation than "a missing
    tab does not explain availability," since here the tab *is* present and
    *is* wrong-by-default. The Beeswarm/Dependence/Waterfall/Force/
    Interaction SHAP sub-tabs and a genuinely-selected pair of clustering
    jobs' Segmentation metric cards were not reached live this rerun
    (time-boxed out); that portion of the finding is **code-evidence-only,
    unchanged since the original audit** (empty `git diff`), not
    independently re-verified live this time.
  - **Delta:** Adds the present-but-wrong-default Segmentation tab behavior
    as new corroborating evidence; no change to the finding's substance
    otherwise.

- **EXP-004 — Inferred: Pipeline Diff lacks an explicit comparison decision
  contract.**
  - **Evidence:** `PipelineDiffView` accepts exactly two jobs in selection
    order, labels them Baseline/Candidate, fetches their saved graphs, and
    lists structural/config differences. It has no baseline chooser, swap,
    run metadata/evaluation context in the header, or explanation of a
    missing/snapshot-less graph beyond the fetch error.
  - **Problem:** Selection order silently determines the baseline and users
    cannot tell whether a structural/config difference corresponds to the
    desired dataset, split, model outcome, or an incomplete historical graph.
  - **Surfaces:** Run sidebar selection; Pipeline Diff tab; graph snapshots;
    change list.
  - **Proposed behavior:** Require an explicit baseline/candidate designation
    (with swap), show each run's dataset, model, timestamp, scoring/split
    context and snapshot availability, and keep structural changes clearly
    separate from outcome differences.
  - **Acceptance criteria:** Exactly-two selection explains how roles are
    assigned and supports swapping; each side names enough metadata to verify
    the intended comparison; no graph/no diff/error states name the affected
    run and recovery next step; a change list preserves direction
    baseline→candidate and is exportable/accessibly navigable.
  - **Validation method:** Mock equal, changed, renamed, missing, malformed,
    and request-failed graph snapshots for ordered job pairs; verify role
    changes, metadata, keyboard controls, change direction, responsive graph
    geometry, and error recovery.
  - **Impact:** Medium. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** saved job graph contract, `graphDiff`, React
    Flow snapshot renderer, and job metadata. **Milestone:** Next.
  - **2026-08-07 status:** Confirmed, plus new cross-referenced evidence.
  - **Current evidence:** Observed (live) at 1440 px: selection order
    determined Baseline (`random_forest_classifier`) vs. Candidate
    (`decision_tree_classifier`) exactly as documented; "Diff summary: 1
    added · 4 modified · 2 unchanged (6 renamed across runs) · edges 4+/3−"
    banner and full graph snapshots rendered with no baseline/candidate swap
    control, and no dataset/timestamp/scoring context in the header beyond
    the model type and a short ID — all as originally described. Source
    unchanged (`PipelineDiffView.tsx`, confirmed via `git diff`). **New,
    related finding this rerun surfaced:** the two short IDs shown in each
    side's header (`27b2bf2b`, `e58ea66c`) are **not** the same identifiers
    shown for the same two jobs anywhere else in the Experiments page
    (sidebar, Detailed Metrics, Visual Comparison all showed
    `f245bcf3`/`6cdfb46e` for the identical pair) — see **EXP-008** below.
    This compounds this finding's "no run metadata in the header" problem:
    a user cannot even use the ID to confirm which sidebar selection
    produced this diff.
  - **Delta:** Substance unchanged; cross-reference to new finding
    **EXP-008** added.

- **EXP-005 — Inferred: threshold exploration, tuning, and prediction-time
  activation need a durable decision record.**
  - **Evidence:** `EvaluationView` distinguishes the client-only slider from
    preview/save/toggle/clear tuning and explains its validation-then-test
    fallback. `InferencePage` retrieves saved thresholds, allows ad-hoc
    overrides, and displays thresholds returned by a prediction. Calls have
    local error strings but no mutation-pending or post-save provenance
    record; current tests only assert threshold-tab visibility/math.
  - **Problem:** A user can change a model-wide prediction rule without a
    compact record of metric, split, values, time/version, activation result,
    and a recoverable state if a mutation fails or the selected run changes.
  - **Surfaces:** Evaluation Threshold Slider/Tuning; confusion matrices;
    saved-threshold API; Inference advanced overrides and results.
  - **Proposed behavior:** Treat tuning as a versioned decision on the current
    Evaluation and Inference surfaces: preview context, confirm/save/enable
    lifecycle, mutation-scoped pending/error/retry feedback, and an immutable
    applied-threshold summary carried into prediction results. Preserve the
    existing distinction between exploratory slider and deployed behavior.
    `FND-004` can later align cross-route fetch retries without blocking this
    threshold lifecycle slice.
  - **Acceptance criteria:** Preview, save, enable, disable, clear, and failed
    states name job/model version, metric, data split, threshold values, and
    whether real prediction is affected; each failed mutation offers one scoped
    retry or dismissal path on the same surface; controls prevent duplicate
    mutations; every result identifies default/saved/override thresholds;
    switching jobs never attributes thresholds to the wrong job.
  - **Validation method:** Mock validation/test fallback, saved-enabled,
    save/toggle/clear success, delayed response, conflict, and failure for two
    jobs. Exercise tune→save→enable→infer→override→clear through component and
    Playwright tests, including tab/run switches and live/status assertions.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** M. **Risk:**
    High. **Dependencies:** threshold-tuning API/version semantics,
    evaluation artifacts, deployment prediction response, and **FND-003**.
    **Milestone:** Now.
  - **2026-08-07 status:** Confirmed (evidence upgraded to Observed for the
    tuning-affects-inference half of the claim; the no-provenance half is
    unchanged).
  - **Current evidence:** Observed (live) at 1440 px, in sequence: Model
    Evaluation's Threshold Slider/Tuning tabs produced per-class confusion
    matrices (0/1/2 vs Rest), ROC/ROC-AUC=1.000, and F1-best-threshold
    badges for Train/Test. Threshold Tuning → Preview (metric=F1) produced
    per-split confusion matrices with the caption "Computed from test split
    (no validation split available — using test split)" — the validation→
    test fallback this finding describes is real and user-visible. The
    Inference page's "Advanced: override thresholds" panel stated verbatim:
    *"This model already has tuned thresholds saved and enabled from the
    Evaluation tab — they're applied automatically to every real prediction
    unless you turn on an override below: 0:1, 1:1, 2:1."* A subsequent real
    prediction run displayed a "THRESHOLDS APPLIED 0:1 1:1 2:1" banner
    directly above the results list — a live, end-to-end confirmation that
    saved thresholds genuinely affect real predictions and are surfaced back
    to the user. No mutation-pending/error/retry affordance was observed on
    the Preview action itself (it completed near-instantly against local
    data); Save was not exercised this run to avoid mutating the shared
    active deployment's threshold state for sibling agents relying on the
    same fixture — a limitation, not a finding of absence.
  - **Delta:** Upgrades to Observed for the tuning-affects-inference half of
    the claim (previously only inferred from separate reads of
    `EvaluationView`/`InferencePage` source). The "no durable decision
    record / no mutation-pending or provenance" half of the problem
    statement was not contradicted — the override panel shows raw threshold
    numbers with no save-time/version/who-changed-it metadata, consistent
    with the original.

  - **2026-08-07 RESOLVED (partial — frontend slice only).**
    - Every threshold mutation (preview, save, enable/disable, clear) now
      carries mutation-scoped pending state, disables its own control while in
      flight so it cannot be double-submitted, and announces the pending
      status. `ExperimentsPage` handlers rethrow so the surface can render the
      failure.
    - A failed mutation offers an in-place retry on the same surface instead
      of a fire-and-forget toast.
    - `InferencePage` now shows provenance for saved thresholds — job, model
      type, the metric they were optimized for, the split they were computed
      from, and `computed_at` — and distinguishes "saved and enabled" from
      "saved but disabled", which the previous copy conflated.
    - Cross-job misattribution is closed by the new `useSavedThresholdInfo`
      hook, which is keyed to the active job and guards with both a cancel
      flag and a request sequence number, so a slow response for a previous
      job can never land on the current one.
    - The verified-working behavior was preserved: the exploratory slider is
      still distinct from deployed tuning, the validation→test fallback
      caption is unchanged, and the applied-threshold results banner still
      renders.
    - **Still open — needs backend work:** the threshold API returns only
      `thresholds`, `classes`, `metric`, `split_used`, `computed_at`, and
      `enabled`. There is **no** save timestamp distinct from `computed_at`,
      **no** model/job version, and **no** actor, so the record is not yet a
      full immutable, attributable decision as the acceptance criteria
      require. Conflict/delayed-response handling was not exercised against a
      real backend, and no Playwright coverage of
      tune→save→enable→infer→override→clear exists. No 390 px measurement.

- **EXP-006 — Observed: inference schema feedback permits a structurally
  incomplete, type-incompatible request.**
  - **Evidence:** The local artifact schema rendered four `unknown`-typed
    fields. With valid JSON containing `"sepal.length": "wrong"` and omitting
    three expected fields, the UI showed “3 missing” and an optional Fix while
    Run Prediction remained enabled. `checkSchema` compares field names only,
    and Fix fills every missing field with numeric `0`; no client type/value
    check exists. This is not a claim that the backend accepted that request.
  - **Problem:** A user can submit data that the available schema visibly
    identifies as incomplete and potentially incompatible, or apply a
    misleading zero-fill before understanding feature semantics.
  - **Surfaces:** Inference editor; artifact schema chips; CSV/sample input;
    schema badges/Fix; Run Prediction.
  - **Proposed behavior:** Obtain a typed, required/nullable input contract
    from the deployed artifact; validate each row before submission; make
    repair an explicit, reviewable transform with per-field defaults/reasons,
    never an unqualified zero-fill; and make the inference editor's own
    invalid/error state explicit even before `FND-005` later normalizes shared
    form primitives.
  - **Acceptance criteria:** Run Prediction is blocked or requires an explicit
    reviewed override for missing, extra, wrong-type, nullability, range,
    categorical, and row-shape violations; each issue names
    row/field/expected/received value and is tied to the current editor state;
    repair previews changed values and preserves the original input; unknown
    schema types are honestly marked unvalidated.
  - **Validation method:** Render deployed-artifact fixtures for numeric,
    string, categorical, nullable, defaultable, and unknown fields; enter
    JSON, CSV, sample, and drag/drop variants; assert request gating/payload,
    repair preview, keyboard feedback, and server-validation reconciliation.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    High. **Dependencies:** deployment artifact schema, prediction validation
    response, and CSV parser. **Milestone:** Now.
  - **2026-08-07 status:** Confirmed (reconfirmed with a materially stronger
    new reproduction; the finding's substance is unchanged).
  - **Current evidence:** Observed (live) at 1440 px, against the real
    active deployment (`random_forest_classifier`, job
    `7c1ec203-dadc-4cb2-8e04-fd4a52c11813`, schema `sepal.length/sepal.width/
    petal.length/petal.width`, all `unknown`-typed): entered
    `[{"sepal.length": "wrong"}]` (1 wrong-type field, 3 missing) — the UI
    showed "3 missing" plus Fix while Run Prediction stayed enabled. Ran it
    anyway: the backend rejected it cleanly — `"Missing required column(s)
    for prediction: ['sepal.width', 'petal.length', 'petal.width']. Expected
    columns: [...]"` — rendered as a raw string in the results pane, directly
    answering the original finding's open question: **the backend does
    reject missing columns**, but only after a client round-trip, with no
    client-side hard gate. Clicked Fix: it zero-filled only the 3 *missing*
    fields, leaving the existing wrong-type field (`"sepal.length":
    "wrong"`) untouched, since `checkSchema`'s name-only comparison cannot
    see it. Ran again: this time the backend **crashed** with a raw Python
    exception surfaced verbatim to the user — **`"Feature engineering
    failed: unsupported operand type(s) for -: 'str' and 'float'"`** — shown
    at the same time as a green "✓ Added 3 missing field(s)" success toast,
    i.e. the UI signals success while the request is actively failing.
  - **Delta:** A materially stronger, previously undemonstrated failure
    mode: the documented "Fix" zero-fill is not just semantically
    questionable (as flagged) — combined with a client-supplied wrong-type
    value, it produces a genuine unhandled backend exception whose raw
    message reaches the end user, while the UI simultaneously reports the
    Fix as a success. This exact reproduction
    (`"unsupported operand type(s) for -: 'str' and 'float'"`) is the
    concrete illustration of "potentially incompatible" and "no client
    type/value check" the original Evidence bullet already claimed.

- **EXP-007 — Inferred: inference execution and recovery are not a complete
  durable run lifecycle.**
  - **Evidence:** `InferencePage` has one `isLoading` flag, captures elapsed
    client time, clears prior results before each request, renders a raw error
    string, and keeps at most five restoreable runs only in component memory.
    It persists input, sample size, and view choice in local storage, but not
    result/error/run provenance. The local mock did not expose a realistic
    deployment or prediction success/failure/long-running payload.
  - **Problem:** A slow or failed prediction has no explicit cancellation,
    retry-with-same-input, durable run record, server timing/version context,
    or result-to-export provenance after reload; users can lose the evidence
    needed to diagnose or reproduce a decision.
  - **Surfaces:** Inference Run Prediction; loading/error/result panes;
    recent runs; JSON/CSV export; deployment card; threshold display.
  - **Proposed behavior:** Model each request as a named inference run with
    submit/pending/success/failure/cancelled status, model/version/schema/
    threshold/input summary, scoped retry/cancel on the same inference surface
    where supported, and a privacy-conscious local/session history that
    distinguishes retained input from retained result. `FND-004` can later
    align broader route-fetch retry affordances without blocking this run
    lifecycle slice.
  - **Acceptance criteria:** Pending work names its run and prevents duplicate
    submission; failure exposes safe cause, unchanged input, one scoped retry,
    and next action; success/export names model version, schema/threshold
    context, row count, client/server latency, and result format;
    reload/history makes retention/expiry explicit; no raw transport object is
    the only recovery copy.
  - **Validation method:** Mock active/no deployment, malformed/server schema
    errors, delayed success, timeout, cancellation, retry, classification and
    regression results, export, reset, and reload. Assert requests, retained
    state, status announcements, CSV/JSON metadata, keyboard behavior, and
    narrow/desktop layouts.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** L. **Risk:**
    Medium. **Dependencies:** deployment/prediction APIs, job/status contract,
    browser storage/privacy policy, exports, and **FND-003**. **Milestone:**
    Now.
  - **2026-08-07 status:** Confirmed (evidence upgraded to Observed for the
    raw-error-string and no-cancel/no-explicit-retry claims; unchanged for the
    rest).
  - **Current evidence:** Observed (live) at 1440 px: both failure
    reproductions above (`Missing required column(s)…` and `Feature
    engineering failed: …`, see **EXP-006**) rendered as raw, unstyled
    strings in the Prediction Results pane with no structured cause, no
    scoped retry button, and no "next action" guidance — directly confirms
    "renders a raw error string" and "no explicit retry." A successful run
    (5 sampled rows, Setosa × 5) produced List/Table toggle, Copy/JSON/CSV
    export buttons, and a "RECENT RUNS" entry (`02:14:11 PM · 5 rows · 57
    ms`) — confirms the recent-runs/export affordances exist as described.
    **Undeploy** requires an explicit confirm dialog ("Are you sure you want
    to undeploy the current model?" / Cancel / Undeploy) — a positive
    finding not previously called out: destructive reset is *not*
    one-click. Tested the Cancel path only, to avoid mutating the shared
    active deployment. Did not exercise a genuinely long-running/timeout
    scenario (no slow endpoint available in this environment) or the
    CSV/drag-drop input path — time-boxed out this rerun; treat as **not
    independently re-verified**, code evidence unchanged per `git diff`.
  - **Delta:** Upgrades the raw-error-string claim to directly observed (two
    concrete verbatim examples now on record). Adds the
    Undeploy-confirmation-dialog detail as new, positive evidence the
    original finding didn't mention (does not change the finding's overall
    thrust, since the core gap — no named run record, no cancel, no
    retry-with-same-input — remains unaddressed).

- **EXP-008 — Observed (new): cross-tab run-identifier mismatch breaks
  selection traceability on the Experiments page.**
  - **Evidence:** The sidebar list, Visual Comparison chart axis, and
    Detailed Metrics & Params column headers all label a run using
    `shortRunId(job)` (`ExperimentsPage/utils/jobMeta.ts` lines 119–123), an
    8-char prefix of the job's **`pipeline_id`** (parent pipeline
    preferred) — confirmed via grep: `JobListSidebar.tsx:78`,
    `ComparisonTableView.tsx:246`. Model Evaluation's job-selector pills
    (`EvaluationView.tsx` line 169), Pipeline Diff's Baseline/Candidate
    badges (`PipelineDiffView.tsx` lines 194/198), Feature Importance's
    chart legend (`FeatureImportanceView.tsx` lines 91/99), and SHAP
    Summary's chart legend (`ShapSummaryView.tsx` lines 82/90) all instead
    label the **same** two selected jobs using `job.job_id.slice(0, 8)` /
    `j.jobId.slice(0, 8)` — an 8-char prefix of the job's own **UUID**, an
    entirely different identifier space (the same `jobId.slice(0, 8)`
    pattern also appears in `ShapBeeswarmView.tsx`, `ShapDependenceView.tsx`,
    `ShapForceView.tsx`, `ShapInteractionView.tsx`, and
    `ShapWaterfallView.tsx`, not independently checked live on every
    sub-view but from the same call-site pattern in each file). Live,
    controlled reproduction at 1440 px (fresh page, single job selected
    first to rule out stale state): selecting exactly the sidebar rows
    labeled `f245bcf3` and `6cdfb46e` (confirmed via `SELECT RUNS (1)` →
    `(2)` header transition tied to each click) caused Model Evaluation,
    Pipeline Diff, Feature Importance, and SHAP Summary to all label the
    *same two jobs* as `27b2bf2b` and `e58ea66c` — with no shared/
    correlating label anywhere on screen.
  - **Problem:** A user selecting runs "f245bcf3" and "6cdfb46e" in the
    sidebar has no way to confirm — short of matching on
    `model_type`/dataset text — that the "27b2bf2b" vs. "e58ea66c" labels
    shown in Model Evaluation, Pipeline Diff, Feature Importance, or SHAP
    are the runs they just selected. This directly compounds **EXP-001**
    (hidden-selection ambiguity) and **EXP-004** (Pipeline Diff header lacks
    run metadata): even when a user does look at the ID shown, it is the
    wrong one to cross-reference against their sidebar action.
  - **Surfaces:** Model Evaluation job-selector pills; Pipeline Diff
    Baseline/Candidate header; Feature Importance chart legend; SHAP
    Summary/Beeswarm/Dependence/Waterfall/Force/Interaction legends.
  - **Proposed behavior:** Use `shortRunId(job)` consistently across every
    tab that identifies a selected run, exactly as the sidebar/Visual
    Comparison/Detailed Metrics already do; if `job_id` must be shown
    anywhere (e.g. for support/debugging), label it explicitly as "Job ID"
    rather than as the bare short ID a user just clicked.
  - **Acceptance criteria:** Given 2+ selected runs, every tab that names a
    run by a short ID uses the same identifier the sidebar used to select
    it; no two different 8-char strings represent the same run within one
    comparison session.
  - **Validation method:** Component/unit test asserting `EvaluationView`,
    `PipelineDiffView`, `FeatureImportanceView`, and `ShapSummaryView` all
    render `shortRunId(job)` (not `job.job_id`) for a fixture job set with
    distinct `job_id` and `pipeline_id` values; Playwright regression
    selecting 2 named runs and asserting the same label string appears in
    the sidebar, Model Evaluation pill row, and Pipeline Diff header.
  - **Impact:** Medium (confusing but not data-corrupting — the underlying
    comparison data is for the correct jobs; only the on-screen label is
    wrong). **Frequency:** Frequent (any 2+-job comparison touching Model
    Evaluation/Pipeline Diff/Feature Importance/SHAP). **Effort:** S (swap
    one function call at ~4-6 call sites). **Risk:** Low. **Dependencies:**
    none beyond `jobMeta.ts`'s existing `shortRunId` export. **Milestone:**
    Next.
  - **Milestone reassessment (Task 7):** originally proposed as `Now`, this
    finding is reassessed to `Next` for consistency with `DAT-008`, which
    shares the identical normalized profile (Impact **Medium**, Frequency
    **Frequent**, Effort **S**, Risk **Low**) and is also a label/selection-
    clarity gap whose underlying data is correct. The `Now` milestone is
    reserved for the smallest dependency-complete **high-impact** set;
    a `Medium`-impact label-consistency fix with no data-loss or reliability
    risk belongs in `Next` alongside `DAT-008`. This diverges from `OPS-008`,
    which keeps `Now` despite also being `Medium`/`S`: `OPS-008` is an active
    data-integrity/React-key-collision defect (React documents key collisions
    as causing list reordering, incorrect DOM reuse, or dropped updates), a
    data-loss/reliability risk the ranking order weighs above raw impact,
    whereas `EXP-008`/`DAT-008` only mislabel correctly-computed data.
  - **2026-08-07 status:** New.
  - **Current evidence:** Observed live against the real backend, as
    described above; independently evidenced by grep-confirmed source
    citations and a controlled two-job reproduction distinct from every
    other `EXP-*` finding (it is about label-identifier consistency across
    components, not selection retention, metric semantics, artifact
    availability, diff roles, threshold provenance, or run lifecycle).
  - **Delta:** New finding; no prior entry to compare against.

### Operations

- **Baseline entry-point mapping:** `/jobs` is the eager `JobsPage`; `/drift`,
  `/registry`, `/deployments`, `/slow-nodes`, and `/audit` are lazy-loaded
  route pages; `/errors` is the eager `ErrorLogPage`.
- **Baseline entry-point mapping:** `Layout.tsx` shows alert badges only on
  `/drift` (`driftAlert`) and `/errors` (`errorAlert`). Current E2E route
  smoke coverage includes `/jobs`; the remaining Operations routes are not
  listed in `routes.spec.ts`, and `a11y.spec.ts` does not cover any
  Operations route.

#### Operations audit evidence and limits

- **Observed:** Local system Chrome at 1440 px rendered historical completed
  and failed rows on Jobs; model rows, version history, a deployment
  confirmation entry point, and one active deployment; Drift's no-report
  state and editable thresholds; 12 Error Log HTTP events with time and
  resolution controls; Slow Nodes aggregates; and Audit Log's
  dataset-and-limit controls with no saves for its selected dataset. The
  reproduced Error Log row expanded to a traceback. No destructive resolve,
  clear, deactivate, deploy, or redeploy request was submitted during this
  documentation-only audit.
- **Observed:** The reproduced registry version dialog named version, date,
  `best_score`, status, and Deploy. The deployment screen named its active
  model, full job ID, date, artifact URI, and Deactivate. These are separate
  route-local displays: the records and identifiers were not interactive links
  to each other or to Jobs in the reproduced UI.
- **Inferred:** `JobsPage.tsx` renders table rows but no row/detail action;
  `JobsDrawer`/`JobDetailsView` separately provide overview, logs, and
  non-terminal cancellation. There is no retry action in either path.
  `DeploymentsPage.tsx`, `DataDriftPage.tsx`, `ErrorLogPage.tsx`,
  `SlowNodesPage.tsx`, and `AuditLogPage.tsx` contain no router navigation
  from operational identifiers. The monitoring contracts carry `job_id`,
  `pipeline_id`, `node_id`, and `sample_node_id`, but the route pages expose
  them as text rather than a common investigation context.
- **Inferred:** Existing Playwright route coverage includes `/jobs`, while
  `a11y.spec.ts` covers only `/`, `/canvas`, `/data`, and `/eda` and does not
  include `/jobs`. Registry, Deployments, Drift, Errors, Slow Nodes, and Audit
  Log are absent from both route-smoke and axe coverage. Unit tests do not
  supply operational history, alert, resolve/reopen, cancellation, retry,
  deployment mutation, or audit-version fixtures. Therefore the audit cannot
  claim reproduced lifecycle transitions, mutation success/failure, or
  populated drift/audit histories. The findings below distinguish that missing
  evidence from the observed static states and code-supported behavior.
- **Inferred:** Although no populated Audit Log fixture was available,
  `AuditLogPage.tsx` renders each entry's actor (`user_id` or anonymous),
  timestamp, save action kind, version, and added/removed/modified node diff.
  Its evidenced gaps are dataset/limit-only filtering, no explicit time-range
  or retention/scope explanation, and no links or correlation to other
  operational records—not missing attribution or change detail.

- **OPS-001 — Observed: the Jobs route presents status history but does not
  enter an actionable job investigation.**
  - **Evidence:** Chrome showed completed and failed job rows with status,
    truncated ID, type, one detail value, duration, and created time. The rows
    have no visible action or detail entry point. In code, `JobsPage.tsx`
    filters and paginates a task pool, while `JobDetailsView.tsx` has a
    different drawer-only overview/logs/cancel experience; it supports
    cancellation for non-terminal jobs, but no retry.
  - **Problem:** A failed or long-running job discovered in the operational
    history cannot be opened into its error, logs, source pipeline, dataset,
    model/version, or a recovery action. Users must locate the same opaque ID
    manually, while terminal and active states provide inconsistent next steps.
  - **Surfaces:** Jobs route; Canvas Job History drawer; job overview/logs;
    completed, failed, queued, running, cancelled, and paginated histories.
  - **Proposed behavior:** Make each job record a single durable investigation
    target, with status phase/progress/timing, task/model/dataset/pipeline
    context, logs/error, and only supported actions (cancel while active;
    retry/clone when the backend can safely supply one). Preserve the current
    Jobs filters and scroll position when closing or following a related
    record.
  - **Acceptance criteria:** Every displayed job has an accessible Details
    action; its detail view names full ID, status transition timestamps,
    progress or an honest “not reported,” inputs, result/model version, and
    error/log state. Active and terminal actions state availability and
    outcome; unavailable retry is not implied. Back returns to the same
    tab/search/status/page. Related model, deployment, dataset, pipeline, and
    error links carry the exact job context.
  - **Validation method:** Seed queued/running/progress/completed/failed/
    cancelled training, tuning, EDA, and ingestion fixtures plus multiple
    pages. Exercise search/status/task filters, Load More, details, logs,
    cancel confirmation/success/failure, retry-supported and retry-unavailable
    outcomes, and Back at 1440 and 390 px with keyboard and live-status checks.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** L. **Risk:**
    Medium. **Dependencies:** jobs/status/log APIs, `useJobStore`,
    `JobsPage`, `JobsDrawer`, job-detail contract, **OPS-007** operational
    context, and **FND-003**. Do not duplicate ingestion/EDA lifecycle work in
    **DAT-003**/**DAT-005**; consume its source-specific provenance.
    **Milestone:** Next. This follows **OPS-007** because the related-record
    links and return context require its serializer/boundary; no independent
    slice is claimed here.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live, 1440 px, `/jobs`):** a fresh
    table load reproduces completed/failed rows with status, truncated ID,
    model type, one metric value, duration, and created time; no row/cell
    click produces navigation, a modal, or any visible affordance, and the
    only other control is a Status-only Filters facet. **Observed (live,
    Canvas Job History drawer):** opening a completed job card reproduces
    the `JobDetailsView` Overview/Live Logs dialog exactly as described —
    Status/Dataset/Duration, Execution Results, full Tuning Configuration,
    Best Score, and an Evaluation Metrics table — with no Retry action
    (terminal job) and no cross-links to the source dataset, registry
    version, or deployment. **Observed (source, `Jobs.tsx`):** `pool`/
    filter/search/tab state remains local `useState`, not URL-synced; row
    rendering has no `onClick`/link wrapper; the status filter remains
    hardcoded to `all|completed|failed`. **Observed (source,
    `JobDetailsView.tsx`/`JobsDrawer.tsx`, `components/panels/jobs/`,
    globally mounted at `MainLayout.tsx`):** the drawer remains
    architecturally decoupled from the `/jobs` route; `useJobStore().
    cancelJob` remains the only mutation exposed, with no retry mutation in
    the store. **Responsive (1440/1024/768/390 px):** the 390 px clipping
    from `FND-001` reconfirms at every width tested; at 768 px the tab row
    and table header text are also clipped. **New this rerun, reported as
    OPS-008 rather than folded here:** a duplicate-row/colliding-React-key
    defect was found in the same Jobs table during this pass; it is a live
    rendering/data-integrity bug distinct from OPS-001's missing-
    investigation-affordance framing, so it is not treated as a revision to
    this finding.
  - **Delta:** No change. All originally cited behavior (no row action,
    drawer-only detail, cancel-only, no retry, no cross-links, local filter
    state, Status-only filter) reproduces exactly on current code and
    current live data.

- **OPS-002 — Observed: model registration and deployment history do not form
  a traceable model-to-deployment decision chain.**
  - **Evidence:** Chrome showed a model's version dialog with its score and a
    Deploy button, then a separate active-deployment record with the same kind
    of job ID and artifact URI. The IDs were text, not links. In code,
    Registry deploys by `version.job_id`; Deployments fetches active/history
    independently and only redeploys an inactive historical job.
  - **Problem:** An operator cannot reliably answer which training job,
    dataset/split/metric, registered version, artifact, deployment event, and
    current inference target belong together—or return to that evidence after
    a deploy/redeploy/deactivate decision.
  - **Surfaces:** Model Registry filters/list/version dialog/artifacts;
    Deployments active card/history/redeploy; Jobs; Experiments/Evaluation;
    Inference deployment picker.
  - **Proposed behavior:** Give the model version a stable lineage record and
    make Registry, Deployments, Jobs, and Inference render the same
    model-version/deployment identity with bidirectional deep links. Retain
    confirmation, pending, success, failure, and inactive/replaced history on
    the initiating surface and on the resulting deployment record.
  - **Acceptance criteria:** A version identifies source job, dataset,
    evaluation provenance, artifact/version, deployment state, actor/time, and
    replacement relationship; the active deployment identifies the same
    version rather than just a job ID. Deploy/redeploy/deactivate confirmations
    name the exact before/after model; success and failure leave a durable,
    refresh-safe record; every related link opens the exact record and offers a
    return path.
  - **Validation method:** Mock two model families, multiple versions, an
    active replacement, inactive history, no deployment, successful and failed
    deploy/redeploy/deactivate mutations, and stale refresh. Assert
    confirmation copy, disabled duplicate actions, route/query context,
    Back/Forward, refresh persistence, and keyboard/modal feedback at desktop
    and 390 px.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** L. **Risk:**
    High. **Dependencies:** registry/deployment version schema, job/evaluation
    provenance, artifact metadata, **OPS-007** operational context, and
    **EXP-005** threshold
    provenance. Reuse **FND-003** for shared async announcements and
    **FND-004** for route-fetch retry rather than creating parallel mechanisms.
    **Milestone:** Next. This follows **OPS-007** because the bidirectional
    links and contextual return require its serializer/boundary; no independent
    slice is claimed here.
  - **2026-08-07 status:** Confirmed, with one addendum.
  - **Current evidence:** **Observed (live, 1440 px, `/registry`):** "View
    Versions" reproduces the version-history dialog (version, date,
    `best_score`, status, per-row Deploy/View Artifacts); "View Artifacts"
    reproduces the plain-text artifact/pipeline-step file list with no link
    to the originating job or dataset. At least one live registry entry
    shows `model_type: "unknown"` and `dataset_id: "unknown"` — a real
    backend data gap that further supports the "cannot reliably answer which
    ... belong together" framing, not a UI regression. **Observed (live,
    1440 px, `/deployments`):** the Active Deployment card and Deployment
    History reproduce exactly as described — full Job ID as plain unlinked
    text, artifact URI, Deactivate, and an empty Actions column for the sole
    history entry; no link from either surface back to Jobs or a Registry
    version detail. **Observed (source, `ModelRegistry.tsx`,
    `useModelRegistry.ts`):** Deploy still calls `POST
    /deployment/deploy/{jobId}` keyed only by `version.job_id`; Deployments
    still fetches active/history independently via separate hooks; IDs
    remain text, not links, throughout. **New addendum, not previously
    documented — client-side-only "manual deployment" tracker
    (source-confirmed, `ModelRegistry.tsx` lines 53–68, 261–318):**
    `ModelRegistry.tsx` maintains a `localStorage` key
    `skyulf_manual_deployments` that lets a user check a "Manual" deployment
    checkbox per registry row (keyed by `${model_type}-${dataset_id}`),
    rendering that row as deployed (`isManuallyDeployed`) purely in this
    browser's local storage. The checkbox is `disabled` (cannot override)
    only when the backend already reports `deployment_count > 0` for that
    row; for every other row a user can locally mark a version "deployed" in
    the Registry UI with no corresponding record in Deployments' active or
    history data, and this marking is invisible to any other browser/session
    since it is not persisted server-side. This is the same lineage-
    consistency surface OPS-002 already targets — a version can appear
    "deployed" in Registry while the real Deployments record disagrees, with
    no reconciliation — so it is treated as an addendum to this finding's
    evidence rather than a new ID. When this finding's Proposed behavior/
    Acceptance criteria are next revised, they should add: "the Registry's
    deployed-state display must derive from the same source of truth as the
    Deployments active record, with no client-only override that other
    sessions cannot see." **Responsive (390 px, `/registry`):** reconfirms
    `FND-001`-style sidebar clipping — not new content, but confirms this
    journey is also blocked by the layout defect at narrow widths.
  - **Delta:** No change to the finding's status or core claim. The
    manual-deployment `localStorage` behavior is a newly documented,
    concrete instance of the same lineage-consistency risk the finding
    already describes, added here as evidence to inform its next revision.

- **OPS-003 — Inferred: drift detection has no durable alert-to-investigation
  and remediation lifecycle.**
  - **Evidence:** Chrome reproduced only the no-report state and locally
    editable PSI, KS p-value, Wasserstein, and KL-divergence thresholds.
    `DataDriftPage.tsx` retains selected job, file, thresholds, and report in
    page state, refreshes per-job history after a calculation, and displays a
    report/filter/export when present. Its API returns history metadata but no
    alert ownership, acknowledgement, severity, deployment, or resolution
    context; no populated drift fixture exists.
  - **Problem:** A drift signal cannot be reliably triaged: the user cannot
    see which deployed model/version and reference/current data it affects,
    why it crossed the threshold, whether thresholds changed since the prior
    check, who investigated it, or the next safe action.
  - **Surfaces:** Data Drift reference-job selector, thresholds, report,
    history, sidebar drift badge, Registry/Deployments, Jobs, Errors, and
    Audit Log.
  - **Proposed behavior:** Persist each drift check as an investigation record
    with reference job/model/deployment, current-data identity, evaluated
    threshold set/version, feature evidence, severity, status, owner, linked
    response actions, and links back to the originating alert. Acknowledge or
    resolve only when an explicit disposition is recorded.
  - **Acceptance criteria:** Every drift alert states affected resource,
    detection time, severity/reason, threshold values, and report/history
    identity; it deep-links to filtered feature evidence and related
    model/deployment/job. Threshold changes are versioned, not silently
    attributed to old reports. Empty, request-failed, partial, acknowledged,
    resolved, and re-opened histories name the next action without treating
    “no report” as “no drift.”
  - **Validation method:** Mock no reference, upload/error, no-drift,
    warning/critical feature drift, schema drift, repeated checks under changed
    thresholds, alert acknowledgement/resolution, and deployment replacement.
    Verify badge→report→deployment/job navigation, history filters, CSV
    provenance, retry behavior, screen-reader status, and narrow/desktop
    layouts.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** L. **Risk:**
    High. **Dependencies:** persisted drift/alert schema, threshold versioning,
    deployment lineage, notification routing, **OPS-007** operational context,
    and **FND-003**. This extends operational investigation; it does not
    duplicate **DAT-007**'s chart interpretation contract. This follows
    **OPS-007** because its alert, report, and related-record links consume the
    shared serializer/boundary; no independent slice is claimed here.
    **Milestone:** Next.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live, 1440 px, `/drift`):** live data
    still has no drift report; the "No Drift Report Yet" empty state
    reproduces exactly, with a reference-job selector, Upload CSV/Parquet
    control, a disabled "Run Analysis" button, a "Refresh jobs" control, and
    a "Drift thresholds" button. **Inferred (source only — thresholds
    dialog contents):** not confirmed via a completed live open this pass
    (tab reclaimed mid-interaction by a sibling agent sharing this session's
    browser — see Task 6 Method above). Confirmed instead via
    `core/api/monitoring.ts`: `DriftThresholds` (PSI/KS/Wasserstein/KL
    fields) remains a page-state object passed per-request to the analysis
    call; `DriftHistoryEntry` still has no threshold-snapshot field,
    confirming thresholds are not versioned against history exactly as this
    finding states. **Observed (source, `DataDriftPage.tsx`):** selected
    job/file/thresholds/report still live in page `useState`, refreshed
    per-job after each calculation; no `severity`, `acknowledged`, `owner`,
    or `resolved` field exists anywhere in the drift API surface.
  - **Delta:** No change. The finding remains correctly framed as Inferred
    for lifecycle states beyond the empty state, since no populated drift
    fixture exists in this environment either.

- **OPS-004 — Observed: Error Log generic identifier search lacks typed
  investigation facets and resource handoffs.**
  - **Evidence:** Chrome reproduced Events and Issues, search, 1h/6h/24h/7d/
    All controls, Show resolved, Resolve, and a route-specific 500 event.
    Expanding that event revealed a traceback but no resource deep link or
    severity filter. `ErrorLogPage.tsx`'s generic search already matches an
    HTTP event's `job_id` and a pipeline log's `node_id` (as well as text
    fields), while the API request only receives time and resolved state. It
    exposes neither typed severity/resource facets nor route navigation from
    those identifiers. Mutation outcomes are untested and were not submitted
    in this audit.
  - **Problem:** Generic search can find a known HTTP `job_id` or pipeline
    `node_id`, but an investigator cannot use dedicated, composable resource
    facets to distinguish the affected record or follow it into Jobs/Canvas
    with its time and filter context. A route string, HTTP code, and raw
    traceback remain the only investigation detail when no exact identifier is
    already known.
  - **Surfaces:** Error Log Events/Issues/timeline/search/time/resolved
    controls; traceback dialog; pipeline logs; Jobs; Canvas; Data Sources;
    Registry/Deployments.
  - **Proposed behavior:** Retain generic search and add typed, composable
    facets for the server-supported time, resolution, severity, error type, and
    resource identities (`job_id` for HTTP events; `pipeline_id`/`node_id` for
    pipeline logs). Provide a contextual View action only when a target
    identity is present; retain safe diagnostic detail and preserve the active
    facets in that link and export.
  - **Acceptance criteria:** Existing generic search still finds exact HTTP
    `job_id` and pipeline `node_id` values. Explicit facets visibly distinguish
    severity, error type, resolution, and each available resource identity and
    compose without ambiguous text matching. Every identifier with a resolvable
    target has a contextual View action; unlinked events say that no target is
    available. Following a link and returning preserves time/facet context;
    detail retains a copyable diagnostic ID and applies product redaction
    policy.
  - **Validation method:** Mock client/4xx/5xx/pipeline errors, exact HTTP
    `job_id` and pipeline `node_id` searches, warning/error/critical
    severities, linked and unlinked resources, resolved rows, and fetch
    failure. Exercise generic search, each facet combination, issue/event
    detail, contextual links, export, and return at 1440 and 390 px with
    keyboard and arrival-focus assertions.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** L. **Risk:**
    Medium. **Dependencies:** monitoring resource-facet schema, correlation/
    resource mapping, **OPS-007** operational context, redaction policy, and
    **FND-003**/**FND-004**. **Milestone:** Next. This follows **OPS-007**
    because the contextual View actions and return state require its
    serializer/boundary; no independent slice is claimed here.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live, 1440 px, `/errors`):** Events(30)/
    Issues(6) tabs, stat cards (HTTP events: 30, Server errors: 30, Pipeline
    failures: 0), an hourly bar chart, a generic Search box, time-range
    buttons (1h/6h/24h/7d/All), a "Show resolved" toggle, and a populated
    table of 500-level exception events with per-row "✓ Resolve" and
    "Traceback" buttons all reproduce exactly. The Node/Route column
    continues to render plain-text values (e.g.
    `/api/pipeline/datasets/273/schema`, `celery/pipeline`) — text, not
    links. **Inferred (source only — Traceback dialog contents, Resolve
    mutation outcome):** the Traceback dialog was not re-opened to
    completion this pass before the tab was reclaimed by a sibling agent; the
    route string + HTTP code already visible in the row is corroborated by
    source: `ErrorLogPage.tsx`'s generic search still matches an HTTP
    event's `job_id` and a pipeline log's `node_id` as substrings of a
    combined searchable text field, while the actual API request
    (`monitoring.ts`) still only accepts time-range and resolved-state
    parameters — confirming no typed severity/resource facet exists
    server-side to expose. No Resolve mutation was submitted, per the
    no-destructive-mutation constraint.
  - **Delta:** No change from the original finding.

- **OPS-005 — Observed: Slow Nodes identifies aggregate cost but cannot lead
  an operator to the slow run, node configuration, or remediation.**
  - **Evidence:** Chrome showed lookback/top-N controls, aggregate metrics,
    sortable columns, and sample node IDs rendered as “e.g.” text. The
    `SlowNodesResponse` only supplies aggregate step statistics and optional
    `sample_node_id`; `SlowNodesPage.tsx` renders no row action or navigation.
  - **Problem:** Aggregate p95/total cost can suggest a candidate, but it
    cannot establish whether a particular pipeline, dataset, run, configuration,
    or deployment caused the cost. “Use it to spot” does not provide an
    investigation or remediation path.
  - **Surfaces:** Slow Nodes lookback/top-N controls, chart/table, Jobs,
    Canvas node properties, Audit Log, Error Log, and affected deployments.
  - **Proposed behavior:** Allow an aggregate to open a time-bound performance
    investigation with contributing run IDs, sample/representative node
    identity, dataset/pipeline/version context, distribution/outlier evidence,
    and context-preserving links to the job and Canvas configuration. Clearly
    distinguish aggregate estimates from one run's measurement.
  - **Acceptance criteria:** Every aggregate states window, run count, unit,
    sort direction, and whether its sample is representative; an accessible
    Investigate action lists contributing runs or honestly says unavailable.
    Links carry step/node/run/time context; returning restores lookback, top-N,
    sort, and row. Empty/error/stale data names its refresh/recovery state.
  - **Validation method:** Mock no runs, one run, many runs, missing sample
    node, outlier p95/max, fetch failure, and a representative run with saved
    pipeline context. Assert sorting, labels, investigation drill-down,
    context-return, keyboard control, chart/table equivalence, and responsive
    geometry.
  - **Impact:** Medium. **Frequency:** Occasional. **Effort:** M. **Risk:**
    Medium. **Dependencies:** slow-node drill-down API, run/pipeline snapshot
    retention, **OPS-007** operational context, and **CAN-002**
    node-addressable diagnosis. This follows **OPS-007** because its
    investigation links and return state consume the shared
    serializer/boundary; no independent slice is claimed here.
    **Milestone:** Later.
  - **2026-08-07 status:** Confirmed, with direct live confirmation added.
  - **Current evidence:** **Observed (live, 1440 px, `/slow-nodes`):**
    lookback controls (24h/7d/30d/90d), Top-10/25/50 controls, a Refresh
    button, summary stats (Step types: 8, Node runs: 65, Jobs scanned: 12,
    Window: 7 days), a "Total time by step type" bar chart, and a sortable
    table all reproduce; each step-type row's `sample_node_id` renders as
    literal "e.g. `<uuid-like string>`" text (e.g. "e.g.
    classification-2b51bfcd-…") confirmed to have no click handler and no
    drill-down. **Observed (source, `SlowNodesPage.tsx`, `monitoring.ts`
    `SlowNodesResponse` type):** the response shape still supplies only
    aggregate step statistics plus an optional `sample_node_id` string; no
    run-ID list, no dataset/pipeline/deployment linkage field exists to
    render a drill-down into even if the UI wanted one.
  - **Limitation:** the responsive (1024/768/390 px) pass for this specific
    page was not completed live this session (shared-browser/time
    constraints); given `FND-001`'s confirmed uniform sidebar/layout
    clipping pattern at 390 px across every other Operations page tested,
    the same clipping is expected here but is **Inferred, not directly
    Observed**, for this page.
  - **Delta:** No change from the original finding. The original audit already
    recorded live Chrome evidence for this page (lookback/top-N controls,
    aggregate metrics, sortable columns, and sample-node-ID `e.g.` text), so
    the prior evidence was **not** source-only; this rerun re-confirms that
    same behavior live at 1440 px and adds source-level confirmation that
    `SlowNodesResponse` still exposes no run-ID/pipeline/deployment drill-down
    field for the UI to render.

- **OPS-006 — Inferred: Audit Log has attributed version/diff entries but lacks
  filter, retention, and cross-record investigation context.**
  - **Evidence:** Chrome rendered Dataset and limit controls and the explicit
    empty state “No saves recorded”; no history fixture was available.
    `AuditLogPage.tsx` calls `pipelineVersionsApi.audit(datasetId, limit)`;
    populated entries render actor, timestamp, action kind, version, and
    added/removed/modified node diffs, but the page exposes no time, actor,
    action, version, resource, or related-record filter. It also does not
    explain the returned history's time scope, ordering, retention, or
    availability, and it supplies no related job/deployment/run links.
  - **Problem:** Operators can see who saved which version and the node-level
    change, but cannot focus that existing history on an incident time window
    or event type, tell whether older records are outside the limit or no
    longer retained, or correlate a version with a related job, deployment, or
    run. The empty state likewise gives no retention or time-scope explanation.
  - **Surfaces:** Audit Log dataset picker/limit/entries/details; Canvas save
    and Save as version; Jobs; Registry/Deployments; Drift; Errors.
  - **Proposed behavior:** Preserve the existing actor/timestamp/action-kind/
    version/diff entry detail. Add server-supported time-range, actor, action,
    version, and resource filters; state the returned history's scope,
    ordering, page limit, and retention/availability policy; and link an entry
    to a related job, deployment, or run only when the API supplies that
    correlation.
  - **Acceptance criteria:** Every populated entry continues to identify its
    actor, timestamp, action kind, version, and node diff. Dataset plus
    server-supported time-range/action/actor/version/resource filters are
    visible, composable, URL-restorable, and retained through Back. The page
    explains result ordering, limit coverage, and retention/availability in
    normal, empty, filtered-empty, access-denied, expired, and request-failed
    states. Related records open contextually when supplied; otherwise the
    entry explicitly identifies the missing correlation.
  - **Validation method:** Mock multiple datasets/pipelines, actors,
    timestamps, action kinds, versions, changed/unchanged node diffs, time
    windows, linked/unlinked jobs and deployments, retention/permission/fetch
    failures, and pagination. Assert existing entry detail, typed filters/query
    restoration, scope/retention copy, contextual links, Back, keyboard
    expansion, and desktop/mobile tables.
  - **Impact:** Medium. **Frequency:** Occasional. **Effort:** L. **Risk:**
    Medium. **Dependencies:** pipeline-version audit filtering/correlation API,
    identity/retention policy, version graph snapshots, **OPS-007** operational
    context, and **EXP-004**'s baseline/candidate snapshot contract. This
    follows **OPS-007** because its contextual filters/links and return state
    consume the shared serializer/boundary; no independent slice is claimed
    here.
    **Milestone:** Later.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Observed (live, 1440 px, `/audit`):** the Dataset
    combobox (populated with dozens of real dataset entries, e.g. "test
    source (3)", "s3 source (5)", "f7009f5b-b59b-…csv (9)") and Limit control
    (25/50/100/200) reproduce, along with the exact empty-state copy "No
    saves recorded for this dataset yet. Saves appear here automatically once
    you click Save on the canvas." for the default-selected dataset. No
    time-range, actor, or action-type filter exists anywhere on the page;
    the only two facets remain Dataset and Limit. **Inferred (source
    only — populated-entry rendering: actor/timestamp/diff detail):** not
    re-confirmed against a populated dataset this pass (a tab reclaim by a
    sibling agent interrupted the dataset switch before a populated entry
    could be captured); confirmed instead via source: `AuditLogPage.tsx`
    still calls `pipelineVersionsApi.audit(datasetId, limit)` and renders
    actor (`user_id` or anonymous), timestamp, save action kind, version,
    and added/removed/modified node diffs when entries exist — matching this
    finding's original Inferred framing exactly.
  - **Delta:** No change from the original finding.

- **OPS-007 — Inferred: Operations lacks a shared typed context-serialization
  and record-link primitive.**
  - **Evidence:** The operations route sources do not call router navigation;
    identifiers in Jobs, Deployments, Drift, Errors, Slow Nodes, and Audit Log
    remain local display/filter state. No shared parser/serializer or link
    helper owns operational identities, origin, or time/filter context. Tests do
    not exercise query parsing, round-tripping, or operational handoffs.
  - **Problem:** If each Operations view adds deep links independently, it will
    hand-roll route keys, parsing, invalid-value handling, and return-state
    semantics. That makes cross-page investigations brittle and raises the risk
    of copying the wrong record, losing origin/time scope, or treating a
    truncated label as identity.
  - **Surfaces:** Shared Operations link/query-state utilities and the future
    Jobs, Registry, Deployments, Drift, Errors, Slow Nodes, and Audit Log
    consumers that adopt them.
  - **Proposed behavior:** Define one typed operational-context contract plus a
    shared serializer/parser and contextual record-link primitive for job,
    pipeline/version, dataset, model version, deployment, drift check,
    incident, node, time range, and origin. Consumer rows/details adopt that
    primitive later rather than inventing their own query keys or return-state
    handling.
  - **Acceptance criteria:** A typed schema exists for supported operational
    identities, origin, and time/filter fields. Valid contexts serialize to a
    copyable URL shape and parse back without losing meaning; partial, unknown,
    invalid, deleted, or unauthorized values degrade safely without inventing a
    target. The shared record-link primitive builds href/return payloads from
    typed input and exposes accessible link text without requiring consumer-
    specific routing logic.
  - **Validation method:** Add focused unit/component tests for schema typing,
    serializer→parser round trips, invalid/partial query inputs, backward-
    compatible parsing, and shared record-link href generation/copy behavior.
    Manual verification uses representative job/model/deployment/error/drift/
    slow-node/audit contexts in a harness or Storybook-style fixture, then
    reloads the generated URL to confirm the same parsed payload.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** shared operational context schema,
    router/query-state serialization, and stable Operations API identities. This
    is the Operations-specific continuity foundation; later views reuse it
    rather than duplicating it beside **FND-003** status semantics or
    **FND-004** retry behavior.
    **Milestone:** Now.
  - **2026-08-07 status:** Confirmed.
  - **Current evidence:** **Source-verified (no live-only claim possible for
    an absence):** grepped Jobs, Registry, Deployments, Drift, Errors, Slow
    Nodes, and Audit Log route/page sources for `useSearchParams`, `<Link`,
    or any shared record-link/query-serialization helper — none exists.
    Every page's filter/search/tab/selection state remains local `useState`,
    confirmed directly for `Jobs.tsx`, `DataDriftPage.tsx`,
    `ErrorLogPage.tsx`, `SlowNodesPage.tsx`, and `AuditLogPage.tsx`, and by a
    dedicated `explore` sub-agent re-check for `ModelRegistry.tsx`/
    `DeploymentsPage.tsx`. No tests exercise query-state parsing or
    round-tripping for any Operations page. **Live corroboration:** every
    page visited in this pass reset its filter/tab/search state on reload or
    route re-entry (e.g., returning to `/jobs` after opening the Canvas Job
    History drawer did not preserve any prior Jobs-table filter selection
    state across the two separate surfaces, consistent with them being fully
    decoupled per OPS-001).
  - **Delta:** No change from the original finding. Still fully absent.
  - **2026-08-07 resolution:** Primitive landed. `core/utils/operationalContext.ts`
    defines the typed `OperationalRef` union (job, pipeline, node, dataset,
    modelVersion, deployment, driftCheck, incident, auditEntry, slowNode) plus
    `origin`, `timeRange`, and `filters`, with `serializeOperationalContext`/
    `parseOperationalContext`/`buildRecordHref`/`describeOperationalRef`.
    Numeric server ids stay numeric across a round trip; unknown kinds, blank
    and malformed identifiers parse to `null` rather than inventing a target;
    unrelated params and unrecognised time ranges are ignored so links stay
    forward/backward compatible. `components/shared/RecordLink.tsx` builds the
    href and return payload from a typed ref and exposes the full identifier as
    its accessible name even when the visible label is truncated. Covered by 35
    tests (`operationalContext.test.ts`, `RecordLink.test.tsx`). **Consumer
    adoption by the individual Operations pages remains outstanding** and is
    tracked by the dependent findings (`OPS-001`–`OPS-006`), which this
    foundation unblocks.

- **OPS-008 — Observed (new): the Jobs table renders duplicate rows with
  colliding React keys on fresh load, driven by a `poolSkip` closure race in
  the auto-load-more effect.**
  - **Evidence:** On a fresh, uncached `/jobs` page load, the table
    intermittently renders duplicate rows for the same `job_id` (e.g.,
    `7c1ec203…`, `a9d86dae…`, `250f18f2…` each appearing twice in the same
    render), accompanied by a live React console error: `Warning:
    Encountered two children with the same key, '<job_id>'. Keys should be
    unique...` at `<tbody><table>`. This was reproduced independently on two
    separate fresh tab loads of `/jobs` in this session, not a one-off
    artifact of a prior action — the visible symptom is a table that briefly
    (or persistently, until a manual Refresh) shows more job rows than
    distinct jobs exist, with the duplicated rows scrolling/paginating
    independently of each other. **Root cause, confirmed by direct source
    reading (`Jobs.tsx`):** an auto-load-more `useEffect` (lines 181–190,
    dependency array `[activeTab, pool, poolHasMore, loading,
    registryItems]`) calls `fetchPool(false)` (an append fetch) whenever the
    active tab's matching-job count is below `LIMIT`. `fetchPool` (lines
    69–92) reads `poolSkip` via component-closure state
    (`const currentSkip = reset ? 0 : poolSkip;`, line 78) rather than a
    functional state updater. Two effect invocations that fire before the
    first fetch's state update commits (plausible under React 18
    `StrictMode`'s intentional development-mode double-invoke of effects —
    confirmed present via `main.tsx` lines 33/47 — or under any
    overlapping-request race) can both read the same stale `poolSkip`, each
    fetch the same page of jobs, and both append that page into the `pool`
    array via `setPool(prev => reset ? fetchedJobs : [...prev,
    ...fetchedJobs])` (line 85). The table's row `key={job.job_id}` (line
    339, confirmed) then collides for every duplicated entry.
  - **Problem:** A user reviewing the Jobs table on a fresh load may see the
    same job listed twice, be uncertain whether two attempts actually ran,
    and encounter a broken/duplicated key state that React explicitly warns
    is unsafe (list reordering, incorrect DOM reuse, or dropped updates are
    the documented risks of duplicate keys in React). This directly
    undermines the same Jobs-investigation trust that **OPS-001** already
    flags as needing durable, accurate job records, independent of
    OPS-001's separate missing-detail-affordance problem.
  - **Surfaces:** Jobs route (`/jobs`) table/pool-loading; the auto-load-more
    effect specifically; potentially the Canvas Job History drawer's job
    list if it shares the same pool/fetch logic (not independently
    confirmed — flagged as an open question, not a claim).
  - **Proposed behavior:** Guard the auto-load-more effect against
    re-entrant/duplicate fetches for the same `poolSkip` (e.g., an
    in-flight-request ref/flag checked before calling `fetchPool`, or a
    functional-updater pattern that reads the latest `poolSkip` atomically),
    and de-duplicate by `job_id` when appending fetched pages into the
    `pool` array (e.g., merge into a `Map` keyed by `job_id` rather than
    concatenating arrays) as a defensive backstop even if the request race
    itself is not fully eliminated.
  - **Acceptance criteria:** A fresh, uncached `/jobs` load under React 18
    `StrictMode` (development) and under production builds never renders two
    rows for the same `job_id`, and the browser console never emits a
    duplicate-key warning for the Jobs table; the auto-load-more effect
    either serializes its fetches or de-duplicates the resulting `pool` by
    `job_id` before render.
  - **Validation method:** Component/integration test that mounts `JobsPage`
    under `StrictMode` with a mocked `jobsApi.getJobs` and asserts the
    rendered row count equals the distinct `job_id` count after the
    auto-load-more effect settles; a second test forces two overlapping
    `fetchPool(false)` calls with the same `poolSkip` and asserts the `pool`
    array contains no duplicate `job_id` values. Reproduced live this rerun
    via two independent fresh-tab loads of `/jobs` at 1440 px with the
    browser console open; not verified across all four widths (the defect
    is state/timing-based, not layout-based, and is expected to reproduce at
    any width) nor confirmed/excluded for the Canvas Job History drawer.
  - **Impact:** Medium (visually confusing and technically unsafe per
    React's own key-collision warning, but the underlying job data itself is
    not corrupted — only the rendered list). **Frequency:** Occasional
    (timing-dependent; reproduced on 2 of a small number of fresh-load
    attempts this session, not on every load). **Effort:** S (a functional
    `poolSkip` updater and/or a `Map`-based de-duplication at one call site).
    **Risk:** Low. **Dependencies:** `Jobs.tsx`'s `fetchPool`/`pool`/
    `poolSkip` state and the auto-load-more `useEffect`; no shared
    Operations primitive required. **Milestone:** Now.
  - **2026-08-07 status:** New.
  - **Current evidence:** Observed live against the real backend on two
    independent fresh-tab loads at 1440 px, with the root cause traced by
    directly reading current `Jobs.tsx` source (line numbers cited above,
    verified against the file as it exists on this branch); this is a live,
    reproducible rendering/data-integrity defect distinct from every other
    `OPS-*` finding, which describe missing investigation affordances,
    lifecycle context, or cross-record links rather than an active rendering
    bug.
  - **Delta:** New finding; no prior entry to compare against.

## Prioritized Findings Inventory

| ID | Evidence | User problem | Surfaces | Impact | Frequency | Effort | Risk | Dependencies | Milestone |
|----|----------|--------------|----------|--------|-----------|--------|------|--------------|-----------|
| FND-001 | Observed | Global navigation and Canvas view controls clip/overlap at 390 px. | Layout; Canvas, Experiments, Inference; Data/EDA; Operations | High | Frequent | M | Medium | Layout; read-only breakpoint | Now |
| FND-002 | Observed | Shell overlays lack a shared focus-containment and focus-return contract. | Canvas, Experiments, Inference overlays; shared Navbar | High | Occasional | S | Low | ModalShell focus helpers | Now |
| FND-003 | Inferred | Async state changes lack shared live-region semantics. | Canvas; Data/EDA; Experiments/Inference; Operations | High | Occasional | S | Low | None | Now |
| FND-004 | Inferred | Route-fetch errors inconsistently offer Retry; Canvas uses a different, toast-scoped pattern. | Dashboard; Data/EDA; Registry; Deployments; Experiments evaluation | Medium | Occasional | S | Low | Page fetch functions | Next |
| FND-005 | Inferred | Canvas, Data/EDA, and Inference forms lack consistent field semantics. | Canvas node forms; Data Sources Add Source; EDA analysis/filter controls; Inference editor; shared controls | High | Frequent | M | Medium | Shared form primitives; node metadata/validation; source/EDA/inference validation | Next |
| FND-006 | Inferred | Shell view selection is not history-restorable or programmatically selected. | Canvas; Experiments; Inference | High | Frequent | M | Medium | useViewStore; retained views | Now |
| FND-007 | Observed | `NotificationCenter` nests an interactive Dismiss button inside another interactive row button. | Shared Navbar `NotificationCenter` (Canvas, Experiments, Inference) | Medium | Occasional | S | Low | None | Next |
| CAN-001 | Observed | Click-added node cards overlap and do not enter configuration. | Canvas palette, graph, Properties panel | High | Frequent | S | Low | Custom-node bounds; Sidebar; selection | Now |
| CAN-005 | Observed | Canvas toolbar clusters overlap and intercept visible actions when Properties narrows the Flow pane. | Canvas Toolbar, Flow viewport, Properties panel | High | Frequent | S | Low | Toolbar responsive layout; panel width | Now |
| CAN-002 | Inferred | Run readiness and failures lack an actionable node-level diagnostic loop. | Canvas run controls, node warnings, Results | High | Occasional | M | Medium | Validators; converter; FND-003 | Now |
| CAN-003 | Inferred | Autosave, recent, and version recovery do not explain unavailable local recovery. | Restore banner; Recent; versions; Toolbar | Medium | Occasional | M | Medium | Persistence; versions; FND-003 | Later |
| CAN-004 | Inferred | Feature Generation exposes a recommendation Apply action that changes nothing. | Feature Generation; Recommendations panel | Medium | Occasional | S | Low | Recommendation schema; FND-005 | Later |
| DAT-001 | Observed | Source onboarding has conflicting destinations and unassociated required fields. | Data Sources; Add Source; Canvas/EDA handoffs | High | Frequent | M | Medium | Source API; router | Now |
| DAT-002 | Observed | Failed preview reports zero-like metadata without recovery context. | Dataset Preview; source profile/sample APIs | High | Occasional | M | Medium | Profile/sample errors; FND-003 | Now |
| DAT-003 | Inferred | Ingestion activity lacks phase/progress/history/retry lifecycle. | Upload; Add Source; jobs; source rows | High | Occasional | M | Medium | Ingestion API; FND-003 | Now |
| DAT-004 | Observed | Data/EDA route controls are clipped at 390 px. | Data table/actions; EDA header/history; Layout | High | Frequent | M | Medium | FND-001; responsive views | Now |
| DAT-005 | Inferred | EDA jobs and history lack durable input/status/recovery context. | EDA selection; jobs; failures; History | High | Frequent | M | Medium | EDA job API; useEDAStore; FND-003 | Now |
| DAT-006 | Inferred | Filters and exclusions apply inconsistently without durable report context. | EDA sidebar; tabs; exports | High | Frequent | M | Medium | EDA schema; FND-005 | Next |
| DAT-007 | Inferred | Charts lose interpretable color/axis/alternative-data context at density and narrow widths. | EDA charts, tables, exports | High | Frequent | L | Medium | Chart themes; exports; report payloads | Next |
| DAT-008 | Observed | EDA Dataset dropdown renders many indistinguishable duplicate-text options. | EDA Dataset selector | Medium | Frequent | S | Low | Dataset list/labeling logic; dataset service | Next |
| DAT-009 | Observed | Chart Download buttons are never disabled for empty/unconfigured charts. | Bivariate/PCA tabs; chart download helper | Low | Occasional | S | Low | `chartUtils.ts` downloadChart; BivariateTab; PCATab | Later |
| EXP-001 | Inferred | Filters can hide selected runs while comparison keeps using them. | Experiments filters, run sidebar, comparison/evaluation/diff tabs | High | Frequent | M | Medium | useJobStore; experiment fixtures | Now |
| EXP-002 | Inferred | Metric comparison does not make direction, comparability, or missingness durable. | Visual/table/branch metric comparison | High | Frequent | M | Medium | Metric metadata; chart/table adapters | Now |
| EXP-003 | Inferred | Conditional explainability/segmentation views conceal availability and comparability. | Feature Importance, SHAP, Segmentation, exports | High | Occasional | M | Medium | Artifact schema; explanation services | Next |
| EXP-004 | Inferred | Pipeline Diff lacks an explicit baseline/candidate decision contract. | Run sidebar, Pipeline Diff, saved graphs | Medium | Occasional | M | Medium | Graph snapshots; graphDiff; job metadata | Next |
| EXP-005 | Inferred | Threshold exploration/tuning/activation lacks a durable decision record. | Evaluation, threshold API, Inference overrides/results | High | Occasional | M | High | Threshold API/version semantics; FND-003 | Now |
| EXP-006 | Observed | Inference permits visibly incomplete/type-incompatible input. | Editor, schema badges/Fix, prediction request | High | Frequent | M | High | Typed artifact schema | Now |
| EXP-007 | Inferred | Inference execution/recovery is not a complete durable run lifecycle. | Run, pending/error/results, history, exports | High | Occasional | L | Medium | Prediction/status API; storage; FND-003 | Now |
| EXP-008 | Observed | Cross-tab run-identifier mismatch (`job_id` vs `pipeline_id`) breaks selection traceability. | Model Evaluation pills; Pipeline Diff header; Feature Importance/SHAP legends | Medium | Frequent | S | Low | `jobMeta.ts` `shortRunId` export | Next |
| OPS-001 | Observed | Jobs history cannot open a unified details/recovery investigation. | Jobs; Job History drawer; logs; related resources | High | Frequent | L | Medium | Job/status/log APIs; useJobStore; DAT-003/DAT-005; OPS-007; FND-003 | Next |
| OPS-002 | Observed | Registered versions and deployments do not form a traceable decision chain. | Registry; Deployments; Jobs; Experiments; Inference | High | Occasional | L | High | Registry/deployment lineage; job/evaluation provenance; OPS-007; FND-003/FND-004 | Next |
| OPS-003 | Inferred | Drift reports lack a durable alert, investigation, and remediation lifecycle. | Drift; alert badge; Registry/Deployments; Jobs; Errors | High | Occasional | L | High | Drift/alert schema; threshold versioning; deployment lineage; OPS-007; FND-003 | Next |
| OPS-004 | Observed | Generic identifier search lacks typed resource facets and contextual deep links. | Error Log; incidents; Jobs; Canvas; Data; Registry/Deployments | High | Frequent | L | Medium | Resource-facet/correlation schema; OPS-007; redaction; FND-003/FND-004 | Next |
| OPS-005 | Observed | Slow-node aggregates cannot lead to the measured run/node or remediation. | Slow Nodes; Jobs; Canvas; Audit Log | Medium | Occasional | M | Medium | Slow-node drill-down API; run snapshots; OPS-007; CAN-002 | Later |
| OPS-006 | Inferred | Attributed version/diff history lacks filters, retention/time clarity, and related-record correlation. | Audit Log; Canvas versions; Jobs; Deployments; Drift; Errors | Medium | Occasional | L | Medium | Audit filtering/correlation API; identity/retention policy; graph snapshots; OPS-007; EXP-004 | Later |
| OPS-007 | Inferred | Operations lacks a shared typed context serializer and record-link primitive. | Shared Operations link/query-state utilities; future rows/details across Operations | High | Frequent | M | Medium | Operational context schema; router/query state; API identities | Now |
| OPS-008 | Observed | Jobs table renders duplicate rows with colliding React keys on fresh load (`poolSkip` closure race). | Jobs table/pool-loading; auto-load-more effect | Medium | Occasional | S | Low | `Jobs.tsx` `fetchPool`/`poolSkip`/auto-load-more effect | Now |

## Component-Boundary Recommendations

Only the following boundaries are recommended. They address a measured
reliability failure or independently testable user-state risk; no split is
recommended merely because a file is large. React Flow continues to own graph
viewport, pan/zoom, handles, and node movement. APIs remain the authoritative
source for ingestion, analysis, deployment, and Operations records.

**Synthesis disclosure — `ClassificationChartsForSplit.tsx` independently
re-reviewed (Task 7):** the Task 5 Experiments/Inference pass flagged that
`ClassificationChartsForSplit.tsx` had not been independently re-reviewed for
component-boundary evidence. This synthesis read the file directly
(`frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx`,
650 lines). Result: **no boundary change.** It is a pure presentational
component that renders all classification charts for one split from props
(`splitName`, `splitData`, `selectedRocClass`, `threshold`, and download
callbacks) plus one memoized `applyThreshold(...)` result. It holds no local
user state, no store subscription, no fetch/mutation lifecycle, and does **not**
reference `job_id`/`shortRunId`/`.slice(0, 8)`, so it is not one of the
components implicated in `EXP-008`'s cross-tab identifier mismatch (that logic
lives in `EvaluationView`, `PipelineDiffView`, `FeatureImportanceView`, and the
SHAP views). Its seven chart Download buttons are each correctly disabled while a
download is in flight (`disabled={downloadingChart === ...}`) and marked
`data-export-ignore`, so it does not exhibit the `DAT-009` empty-chart-download
pattern either. It therefore introduces no measured reliability failure or
independently testable user-state risk and does not warrant its own
component-boundary recommendation; the existing `InferencePage.tsx` and
Experiments boundary reasoning is unchanged.

### `Toolbar.tsx`

- **User-facing risk:** The independently positioned action groups overlap and
  intercept Undo at a measured desktop pane width (`CAN-005`).
- **Current responsibilities:** It calculates two absolute action clusters,
  chooses responsive visibility, and wires graph actions.
- **Proposed boundaries:** `CanvasToolbarLayout` owns one measured pane-width
  allocation and overflow decision; `PrimaryToolbarActions` and
  `SecondaryToolbarActions` render only their supplied actions.
- **Required behavior preservation:** Undo/Redo shortcuts, current action
  labels, disabled states, panel breakpoints, and every existing action remain
  available; React Flow controls do not move into this boundary.
- **Validation:** Unit-test allocation/overflow thresholds; Playwright measures
  non-intersecting rectangles and activates every visible/overflow action with
  both panels at 1440, 768, and 390 px, by pointer and keyboard.

### `EDAPage.tsx`

- **User-facing risk:** Selection, draft inputs, polling, loaded history, and
  exports can describe different analyses (`DAT-005`, `DAT-006`).
- **Current responsibilities:** It coordinates report/job queries, editable
  analysis context, lifecycle actions, header/sidebar state, history, tabs,
  and export context.
- **Proposed boundaries:** `useAnalysisRecord` derives one immutable
  current-report/input/lifecycle record from React Query and `useEDAStore`;
  `AnalysisContextSummary`, `AnalysisLifecycleActions`, and
  `AnalysisHistoryPanel` consume it. Tabs keep chart-specific rendering and
  explanation.
- **Required behavior preservation:** Existing dataset handoff, drafts,
  polling cadence, cancellation/retry permissions, history loading, and
  tab-specific content remain unchanged; server report truth stays in React
  Query.
- **Validation:** Unit-test record derivation for no-report, pending, failed,
  completed, stale-history, and dataset-switch fixtures; Playwright validates
  submit/cancel/retry/history/export context, keyboard flow, and VoiceOver
  announcements at 1440, 768, and 390 px.

### `InferencePage.tsx`

- **User-facing risk:** Schema/import/repair, thresholds, transport, results,
  and transient history can attribute an error or result to the wrong
  deployment (`EXP-005`–`EXP-007`).
- **Current responsibilities:** It loads deployment/schema data, imports and
  repairs input, manages overrides, submits predictions, renders results and
  exports, and retains recent-run state.
- **Proposed boundaries:** `useInferenceRunController` owns the immutable
  request snapshot and mutation lifecycle; `InferenceInputPanel` owns
  editing/repair; `InferenceRunResult` and `InferenceRunHistory` render only
  controller snapshots. API clients remain authoritative.
- **Required behavior preservation:** JSON/CSV/sample entry, current
  thresholds, reset, export formats, privacy/retention choices, and existing
  keyboard actions remain available; the split must not persist results that
  are currently deliberately transient.
- **Validation:** Unit-test deployment switches, typed validation, repairs,
  delayed success, failure, cancellation, retry, and export snapshots;
  Playwright completes the inference journey with keyboard and screen-reader
  checks at 1440, 768, and 390 px.

### Operations context utility boundary

- **User-facing risk:** Separate route pages would otherwise invent incompatible
  query keys and return context, risking an investigation of the wrong record
  (`OPS-007`).
- **Current responsibilities:** Identity, origin, time/filter, href, and
  return-state handling are absent or route-local rather than owned by a
  common boundary.
- **Proposed boundaries:** `OperationalContextSchema` owns typed identities and
  safe parse/serialize behavior; `createOperationalRecordLink` owns href and
  accessible link payload generation. Future page rows/details consume them;
  they do not parse query state themselves.
- **Required behavior preservation:** Existing route-local filters and record
  displays remain unchanged until their respective `OPS-001`–`OPS-006`
  consumer work; partial, deleted, and unauthorized values never invent a
  target.
- **Validation:** Unit-test valid, partial, invalid, and backward-compatible
  round trips plus href/copy output; manually reload representative contexts
  with keyboard and screen-reader link names in a desktop, tablet, and mobile
  harness before any consumer integration.

## Now / Next / Later Roadmap

### Now

This is the smallest dependency-complete high-impact set: four shared
foundations, the Operations primitive, and independently complete journey
slices. Re-evaluation finds **19** Now items; every finding-ID dependency named
by one is also in Now (or is an external API/product contract rather than a
later roadmap item). Work foundations first: `FND-001`/`FND-002`/`FND-003`/
`FND-006` and `OPS-007`; then land the dependent Canvas, Data/EDA, and
Experiments/Inference slices. `OPS-008` is the one Medium-impact item admitted
to Now, because its React-key collision is an active data-integrity/reliability
defect (not a cosmetic gap) whose fix is self-contained in `Jobs.tsx`.

- **FND-001:** Make the shared shell and Canvas subview navigation usable at
  390 px without degrading 768 px and desktop workflows.
- **FND-002:** Give shared shell overlays contained, returnable keyboard focus.
- **FND-003:** Add semantic async status/error feedback across shared states.
- **FND-006:** Make shared Canvas, Experiments, and Inference view navigation
  restorable and programmatically selected.
- **CAN-001:** Make click-to-add create a visible, selected, configurable node
  without card collisions.
- **CAN-005:** Keep Canvas toolbar actions non-overlapping and operable with
  both panels open at every responsive width.
- **CAN-002:** Turn Canvas validation and run failures into node-addressable
  diagnosis and recovery.
- **DAT-001:** Make source onboarding accessible and make Canvas/EDA handoffs
  explicit without waiting on shared form-primitives cleanup.
- **DAT-002:** Differentiate unavailable preview metadata from an empty
  dataset and give recovery context.
- **DAT-003:** Make ingestion phase, progress, failure, and recovery coherent.
- **DAT-004:** Make Data and EDA controls reachable at 390 px through
  **FND-001**'s compact-shell work.
- **DAT-005:** Preserve dataset/input/job/history context across EDA outcomes.
- **EXP-001:** Keep selected-run context visible when filters change.
- **EXP-002:** Make metric direction, split/population, availability, and
  winner logic explicit in every comparison.
- **EXP-005:** Make threshold preview/save/enable state versioned, attributable,
  recoverable, and visible at prediction time on the existing
  Experiments/Inference surfaces.
- **EXP-006:** Validate typed deployed-schema input before inference, with
  route-local issue reporting and reviewable repairs.
- **EXP-007:** Give prediction runs durable pending/failure/retry/result/export
  context within the Inference journey.
- **OPS-007:** Define a typed, URL-restorable operational-context schema plus
  serializer/parser round-trip behavior and a shared contextual record-link
  primitive; leave view-specific row/detail adoption to `OPS-001`–`OPS-006`.
- **OPS-008:** Eliminate the duplicate Jobs-row/colliding-React-key defect on
  fresh `/jobs` load by serializing the auto-load-more fetch and/or
  de-duplicating the pool by `job_id`, so the table never renders two rows for
  one job. Self-contained in `Jobs.tsx`; no shared Operations primitive
  required.

### Next

- **FND-004:** Normalize recoverable route-fetch retries after route-specific
  threshold and inference recovery flows are complete.
- **FND-005:** Normalize labels, required-state messaging, and field-error
  relationships in Canvas, Data/EDA, and Inference after route-specific source
  and inference validation fixes land.
- **FND-007:** Remove the invalid button-in-button DOM nesting in
  `NotificationCenter` so the Dismiss control is not a descendant of the row
  button, restoring a valid accessible name and keyboard focus order.
- **DAT-006:** Make filter and exclusion application accessible, explicit, and
  visible in every result/export.
- **DAT-007:** Establish interpretable, responsive, theme-safe chart and
  alternative-data behavior.
- **DAT-008:** Give each EDA Dataset-dropdown option a disambiguating label
  (id fragment, date, or row/column count) so no two options render identical
  visible text.
- **EXP-003:** Explain absent/partial explainability artifacts, missing versus
  zero values, and segmentation metric direction.
- **EXP-004:** Make baseline/candidate roles, snapshot availability, and
  structural-versus-outcome differences explicit in Pipeline Diff.
- **EXP-008:** Use `shortRunId(job)` consistently across every Experiments tab
  that names a selected run, so Model Evaluation, Pipeline Diff, Feature
  Importance, and SHAP use the same identifier the sidebar used to select it.
- **OPS-001:** After `OPS-007`, make every Jobs history record open a
  context-rich details and supported-recovery view with contextual return.
- **OPS-002:** After `OPS-007`, trace every registered model version through
  source job, evaluation/artifact, deployment event, active state, and
  inference target.
- **OPS-003:** After `OPS-007`, persist drift alerts, threshold versions,
  severity, ownership, disposition, and links to the affected
  model/deployment/job.
- **OPS-004:** After `OPS-007`, retain generic identifier search and add typed
  Error Log resource/severity facets with contextual links.

### Later

- **CAN-003:** Make Canvas recovery sources and unavailable autosaves
  understandable before replacing work.
- **CAN-004:** Make Feature Generation recommendations apply once, or remove
  the unsupported Apply action.
- **OPS-005:** After `OPS-007` and `CAN-002`, let Slow Nodes open a
  time-bound contributing-run and Canvas-node investigation.
- **OPS-006:** After `OPS-007` and `EXP-004`, preserve attributed version/diff
  history while adding filters, retention/time-scope clarity, and
  API-supplied related-record correlation.
- **DAT-009:** Disable or hide each EDA chart Download button (Bivariate, PCA,
  and any tab sharing the pattern) until its chart's required inputs are
  selected and rendered, or make `downloadChart` refuse a container holding
  only empty-state content, so a download never silently captures placeholder
  text.

## Historically Resolved Findings

**2026-08-08 implementation pass.** The `2026-08-07` audit rerun recorded
`Resolved` count `0` because no *prior* finding's user problem had been
demonstrated fixed at audit time. Since then an implementation pass has landed
fixes for the findings below. This section is the auditable record; the
`## Finding Status Summary` counts above are deliberately left at their
as-audited values, because they describe the state of the `2026-08-07` audit,
not the state of the codebase today. A future audit rerun should recount from
scratch and move these into its own `Resolved` bucket.

Each entry names the finding, the release the fix shipped in, and where to find
the change. Per-finding evidence lives in `changelog/0.7.x.md`; the finding's
full original entry remains in place above so the problem statement and
acceptance criteria stay readable.

### Resolved in v0.7.5

| Finding | Rank | What changed |
|---------|------|--------------|
| FND-001 | 1 | Responsive app shell and navigation down to 390 px; global controls no longer clip. |
| FND-006 | 2 | Shell-view selection restored through browser history. |
| CAN-001 | 3 | Canvas click-to-add no longer overlaps an existing node; the new node is selected and its settings open. |
| CAN-005 | 4 | Container-aware toolbar; every visible enabled toolbar target has an independent hit area with either panel open. |
| EXP-006 | 5 | Typed inference input validation blocks schema violations before submit. |
| DAT-001 | 6 | Accessible Add Source form; source fields are labelled and success names the next step. |
| OPS-007 | 7 | `RecordLink` operational-context primitive (35 tests) — the shared contract the other OPS findings depend on. |
| DAT-004 | 8 | Data/EDA controls reachable at narrow widths. |
| DAT-005 | 9 | EDA job context: dataset, target, task, filter, and exclusion inputs persist through pending/fail/history outcomes; deep-link fixes. |
| EXP-001 | 10 | Experiments discloses hidden selected runs; evaluation tabs target renderable runs. |
| EXP-002 | 11 | Metric direction explicit across run comparison; missing metric values state why. |
| FND-005 | 12 | `FormField` primitive; shared form semantics across Canvas, Data, EDA, and Inference. |
| FND-003 | 13 | Shared async-state live regions (`LoadingState` / `EmptyState` / `ErrorState`); removed the duplicate toast announcement. |
| FND-002 | 14 | Shared `useModalFocus` focus contract across all overlays. |
| CAN-002 | 15 | `validateGraph` returns structured per-node issues; Results renders a selectable issue list that opens the offending node's settings. |
| DAT-002 | 16 | Dataset preview stops rendering unavailable metadata as `0` and surfaces the backend's real error; sample and profile failures reported independently. |
| DAT-003 | 17 | Ingestion Jobs separates active ingestions from completed history; failed sources get a Retry route. |
| EXP-005 | 18 | Threshold mutations expose scoped pending state, cannot be double-submitted, retry in place, and display their provenance. |
| OPS-008 | 28 | Verified already fixed: `Jobs.tsx` keys on `job.job_id` and dedupes by `job_id`. No code change was required — recorded here so a future audit does not re-open it without re-checking. |
| DAT-008 | 29 | EDA dataset dropdown disambiguates same-named datasets with a stable id suffix (`core/utils/edaDatasetOptions.ts`). |
| FND-007 | 32 | `NotificationCenter` no longer nests the Dismiss `<button>` inside the row `<button>`; `markAllRead()` moved out of the render phase. |
| FND-004 | 31 | Model Registry, Model Evaluation, and Segmentation pass `onRetry` to the shared `ErrorState`; the button disables itself while an async retry is in flight. |
| CAN-004 | 35 | Feature Generation's dead "Apply Recommendation" control removed — recommendations are informational, because the payload does not map onto the node's multi-operation config. |
| DAT-009 | 37 | Bivariate and PCA Download buttons disable until the chart has renderable data and state why. |
| EXP-008 | 30 | Every Experiments tab naming a run by short ID now uses `shortRunId(job)`; an unavailable pipeline id is labelled `Job ID:` rather than shown bare. |
| DAT-006 | 22 | EDA filters use an explicit draft/apply model instead of re-running analysis on every add/remove; filter controls have accessible names and linked numeric validation. **Partial:** the finding's applied-context banner across tabs/exports/history is not implemented. |
| EXP-004 | 27 | Pipeline Diff gained a Swap control, per-side dataset/model/timestamp metadata, and run-specific missing-snapshot messaging. **Partial:** change-list export is not implemented. |
| OPS-006 | 33 | Audit Log gained actor/action/time filters plus scope, limit, and ordering copy. Filters are applied **server-side across the dataset's full history** (`actor`, `kind`, `created_after`, `created_before` on `/versions/{id}/audit`), with server-computed `facets` keeping the dropdowns complete and `total_unfiltered` distinguishing "no history" from "nothing matched". **Partial:** no cross-record links were added, because the payload carries no such correlation. Also fixed a latent diff-walk bug where a pinned version made the route diff every entry against the wrong predecessor. |

**Known gap in this record:** these fixes were validated by unit/component
tests plus `tsc`, `lint`, and `build`, not by re-running each finding's full
`## Validation Matrix` row. In particular the responsive (1440/1024/768/390 px)
and screen-reader passes specified for `FND-001`, `CAN-005`, and `DAT-004` have
not been repeated end-to-end since the fixes landed. Treat the entries above as
"fix implemented and unit-verified", not "acceptance criteria fully
re-validated".

### Resolved in v0.7.6

The last 9 open findings. Same caveat as above: validated by unit/component and
backend tests plus `lint`, `ty`, `tsc`, and `build`, not by re-running each
finding's full `## Validation Matrix` row — no Playwright, responsive, or
screen-reader pass was repeated end-to-end.

| Finding | Rank | What changed |
|---------|------|--------------|
| EXP-007 | 19 | Named prediction runs with pending/cancel/timeout/failure/retry states, 24h persisted history, reload rehydration, and per-run provenance. **Partial:** latency is client-observed only — the backend `PredictionResponse` carries no server-side timing. `shortRunId` was not reused because it keys on training-job identifiers a prediction run does not have. |
| OPS-001 | 20 | Unified Jobs investigation view (timeline, input, error, logs, related `RecordLink`s), `POST /jobs/{id}/retry`, scoped cancel/retry guard, explained unavailable actions, and URL-persisted list-state restoration. **Partial:** deep-link retry for EDA/ingestion job types is limited to cached entries, which are synthesized client-side with no `/pipeline/jobs/{id}` backing. |
| OPS-004 | 21 | Server-side Error Log facets (`severity`/`error_type`/`job_id`/`node_id`) with `facets` + `total_unfiltered`, derived severity classification, and per-error `RecordLink`s. **Partial:** no dataset/model-version links — the `ErrorEvent`/`PipelineRunLog` schema carries no such ids; no traceback redaction, because no shared redaction primitive exists to apply. |
| DAT-007 | 22 | Shared `ChartLegend` (color **and** marker shape) and `ChartDataTable` (table + CSV) primitives across EDA chart families; persistent heatmap gradient legend and always-visible axis labels; 3D scatter theming. **Partial:** export context is carried by the adjacent table/CSV controls rather than embedded in the PNG itself. |
| EXP-003 | 23 | Per-run artifact coverage (`available`/`unsupported`/`not_computed`/`failed` with a reason), explicit normalization scale, hatched not-reported bars, cluster metric direction, and data-table alternatives. **Partial:** Segmentation cannot yet distinguish "not computed for this exact run" per run, because clustering evaluation is fetched only for the active job. |
| OPS-002 | 24 | `previous_deployment_id` lineage, dataset/version enrichment on every history row, `RecordLink` chains with explicit "no target available", scoped mutation errors that retry in place, and removal of a `localStorage` manual-deploy shim. **Partial:** evaluation provenance linking to `EXP-005` threshold data was not included. |
| OPS-003 | 25 | Durable drift alerts: severity, immutable `DriftThresholdVersion` pinning, disposition state machine with audit trail, per-outcome persistence (`completed`/`no_baseline`/`failed`), evidence table, and deployment/model/job links. Also fixed double-nested evidence storage. |
| OPS-005 | 26 | Slow-node aggregates state unit, window, run count, single-run and unrepresentative-sample flags, outliers, and contributing runs, with URL-synced list state and job/Canvas drill-downs. **Partial:** the Canvas link carries node context correctly, but `CanvasPage` does not yet consume it to select the node on arrival. |
| CAN-003 | 27 | `loadCanvasSnapshotDiagnostic()` distinguishes available/empty/corrupt/version-mismatch/storage-error; labelled recovery sources, re-probing restore prompt, non-blocking failure explanation, and fit-and-focus on restore. **Partial:** server versions remain in the Toolbar's Load menu rather than a single unified widget — an empty canvas has no dataset context to query them from. |

Rank numbers in this table are sequential within the v0.7.6 batch and are not
the `### Normalized ranking` positions used by the v0.7.5 table.

### Still open (0)

All 37 findings from the `2026-08-07` audit rerun now have an implemented fix.
The final 9 shipped in `v0.7.6` — see `### Resolved in v0.7.6` above.

Several resolved entries (`DAT-006`, `EXP-004`, `OPS-006`, plus the rows marked
**Partial** in the `v0.7.6` table) shipped a scoped slice rather than the
finding's full proposed behavior; each names its remaining gap in its table row.
Those gaps are deliberately *not* tracked as separate open findings — a future
audit rerun should re-evaluate them against live evidence and decide whether the
residue still constitutes a user problem.

## Validation Matrix

All user-interface rows require desktop (**D**, 1440 px), tablet (**T**, 768
px), and mobile (**M**, 390 px) checks unless the row is a shared utility with
no standalone layout. Each row's accessibility coverage includes keyboard-only
operation (Tab/Shift+Tab plus relevant Enter, Space, and Escape) and a
screen-reader pass with the supported local screen reader; axe is an automated
supplement, not a substitute. The table states any additional coverage.

| Roadmap item | Acceptance criteria | Automated validation | Manual validation | Responsive coverage | Accessibility coverage |
|--------------|---------------------|----------------------|-------------------|---------------------|------------------------|
| FND-001 shared compact shell | No clipped/overlapping global controls; navigation remains available | Playwright viewport geometry and screenshot checks | Navigate every route and Canvas subview | 1440, 1024, 768, 390 px | Keyboard drawer and target-size check |
| FND-002 shell-overlay focus | Focus remains in overlay and returns to invoker | Playwright Tab/Shift+Tab/Escape tests | Shortcuts, Command Palette, notification detail from all shell views | 1440 and 390 px | Focus-order assertions |
| FND-003 async semantics | Status/alert messages announce transitions | Component role tests and axe | Success, empty, error, retry, unavailable action | 1440 and 390 px | Live-region review |
| FND-004 retry consistency | Every recoverable route fetch error retries in place | Page request-failure tests | Preserve filters and selection after retry | 1440 and 390 px | Retry button keyboard operation |
| FND-005 shared form semantics | Controls have labels, required/invalid states, and linked errors | Component accessibility tests and axe | Keyboard-only Canvas configuration, Add Source, EDA filter/setup, and Inference entry | 1440 and 390 px | Accessible-name/error relationship review |
| FND-006 shell-view history | Back/Forward restores selected Canvas, Experiments, or Inference view | Playwright history tests | Verify retained local state | 1440, 1024, 768, 390 px | Selected-state snapshot |
| FND-007 notification button nesting | `NotificationCenter` renders no interactive control as a descendant of another button; each row and its Dismiss are separate focus stops with correct accessible names | Component test asserting no nested `<button>`; render/DOM-nesting assertion (no `validateDOMNesting` warning) | Seed a notification, Tab through the row and Dismiss, confirm distinct focus stops and no concatenated accessible name | 1440 and 390 px | Keyboard focus order and accessible-name review; axe |
| CAN-001 Canvas click-add | New nodes never overlap, are selected, and expose required settings | Playwright palette/drag/palette placement checks | Build a representative pipeline by each insertion method | 1440, 1024, 768, 390 px | Keyboard reachability and focus check |
| CAN-005 Canvas toolbar collision | Every visible enabled toolbar target has an independent hit area with either panel open | Playwright rectangle-intersection and pointer-action checks | Open/close both panels, then undo/redo/load/save and overflow actions | 1440, 1024, 768, 390 px | Focus, menu role, and keyboard Undo/Redo check |
| CAN-002 Canvas diagnosis | Invalid/failing nodes identify a next action and open their settings | Validator and mocked-failure tests | Fix every issue from the Canvas summary | 1440 and 390 px | Summary role, focus, and live feedback |
| CAN-003 Canvas recovery | Local, recent, and server recovery sources and failures are explained | Persistence and version-load tests | Restore/cancel from empty and nonempty canvases | 1440 and 390 px | Keyboard recovery controls and status review |
| CAN-004 Feature Generation recommendations | Apply changes state once or is absent when unsupported | Component recommendation state/undo tests | Compare representative node configuration behavior | 1440 and 390 px | Accessible feedback after apply |
| DAT-001 source onboarding | Every source field is labelled in the Data Sources journey and success names the selected source's Canvas/EDA next step | Form/API outcome and router-handoff Playwright tests | Create file/S3 source; follow both handoffs | 1440 and 390 px | Labels, errors, modal focus, axe |
| DAT-002 preview recovery | Unavailable metadata is never presented as zero; retry has scoped context | Mock sample/profile partial/full failure tests | Empty, deleted, and transient source preview | 1440 and 390 px | Status/alert, retry focus, table scrolling |
| DAT-003 ingestion lifecycle | Each job exposes phase/progress/error/cancel/retry and source context | Upload-progress and status-transition tests | Successful, stalled, failed, cancelled ingestion | 1440 and 390 px | Live updates and duplicate-submit checks |
| DAT-004 Data/EDA compact journey | All source and analysis controls remain in viewport | Geometry, overflow, and compact-menu Playwright tests | Complete filter/handoff/analyze/history tasks | 1440, 1024, 768, 390 px | Keyboard order and 44 px targets |
| DAT-005 EDA job context | Dataset plus target/task/filter/exclusion inputs persist through pending/fail/history outcomes | Mock job/report/history and polling tests | Submit, cancel, retry, load history, switch dataset | 1440 and 390 px | Status/alert and current-report context |
| DAT-006 EDA applied context | Filters/exclusions use labelled draft/apply/reset and annotate all outputs | Component payload/context/export tests | Refine analysis and inspect every tab/export | 1440 and 390 px | Labels, errors, keyboard, axe |
| DAT-007 visualization contract | Legends/axes/tooltips/context/alternatives work in themes and at density | Deterministic chart fixture and visual/axe tests | Interpret, scroll, tabulate, and download each chart family | 1440 and 390 px, light/dark | Non-color meaning and data-table alternatives |
| DAT-008 Dataset dropdown labels | No two Dataset selector options render identical visible text without a disambiguating detail | Component test with a duplicate-name fixture; Playwright option-text uniqueness check | Inspect the selector against a realistic duplicate-heavy dataset list | 1440 and 390 px | Accessible option text and axe |
| DAT-009 chart Download gating | Download is disabled/hidden for empty/unconfigured charts; enabled Download always captures the rendered chart | Component tests for Bivariate/PCA empty and configured states | Attempt Download before and after selecting chart inputs | 1440 and 390 px | Disabled-state semantics and focus |
| EXP-001 selected-run context | Filtered views identify all retained selected runs and their use | Mixed-job selection/filter Playwright tests | Filter, compare, switch tabs/views, clear/restore selection | 1440 and 390 px | Selection summary and keyboard operation |
| EXP-002 metric decision contract | Metric direction/split/units/missingness/winner are explicit | Deterministic metric/branch fixture tests | Compare classification, regression, CV, and partial jobs | 1440 and 390 px, light/dark | Tooltip-independent labels and table semantics |
| EXP-003 explanation/segmentation availability | Artifact coverage, missingness, normalization, and cluster metric direction are explicit | Artifact-state component/visual tests | Compare supported/unsupported/partial SHAP and clustering jobs | 1440 and 390 px, light/dark | Non-color and data/export alternatives |
| EXP-004 Pipeline Diff roles | Baseline/candidate, graph status, and difference direction are unambiguous | Ordered graph-pair, missing/error snapshot tests | Swap roles and inspect equal/changed/failed pairs | 1440 and 390 px | Keyboard role controls and change-list semantics |
| EXP-005 threshold decision lifecycle | Preview/save/enable/clear/provenance cannot be misattributed and failed mutations retry in place on the current surface | Two-job threshold API state-transition tests | Tune, enable, infer, override, clear, retry | 1440 and 390 px | Status/error announcements and control labels |
| EXP-006 typed inference input | Invalid field/value/row shapes and editor-local issue state are actionable before submit | Typed-schema JSON/CSV request-gating tests | Review repair/default/unknown-type input | 1440 and 390 px | Field/error relationships and keyboard repair |
| EXP-007 inference run lifecycle | Pending/failure/retry/results/export/history retain clear provenance on the Inference surface | Delayed/success/failure/cancel/reload/export tests | Execute, reset, retry, reload, restore, export | 1440 and 390 px | Live status, error recovery, and focus review |
| EXP-008 cross-tab run identifiers | Every tab naming a selected run by short ID uses the same identifier the sidebar used to select it | Component test asserting `shortRunId(job)` usage across EvaluationView/PipelineDiffView/FeatureImportanceView/ShapSummaryView; Playwright label-consistency check | Select 2 named runs and compare the label shown in sidebar, Model Evaluation, and Pipeline Diff | 1440 and 390 px | Accessible label consistency |
| OPS-001 job investigation lifecycle | Job details name lifecycle/input/error/log/result context; supported actions recover in place and Back restores list state | After OPS-007, paginated multi-type job, cancel/retry/unavailable-action component and Playwright fixtures | Search/filter/load/details/log/cancel/retry/return | 1440 and 390 px | Detail/action names, live status, keyboard return |
| OPS-002 model deployment lineage | Model/job/version/artifact/deployment/inference identities remain traceable across action outcomes | After OPS-007, multi-version/deployment mutation and deep-link Playwright tests | Deploy, replace, deactivate, redeploy, refresh, follow links | 1440 and 390 px | Confirmation/modal focus and action status |
| OPS-003 drift investigation lifecycle | Alert, severity, threshold version, evidence, owner/disposition, and related resources persist per check | After OPS-007, drift/alert history and transition fixtures | Alert→report→job/deployment, threshold change, acknowledge/reopen | 1440 and 390 px | Alert/status semantics and feature-table navigation |
| OPS-004 Error Log investigation facets | Generic search retains exact HTTP job/pipeline node IDs; typed severity/resource facets and contextual links remain unambiguous | After OPS-007, HTTP `job_id`/pipeline `node_id` search plus facet/link/export Playwright tests | Search, facet, expand, export, follow and return | 1440 and 390 px | Facet names, arrival focus, readable detail |
| OPS-005 slow-node diagnosis | Aggregate source, unit/window, contributing run context, and returnable remediation links are explicit | After OPS-007, aggregate/outlier/no-data/drill-down tests | Sort, investigate, open job/Canvas, return with controls retained | 1440 and 390 px | Sort/button names and chart/table alternatives |
| OPS-006 version audit trail | Existing actor/timestamp/action/version/diff detail remains visible; filters, time/retention scope, and supplied correlations are clear | After OPS-007, multi-dataset/version/actor/time/filter/query-state and linked/unlinked-record tests | Filter, inspect scope copy, reload link, follow and return | 1440 and 390 px | Expandable audit detail and filter semantics |
| OPS-007 operational context primitive | Typed operational identities, origin, and time/filter context round-trip without loss and build shared href/return payloads | Schema/serializer/parser round-trip and shared record-link component tests | Generate representative job/model/deployment/error/drift/slow/audit contexts, copy the link, reload, and confirm the same parsed payload | 1440 and 390 px for primitive rendering | Accessible link names and copyable target semantics |
| OPS-008 duplicate Jobs rows | A fresh, uncached `/jobs` load never renders two rows for the same `job_id`, under both `StrictMode` and production builds | Component/integration test mounting `JobsPage` under `StrictMode`; forced-overlapping-fetch test asserting no duplicate `job_id` in `pool` | Load `/jobs` fresh, inspect rendered row count vs. distinct `job_id` count, watch console for key warnings | 1440 px (timing-based, not layout-based) | No duplicate-key console warning; accurate row count |
