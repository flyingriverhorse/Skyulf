# Frontend UX Roadmap

## Executive Summary

Task 1 establishes the roadmap scaffold and records objective frontend baseline
evidence. Later audit tasks will add findings, prioritization, and delivery
milestones using the evidence labels below.

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
  click it (`preview.spec.ts:113`). The accessibility suite passed its
  critical-only gate while logging two non-blocking serious findings:
  dashboard `color-contrast` and canvas `scrollable-region-focusable`. These
  remain pre-existing UX/test evidence, not fixes.
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

### Async and Feedback States

### Forms and Validation

### Accessibility and Keyboard UX

### Responsive Behavior

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

### Data and EDA

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

### Operations

## Prioritized Findings Inventory

| ID | Evidence | User problem | Surfaces | Impact | Frequency | Effort | Risk | Dependencies | Milestone |
|----|----------|--------------|----------|--------|-----------|--------|------|--------------|-----------|

## Component-Boundary Recommendations

## Now / Next / Later Roadmap

### Now

### Next

### Later

## Validation Matrix

| Roadmap item | Acceptance criteria | Automated validation | Manual validation | Responsive coverage | Accessibility coverage |
|--------------|---------------------|----------------------|-------------------|---------------------|------------------------|
