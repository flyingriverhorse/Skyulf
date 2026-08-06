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

- **FND-006 — Observed: Canvas subviews do not create navigation history or
  announce their selected state.**
  - **User problem:** After opening Experiments, the address remains
    `/canvas`; browser Back leaves Canvas for the prior route rather than
    returning to the Canvas view. The Canvas/Experiments/Inference controls
    also expose three unselected buttons to assistive technology, so the
    current subview is communicated by color only.
  - **Affected surfaces:** Canvas, Experiments, Inference; `Navbar.tsx`,
    `MainLayout.tsx`, and `Breadcrumb.tsx`.
  - **Proposed behavior:** Represent the selected Canvas subview in
    navigation state that participates in Back/Forward (for example, a
    query parameter), and expose the control group with the appropriate
    selected state and accessible name.
  - **Acceptance criteria:** Switching views updates a restorable URL/state;
    Back/Forward returns to the prior Canvas subview without resetting its
    retained local state; the active control is programmatically identified
    without depending on its color.
  - **Validation method:** Playwright navigates Canvas → Experiments →
    Inference → Back/Forward at 1440, 1024, 768, and 390 px; an accessibility
    snapshot asserts one selected/current subview.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** `useViewStore` and retained-view behavior.
    **Milestone:** Now.

### Async and Feedback States

- **FND-003 — Inferred: Shared loading and error states do not announce
  async changes.**
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

- **FND-004 — Inferred: Retry affordances are inconsistent for equivalent
  request failures.**
  - **User problem:** Users can retry failed dashboard, EDA, and deployment
    loads in place, but Model Registry and evaluation error uses can render
    `ErrorState` without a retry, forcing a reload or a route change.
  - **Affected surfaces:** Dashboard, EDA, Model Registry, Deployments,
    Experiments evaluation/segmentation, and their shared `ErrorState`.
  - **Proposed behavior:** Pass a safe, idempotent retry action to every
    recoverable request error, retain the prior selection/filter context, and
    distinguish unavailable actions from a failed request.
  - **Acceptance criteria:** Every recoverable top-level and subview fetch
    error presents a Retry action; activation performs one new request,
    disables duplicate submission while pending, and restores the same
    context on success or a useful error on failure.
  - **Validation method:** Add focused page tests for each current
    `ErrorState` use and Playwright request-failure/retry checks for the four
    journeys.
  - **Impact:** Medium. **Frequency:** Occasional. **Effort:** S. **Risk:**
    Low. **Dependencies:** Existing page fetch functions. **Milestone:** Next.

### Forms and Validation

- **FND-005 — Inferred: Representative node forms do not consistently
  programmatically label controls or communicate required validation.**
  - **User problem:** Visual `span` labels such as “Model Type”, “Encoding
    Method”, and “Selection Method” are not associated with their adjacent
    selects/inputs; required fields are often explanatory text instead of
    required/error semantics. Keyboard and assistive-technology users can
    reach a control without its purpose or invalid state.
  - **Affected surfaces:** Canvas Training, Ensemble, Encoding, Feature
    Generation, and Feature Selection node settings; shared `Input` and
    `Button` primitives. The same convention is available to Data/EDA,
    Experiments/Inference, and Operations forms.
  - **Proposed behavior:** Require an explicit label association (or
    `aria-labelledby`) for shared and node-form controls, expose required and
    invalid states programmatically, and place a persistent error beside the
    field while preserving helpful defaults.
  - **Acceptance criteria:** Every interactive form control has a unique
    accessible name; required fields announce as required before submission;
    invalid fields expose `aria-invalid` and an associated error; Enter
    submits only forms with a defined submit action.
  - **Validation method:** Render representative forms in component tests,
    assert accessible names/required/error relationships, and complete
    keyboard-only configuration at desktop and 390 px; run axe on rendered
    form panels.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** Node configuration metadata and existing
    validation rules. **Milestone:** Next.

### Accessibility and Keyboard UX

- **FND-002 — Observed: the Keyboard Shortcuts overlay lets Tab escape to
  covered Canvas controls.**
  - **User problem:** With the shortcuts overlay open, the next Tab focused
    the covered “More canvas tools” button instead of an element in the
    dialog. Keyboard users can operate hidden controls and lose their
    expected modal focus flow.
  - **Affected surfaces:** Canvas `ShortcutsOverlay`; the same custom-overlay
    pattern is used by Command Palette and Notification detail in Canvas,
    Experiments, and Inference.
  - **Proposed behavior:** Apply the existing modal focus containment and
    focus-return behavior to these custom overlays, with an initial focus on
    the first useful dialog control and Escape/backdrop close where allowed.
  - **Acceptance criteria:** Tab and Shift+Tab remain within each open
    overlay; Escape closes it; focus returns to the invoker; no interactive
    element remains reachable behind a modal. Command Palette retains arrow
    navigation and search focus.
  - **Validation method:** Playwright keyboard tests open/close Shortcuts,
    Command Palette, and Notification detail and assert the active element
    throughout; run existing overlay/component tests.
  - **Impact:** High. **Frequency:** Occasional. **Effort:** S. **Risk:**
    Low. **Dependencies:** `ModalShell` focus helpers. **Milestone:** Now.

- **FND-007 — Observed: Dashboard card metadata fails the minimum text
  contrast ratio.**
  - **User problem:** The small dashboard metadata text uses
    `text-slate-400` on white (2.56:1), so activity context is difficult to
    read for low-vision users and fails WCAG AA.
  - **Affected surfaces:** Dashboard status cards; the repeated muted-text
    visual hierarchy used by summary cards.
  - **Proposed behavior:** Use a muted foreground token/value that reaches
    at least 4.5:1 on the card background without changing the information
    hierarchy.
  - **Acceptance criteria:** Every normal-size status-card metadata label
    reaches 4.5:1 in light and dark themes, and color is not its only status
    cue.
  - **Validation method:** Run axe on Dashboard in both themes and manually
    inspect the cards at 1440, 1024, 768, and 390 px.
  - **Impact:** Medium. **Frequency:** Frequent. **Effort:** S. **Risk:**
    Low. **Dependencies:** Dashboard color tokens. **Milestone:** Now.

### Responsive Behavior

- **FND-001 — Observed: the shared shell is not usable at 390 px.**
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

- **Baseline entry-point mapping:** `/data` mounts the eager `DataSources`
  page, while `/eda` mounts `EDAPage` through `LazyRoute` / `React.lazy`.
  `EDAPage` owns the `EDASidebar` and analysis tabs, and the shared
  `Layout.tsx` also collapses the shell sidebar on `/eda` (same compact shell
  treatment used for `/canvas`).
- **Baseline entry-point mapping:** The only current route/a11y smoke
  coverage for these entry points is `e2e/routes.spec.ts` and
  `e2e/a11y.spec.ts`, both of which exercise `/data` and `/eda`.

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

- **Baseline entry-point mapping:** `/jobs` is the eager `JobsPage`; `/drift`,
  `/registry`, `/deployments`, `/slow-nodes`, and `/audit` are lazy-loaded
  route pages; `/errors` is the eager `ErrorLogPage`.
- **Baseline entry-point mapping:** `Layout.tsx` shows alert badges only on
  `/drift` (`driftAlert`) and `/errors` (`errorAlert`). Current E2E route
  smoke coverage only includes `/jobs`; the remaining Operations routes are
  not listed in the route or a11y specs.

## Prioritized Findings Inventory

| ID | Evidence | User problem | Surfaces | Impact | Frequency | Effort | Risk | Dependencies | Milestone |
|----|----------|--------------|----------|--------|-----------|--------|------|--------------|-----------|
| FND-001 | Observed | Global navigation and Canvas view controls clip/overlap at 390 px. | Layout; Canvas, Experiments, Inference; Data/EDA; Operations | High | Frequent | M | Medium | Layout; read-only breakpoint | Now |
| FND-002 | Observed | Tab escapes the keyboard-shortcuts dialog to covered controls. | Canvas overlays; Command Palette; notifications | High | Occasional | S | Low | ModalShell focus helpers | Now |
| FND-003 | Inferred | Async state changes lack shared live-region semantics. | Canvas; Data/EDA; Experiments/Inference; Operations | High | Occasional | S | Low | None | Now |
| FND-004 | Inferred | Equivalent request failures do not consistently offer Retry. | Dashboard; EDA; Registry; Deployments; evaluation views | Medium | Occasional | S | Low | Page fetch functions | Next |
| FND-005 | Inferred | Form labels, required states, and errors are not consistently programmatic. | Canvas node forms; shared controls | High | Frequent | M | Medium | Node metadata/validation | Next |
| FND-006 | Observed | Canvas subview selection is not restorable with Back/Forward or exposed as selected. | Canvas; Experiments; Inference | High | Frequent | M | Medium | useViewStore; retained views | Now |
| FND-007 | Observed | Dashboard card metadata has 2.56:1 contrast. | Dashboard status cards | Medium | Frequent | S | Low | Dashboard color tokens | Now |

## Component-Boundary Recommendations

## Now / Next / Later Roadmap

### Now

- **FND-001:** Make the shared shell and Canvas subview navigation usable at
  390 px without degrading 768 px and desktop workflows.
- **FND-002:** Contain and return keyboard focus for custom overlays.
- **FND-003:** Add semantic async status/error feedback across shared states.
- **FND-006:** Make Canvas subview navigation restorable and programmatically
  selected.
- **FND-007:** Correct the documented Dashboard metadata contrast failure.

### Next

- **FND-004:** Normalize recoverable request retries.
- **FND-005:** Normalize labels, required-state messaging, and field-error
  relationships in shared and node forms.

### Later

## Validation Matrix

| Roadmap item | Acceptance criteria | Automated validation | Manual validation | Responsive coverage | Accessibility coverage |
|--------------|---------------------|----------------------|-------------------|---------------------|------------------------|
| FND-001 shared compact shell | No clipped/overlapping global controls; navigation remains available | Playwright viewport geometry and screenshot checks | Navigate every route and Canvas subview | 1440, 1024, 768, 390 px | Keyboard drawer and target-size check |
| FND-002 overlay focus | Focus remains in overlay and returns to invoker | Playwright Tab/Shift+Tab/Escape tests | Command Palette, Shortcuts, notification detail | 1440 and 390 px | Focus-order assertions |
| FND-003 async semantics | Status/alert messages announce transitions | Component role tests and axe | Success, empty, error, retry, unavailable action | 1440 and 390 px | Live-region review |
| FND-004 retry consistency | Every recoverable fetch error retries in place | Page request-failure tests | Preserve filters and selection after retry | 1440 and 390 px | Retry button keyboard operation |
| FND-005 form semantics | Controls have labels, required/invalid states, and linked errors | Component accessibility tests and axe | Keyboard-only node configuration | 1440 and 390 px | Accessible-name/error relationship review |
| FND-006 Canvas view history | Back/Forward restores selected Canvas subview | Playwright history tests | Verify retained local state | 1440, 1024, 768, 390 px | Selected-state snapshot |
| FND-007 contrast | Status-card metadata is at least 4.5:1 | Axe in light/dark modes | Inspect hierarchy without color-only cues | 1440, 1024, 768, 390 px | WCAG AA contrast check |
