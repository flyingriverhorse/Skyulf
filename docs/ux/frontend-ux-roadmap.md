# Frontend UX Roadmap

## Executive Summary

Task 1 establishes the roadmap scaffold and records objective frontend baseline
evidence. Task 2 adds six normalized shared-foundation findings: one directly
observed shared-shell issue (`FND-001`), one shared-component risk
(`FND-003`), and four code-supported risks spanning multiple shell views or
route journeys (`FND-002`, `FND-004`, `FND-005`, and `FND-006`). Task 3 adds
five Canvas findings: two directly observed placement/toolbar failures
(`CAN-001`, `CAN-005`) and three code-supported recovery, diagnosis, and node
configuration risks (`CAN-002` through `CAN-004`). Task 4 adds seven Data and
EDA findings: three directly reproduced journey/responsive outcomes
(`DAT-001`, `DAT-002`, and `DAT-004`) and four code-supported lifecycle,
configuration, and visualization risks (`DAT-003`, `DAT-005` through
`DAT-007`). Task 5 adds seven Experiments and Inference findings: two
comparison-context issues kept in Now (`EXP-001`, `EXP-002`), two artifact/diff
clarity items sequenced to Next (`EXP-003`, `EXP-004`), and three high-impact
threshold/inference items (`EXP-005` through `EXP-007`) explicitly scoped so
their Now slice is independently complete while shared retry (`FND-004`) and
shared field-semantic (`FND-005`) normalization remain Next follow-ons. The
Dashboard-only contrast result is intentionally deferred to the Dashboard
journey rather than represented as a shared-foundation finding.

Task 6 adds seven Operations findings. `OPS-007` now limits the Now slice
to a shared typed operational-context schema, serializer/parser round-trip
contract, and contextual record-link primitive. The consumer integrations for
Jobs, Registry/Deployments, Drift, Error Log, Slow Nodes, and Audit Log
(`OPS-001`–`OPS-006`) remain Next so each view can adopt that boundary in its
own rows/details. `OPS-006` does not claim Audit Log lacks attribution or
change detail: its existing entries render actor, timestamp, action kind,
version, and node diffs. It instead targets missing filters,
time-range/retention clarity, and cross-record correlation.

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

### Accessibility and Keyboard UX

- **FND-002 — Inferred: shell overlays lack a shared focus-containment and
  focus-return contract.**
  - **Evidence:** **Observed** only in Canvas `ShortcutsOverlay`: Tab reached
    the covered “More canvas tools” control instead of a dialog control.
    **Inferred** for the other shell overlays: `ShortcutsOverlay.tsx`,
    `CommandPalette.tsx`, and the notification detail modal in
    `NotificationCenter.tsx` each render custom dialogs without
    `ModalShell.tsx`'s containment/return helpers. `MainLayout.tsx` mounts
    Shortcuts and Command Palette alongside Canvas, Experiments, and
    Inference, and `Navbar.tsx` renders NotificationCenter for the same three
    views. Command Palette and notification detail were not separately
    reproduced, so the inventory label is Inferred.
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
  outcomes.

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
    pipeline versions API, and **FND-003**. **Milestone:** Next.

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
    **Milestone:** Next.

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
    **Milestone:** Next.

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
    **Milestone:** Next.

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

## Prioritized Findings Inventory

| ID | Evidence | User problem | Surfaces | Impact | Frequency | Effort | Risk | Dependencies | Milestone |
|----|----------|--------------|----------|--------|-----------|--------|------|--------------|-----------|
| FND-001 | Observed | Global navigation and Canvas view controls clip/overlap at 390 px. | Layout; Canvas, Experiments, Inference; Data/EDA; Operations | High | Frequent | M | Medium | Layout; read-only breakpoint | Now |
| FND-002 | Inferred | Shell overlays lack a shared focus-containment and focus-return contract. | Canvas, Experiments, Inference overlays; shared Navbar | High | Occasional | S | Low | ModalShell focus helpers | Now |
| FND-003 | Inferred | Async state changes lack shared live-region semantics. | Canvas; Data/EDA; Experiments/Inference; Operations | High | Occasional | S | Low | None | Now |
| FND-004 | Inferred | Route-fetch errors inconsistently offer Retry; Canvas uses a different, toast-scoped pattern. | Dashboard; Data/EDA; Registry; Deployments; Experiments evaluation | Medium | Occasional | S | Low | Page fetch functions | Next |
| FND-005 | Inferred | Canvas, Data/EDA, and Inference forms lack consistent field semantics. | Canvas node forms; Data Sources Add Source; EDA analysis/filter controls; Inference editor; shared controls | High | Frequent | M | Medium | Shared form primitives; node metadata/validation; source/EDA/inference validation | Next |
| FND-006 | Inferred | Shell view selection is not history-restorable or programmatically selected. | Canvas; Experiments; Inference | High | Frequent | M | Medium | useViewStore; retained views | Now |
| CAN-001 | Observed | Click-added node cards overlap and do not enter configuration. | Canvas palette, graph, Properties panel | High | Frequent | S | Low | Custom-node bounds; Sidebar; selection | Now |
| CAN-005 | Observed | Canvas toolbar clusters overlap and intercept visible actions when Properties narrows the Flow pane. | Canvas Toolbar, Flow viewport, Properties panel | High | Frequent | S | Low | Toolbar responsive layout; panel width | Now |
| CAN-002 | Inferred | Run readiness and failures lack an actionable node-level diagnostic loop. | Canvas run controls, node warnings, Results | High | Occasional | M | Medium | Validators; converter; FND-003 | Now |
| CAN-003 | Inferred | Autosave, recent, and version recovery do not explain unavailable local recovery. | Restore banner; Recent; versions; Toolbar | Medium | Occasional | M | Medium | Persistence; versions; FND-003 | Next |
| CAN-004 | Inferred | Feature Generation exposes a recommendation Apply action that changes nothing. | Feature Generation; Recommendations panel | Medium | Occasional | S | Low | Recommendation schema; FND-005 | Next |
| DAT-001 | Observed | Source onboarding has conflicting destinations and unassociated required fields. | Data Sources; Add Source; Canvas/EDA handoffs | High | Frequent | M | Medium | Source API; router | Now |
| DAT-002 | Observed | Failed preview reports zero-like metadata without recovery context. | Dataset Preview; source profile/sample APIs | High | Occasional | M | Medium | Profile/sample errors; FND-003 | Now |
| DAT-003 | Inferred | Ingestion activity lacks phase/progress/history/retry lifecycle. | Upload; Add Source; jobs; source rows | High | Occasional | M | Medium | Ingestion API; FND-003 | Now |
| DAT-004 | Observed | Data/EDA route controls are clipped at 390 px. | Data table/actions; EDA header/history; Layout | High | Frequent | M | Medium | FND-001; responsive views | Now |
| DAT-005 | Inferred | EDA jobs and history lack durable input/status/recovery context. | EDA selection; jobs; failures; History | High | Frequent | M | Medium | EDA job API; useEDAStore; FND-003 | Now |
| DAT-006 | Inferred | Filters and exclusions apply inconsistently without durable report context. | EDA sidebar; tabs; exports | High | Frequent | M | Medium | EDA schema; FND-005 | Next |
| DAT-007 | Inferred | Charts lose interpretable color/axis/alternative-data context at density and narrow widths. | EDA charts, tables, exports | High | Frequent | L | Medium | Chart themes; exports; report payloads | Next |
| EXP-001 | Inferred | Filters can hide selected runs while comparison keeps using them. | Experiments filters, run sidebar, comparison/evaluation/diff tabs | High | Frequent | M | Medium | useJobStore; experiment fixtures | Now |
| EXP-002 | Inferred | Metric comparison does not make direction, comparability, or missingness durable. | Visual/table/branch metric comparison | High | Frequent | M | Medium | Metric metadata; chart/table adapters | Now |
| EXP-003 | Inferred | Conditional explainability/segmentation views conceal availability and comparability. | Feature Importance, SHAP, Segmentation, exports | High | Occasional | M | Medium | Artifact schema; explanation services | Next |
| EXP-004 | Inferred | Pipeline Diff lacks an explicit baseline/candidate decision contract. | Run sidebar, Pipeline Diff, saved graphs | Medium | Occasional | M | Medium | Graph snapshots; graphDiff; job metadata | Next |
| EXP-005 | Inferred | Threshold exploration/tuning/activation lacks a durable decision record. | Evaluation, threshold API, Inference overrides/results | High | Occasional | M | High | Threshold API/version semantics; FND-003 | Now |
| EXP-006 | Observed | Inference permits visibly incomplete/type-incompatible input. | Editor, schema badges/Fix, prediction request | High | Frequent | M | High | Typed artifact schema | Now |
| EXP-007 | Inferred | Inference execution/recovery is not a complete durable run lifecycle. | Run, pending/error/results, history, exports | High | Occasional | L | Medium | Prediction/status API; storage; FND-003 | Now |
| OPS-001 | Observed | Jobs history cannot open a unified details/recovery investigation. | Jobs; Job History drawer; logs; related resources | High | Frequent | L | Medium | Job/status/log APIs; useJobStore; DAT-003/DAT-005; OPS-007; FND-003 | Next |
| OPS-002 | Observed | Registered versions and deployments do not form a traceable decision chain. | Registry; Deployments; Jobs; Experiments; Inference | High | Occasional | L | High | Registry/deployment lineage; job/evaluation provenance; OPS-007; FND-003/FND-004 | Next |
| OPS-003 | Inferred | Drift reports lack a durable alert, investigation, and remediation lifecycle. | Drift; alert badge; Registry/Deployments; Jobs; Errors | High | Occasional | L | High | Drift/alert schema; threshold versioning; deployment lineage; OPS-007; FND-003 | Next |
| OPS-004 | Observed | Generic identifier search lacks typed resource facets and contextual deep links. | Error Log; incidents; Jobs; Canvas; Data; Registry/Deployments | High | Frequent | L | Medium | Resource-facet/correlation schema; OPS-007; redaction; FND-003/FND-004 | Next |
| OPS-005 | Observed | Slow-node aggregates cannot lead to the measured run/node or remediation. | Slow Nodes; Jobs; Canvas; Audit Log | Medium | Occasional | M | Medium | Slow-node drill-down API; run snapshots; OPS-007; CAN-002 | Next |
| OPS-006 | Inferred | Attributed version/diff history lacks filters, retention/time clarity, and related-record correlation. | Audit Log; Canvas versions; Jobs; Deployments; Drift; Errors | Medium | Occasional | L | Medium | Audit filtering/correlation API; identity/retention policy; graph snapshots; OPS-007; EXP-004 | Next |
| OPS-007 | Inferred | Operations lacks a shared typed context serializer and record-link primitive. | Shared Operations link/query-state utilities; future rows/details across Operations | High | Frequent | M | Medium | Operational context schema; router/query state; API identities | Now |

## Component-Boundary Recommendations

- **Canvas frame and toolbar:** Keep panel-breakpoint ownership in
  `MainLayout`/Canvas layout, but make `Toolbar` consume the resulting Canvas
  pane width as one collision contract rather than independently absolutely
  positioning left and right clusters (`CAN-005`). The shared shell remains
  responsible for the 390 px view switcher in `FND-001`.
- **Graph-library boundary:** Leave React Flow responsible for graph viewport,
  pan/zoom, handles, and node movement. Keep Sidebar insertion placement,
  selected-node handoff, custom-card semantics, and keyboard insertion
  discoverability in Skyulf (`CAN-001`); do not classify these custom controls
  as React Flow defects.
- **Settings and recovery boundary:** `PropertiesPanel` hosts node-specific
  forms, which should receive shared label/error primitives from `FND-005`;
  `useRunControls`, graph validators, `ResultsPanel`, persistence, and restore
  UI must exchange structured node/recovery state for `CAN-002`/`CAN-003`.
- **Data journey boundary:** Keep ingestion transport/status truth in the
  source and ingestion APIs, but make Data Sources own a coherent
  source-to-Canvas/EDA handoff and preview recovery surface
  (`DAT-001`–`DAT-003`). Reuse **FND-003** status semantics now and treat
  **FND-005** as later shared-primitive normalization rather than a blocker for
  the route-specific source journey.
- **EDA analysis boundary:** Keep server report/job/history state in React
  Query and editable analysis context in `useEDAStore`, but expose one
  user-facing analysis record to the header, sidebar, History, tabs, and
  export functions (`DAT-005`, `DAT-006`). Chart adapters own rendering and
  theme mechanics; EDA tabs own explanations, data alternatives, and applied
  context (`DAT-007`).
- **Inference run boundary:** `InferencePage.tsx` currently combines
  deployment/schema loading, input import/repair, per-request threshold
  overrides, prediction transport, result/export rendering, and transient
  history. Split a testable inference-run controller from input and
  results/history views only to prevent those independently changing states
  from attributing a result, error, or threshold to the wrong deployment
  (`EXP-006`, `EXP-007`). Keep API truth in the deployment/threshold clients;
  treat shared retry and field-semantics cleanup in `FND-004`/`FND-005` as
  later normalization, and do not split presentational helpers merely for file
  size.
- **Operations record boundary:** Keep authoritative job, registry/deployment,
  drift, incident, performance, and version-audit data in their existing APIs,
  but deliver `OPS-007`'s small typed operational-context schema,
  serializer/parser, and record-link primitive first. Jobs (`OPS-001`),
  Registry/Deployments (`OPS-002`), Drift (`OPS-003`), Error Log (`OPS-004`),
  Slow Nodes (`OPS-005`), and Audit Log (`OPS-006`) consume that
  identity/origin contract afterward; they do not independently invent query
  keys, parse deep-link state, or serialize return context. **FND-003** and
  **FND-004** remain the shared status and retry mechanisms rather than
  parallel Operations implementations.

## Now / Next / Later Roadmap

### Now

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

### Next

- **FND-004:** Normalize recoverable route-fetch retries after route-specific
  threshold and inference recovery flows are complete.
- **FND-005:** Normalize labels, required-state messaging, and field-error
  relationships in Canvas, Data/EDA, and Inference after route-specific source
  and inference validation fixes land.
- **CAN-003:** Make Canvas recovery sources and unavailable autosaves
  understandable before replacing work.
- **CAN-004:** Make Feature Generation recommendations apply or stop presenting
  an Apply action.
- **DAT-006:** Make filter and exclusion application accessible, explicit, and
  visible in every result/export.
- **DAT-007:** Establish interpretable, responsive, theme-safe chart and
  alternative-data behavior.
- **EXP-003:** Explain absent/partial explainability artifacts, missing versus
  zero values, and segmentation metric direction.
- **EXP-004:** Make baseline/candidate roles, snapshot availability, and
  structural-versus-outcome differences explicit in Pipeline Diff.
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
- **OPS-005:** After `OPS-007`, let Slow Nodes open time-bound contributing-run
  and Canvas-node investigation rather than presenting aggregates alone.
- **OPS-006:** After `OPS-007`, preserve attributed version/diff history while
  adding filters, retention/time-scope clarity, and related-record
  correlation.

### Later

## Validation Matrix

| Roadmap item | Acceptance criteria | Automated validation | Manual validation | Responsive coverage | Accessibility coverage |
|--------------|---------------------|----------------------|-------------------|---------------------|------------------------|
| FND-001 shared compact shell | No clipped/overlapping global controls; navigation remains available | Playwright viewport geometry and screenshot checks | Navigate every route and Canvas subview | 1440, 1024, 768, 390 px | Keyboard drawer and target-size check |
| FND-002 shell-overlay focus | Focus remains in overlay and returns to invoker | Playwright Tab/Shift+Tab/Escape tests | Shortcuts, Command Palette, notification detail from all shell views | 1440 and 390 px | Focus-order assertions |
| FND-003 async semantics | Status/alert messages announce transitions | Component role tests and axe | Success, empty, error, retry, unavailable action | 1440 and 390 px | Live-region review |
| FND-004 retry consistency | Every recoverable route fetch error retries in place | Page request-failure tests | Preserve filters and selection after retry | 1440 and 390 px | Retry button keyboard operation |
| FND-005 shared form semantics | Controls have labels, required/invalid states, and linked errors | Component accessibility tests and axe | Keyboard-only Canvas configuration, Add Source, EDA filter/setup, and Inference entry | 1440 and 390 px | Accessible-name/error relationship review |
| FND-006 shell-view history | Back/Forward restores selected Canvas, Experiments, or Inference view | Playwright history tests | Verify retained local state | 1440, 1024, 768, 390 px | Selected-state snapshot |
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
| EXP-001 selected-run context | Filtered views identify all retained selected runs and their use | Mixed-job selection/filter Playwright tests | Filter, compare, switch tabs/views, clear/restore selection | 1440 and 390 px | Selection summary and keyboard operation |
| EXP-002 metric decision contract | Metric direction/split/units/missingness/winner are explicit | Deterministic metric/branch fixture tests | Compare classification, regression, CV, and partial jobs | 1440 and 390 px, light/dark | Tooltip-independent labels and table semantics |
| EXP-003 explanation/segmentation availability | Artifact coverage, missingness, normalization, and cluster metric direction are explicit | Artifact-state component/visual tests | Compare supported/unsupported/partial SHAP and clustering jobs | 1440 and 390 px, light/dark | Non-color and data/export alternatives |
| EXP-004 Pipeline Diff roles | Baseline/candidate, graph status, and difference direction are unambiguous | Ordered graph-pair, missing/error snapshot tests | Swap roles and inspect equal/changed/failed pairs | 1440 and 390 px | Keyboard role controls and change-list semantics |
| EXP-005 threshold decision lifecycle | Preview/save/enable/clear/provenance cannot be misattributed and failed mutations retry in place on the current surface | Two-job threshold API state-transition tests | Tune, enable, infer, override, clear, retry | 1440 and 390 px | Status/error announcements and control labels |
| EXP-006 typed inference input | Invalid field/value/row shapes and editor-local issue state are actionable before submit | Typed-schema JSON/CSV request-gating tests | Review repair/default/unknown-type input | 1440 and 390 px | Field/error relationships and keyboard repair |
| EXP-007 inference run lifecycle | Pending/failure/retry/results/export/history retain clear provenance on the Inference surface | Delayed/success/failure/cancel/reload/export tests | Execute, reset, retry, reload, restore, export | 1440 and 390 px | Live status, error recovery, and focus review |
| OPS-001 job investigation lifecycle | Job details name lifecycle/input/error/log/result context; supported actions recover in place and Back restores list state | After OPS-007, paginated multi-type job, cancel/retry/unavailable-action component and Playwright fixtures | Search/filter/load/details/log/cancel/retry/return | 1440 and 390 px | Detail/action names, live status, keyboard return |
| OPS-002 model deployment lineage | Model/job/version/artifact/deployment/inference identities remain traceable across action outcomes | After OPS-007, multi-version/deployment mutation and deep-link Playwright tests | Deploy, replace, deactivate, redeploy, refresh, follow links | 1440 and 390 px | Confirmation/modal focus and action status |
| OPS-003 drift investigation lifecycle | Alert, severity, threshold version, evidence, owner/disposition, and related resources persist per check | After OPS-007, drift/alert history and transition fixtures | Alert→report→job/deployment, threshold change, acknowledge/reopen | 1440 and 390 px | Alert/status semantics and feature-table navigation |
| OPS-004 Error Log investigation facets | Generic search retains exact HTTP job/pipeline node IDs; typed severity/resource facets and contextual links remain unambiguous | After OPS-007, HTTP `job_id`/pipeline `node_id` search plus facet/link/export Playwright tests | Search, facet, expand, export, follow and return | 1440 and 390 px | Facet names, arrival focus, readable detail |
| OPS-005 slow-node diagnosis | Aggregate source, unit/window, contributing run context, and returnable remediation links are explicit | After OPS-007, aggregate/outlier/no-data/drill-down tests | Sort, investigate, open job/Canvas, return with controls retained | 1440 and 390 px | Sort/button names and chart/table alternatives |
| OPS-006 version audit trail | Existing actor/timestamp/action/version/diff detail remains visible; filters, time/retention scope, and supplied correlations are clear | After OPS-007, multi-dataset/version/actor/time/filter/query-state and linked/unlinked-record tests | Filter, inspect scope copy, reload link, follow and return | 1440 and 390 px | Expandable audit detail and filter semantics |
| OPS-007 operational context primitive | Typed operational identities, origin, and time/filter context round-trip without loss and build shared href/return payloads | Schema/serializer/parser round-trip and shared record-link component tests | Generate representative job/model/deployment/error/drift/slow/audit contexts, copy the link, reload, and confirm the same parsed payload | 1440 and 390 px for primitive rendering | Accessible link names and copyable target semantics |
