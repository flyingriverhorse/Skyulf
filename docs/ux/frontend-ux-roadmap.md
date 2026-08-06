# Frontend UX Roadmap

## Executive Summary

Task 1 establishes the roadmap scaffold and records objective frontend baseline
evidence. Task 2 adds six normalized shared-foundation findings: one directly
observed shared-shell issue (`FND-001`), one shared-component risk
(`FND-003`), and four code-supported risks spanning multiple shell views or
route journeys (`FND-002`, `FND-004`, `FND-005`, and `FND-006`). Task 3 adds
four Canvas findings: one directly observed click-to-add placement/configuration
failure (`CAN-001`) and three code-supported recovery, diagnosis, and node
configuration risks (`CAN-002` through `CAN-004`). The Dashboard-only contrast
result is intentionally deferred to the Dashboard journey rather than
represented as a shared-foundation finding.

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

- **FND-005 — Inferred: Canvas node settings and the Inference prediction
  editor lack consistent programmatic field semantics.**
  - **Evidence:** In Canvas, `EncodingNode.tsx` places the visible “Encoding
    Method” `span` beside a `select` without `htmlFor`, `id`, or ARIA
    association; the representative node settings use the same visual-label
    convention. In Inference, `InferencePage.tsx` renders the JSON
    `textarea` without a `label`, `aria-label`, or `aria-labelledby`, and
    displays parse status without associating it to the editor or setting its
    invalid state. This is source evidence across two journeys, not a claim
    about uninspected Data/EDA, Experiments, or Operations forms.
  - **User problem:** Keyboard and assistive-technology users can reach the
    Canvas encoding control or Inference prediction editor without a
    programmatically conveyed purpose or invalid/error relationship.
  - **Affected surfaces:** Canvas node settings, including Encoding; Inference
    prediction input; shared `Input` and `Button` primitives where adopted.
  - **Proposed behavior:** Require an explicit label association (or
    `aria-labelledby`) for the affected Canvas node and Inference controls,
    expose required and invalid states programmatically, and place a
    persistent error beside the field while preserving helpful defaults.
  - **Acceptance criteria:** Every interactive control in the affected Canvas
    node panels and Inference prediction editor has a unique accessible name;
    required fields announce as required before submission; invalid fields
    expose `aria-invalid` and an associated error; Enter submits only forms
    with a defined submit action.
  - **Validation method:** Render representative forms in component tests,
    assert accessible names/required/error relationships, and complete
    keyboard-only configuration at desktop and 390 px; run axe on rendered
    Canvas node panels and the Inference editor.
  - **Impact:** High. **Frequency:** Frequent. **Effort:** M. **Risk:**
    Medium. **Dependencies:** Node configuration metadata and existing
    validation rules. **Milestone:** Next.

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

- **CAN-001 — Observed: click-to-add places full-size cards on top of each
  other and does not take the user to configuration.**
  - **Evidence:** `Sidebar.tsx` increments click-to-add placement by only
    30 px from `(100, 100)` while `CustomNodeWrapper.tsx` renders a
    `min-w-[200px]` card. In the running Canvas, click-adding Dataset then
    Encoding visibly overlapped the two cards; the Dataset card intercepted
    the attempt to select Encoding. The first click-added Dataset was not
    selected, so its required setting was unavailable until a second click on
    the new card opened Properties.
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
    **Effort:** S. **Risk:** Low. **Dependencies:** React Flow node bounds,
    Sidebar placement, and Properties panel selection. **Milestone:** Now.

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
| FND-002 | Inferred | Shell overlays lack a shared focus-containment and focus-return contract. | Canvas, Experiments, Inference overlays; shared Navbar | High | Occasional | S | Low | ModalShell focus helpers | Now |
| FND-003 | Inferred | Async state changes lack shared live-region semantics. | Canvas; Data/EDA; Experiments/Inference; Operations | High | Occasional | S | Low | None | Now |
| FND-004 | Inferred | Route-fetch errors inconsistently offer Retry; Canvas uses a different, toast-scoped pattern. | Dashboard; Data/EDA; Registry; Deployments; Experiments evaluation | Medium | Occasional | S | Low | Page fetch functions | Next |
| FND-005 | Inferred | Canvas node settings and Inference prediction input lack consistent field semantics. | Canvas node forms; Inference editor; shared controls | High | Frequent | M | Medium | Node metadata/validation | Next |
| FND-006 | Inferred | Shell view selection is not history-restorable or programmatically selected. | Canvas; Experiments; Inference | High | Frequent | M | Medium | useViewStore; retained views | Now |
| CAN-001 | Observed | Click-added node cards overlap and do not enter configuration. | Canvas palette, graph, Properties panel | High | Frequent | S | Low | React Flow bounds; Sidebar; selection | Now |
| CAN-002 | Inferred | Run readiness and failures lack an actionable node-level diagnostic loop. | Canvas run controls, node warnings, Results | High | Occasional | M | Medium | Validators; converter; FND-003 | Now |
| CAN-003 | Inferred | Autosave, recent, and version recovery do not explain unavailable local recovery. | Restore banner; Recent; versions; Toolbar | Medium | Occasional | M | Medium | Persistence; versions; FND-003 | Next |
| CAN-004 | Inferred | Feature Generation exposes a recommendation Apply action that changes nothing. | Feature Generation; Recommendations panel | Medium | Occasional | S | Low | Recommendation schema; FND-005 | Next |

## Component-Boundary Recommendations

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
- **CAN-002:** Turn Canvas validation and run failures into node-addressable
  diagnosis and recovery.

### Next

- **FND-004:** Normalize recoverable request retries.
- **FND-005:** Normalize labels, required-state messaging, and field-error
  relationships in Canvas node and Inference prediction forms.
- **CAN-003:** Make Canvas recovery sources and unavailable autosaves
  understandable before replacing work.
- **CAN-004:** Make Feature Generation recommendations apply or stop presenting
  an Apply action.

### Later

## Validation Matrix

| Roadmap item | Acceptance criteria | Automated validation | Manual validation | Responsive coverage | Accessibility coverage |
|--------------|---------------------|----------------------|-------------------|---------------------|------------------------|
| FND-001 shared compact shell | No clipped/overlapping global controls; navigation remains available | Playwright viewport geometry and screenshot checks | Navigate every route and Canvas subview | 1440, 1024, 768, 390 px | Keyboard drawer and target-size check |
| FND-002 shell-overlay focus | Focus remains in overlay and returns to invoker | Playwright Tab/Shift+Tab/Escape tests | Shortcuts, Command Palette, notification detail from all shell views | 1440 and 390 px | Focus-order assertions |
| FND-003 async semantics | Status/alert messages announce transitions | Component role tests and axe | Success, empty, error, retry, unavailable action | 1440 and 390 px | Live-region review |
| FND-004 retry consistency | Every recoverable route fetch error retries in place | Page request-failure tests | Preserve filters and selection after retry | 1440 and 390 px | Retry button keyboard operation |
| FND-005 Canvas/Inference form semantics | Controls have labels, required/invalid states, and linked errors | Component accessibility tests and axe | Keyboard-only Canvas configuration and Inference entry | 1440 and 390 px | Accessible-name/error relationship review |
| FND-006 shell-view history | Back/Forward restores selected Canvas, Experiments, or Inference view | Playwright history tests | Verify retained local state | 1440, 1024, 768, 390 px | Selected-state snapshot |
| CAN-001 Canvas click-add | New nodes never overlap, are selected, and expose required settings | Playwright palette/drag/palette placement checks | Build a representative pipeline by each insertion method | 1440, 1024, 768, 390 px | Keyboard reachability and focus check |
| CAN-002 Canvas diagnosis | Invalid/failing nodes identify a next action and open their settings | Validator and mocked-failure tests | Fix every issue from the Canvas summary | 1440 and 390 px | Summary role, focus, and live feedback |
| CAN-003 Canvas recovery | Local, recent, and server recovery sources and failures are explained | Persistence and version-load tests | Restore/cancel from empty and nonempty canvases | 1440 and 390 px | Keyboard recovery controls and status review |
| CAN-004 Feature Generation recommendations | Apply changes state once or is absent when unsupported | Component recommendation state/undo tests | Compare representative node configuration behavior | 1440 and 390 px | Accessible feedback after apply |
