# Canvas UX improvement backlog

Date: 2026-09-06
Status: CUX-01 complete; CUX-03, CUX-04, and CUX-08 in progress. Completed portions are recorded below.

## Purpose and review scope

Improve the experience of building a pipeline, connecting nodes, configuring
settings, resolving problems, and running previews or training jobs.

The review combined source inspection with browser checks at 1440 × 900 and
1100 × 800 using mocked API responses. It assessed UI behavior and layout,
not live training or end-to-end backend correctness. The frontend build passed
during the review, with circular-chunk and empty-chunk warnings. These checks
are historical evidence, not validation of future changes.

The canvas already provides templates, undo/redo, auto-layout, schema previews,
connection validation, searchable components, and a clickable validation list.
Build on these features. The highest-value work is improving the editing flow
before expanding the node catalog.

Paths below are relative to `frontend/ml-canvas/`. Recheck the current source
before implementation because other work may have changed these components.

## Prioritized backlog

| ID | Priority | Improvement | Status |
|---|---|---|---|
| CUX-01 | High | Preserve canvas space with resizable panels | Complete at checked desktop/laptop sizes |
| CUX-02 | High | Guide node connections and adding the next step | Open |
| CUX-03 | High | Clearly distinguish previewing data from training | In progress; visible action labels and training guidance complete |
| CUX-04 | Medium | Improve component discovery | In progress; description search and collapsible categories complete |
| CUX-05 | Medium | Navigate from validation issues to the exact setting | Open |
| CUX-06 | Medium | Reduce connection and settings visual noise | Open |
| CUX-07 | Medium | Inspect a selected node's input and output | Open |
| CUX-08 | High, alongside related work | Fix keyboard and accessible-name gaps | In progress; panel labels, sidebar keyboard access, and primary field labels complete |

### CUX-01 — Preserve canvas space

**Observed during review:** The properties panel switched between 320px and almost the full
available width. At 1100px, the 64px navigation, 256px component library, and
320px properties panel leave approximately 460px for the canvas. Expanded
results occupy another 384px vertically. Users can lose sight of the pipeline
while editing it.

**Proposal:** Allow resizing of the properties and results panels. Collapse
the component library when space becomes constrained, with a clear way to
reopen it. Preserve explicit user layout choices where practical.

**Progress (2026-09-07):** The docked settings panel has a draggable left edge
and a focusable resize handle. Left/Right arrows widen/narrow it in 20px steps;
Home resets to 320px and End uses the available maximum. Width is capped at
720px and reduced to reserve 400px of canvas when space permits, with a 320px
minimum for settings. The preferred width survives selection changes,
close/reopen, and expand/collapse within the current app session; it returns
when the window grows or the component library closes. Refresh resets it.
Results now have a draggable top edge and a focusable resize handle. Up/Down
arrows grow/shrink the panel in 20px steps; Home resets the preferred height
to 384px and End uses the docked maximum. Height ranges from 200px to 720px,
with the maximum reduced to reserve 240px above results when space permits.
The preferred height survives collapse, maximize/restore, close/new results,
and viewport changes within the session; refresh resets it. Canvas zoom
controls follow the visible height and return when maximized results collapse.
The component library now defaults to collapsed below 1280px and reopens
automatically at larger widths. Explicit open/close choices override this
default for the app session. Search and category choices survive automatic
collapse/reopen while editing. Focus moves to the reopen button only when a
sidebar control was focused; it returns to search when the focused reopen
button disappears. Settings fields keep focus when the sidebar changes.
Sidebar visibility, toolbar placement, and settings sizing share one resolver.
The existing read-only transition below 1024px still unmounts the sidebar:
search/category state resets on returning, but explicit open/close choices
persist. Settings now has a "Show node on canvas" button with a tooltip and
visible keyboard focus. It restores expanded settings and maximized results,
then fits the node clear of toolbar controls and the visible results panel.
Sidebar additions and existing deep-link reveals use the same placement.
The action preserves settings and preview data, honors reduced motion, and
keeps focus at its origin unless a deep link explicitly requests canvas focus.
Ordinary resizing and manual panning do not continually recenter the node.

**Acceptance criteria:**

- [x] Settings-panel resizing preserves configuration and selection, with mouse and keyboard controls.
- [x] Docked settings width adapts to available space at checked 1440px and 1100px widths.
- [x] Users can adjust panel sizes without losing their selected node or form state.
- [x] Panel limits preserve usable canvas space at checked desktop and laptop sizes.
- [x] A selected node can be brought into the unobscured canvas area.
- [x] Resizing has a keyboard-accessible alternative.
- [x] Existing expand, collapse, results maximize, and read-only behaviors remain coherent.
- [x] The component library collapses automatically when space becomes constrained, preserving explicit choices.

**Starting points:** `src/components/layout/PropertiesPanel.tsx`,
`MainLayout.tsx`, `ResultsPanel.tsx`, `Sidebar.tsx`,
`src/components/canvas/FlowCanvas.tsx`, and `src/core/store/useViewStore.ts`.

### CUX-02 — Guide connections and the next step

**Observed:** Connections reject cycles and invalid model endpoints and explain
some rejected drops with toasts. Ports use small handles and labels. The user
still needs to know which step or input is appropriate before connecting it.

**Proposal:** Highlight compatible destinations during connection creation,
show the reason for incompatibility near the attempted destination, and offer
an output-handle “Add next step” action with compatible suggestions.

**Acceptance criteria:**

- [ ] Compatibility guidance agrees with the existing graph validation rules.
- [ ] Invalid connections explain the cause and the next action.
- [ ] An output can open a relevant node picker and connect the chosen node.
- [ ] Adding and connecting a node can be undone cleanly.
- [ ] Keyboard users have an equivalent way to choose endpoints.
- [ ] Train/validation/test ports and ensemble model inputs remain distinguishable.

**Starting points:** `src/components/canvas/FlowCanvas.tsx`,
`CustomNodeWrapper.tsx`, `src/core/store/useGraphStore.ts`,
`src/core/registry/NodeRegistry.ts`, and `src/core/types/nodes.ts`.

### CUX-03 — Separate preview and training actions clearly

**Observed:** Preview is in the canvas toolbar, training is in a selected
model's settings, and multi-experiment execution has another toolbar action.
The preview control can become a play icon without a visible label. Users may
interpret it as “train this pipeline.”

**Proposal:** Keep “Preview data” visibly identifiable and place model training
in a consistent, clearly labeled location. Explain whether an action previews
data, trains the selected model, or queues multiple experiments.

**Progress (2026-09-07):** The toolbar keeps "Preview data" visible, including
at 1100px with settings open, and displays "Previewing data..." while running.
Training settings use "Train model" / "Tune model," show the selected model,
and explain the action below the button. Missing upstream dataset selection
has a visible explanation associated with the disabled button. The help guide
and shortcut overlay use the same preview label.

**Acceptance criteria:**

- [x] Preview and training have distinct visible labels at checked desktop/laptop widths (1440px and 1100px).
- [x] Selected-model training identifies its model and explains how to enable the action when no dataset is connected.
- [ ] Training identifies the model or set of experiments it will run.
- [ ] Blocked actions expose actionable reasons without requiring a tooltip.
- [x] Keyboard shortcuts use the same action names and behavior.
- [ ] Loading, queued, running, and completed states identify the relevant action.
- [ ] Existing run handlers remain the source of execution behavior.

**Starting points:** `src/components/layout/Toolbar.tsx`,
`src/components/layout/toolbar/_hooks/useRunControls.ts`,
`src/modules/nodes/modeling/TrainingSettings.tsx`, and
`src/core/hooks/useTrainingNodeContext.ts`.

### CUX-04 — Improve component discovery

**Observed during review:** A long preprocessing list pushes modeling and evaluation far down
the library. Search matched node labels and categories. Descriptions
are truncated, and the instruction mentions dragging although clicking also
adds a node.

**Proposal:** Introduce collapsible, understandable preprocessing groups and
search descriptions or curated synonyms. Include task-oriented terms such as
“missing values,” “normalize,” and “predict.” Explain both click and drag.

**Progress (2026-09-07):** Sidebar search now matches descriptions as well as
labels and categories, ignoring case and surrounding whitespace. For example,
"missing values" finds Imputation and Missing Indicator. The sidebar helper
text already explains click and drag. Category headings are now buttons with
chevrons and node counts; mouse, Enter, and Space toggle their node lists.
Search keeps matching categories expanded and temporarily disables their
toggles. Clearing search restores the previous collapsed choices, which also
survive closing/reopening the sidebar while it remains mounted. Categories
start expanded. Preprocessing subgroups, synonyms, and fuller result
descriptions remain open.

**Acceptance criteria:**

- [x] Sidebar search matches node descriptions, names, and categories; blank queries restore the library.
- [x] Modeling and evaluation are reachable by collapsing the preceding categories.
- [ ] Common task terms return relevant nodes without flooding results.
- [ ] Search results expose the reason a node is useful and enough description to choose it.
- [x] Collapsed categories do not hide matches during a search.
- [ ] Existing click-to-add, drag, and command-palette flows remain available.

**Starting points:** `src/components/layout/Sidebar.tsx`,
`CommandPalette.tsx`, and `src/core/registry/NodeRegistry.ts`.

### CUX-05 — Take users to the exact invalid setting

**Observed:** Clicking a validation issue selects its node. It does not
explicitly center the node or focus the relevant field. Some configuration
problems are represented by small badges with tooltip explanations.

**Proposal:** Complete the existing issue-navigation flow: select and reveal
the node, open the relevant settings section, and focus the invalid field.
Keep the explanation beside the setting while it remains invalid.

**Acceptance criteria:**

- [ ] Issue activation selects and reveals the correct node.
- [ ] Hidden settings sections open when needed.
- [ ] Field-specific problems focus the field; general problems focus an appropriate summary.
- [ ] Keyboard focus is visible and not covered by results or settings panels.
- [ ] Fixed issues disappear without unexpectedly moving focus.

**Starting points:** `src/components/layout/ResultsPanel.tsx`,
`PropertiesPanel.tsx`, `src/components/canvas/CustomNodeWrapper.tsx`,
`FlowCanvas.tsx`, and `src/core/hooks/useKeyboardShortcuts.ts`.

### CUX-06 — Reduce visual noise

**Observed:** Every connection permanently shows an × delete button. Edge
tooltips display source/target IDs. The properties header prominently displays
a long UUID, which consumes space without helping most settings decisions.

**Proposal:** Reveal connection deletion on hover, selection, or keyboard focus.
Use readable node names in tooltips and accessible labels. Move technical IDs
into a details area with a copy action.

**Acceptance criteria:**

- [ ] Unselected connections remain readable without permanent delete controls.
- [ ] Connection deletion remains discoverable and keyboard accessible.
- [ ] Tooltips identify nodes using meaningful names, with disambiguation when needed.
- [ ] Technical IDs remain available for troubleshooting without dominating settings.
- [ ] Branch labels, merge-winner indicators, and undo behavior remain intact.

**Starting points:** `src/components/canvas/CustomEdge.tsx`,
`CustomNodeWrapper.tsx`, and `src/components/layout/PropertiesPanel.tsx`.

### CUX-07 — Inspect input and output for the selected node

**Opportunity:** Existing schema badges, data previews, node summaries, and
inspection tools provide parts of the answer to “What did this step do?” A
single selected-node view could make those answers easier to find.

**Proposal:** Add an Input / Output view with a small data sample, column
changes, row-count changes, and a plain-language summary when measured data
supports one. Example: “Filled 23 missing values in income; no rows removed.”
This example is a proposed display, not an assertion that every node currently
reports those measurements.

**Acceptance criteria:**

- [ ] Users can inspect the selected node without adding a separate preview node for each step.
- [ ] Predicted schemas are clearly distinguished from measured execution results.
- [ ] Results identify their run and indicate when settings have changed since that run.
- [ ] Samples are bounded; loading, unavailable, and failed states are explicit.
- [ ] Multi-input and split nodes identify the branch/split being inspected.
- [ ] Summaries use available measurements and never invent transformation counts.

**Starting points:** `src/components/shared/NodeInspectorModal.tsx`,
`src/modules/nodes/inspection/DataPreviewNode.ts`,
`src/components/layout/ResultsPanel.tsx`,
`src/components/canvas/CustomNodeWrapper.tsx`, and
`src/core/hooks/useSchemaPreview.ts`.

**Scope check before implementation:** Inventory existing preview/inspection
payloads. Input samples, row counts, or transformation counts may require
backend work; this item is not assumed to be frontend-only.

### CUX-08 — Accessibility fixes

**Observed during review:** Properties-panel expand/close buttons lacked accessible names.
Sidebar cards were clickable draggable divs without keyboard activation.
Some settings use visual text rather than associated form labels.

**Progress (2026-09-06):** Properties-panel buttons now have accessible names
and native hover tooltips: "Expand settings panel," "Collapse settings panel,"
and "Close settings panel." The expansion label follows the current state.
Sidebar cards now use native buttons with accessible action names and visible
keyboard focus. Tab reaches them; Enter or Space adds a node using the existing
placement and viewport behavior. Mouse clicks and dragging remain supported.
Select Dataset, Model Type, and Target Column now use associated form labels
with instance-specific IDs. Target Column keeps its label when switching
between a text input and a dropdown populated from the connected dataset.
The eight tuning/CV fields also have associated labels: Search Method, Metric,
Trials, Random State, Folds, Method, Time Column, and Fold Split Seed.
Results-panel resizing has a named keyboard-focusable separator. Its header
now uses separate native buttons, so keyboard activation of maximize/restore
does not also toggle collapse. These controls have visible focus rings; the
title button uses the theme foreground color. A targeted axe scan of the
dark results panel with an error and reduced motion passes.

**Proposal:** Address these gaps alongside the affected interaction changes.
Use semantic controls, associated labels, and visible keyboard focus.

**Acceptance criteria:**

- [x] Properties-panel expand/collapse and close buttons have accessible names and tooltips.
- [ ] Remaining icon-only controls have meaningful accessible names.
- [x] Sidebar nodes can be reached and added using the keyboard.
- [x] Select Dataset, Model Type, and both Target Column variants have associated labels.
- [x] Tuning and cross-validation fields have associated labels, including conditional fields.
- [ ] Remaining form controls have programmatically associated labels.
- [ ] Focus remains visible through panel changes and issue navigation.
- [ ] Targeted accessibility checks cover the changed states and interactions.

**Starting points:** `src/components/layout/PropertiesPanel.tsx`,
`Sidebar.tsx`, `src/modules/nodes/modeling/TrainingSettings.tsx`, and
`src/modules/nodes/data/DatasetNode.tsx`.

Reference used in the review:
[Web Interface Guidelines](https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md).

## Suggested implementation order

1. **Workspace and actions:** CUX-01 and CUX-03, with relevant CUX-08 fixes.
2. **Connecting and correcting:** CUX-02 and CUX-05, followed by CUX-06.
3. **Discovery:** CUX-04, with keyboard support from CUX-08.
4. **Understanding results:** CUX-07 after checking available backend payloads.

For each item, review the current behavior, settle the interaction details,
implement a bounded change, and record verification here. Only the progress
explicitly recorded above is complete; remaining interaction details need review.

## Verification for future implementation

- Run focused Vitest checks for changed state, validation, and component behavior.
- Use Playwright for panel resizing, connection creation, focus navigation, and responsive layout.
- Check populated canvases and error states, not only the empty screen.
- Check desktop/laptop widths, dark/light themes, keyboard interaction, and reduced motion.
- Run `npm run lint`, relevant tests, and `npm run build` in `frontend/ml-canvas/`.
- For backend changes required by CUX-07, run the relevant backend tests and lint/type checks.
- New tests need a one-line behavioral docstring/comment as appropriate and real assertions.
- Update each item's status only after its acceptance criteria and checks are satisfied.

## Implementation log

- 2026-09-06: Recorded the canvas UX review and acceptance criteria. No source
  implementation changes were made as part of this document.
- 2026-09-06: Completed the settings-panel button-label subset of CUX-08 in
  `src/components/layout/PropertiesPanel.tsx`. Added state-aware accessible
  names, hover tooltips, and explicit button types. Verification: all 6 existing
  PropertiesPanel tests passed, `npm run lint` passed, and `npm run build`
  passed. Tests emitted connection-error logs; the build retained its existing
  circular-chunk and empty-chunk warnings. Other CUX-08 work remains open.
- 2026-09-06: Completed sidebar keyboard access in `src/components/layout/Sidebar.tsx`.
  Replaced clickable card divs with native buttons, added accessible action names
  and focus rings, and updated the helper text to mention click-to-add.
  Verification: a Playwright browser check with mocked API responses passed Tab
  navigation, visible focus, Enter/Space adding exactly one node per activation,
  retained focus, mouse click, and a synthesized drag/drop adding a node.
  `npm run lint` and `npm run build` passed. Remaining CUX-08 criteria stay open.
- 2026-09-06: Connected the primary dataset/training labels to their controls in
  `src/modules/nodes/data/DatasetNode.tsx` and
  `src/modules/nodes/modeling/TrainingSettings.tsx`, using React `useId`.
  Verification: a Playwright browser check with mocked dataset/schema/model
  responses passed accessible-name lookup and label-click focus for Select
  Dataset, Model Type, and both Target Column variants; values persisted when
  the target input changed to a dropdown after connecting a dataset.
  `npm run lint` and `npm run build` passed. Other settings labels remain open.
- 2026-09-07: Connected all eight tuning/CV field labels in
  `src/modules/nodes/modeling/TrainingSettings.tsx` using instance-specific IDs.
  A Playwright browser check with mocked model metadata passed accessible-name
  lookup and label-click focus for every changed field, edited-value persistence,
  conditional time-series controls, and disabled trial counts for grid search.
  `npm run lint` and `npm run build` passed. Dynamic hyperparameter controls and
  other remaining form-label gaps still need review; CUX-08 remains in progress.
- 2026-09-07: Implemented the action-label and selected-model guidance portion
  of CUX-03 in Toolbar, TrainingSettings, HelpGuideModal, and ShortcutsOverlay.
  Updated the existing preview browser test to use the new accessible name.
  Browser checks at 1440px and 1100px confirmed visible preview text with
  settings open, Train/Tune labels, selected-model text, and accessible visible
  guidance for the disabled training action. The existing preview E2E test
  passed request submission and displayed results; the 3 run-control/shortcut
  unit tests passed, with the shortcut test rerun after its wording update.
  `npm run lint` and `npm run build` passed.
  Broader experiment guidance and queued/completed-state work remain open.
- 2026-09-07: Added settings-panel resizing in PropertiesPanel and session width
  state in useViewStore. Added `e2e/settings-panel-resize.spec.ts`; it first
  failed because the resize handle was absent, then passed with the feature.
  It covers dragging, arrow/Home/End keys, form-value retention, close/reopen,
  expand/collapse, sidebar changes, and the laptop canvas-space limit.
  All 6 existing PropertiesPanel tests, `npm run lint`, and `npm run build`
  passed. The focusable separator uses the
  [ARIA window-splitter semantics](https://www.w3.org/WAI/ARIA/apg/patterns/windowsplitter/);
  a targeted lint exception allows this interactive separator. CUX-01 remains
  in progress for results resizing and the other unchecked criteria.
- 2026-09-07: Extended Sidebar search to descriptions and normalized the query
  once by trimming surrounding whitespace and lowercasing. Browser checks
  passed description-only matches, case/space handling, existing name/category
  matching, keyboard addition from filtered results, the no-results state, and
  clearing the query. `npm run lint` and `npm run build` passed. CUX-04 remains
  in progress for grouping and the other discovery improvements.
- 2026-09-07: Added collapsible category headings and node counts in Sidebar.
  Added `e2e/sidebar-categories.spec.ts`, which failed before the category
  toggles existed and passed after implementation. It covers Enter/Space
  toggling, reaching Modeling at 1100px, searching collapsed groups, keyboard
  node addition, restoring collapsed choices after search, and sidebar
  close/reopen. The new test and 2 existing canvas smoke tests passed;
  `npm run lint` and `npm run build` passed. CUX-04 remains in progress for
  the unchecked discovery criteria and finer preprocessing groups.
- 2026-09-07: Added results-panel resizing in ResultsPanel, preferred height in
  useViewStore, and viewport limits in MainLayout. FlowCanvas uses the same
  rendered height to keep zoom controls above results. Replaced the nested
  interactive results header with separate native buttons and explicit title
  color. Added `e2e/results-panel-resize.spec.ts`; it failed first because the
  handle was missing, then caught and verified a compact-viewport Home-reset
  fix. Its two tests cover pointer/keyboard resizing, retained settings and
  rows, viewport preference restoration, minimum height, collapse while
  maximized, maximize/restore, close/new results, and dark read-only error
  content with reduced motion. Desktop/laptop checks used 1440x900,
  1100x700, and 1100x600; read-only used 900x800. A scoped axe scan passes.
  All 6 focused browser tests (new resizing checks, settings resizing,
  preview submission, and canvas smoke) and all 12 ResultsPanel/PropertiesPanel
  unit tests passed. `npm run lint` and `npm run build` passed; generated
  frontend assets were refreshed. Mocked checks still emit backend connection
  logs; the build retains circular-chunk and empty-chunk warnings. CUX-01
  remains in progress for sidebar auto-collapse and selected-node visibility.
- 2026-09-07: Committed the accumulated resizing, action-label, and discovery
  work as `42da3904`. Commit verification passed all 873 frontend unit tests,
  7 focused browser tests, and the applicable pre-commit hooks.
- 2026-09-07: Added responsive component-library defaults through
  `useSidebarOpen`, with a session override in useViewStore. Sidebar,
  PropertiesPanel, and Toolbar use the same effective visibility. Sidebar
  replacement preserves keyboard focus and mounted search/category state.
  Added `e2e/sidebar-responsive.spec.ts`; the initial tests failed before
  implementation and all three pass afterward. Coverage includes initial
  laptop layout, automatic collapse/reopen, manual overrides across viewport
  changes, focused settings retention, toolbar clearance, panel-width limits,
  and dark/reduced-motion/read-only transitions. The existing settings-resize
  and category tests now explicitly open the library where needed.
  All 10 focused browser tests and 24 focused unit tests passed, along with
  `npm run lint` and `npm run build`; generated assets were refreshed. Existing
  circular-chunk/empty-chunk build warnings remain. Read-only still unmounts
  local sidebar browsing state, as recorded above. CUX-01 remains in progress
  for selected-node visibility.
- 2026-09-07: Committed responsive sidebar defaults as `f6d58255`; all 873
  frontend unit tests and the applicable pre-commit hooks passed.
- 2026-09-07: Completed selected-node reveal for CUX-01. PropertiesPanel adds
  "Show node on canvas"; FlowCanvas handles the shared reveal event after
  panel resizing settles and uses React Flow's asymmetric padding to exclude
  toolbar/zoom controls and the rendered results height. Expanded settings
  and maximized results restore before fitting. Added
  `e2e/reveal-node.spec.ts`; the first two tests failed before implementation
  and all three pass afterward. Coverage includes a distant selected node,
  keyboard activation, retained configuration/preview rows, desktop/laptop
  bounds, sidebar additions with enlarged results, reduced motion, read-only
  canvas focus, and manual pan retention after a later viewport resize.
  All 12 focused browser checks and 24 focused unit tests passed, as did
  `npm run lint` and `npm run build`; generated assets were refreshed. Existing
  mocked backend connection logs and circular-chunk/empty-chunk build warnings
  remain. CUX-01 is complete for its recorded acceptance criteria. Exact-field
  navigation from validation issues remains separate CUX-05 work.
- 2026-09-07: Audited responsive canvas controls after reports of overlapping
  buttons. Toolbar groups now share a flex layout and move secondary actions
  into More based on the actual canvas width. Preview retains its visible
  label; overflow actions reuse the existing execution handlers. Navigation
  uses grid columns and truncates long dataset context. Results titles can
  shrink without crowding controls. Legend, load/recent, overflow, and
  notification menus have viewport bounds and scrolling; open toolbar menus
  render above results. Node reveal measures the rendered toolbar height.
  Added four tests in `e2e/canvas-layout.spec.ts`, which reproduced control
  collisions, a menu hidden behind results, and phone legend overflow before
  the fixes. Coverage spans 320px to 1920px, pinned desktop side panels,
  long dataset names, parallel actions, keyboard menu access, and dark phone
  popovers. All 17 focused browser tests and all 873 frontend unit tests
  passed, along with `npm run lint` and `npm run build`; served assets were
  rebuilt. Existing mocked connection logs and build chunk warnings remain.
  Review also identified lost keyboard focus when More opened a replacement
  menu. New assertions reproduced it; focus now moves into that content and
  returns to More on dismissal. All four layout tests passed again, including
  keyboard opening and dismissal of the load menu and legend.
- 2026-09-07: Unified light/dark switching and applied the root site's galaxy
  night / dawn palette to shared theme tokens, neutral utilities, canvas
  controls, chart surfaces, and primary actions. Theme changes temporarily
  suppress component transitions so surfaces update together; normal hover
  and layout motion resumes after the paint. Replaced the navigation's S
  placeholder and Vite favicon with the existing wolf logo, bundled by Vite.
  First-paint HTML backgrounds match the app palette. Gold buttons use dark
  text; a dataset upload link now uses periwinkle with adequate dark contrast.
  Added three `e2e/theme.spec.ts` tests. They reproduced staggered color
  transitions, the missing logo, and white-on-gold upload text before fixes.
  Checks cover populated-canvas contrast in both modes, mobile/reduced motion,
  loaded logo pixels, restored normal transitions, preference persistence,
  and upload actions outside the canvas. All 20 focused browser tests and
  873 frontend unit tests passed; 14 affected page tests passed again after
  review corrections. Final lint, TypeScript, and production build passed.
  Existing mocked connection logs and build chunk warnings remain.
- 2026-09-07: Revised the app's dark palette after visual feedback: neutral
  charcoal surfaces and soft blue actions replace plum and gold in dark mode.
  Light mode retains the dawn palette and gold actions. Neutral gray/slate
  utilities now resolve through theme-specific variables, including chart and
  canvas surfaces. Shared primary/secondary action classes align preview,
  training, prediction, upload, create, apply, confirm, and secondary tools;
  shared Button variants use the same colors. The logo header now has the
  same 56px height as the canvas navbar, fixing the 9px divider mismatch.
  Four theme browser tests cover both palettes, text contrast, action colors
  across pages, persistence, reduced motion, and divider alignment at 1440px,
  1100px, and 768px. All 21 focused browser checks passed. Review found no
  action-style regressions; remaining prediction/threshold action colors were
  brought into the same styles and their focused unit tests passed.
  All 873 frontend unit tests passed on the final run; the earlier parallel
  run missed a pre-existing 20ms mocked deployment loading state, which also
  passed in isolation. Final lint, TypeScript, and production build passed,
  and served assets were rebuilt. Existing build chunk warnings remain.
