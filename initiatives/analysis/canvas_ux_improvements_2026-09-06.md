# Canvas UX improvement backlog

Date: 2026-09-06
Status: CUX-08 in progress; panel labels, sidebar keyboard access, and primary field labels complete.

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
| CUX-01 | High | Preserve canvas space with resizable panels | Open |
| CUX-02 | High | Guide node connections and adding the next step | Open |
| CUX-03 | High | Clearly distinguish previewing data from training | Open |
| CUX-04 | Medium | Improve component discovery | Open |
| CUX-05 | Medium | Navigate from validation issues to the exact setting | Open |
| CUX-06 | Medium | Reduce connection and settings visual noise | Open |
| CUX-07 | Medium | Inspect a selected node's input and output | Open |
| CUX-08 | High, alongside related work | Fix keyboard and accessible-name gaps | In progress; panel labels, sidebar keyboard access, and primary field labels complete |

### CUX-01 — Preserve canvas space

**Observed:** The properties panel switches between 320px and almost the full
available width. At 1100px, the 64px navigation, 256px component library, and
320px properties panel leave approximately 460px for the canvas. Expanded
results occupy another 384px vertically. Users can lose sight of the pipeline
while editing it.

**Proposal:** Allow resizing of the properties and results panels. Collapse
the component library when space becomes constrained, with a clear way to
reopen it. Preserve explicit user layout choices where practical.

**Acceptance criteria:**

- [ ] Users can adjust panel sizes without losing their selected node or form state.
- [ ] Panel limits preserve usable canvas space at desktop and laptop widths.
- [ ] A selected node can be brought into the unobscured canvas area.
- [ ] Resizing has a keyboard-accessible alternative.
- [ ] Existing expand, collapse, results maximize, and read-only behaviors remain coherent.

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

**Acceptance criteria:**

- [ ] Preview and training have distinct visible labels at supported editing widths.
- [ ] Training identifies the model or set of experiments it will run.
- [ ] Blocked actions expose actionable reasons without requiring a tooltip.
- [ ] Keyboard shortcuts use the same action names and behavior.
- [ ] Loading, queued, running, and completed states identify the relevant action.
- [ ] Existing run handlers remain the source of execution behavior.

**Starting points:** `src/components/layout/Toolbar.tsx`,
`src/components/layout/toolbar/_hooks/useRunControls.ts`,
`src/modules/nodes/modeling/TrainingSettings.tsx`, and
`src/core/hooks/useTrainingNodeContext.ts`.

### CUX-04 — Improve component discovery

**Observed:** A long preprocessing list pushes modeling and evaluation far down
the library. Search currently matches node labels and categories. Descriptions
are truncated, and the instruction mentions dragging although clicking also
adds a node.

**Proposal:** Introduce collapsible, understandable preprocessing groups and
search descriptions or curated synonyms. Include task-oriented terms such as
“missing values,” “normalize,” and “predict.” Explain both click and drag.

**Acceptance criteria:**

- [ ] Modeling and evaluation are easy to reach without scrolling past every preprocessing node.
- [ ] Common task terms return relevant nodes without flooding results.
- [ ] Search results expose the reason a node is useful and enough description to choose it.
- [ ] Grouping does not hide matches during a search.
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
