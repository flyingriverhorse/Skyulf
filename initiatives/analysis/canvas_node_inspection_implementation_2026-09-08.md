# CUX-07 selected-node inspection implementation plan

**Goal:** Inspect the selected node's measured input/output from a data preview,
with bounded samples, branch/split identity and honest stale-result feedback.

**Architecture:** Add optional selected-node capture to the existing preview
endpoint and execution engine. Capture input at resolution time, before a
transformer can mutate it, and output immediately after execution. Keep a single
latest inspection receipt in frontend session state and compare the submitted
semantic configuration against the current graph.

**Tech stack:** FastAPI/Pydantic, existing Python pipeline engine, React,
TypeScript, Zustand, Vitest and Playwright. No new dependencies or migration.

**Spec:** `canvas_node_inspection_payloads_2026-09-08.md` and CUX-07 in
`canvas_ux_improvements_2026-09-06.md`. The user approved Input / Output tabs,
samples, shape/schema details and stale results on 2026-09-08.

## Constraints and decisions

### Follow-up approved on 2026-09-08

The user requested one preview to capture every node, so changing selection
does not require rerunning. Inspector Refresh preview should preserve the
Results panel's visibility and content. The changes below supersede the
single-selected-node request/cache decisions in the original plan.

- [x] Backend: add optional `inspect_all=true` query while preserving the
  existing `inspect_node_id` and no-capture default. Capture all runnable
  nodes during each existing branch execution; include explicit unavailable
  entries for skipped nodes. Preserve run/branch/node/split identity.
- [x] Backend: share an 8 MiB sample-row budget across all captured node sides
  and branches, also retaining the existing per-side/table limits. Preserve
  measured counts/schema and indicate truncation when samples are reduced.
  Add tests for single-pass capture, shared branches, failure/skipped nodes,
  total budget and backward compatibility.
- [x] Frontend: request all-node capture from toolbar and inspector; store a
  graph-scoped receipt instead of a selected-node receipt. Selecting another
  captured node reuses the same run/configuration without a network call.
- [x] Frontend: inspector refresh updates only the inspection receipt/error;
  it does not publish a new global Results panel result or run error. Toolbar
  continues to publish global results. Test closed/collapsed/expanded Results
  states and failures, including selection changes during a request.
- [x] Verify relevant Python/Vitest suites, browser interactions and build;
  update backlog/changelog. No commit or push requested for this follow-up.

### Single preview action and node-local paths approved on 2026-09-08

This follow-up supersedes the inspector Refresh preview action above. Toolbar
Preview data is the single execution action; Input / Output only displays the
latest receipt. Empty/stale states direct users to the toolbar.

Inspection choices describe paths reaching the selected node, not every training
terminal below it. Backend receipts add optional `path_id` / `path_label` from
each node's actual partition ancestry, preserving ordered inputs and parameters.
Descendants, pipeline IDs and display names do not change path identity. Labels
describe upstream steps and are bounded to 240 characters. Raw branch receipts
remain available. The UI groups matching path IDs and captured input/output,
while keeping different paths, measurements, errors or unknown provenance apart.
One resulting capture has no selector; multiple captures use Data path.

| Relevant files | Change and verification |
|---|---|
| Backend `preview.py`, `_schemas.py` | Add optional node-local provenance without changing execution/partitioning; integration tests cover shared models, real parallel inputs and ordered merges |
| `nodeInspectionPaths.ts`, inspection types/hook | Group recorded provenance, retain differing measurements, remove inspector execution; utility/hook tests cover reuse and visibility |
| `NodeInspectionPanel.tsx`, PropertiesPanel tests | Remove Refresh, update empty/stale guidance, hide single-choice selector; component and browser checks |

No database/configuration migration or change to the core ML library is needed.

### Original implementation decisions

- Extend data preview only. Do not run training to inspect a model.
- Preserve existing request bodies and terminal preview consumers.
- Add query parameter `inspect_node_id: str | None` to POST /pipeline/preview.
- New response fields: `run_id: str | None` and `node_inspections`, default [].
- At most one requested node; each branch gets a separate captured execution.
- Return at most 50 sample rows and 100 columns per table, at most six tables
  per side (train/test/validation, each X/y), and bounded displayed cell values.
- Row/column counts describe the actual preview data, not the entire source
  dataset. Preview loaders currently take up to 1,000 source rows.
- Each table identifies its resolved port and split. A merged input is captured
  after the merge; never substitute an arbitrary upstream output.
- Missing source input, skipped model nodes, failed execution and unsupported
  artifact shapes have explicit availability reasons. Empty frames retain schema.
- No HTTP, persistence, filesystem or capture work belongs in skyulf-core.
- Position, selection, labels, panel layout and generated pipeline UUIDs do not
  invalidate results. Parameters, connections and dataset selection do.
- Keep existing UI styling, panel resizing and validation navigation. Keyboard
  users can switch tabs/selectors and refresh without focus being displaced.
- No commits or pushes are part of this implementation turn.

## Wire contract

```typescript
interface InspectionTable {
  port: string;
  split: string | null;
  row_count: number;
  column_count: number;
  columns: { name: string; dtype: string }[];
  rows: Record<string, unknown>[];
  truncated: boolean;
}
interface InspectionSide {
  status: 'available' | 'unavailable' | 'error';
  reason: string | null;
  tables: InspectionTable[];
}
interface NodeInspection {
  node_id: string;
  branch_id: string;
  branch_label: string;
  input: InspectionSide;
  output: InspectionSide;
}
```

`run_id` scopes every entry in a response. Branch identifiers must be distinct
even when display labels or selected node IDs coincide. A request for a node
that does not execute still returns an explicit unavailable entry.

## Context map

| File | Responsibility and change |
|---|---|
| backend/ml_pipeline/_execution/engine/_inspection.py (new) | Bounded snapshot serialization and side availability |
| backend/ml_pipeline/_execution/engine/__init__.py | Optional capture at actual input resolution and node completion/failure |
| backend/ml_pipeline/_internal/_routers/preview.py | Optional request, per-branch capture collection, receipt and cleanup |
| backend/ml_pipeline/_internal/_schemas.py | Additive response fields and inspection response models |
| frontend/ml-canvas/src/core/types/nodeInspection.ts (new) | Wire types |
| frontend/ml-canvas/src/core/api/client.ts | Optional request query and additive response fields |
| frontend/ml-canvas/src/core/store/useNodeInspectionStore.ts (new) | Single-flight preview request and latest captured receipt |
| frontend/ml-canvas/src/core/hooks/useNodeInspection.ts (new) | Current node's measured state, refresh and predicted output schema |
| frontend/ml-canvas/src/core/utils/previewConfiguration.ts (new) | Shared graph filtering and semantic configuration identity |
| frontend/ml-canvas/src/components/layout/toolbar/_hooks/useRunControls.ts | Shared preview transport and selected-node capture |
| frontend/ml-canvas/src/components/layout/NodeInspectionPanel.tsx (new) | Samples, shape, schema and state presentation |
| frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx | Settings / Input / Output navigation |

Dependencies: existing graph store, registry converter and schema preview store;
existing engine input merge, SplitDataset and temporary artifact store. Reference
patterns: DataPreviewComponents tables, ValidationNavigation/useValidationReveal,
PropertiesPanel keyboard controls and preview-router integration tests.

Risk review: response changes are additive; no migrations or configuration
changes. Main risks are mutating inputs, shared branch artifacts, stale async
responses and settings hidden during validation navigation. Tests below pin them.

## Task 1: Backend capture (backend worker)

- [x] Add failing tests in `tests/integration/test_node_inspection.py` for the
  actual before/after transformation, merged input, source/no input, failed and
  skipped nodes, row/column caps, empty schema, splits/X-y and branch separation.
- [x] Execute these with `.venv/Scripts/python.exe -m pytest` and confirm the
  missing contract/capture failure before implementing.
- [x] Implement the wire contract above with optional capture at `_get_input`
  and execution completion. Snapshot input before running the transformer.
- [x] Collect snapshots immediately per branch, before another branch overwrites
  shared artifacts. Include a new UUID run receipt and predictable branch IDs.
- [x] Verify temporary store cleanup even when resolution or capture fails.
- [x] Run inspection and existing preview tests, ruff and ty. Report scope and
  evidence to the controller; do not commit.

## Task 2: Preview receipt and refresh (controller)

- [x] Add failing Vitest tests proving semantic edits invalidate results while
  moving/renaming/selecting nodes does not, and late results retain the submitted
  configuration. Verify dataset changes and connection order/merge behavior.
- [x] Implement the shared request in `useNodeInspectionStore` and transport.
  One request at a time; panel and toolbar share pending state.
- [x] Implement `useNodeInspection(nodeId)` returning `branches`, `runId`,
  `isStale`, `isLoading`, `error`, `predictedSchema`, and `refresh`.
  `predictedSchema` uses the existing `PredictedSchema | null` type;
  `branches` is `NodeInspection[]`; refresh returns `Promise<void>`.
- [x] Store the request's semantic configuration before awaiting the API.
  Match both node ID and configuration when presenting the latest receipt.
- [x] Use the shared request in toolbar Preview data; include the current
  selected node when present, retain existing notifications/results behavior.
- [x] Run focused hooks/store/converter and toolbar tests.

## Task 3: Inspection UI (frontend worker)

- [x] Add failing component tests for measured counts/samples, schema on empty
  frames, branch/split selection, pending/error/unavailable/stale states and
  accessible navigation; use the hook interface defined in Task 2.
- [x] Build `NodeInspectionPanel({ nodeId, side: 'input' | 'output' })`.
  Show a Refresh preview button, preview scope, selected branch/table, shape,
  dtype headers and bounded scrollable sample table. Explicitly label predicted
  output schema when no measured output exists; do not invent predicted input.
- [x] Add Settings / Input / Output tabs to PropertiesPanel, keeping settings
  state mounted when hidden. Validation requests reveal Settings before focus.
- [x] Verify labels/focus, light/dark themes and compact panel bounds.
- [x] Run focused component tests and report evidence; do not commit or rebuild
  shared served assets until the controller's final integration build.

## Task 4: Integration and completion (controller with independent review)

- [x] Review backend and frontend changes against this contract and fix findings.
- [x] Add Playwright cases covering real UI requests, selected node changes,
  stale semantic edits versus movement, tab keyboard controls and themes.
- [x] Run relevant backend suites, full frontend unit suite, focused browser
  suite, ruff, ty, eslint, TypeScript/build and bundle-size checks.
- [x] Update CUX backlog, inventory and changelog with implemented behavior and
  actual verification evidence. Rebuild served assets in static/ml_canvas.
- [x] Report what is visible and any verified limitations. Leave changes ready
  for review in the user's workspace.

## Execution record

- Baseline: b67ee0d1; clean tracked tree; previous full frontend suite 1,212 passed.
- Worktree creation was denied by the sandbox; skill fallback uses branch 0817.
- Local offloader unavailable; health check could not start its server because
  its log directory is outside the writable workspace. No retry required.
- Context map and task interfaces reviewed: backend and UI own separate files;
  controller owns the shared frontend types, state, hooks and integration.

## Completion evidence

- Task 1 complete: 27 new inspection tests; final relevant backend sweep passed
  228 tests and seven configuration snapshots. Ruff/format and backend ty passed.
- Task 2 complete: semantic receipts, single-flight transport and toolbar
  activation snapshots covered by 30 focused hook/toolbar tests.
- Task 3 complete: tab state, schemas, measured summaries, branch and plain/X-y
  split matching, blocked states and pending-focus handling covered by component
  tests. Independent review findings were reproduced, fixed and re-reviewed.
- Task 4 complete: full frontend suite passed 1,243 tests; final focused suite
  passed 61 after the final selection refinement. All 26 targeted browser cases
  passed. Light/dark screenshots inspected; lint, TypeScript/build and bundle
  budget passed (main gzip 304.8 KB / 325 KB). Served assets rebuilt.
- Verification limitations: real backend router/engine with fixture catalog;
  browser APIs mocked. No live training, hosted CI, commit or push performed.

### All-node follow-up evidence

- Bulk capture and global sample budgets passed 36 inspection tests, including
  shared branches, failures, skipped nodes, bounded execution-error details and
  backward-compatible single-node requests. Final backend sweep: 237 tests and
  seven configuration snapshots passed; scoped Ruff/format and backend ty passed.
- Full frontend suite: 1,248 tests passed. The final focused hook/toolbar/settings
  sweep passed 46 tests after removing obsolete selected-node receipt fields.
- Browser regression sweep: 28 scenarios passed. After adding branch-specific
  HTTP-200 runtime-error coverage, all five inspection browser scenarios passed.
  Tests exercise node changes during a request, shared receipts and Results
  visibility/content preservation. Existing request mocks now accept the bulk
  query; the viewport test uses reduced motion to avoid reading mid-animation.
- Independent review identified the hidden runtime diagnostic and legacy skipped
  training-node explanation; both have regressions and passed re-review.
- Frontend lint, TypeScript/build and bundle budgets passed. Served assets were
  rebuilt; main gzip is 304.7 KB against 325 KB. Browser APIs remain mocked.
- Backlog, payload inventory and changelog updated. No commit or push performed.

### Single-action and node-local path evidence

- Backend: 154 targeted preview/API/partition tests passed, including all 41
  inspection tests and seven configuration snapshots. Scoped Ruff/format and
  backend ty passed. Optional provenance leaves raw branch capture unchanged.
- Frontend: 70 focused tests and the full 1,254-test suite passed; all five path
  grouping tests passed after the final test-fixture type correction.
- Browser: 17 inspection, preview and execution-feedback scenarios passed,
  including shared-source deduplication, different paths, retained side selection,
  toolbar-only execution, closed Results and captured failures. Dark screenshot
  inspected; browser APIs remain mocked.
- Independent review found no actionable issues. Lint, TypeScript/build and
  bundle budgets passed; served assets rebuilt (main gzip 304.6 KB / 325 KB).
- Backlog, inventory and changelog updated. No commit or push performed.
