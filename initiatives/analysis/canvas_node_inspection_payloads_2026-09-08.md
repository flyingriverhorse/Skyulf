# CUX-07 selected-node inspection: payload inventory

Date: 2026-09-08
Status: Inventory and CUX-07 implementation complete; see the implementation record below.

## Contracts recorded before implementation

| Source | Available today | Limit for CUX-07 |
|---|---|---|
| `POST /api/pipeline/schema-preview` | Per-node predicted column names and dtypes; unknown/data-dependent schemas are `null`; broken column references | Predictions contain no measured row counts or samples |
| `POST /api/pipeline/preview` | Per-node status, errors, metrics, timing and summary metadata; terminal samples capped at 50 rows, true totals, branch labels and split keys | Samples describe branch terminals, not every node's actual input/output; response has a pipeline ID but no distinct preview-run receipt or configuration fingerprint |
| Execution engine artifacts | Intermediate data saved under node IDs during execution | Preview's temporary store is removed after the request; the UI cannot fetch those artifacts later |
| Data Preview node | Dedicated background jobs; summary/sample/shape data for that inspection point; branch job IDs | Requires inserting a separate node; does not give paired input/output for any selected node |
| Monitoring Node Inspector | Stored job graph, node parameters, neighbors, status, timing, job/branch identity and logs | Historical graph inspection has no paired data samples or column/row deltas |
| Canvas graph store | Current graph, latest preview response and predicted schemas | No captured semantic configuration alongside the result to determine whether edits made it stale |

## Files and responsibilities

- `frontend/ml-canvas/src/core/api/client.ts`: `NodeExecutionResult` and
  `PreviewResponse` wire types; samples and totals are currently terminal-level.
- `frontend/ml-canvas/src/core/api/schemaPreview.ts` and
  `src/core/hooks/useSchemaPreview.ts`: predicted schema transport and storage.
- `frontend/ml-canvas/src/core/store/useGraphStore.ts`: latest result storage.
- `frontend/ml-canvas/src/components/shared/NodeInspectorModal.tsx` and
  `src/core/api/monitoring.ts`: existing historical graph inspector.
- `frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx`:
  separate preview-node jobs and bounded rendered samples.
- `backend/ml_pipeline/_internal/_routers/preview.py`: branch execution,
  `_extract_preview`, terminal sample aggregation and temporary-store cleanup.
- `backend/ml_pipeline/_execution/schemas.py`: engine node result metadata.
- `backend/ml_pipeline/_execution/engine/_node_runners.py`: intermediate
  artifact writes and preprocessing metrics.
- `backend/monitoring/router.py`: historical inspector response models/routes.

## Implementation direction

Extend preview with an optional, bounded inspection request for the selected
node. Capture its actual input after merge/input resolution and its output
during that execution, before temporary artifacts disappear. Return samples,
full row counts, column names/dtypes and explicit availability/error states.
Keep capture in the backend; core nodes remain independent of HTTP and storage.

Key inspection results by run, branch, node, input port and split. The current
preview aggregator assigns `combined_node_results[node_id]` repeatedly across
branches, so that map cannot distinguish multiple executions of a shared node.
Do not infer an input by taking an arbitrary upstream terminal sample: merges,
splits and parallel execution can change which data the node actually receives.

Give each preview a distinct receipt and record the semantic configuration used
for it. Compare that configuration with current settings and connections to
mark results stale; position, selection and panel changes must not count as
data changes. Display predicted schemas separately from measured results.

Reuse measured preprocessing metrics when their meanings are known. Otherwise
show only observed column/row changes and an explicit unavailable explanation;
do not infer counts such as missing values filled from the sample.

The UI can then expose Input / Output in selected-node settings, with branch
and split selectors where required. A new endpoint for historical training
artifacts is a separate retention/API decision; the existing historical graph
inspector should retain its current behavior.

## Verification needed when implementing

- Backend: sample cap and full totals, empty/failing/source/model nodes, actual
  merged input, split/X-y shapes, shared node across branches, temp cleanup.
- Frontend: predicted versus measured state, stale results after semantic edits,
  no stale marker after panning, run/branch identity, bounded tables and explicit
  loading/unavailable/error states.
- Browser: keyboard switching, light/dark layouts, compact settings and results
  panels, focus retained through refresh and selection changes.

## Implementation record

The extension is implemented in
`canvas_node_inspection_implementation_2026-09-08.md`. POST /pipeline/preview
accepts optional `inspect_all=true` or `inspect_node_id` and returns `run_id` plus
branch-specific `node_inspections`. Bulk mode takes precedence when both are
supplied. Default callers do not capture intermediate data. Actual
input is captured at resolution after merging; output is captured before another
branch can replace artifacts. Capture failures do not fail execution, and
request cleanup now also covers failures during source resolution.

Settings / Input / Output tabs show measured schema, bounded samples and shape
changes, separate predicted schemas and explicit availability/error states.
The frontend requests all nodes and stores the submitted semantic configuration
with its latest receipt. Changing selection reuses captured data without another
request or modifying global Results. Toolbar Preview data is the single execution
action; Input / Output has no Refresh button and directs empty/stale states to
the toolbar. Optional `path_id` and `path_label` identify each node's actual
partition ancestry independently of downstream training choices. The UI groups
matching path IDs with equal captured input/output, preserves distinct paths and
measurements/errors, and hides the selector for one result. Missing provenance
is kept separate. Path labels describe upstream steps and are capped at 240 chars.
Data path and compatible split selection survive side switches; settings retain
keyboard focus. Captured data remains in session memory, not a historical
training-artifact API. Limits: 50 rows, 100 columns, six tables per side, 500
characters per captured display value and 256 KiB of sample-row JSON per side.
Bulk capture shares an 8 MiB sample-row budget fairly across all branch-node
sides; bounded schema metadata is separate. Capture uses existing branch runs
without additional pipeline executions or catalog reads.

Follow-up verification passed 237 backend tests, seven configuration snapshots,
the full 1,248-test frontend suite and 46 final focused frontend tests. The
28-scenario browser sweep passed; all five inspection scenarios passed after
adding HTTP-200 runtime-error coverage. Ruff, ty, eslint, TypeScript/build and
bundle-size checks passed; served assets rebuilt. Runtime errors retain bounded
detail within the failing branch, and legacy selected-model requests retain
their explicit skipped-training explanation.
The subsequent single-action/data-path follow-up passed 154 targeted backend
tests (41 inspection cases and seven snapshots), 1,254 frontend tests, 70 focused
tests and 17 browser scenarios. Independent review, Ruff/format, ty, eslint,
TypeScript/build and bundle budgets passed; served assets rebuilt. These checks
use a fixture catalog and mocked browser APIs, not live training.
