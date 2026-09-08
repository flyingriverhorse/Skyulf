# CUX-07 selected-node inspection: payload inventory

Date: 2026-09-08
Status: Inventory complete; Input / Output UI and payload extension remain open.

## Existing contracts

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

This inventory is based on source inspection. It does not claim that CUX-07 is
implemented or that live training was exercised.
