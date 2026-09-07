# CUX-02 — Guided connections implementation

Scope approved in the canvas UX conversation on 2026-09-07.
Backlog: `canvas_ux_improvements_2026-09-06.md`, CUX-02.

## Interaction

Output labels open a small keyboard-accessible picker with Existing node and
Add node choices. Each choice identifies its input port. Search reuses the
component vocabulary; hidden legacy types are excluded from new suggestions.
The picker closes after a successful action. Dismissal preserves graph state.
Adding and connecting is a single graph update and a single undo operation.

During a connection drag, compatible ports receive a visible ring. The hovered
incompatible target shows its reason in a floating message above canvas nodes.
Guidance supports dragging from either end. Read-only mode allows inspection
but disables handles and hides mutation controls.

Connecting any Train/Test output includes both outputs; Validation joins only
when enabled. X/y follows the same rule. Visible port lines converge at one
junction and continue as a single connection to the downstream input. Manual
dragging, existing-node selection, and adding a node share this behavior.
The graph stores one canonical edge per group, so deletion, undo/redo, and
copy/paste operate on the whole connection. Loading older graphs folds duplicate
split edges while retaining distinct downstream branches. Validation changes
update the visible group without requiring reconnection. Split port labels share
the existing body area with the summary: summary on the left, compact port labels
on the right. No extra spacer rows or fixed header height enlarge these cards.

New nodes use the closest available position to the right of their source,
starting with a 64px gap and avoiding existing nodes. The canvas reveals the
source and new node together.

## Files and implementation sequence

1. `src/core/utils/connectionValidation.ts`, `useGraphStore.ts`, and store tests:
   share cycle/model/port checks across interactive paths; retain confirmation
   warnings and existing public helper exports; add an atomic connected-node
   insertion action. Test rejection, cancellation, undo/redo, and split handles.
2. `src/components/canvas/ConnectionPicker.tsx`, `CustomNodeWrapper.tsx`, and
   `FlowCanvas.tsx`: add the port picker, compatible-port styling, and a bounded
   floating rejection message using the installed React Flow connection state.
   Reuse existing node reveal, search, naming, and Radix popover behavior.
3. Browser checks: drag guidance in both directions, nearby-node stacking,
   keyboard existing/new choices, undo, split/model ports, read-only mode,
   and light/dark laptop layouts. Run relevant unit tests, full frontend units,
   lint, TypeScript/build, then update the backlog and v0.8.16 notes.

No backend payloads, execution handlers, or new dependencies are required.
