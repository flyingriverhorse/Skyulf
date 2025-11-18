# Feature Canvas Module

The `canvas` package contains everything required to render and operate the drag-and-drop feature engineering canvas: UI components, orchestration hooks, domain helpers, and shared types.

## Directory map

| Path | Description |
| --- | --- |
| `components/` | React components for the canvas shell, viewport, side panel, and individual nodes. They stay thin and delegate heavy logic to hooks/services. |
| `constants/` | Static configuration such as default node graphs, connection-handle metadata, and split definitions. |
| `hooks/` | Reusable logic slices (catalog loaders, node editor, pipeline hydration, split propagation, etc.) extracted from `CanvasShell`. These are the main integration points with `react-flow-renderer` and the API layer. |
| `services/` | Pure helper modules for layout math, graph serialization, config sanitisation, and node factories. No React dependency so they are easy to unit-test. |
| `types/` | TypeScript declarations for node data, pipeline payloads, and feedback widgets consumed across the canvas. |
| `utils/` | Small utility helpers (e.g., building save payloads, formatting timestamps) that do not warrant a full service module. |

Use the sections below to understand the roles of every file/folder.

## Components

| Component | Purpose |
| --- | --- |
| `CanvasShell/CanvasShell.tsx` | Top-level orchestrator rendered by the feature canvas route. Composes every hook (node catalog, node editor, pipeline loader, split propagation, connection handlers, sidebar view model) and exposes an imperative API (`openCatalog`, `closeCatalog`, `clearGraph`) to parent pages. |
| `CanvasViewport/CanvasViewport.tsx` | Thin presentational wrapper around `ReactFlow`. Receives prepared props (nodes, edges, handlers) and renders the canvas, controls, reset/catalog FABs, and connection line component. This keeps React Flow usage confined to one place. |
| `CanvasSidepanel/CanvasSidepanel.tsx` | Hosts the inspector/sidebar UI (dataset details, pipeline stats, history tabs, save feedback). Driven by `useSidepanelViewModel`, toggled via `useSidepanelToggle`, and consumes snapshot/save state hooks. |
| `FeatureCanvasNode/FeatureCanvasNode.tsx` | Custom node renderer injected into React Flow’s `nodeTypes`. Handles status badges, drag handles, chip indicators, and node-level action buttons (open settings, remove, duplicate, etc.). |

Each component folder currently contains a single file, but the structure leaves room to add stories/tests or split components further.

## Hooks catalog

The `hooks/` folder is the largest part of the module. Each file is named after the capability it delivers:

| Hook | Responsibility |
| --- | --- |
| `useCanvasSidebar.tsx` | Memoises the catalog drawer body (loading/errors vs. `FeatureCanvasSidebar`). Returns React elements so `CanvasShell` can render the drawer without inline logic. |
| `useCanvasSnapshotState.ts` | Tracks auto-saved snapshots, their timestamps, and the current snapshot ID shown in the side panel. |
| `useClearCanvasHandler.ts` | Provides the handler that resets the canvas back to a dataset-only state and closes modals. Useful for toolbars outside `CanvasShell`. |
| `useConnectionHandlers.ts` | Supplies `onConnect` and `isValidConnection` callbacks for React Flow along with side effects (schedule node-internal updates). Encapsulates handle matching rules from `constants/nodeHandles.ts`. |
| `useDatasetSelection.ts` | Derives the active dataset context (source ID, display label) based on page props plus local overrides. |
| `useDatasetSelectionHandler.ts` | Offers callbacks for switching datasets from the side panel while keeping the canvas state coherent. |
| `useNodeCatalogDrawer.ts` | Fetches the node catalog via the API layer, exposes loading/error state, a `Map` of catalog entries for quick lookup, and imperative `openCatalog`/`closeCatalog` helpers. |
| `useNodeEditor.ts` | Central node CRUD facility. Registers nodes with interaction handlers, creates/removes nodes, updates configs/data, handles bulk resets, and keeps the node ID counter in sync with React Flow. |
| `usePipelineHistory.ts` | Loads and formats pipeline history (previous saves) for the side panel timeline. |
| `usePipelineHydration.ts` | Takes pipeline payloads and produces React Flow nodes/edges (used by save/load flows that already have the graph in memory). |
| `usePipelineLoader.ts` | Wraps the backend fetch + hydration flow given a `sourceId`. Handles default/sample graphs, error fallbacks, fit-view scheduling, and hydration callbacks. |
| `usePipelineSave.ts` | Builds `FeatureGraph` payloads with `buildPipelineSavePayload`, invokes the API, manages optimistic state, and surfaces success/error feedback. |
| `useSaveFeedbackState.ts` | Houses toast/banner state (success, pending, error) used after saves and previews. |
| `useSidepanelToggle.ts` | Simple hook that keeps track of whether the inspector side panel is open/closed and exposes toggle helpers. |
| `useSidepanelViewModel.ts` | Combines dataset, pipeline, snapshot, and job info into a single view model consumed by `CanvasSidepanel`. Prevents prop drilling and repeated selectors. |
| `useSplitPropagation.ts` | Watches edges and recomputes `activeSplits`, `connectedSplits`, and `hasRequiredConnections` per node. Triggers `useUpdateNodeInternals` for nodes whose handles change. |

> **Tip:** If you add a new hook, keep it focused on one concern (fetching, state derivation, or UI wiring). Smaller hooks compose better inside `CanvasShell`.

## Constants

- `defaults.ts` – default `nodes` and `edges` used when the canvas first loads or when a pipeline reset is triggered.
- `nodeHandles.ts` – per-node connection handle definitions and matcher utilities used by `useConnectionHandlers`.
- `splits.ts` – split type enumerations, ordering, and helpers for `useSplitPropagation`.

## Services

| File | Responsibility |
| --- | --- |
| `configSanitizer.ts` | Sanitises node configs before persistence (removes transient flags, normalises defaults, clones objects safely). |
| `graphSerialization.ts` | Builds shareable graph snapshots, used for the settings modal and save payloads. |
| `layout.ts` | Layout helpers (default positioning, drop-position calculations, sample graphs). |
| `nodeFactory.ts` | Creates new nodes with the correct data shape and wiring, plus helpers to register interaction callbacks. |
| `splitPropagation.ts` | Low-level math for computing active/connected splits and validating connection requirements. |

## Types & Utils

- `types/nodes.ts` – definitions for `FeatureNodeData`, catalog metadata, and node-specific flags.
- `types/pipeline.ts` – interface for `CanvasShell` props/handles and pipeline hydration payloads.
- `types/feedback.ts` – shapes for user feedback/toast messaging around saves/loads.
- `utils/buildPipelineSavePayload.ts` – assembles the backend payload from canvas state, including sanitised snapshot IDs.
- `utils/time.ts` – date/time helpers (formatting, countdowns) used in history/feedback components.

## Working in this folder

1. **Add logic via hooks/services first.** Components should remain declarative wrappers; if a component grows too large, extract the logic into `hooks/` or `services/` the way `CanvasShell` now delegates node editing, pipeline loading, split propagation, and connection validation.
2. **Keep React Flow specifics centralised.** `CanvasViewport` and `useConnectionHandlers` are the only places that touch `react-flow-renderer` APIs directly. Feed them simple props to keep type-surface manageable.
3. **Reuse shared constants/types.** Before introducing new IDs or enumerations, check `constants/` and `types/` so the entire canvas remains consistent.
