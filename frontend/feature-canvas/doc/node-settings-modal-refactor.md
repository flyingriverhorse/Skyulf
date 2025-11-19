# NodeSettingsModal Modularization Plan

## Context
- Source: `frontend/feature-canvas/src/components/NodeSettingsModal.tsx` (~8k lines)
- Responsibilities today: metadata rendering, catalog-type gating, 40+ node-specific sections, cross-node recommendation hooks, connection validation, preview orchestration, save/reset handlers, utility sanitizers.
- Pain: tight coupling of UI, derived state, network orchestration; difficult to reason about regressions or to reuse logic outside the modal.

## Current Responsibility Map
| Concern | Examples in file | Notes |
| --- | --- | --- |
| Metadata + title | `metadata` / `title` memo (lines ~120-170) | Pure presentation, no dependencies. |
| Catalog capability flags | `useCatalogFlags` destructuring (~180-230) | Shared across many sections; should stay in a dedicated hook. |
| Parameter cataloging | `parameters` memo, dozens of `...Parameter` memos | Should be extracted into a selector module. |
| Sanitizers/util fns | `sanitizeStringList`, `sanitizeNumberValue`, `toHandleKey`, etc. | Move to `node-settings/utils/` to avoid redeclaring. |
| Connection diagnostics | `connectionInfo`, `connectedHandleKeys`, `missingRequiredInputs` | Candidate for `useConnectionReadiness()` hook. |
| Modeling-specific state | `useModelingConfiguration`, `featureMathOperations`, `trainModel*` handles | Already isolated via hooks; just wire in from orchestrator. |
| Async data fetches | Hooks for drop columns, encoding suggestions, preview, skewness, etc. | Should be triggered from a thin orchestrator hook instead of inline `useEffect`. |
| Section rendering | Hundreds of `<Section />` components with prop plumbing | Should be composed from dedicated "renderer" files grouped by domain (data hygiene, encoding, modeling, insights). |
| Save/reset orchestration | `handleSave`, `handleResetNode`, `handleParameterChange` | Move to `useNodeSettingsActions`. |

## Target Module Topology
```
frontend/feature-canvas/src/components/node-settings/
├─ modal/
│  ├─ NodeSettingsModal.tsx              # thin view orchestrator
│  ├─ useNodeSettingsState.ts           # aggregates hooks, derived data, sections to render
│  ├─ useNodeSettingsActions.ts         # save/reset/connection gating handlers
│  ├─ useNodeSettingsLayout.ts          # metadata, breadcrumbs, busy labels, CTA visibility
│  └─ sections/
│     ├─ metadata/MetadataPanel.tsx
│     ├─ connectivity/ConnectivityAlerts.tsx
│     ├─ hygiene/HygieneSections.tsx    # drop missing, duplicates, text cleanup...
│     ├─ encoding/EncodingSections.tsx
│     ├─ modeling/ModelingSections.tsx
│     ├─ insights/InsightsSections.tsx
│     └─ preview/PreviewPanel.tsx
├─ utils/
│  ├─ sanitizers.ts                     # shared sanitize* fns & constants
│  ├─ connections.ts                    # toHandleKey, handle-key helpers
│  └─ formatting.ts                     # wrappers around formatters used in modal only
├─ hooks/
│  ├─ useConnectionReadiness.ts
│  ├─ useParameterCatalog.ts
│  ├─ useNodeConfigState.ts             # wraps current `configState`/`setConfigState`
│  └─ useSectionRegistry.ts             # maps catalog flags to section configs
```

## Section Registry Concept
- Define a `SectionConfig` type: `{ id, when: (catalogFlags) => boolean, component: React.FC<SectionProps> }`.
- Build registries per domain (hygiene, encoding, modeling, insights) so `NodeSettingsModal` just iterates over active configs.
- Each section component receives only the slice of state it needs (e.g., `imputerStrategies`, `handleImputer...`).

## Incremental Refactor Plan
1. **Utility Extraction (PR1)**
   - Move `sanitize*`, `toHandleKey`, histograms into `utils/`.
   - Replace local implementations with imports; add tests for sanitizers.
2. **Connection + Metadata Hooks (PR2)**
   - Create `useConnectionReadiness` (wraps connection memos) and `useNodeMetadata`.
   - Update modal to use new hooks; verify snapshots unaffected.
3. **State Isolation (PR3)**
   - Build `useNodeConfigState` that encapsulates `configState`, `stableInitialConfig`, and reset logic.
   - Move `handleParameterChange`, numeric/text helpers into a dedicated actions hook.
4. **Section Registry (PR4)**
   - Introduce registry + domain section components while still defined inside modal file (to limit churn).
   - Ensure props-only data path (no direct access to modal internals).
5. **File Split (PR5)**
   - Move new hooks/sections into separate files per topology.
   - Keep `NodeSettingsModal.tsx` as orchestrator (<300 lines) that:
     1. calls hooks (`useNodeSettingsState`, `useNodeSettingsActions`)
     2. renders layout shell
     3. maps registry sections.
6. **Cleanup + Types (PR6)**
   - Delete dead imports, ensure barrel files exist.
   - Add unit tests for hooks (especially `useNodeSettingsState`).

## Data Flow After Refactor
1. **Inputs**: `node`, `graphSnapshot`, `sourceId`, `defaultConfigTemplate`.
2. **`useNodeSettingsState`** gathers:
   - Catalog flags
   - Parameter catalog (maps + proxies per node type)
   - Derived data (metadata, connection state, preview state, async statuses)
   - Section view-models (per domain)
3. **`NodeSettingsModal`** renders header, `MetadataPanel`, `ConnectivityAlerts`, `SectionRenderer`, and footer.
4. **Actions hook** exposes `handleSave`, `handleReset`, `handleParameterChange`, domain-specific handlers; sections only receive the functions they need.

## Testing Strategy
- Preserve existing Playwright coverage for the feature canvas.
- Add Jest/RTL tests for new hooks (mock node objects to ensure selectors behave).
- Smoke-test section registry to confirm `when` guards include intended node types.
- Regression checklist per PR: confirm saving nodes still calls `onUpdateConfig`, ensure async sections continue to hydrate.

## Risks & Mitigations
| Risk | Mitigation |
| --- | --- |
| Breaking save/reset flows while moving handlers | Introduce hook-level tests + feature-flag new hook path before deleting legacy code. |
| Section props explosion | Use typed view-models per domain to keep prop shapes small. |
| Merge conflicts with ongoing feature work | Stage refactor per domain; coordinate with feature authors via feature flags. |

## Open Questions
1. Do we want suspense-based loading for async sections once modularized?
2. Should section registry be configurable at runtime (e.g., for enterprise builds)?
3. Are there upcoming node types that require additional domains (e.g., LLM integration) we should reserve slots for?
