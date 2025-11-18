# API Layer Overview

This folder hosts every browser-side helper that talks to the ML workflow backend. Each module wraps a concrete REST endpoint (or a small family of endpoints) and normalises inputs/outputs so the rest of the UI can stay declarative.

## Directory map

| Path | Description |
| --- | --- |
| `catalog.ts` | Fetches the feature-node catalog that powers the canvas sidebar. Throws if the backend call fails so React Query can surface the error. |
| `datasets.ts` | Lists available dataset sources. Accepts an optional `limit` query param to keep landing pages lightweight. |
| `utils.ts` | Helpers shared by higher-level API calls. Today it provides deterministic graph hashing via `generatePipelineId`, which is used when saving/retrieving pipelines. |
| `analytics/` | Feature quality and diagnostics requests (binning, outlier checks, quick profile, etc.). Each file exports a small `fetch*` helper and `analytics/index.ts` re-exports them. |
| `jobs/` | CRUD helpers for async jobs: training requests, hyperparameter sweeps, evaluation runs. Splitting API files keeps the payload types focused. |
| `pipelines/` | Pipeline-focused endpoints such as create/update (`crud.ts`) and preview rendering. Imported heavily by the canvas save/load hooks. |
| `recommendations/` | Encapsulates services that recommend encodings, outlier treatments, and statistical insights for nodes. |
| `types/` | Source of shared API TypeScript types (dataset summaries, graph shapes, catalog entries, etc.). All other files import from here instead of redefining shapes. |

## Module notes

### `analytics/`
- `binnedDistribution.ts` – pulls histogram-style summaries for selected features.
- `outlierDiagnostics.ts` – fetches influence/outlier metrics used in the canvas inspector.
- `quickProfile.ts` – lightweight profile stats when users hover over dataset columns.
- `index.ts` – convenience barrel so consumers can `import { fetchQuickProfile } from '@/api/analytics'`.

### `jobs/`
- `training.ts` – start/cancel training jobs and poll their status.
- `hyperparameters.ts` – endpoints for hyperparameter tuning queues.
- `evaluation.ts` – submit evaluation runs against saved models.
- `index.ts` – exports every job helper for ergonomic imports.

### `pipelines/`
- `crud.ts` – create, read, update, delete operations for stored feature pipelines.
- `preview.ts` – requests backend previews (e.g., sample outputs or validation results) before persisting.
- `index.ts` – barrel file.

### `recommendations/`
- `encoding.ts` – suggests encoding strategies per column.
- `outliers.ts` – anomaly-handling recommendations.
- `stats.ts` – pulls summary stats for recommendation widgets.
- `index.ts` – barrel file.

## Integration tips
- Every helper returns a Promise and throws on HTTP failure; pair them with React Query so retries and caching are handled centrally.
- Use the exported types from `types/index.ts` when wiring new hooks to avoid hand-maintaining duplicate interfaces.
- Utilities in `utils.ts` are intentionally dependency-free. If you add new helpers, keep them browser-safe (no Node-specific APIs) so they work during SSR/static builds.
