# Pipeline Draft Persistence Flow

This document describes how feature-engineering pipeline drafts are saved from the React canvas and how the FastAPI service persists them.

## Frontend snapshotting (`frontend/feature-canvas/src/App.tsx`)
- The canvas layer keeps an immutable copy of the current graph in `graphSnapshotRef`. `handleGraphChange` updates it whenever nodes/edges move and flips the `isDirty` flag unless we are hydrating from a stored draft.
- Clicking **Save draft** triggers `handleSaveClick`. It refuses to save when no dataset is selected or the canvas only contains the `dataset-source` stub.
- The save payload follows `FeaturePipelinePayload` (`src/api.ts`):
  - `name`: defaults to `Draft pipeline for <dataset>` so each dataset keeps a readable label.
  - `graph`: raw `nodes` plus `edges`, where edges are normalized to `type: "animatedEdge"` and `animated: true` to keep consistent rendering after reloads.
  - `metadata`: lightweight client annotations (timestamp, node count, edge count). Callers can extend this map; it is forwarded to the API untouched.
- `useMutation` (`triggerSave`) posts the payload through `savePipeline`. On success the hook clears the dirty flag, caches the API response under the pipeline query key, and invalidates history so the revision list refreshes.
- `fetchPipeline` and `fetchPipelineHistory` reuse the same REST endpoints to hydrate the canvas at startup or when switching revisions.

## REST contract (`frontend/feature-canvas/src/api.ts`)
- Drafts travel over `/ml-workflow/api/pipelines/{dataset_source_id}` as JSON. The response type `FeaturePipelineResponse` mirrors the request fields and adds `id`, `dataset_source_id`, `is_active`, and timestamps.
- The client never mutates the returned graph before storing it; future renders expect the backend to echo the same node/edge schema.

## Backend schema & persistence (`core/feature_engineering/routes.py`)
- The POST handler `upsert_pipeline` fetches the latest `FeatureEngineeringPipeline` row for the dataset. If one exists it updates `name`, `description`, `graph`, and `pipeline_metadata`; otherwise it inserts a new row.
- `FeaturePipelineCreate` (`core/feature_engineering/schemas.py`) validates the payload:
  - `graph` is a `FeatureGraph` with arbitrary node/edge dictionaries (typically containing `id`, `type`, `position`, and `data.catalogType`/`data.config`).
  - `metadata` is optional and stored verbatim in the `metadata` column.
- `FeatureEngineeringPipeline` (`core/database/models.py`) persists drafts with columns:
  - `dataset_source_id` (string key), `name`, `description`.
  - `graph` (JSON) capturing the entire canvas structure.
  - `metadata` (JSON) for client-supplied annotations.
  - `is_active`, `created_at`, `updated_at` for lifecycle tracking.
- `get_pipeline` returns the most recent revision ordered by `updated_at`, while `/history` exposes multiple revisions for manual rollback.

## Data shape in practice
- Nodes round-trip with their canvas metadata, e.g.:
  ```json
  {
    "id": "node-2",
    "type": "feature-node",
    "position": {"x": 420, "y": 160},
    "data": {
      "label": "Train/test split",
      "catalogType": "train_test_split",
      "config": {"test_size": 0.2, "validation_size": 0.1}
    }
  }
  ```
- Edge records include connection handles (e.g. `sourceHandle: "node-2-train"`). The backend keeps them intact so `_determine_node_split_type` can identify which downstream branch represents the train/test/validation view.
- Metadata currently records client-side hints (`lastClientSave`, `nodeCount`, `edgeCount`) but can be enriched without backend changes because the column is schemaless JSON.
