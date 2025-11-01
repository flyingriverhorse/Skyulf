# ML Workflow Pipeline Internals

This note summarizes how the feature-engineering "ml-workflow" pipeline is executed on the backend. Code references use repository-relative paths.

## Data sourcing & graph input
- Entry points live in `core/feature_engineering/routes.py`. Requests resolve a dataset sample via `_load_dataset_frame`, which either pulls a sampled preview through `FeatureEngineeringEDAService` or loads the entire file with `FullDatasetCaptureService` when the caller requests a full run.
- Canvas graphs arrive as JSON (`FeatureGraph`), then `_sanitize_graph_nodes` and `_sanitize_graph_edges` strip non-serializable values and preserve `sourceHandle`/`targetHandle` so split routes remain traceable.
- `_ensure_dataset_node` injects the fixed `dataset-source` node when the client omits it. `_generate_pipeline_id` (routes.py:204) hashes the dataset id and normalized graph so we can key transformer storage per unique pipeline topology.

## Resolving execution order
- `_build_predecessor_map` and `_build_successor_map` derive adjacency lists from the edge list.
- `_execution_order` performs a depth-first walk that ensures parents execute before children and prunes nodes that are not reachable from `dataset-source`. When a `target_node_id` is supplied (previewing an intermediate node) the order is truncated just before that node.
- `_determine_node_split_type` walks upstream edges to see whether the target node hangs off a specific `train/test/validation` outlet of a split node. The preview response is then filtered to that split only.

## Main execution loop
- `_run_pipeline_execution` owns the preview transformations. It iterates the resolved order, looks up each node’s `catalogType`, and dispatches to the implementation in `core/feature_engineering/nodes/**`.
- Each node returns the transformed frame plus a textual summary and a structured signal that feeds `PipelinePreviewSignals`. Signals include quality diagnostics, encoding summaries, and `TrainModelDraftReadinessSnapshot` for the lightweight modeling readiness node.
- `_apply_graph_transformations_before_node` mirrors this loop for recommendation APIs that need to simulate upstream nodes before running analytics.
- After all nodes run, `remove_split_column` strips internal metadata (`__split_type__`) so responses look like standard DataFrames.

## Split-aware processing
- `nodes/modeling/dataset_split.py::apply_train_test_split` introduces the `__split_type__` marker and tracks ratios in `TrainTestSplitNodeSignal`.
- `split_handler.py` defines `SplitType`, `NodeCategory`, and the `NODE_CATEGORY_MAP` that classifies every catalog entry as a transformer, filter, splitter, model, resampling step, or passthrough.
- `SplitAwareProcessor` (split_handler.py:247) centralizes the policies:
  - Filters (`drop_missing_rows`, `remove_duplicates`, etc.) run on each split independently before the results get merged.
  - Transformers and models fit on the train partition and reuse the learned state (`transform`) on test/validation. Resampling nodes are forced to stay on the train split.
  - Passthrough nodes ignore split metadata, and splitter nodes (feature/target split, train/test split) are allowed to create or mutate the split column directly.
- Encoding nodes such as `one_hot_encoding` and `target_encoding` consume the split metadata to decide whether to fit or reuse an encoder. They store fitted artifacts via `TransformerStorage` when a `pipeline_id` is present.

## Transformer persistence & audit
- `transformer_storage.py` keeps an in-memory registry keyed by `{pipeline_id}:{node_id}:{transformer_name}:{column}`. Metadata tracks when the object was created and which split it interacted with.
- Nodes call `store_transformer`, `has_transformer`, and `record_split_activity` so repeat preview runs (or full executions) reuse the exact same encoder/scaler objects. The `transformer_audit` node exposes this state through preview signals.

## Preview vs. full execution
- Preview responses run on sampled data unless the dataset is already small. When the estimated row count exceeds `FULL_DATASET_EXECUTION_ROW_LIMIT` the system defers a full run.
- `FullExecutionJobStore` queues background jobs (`FullExecutionJob`) that re-run `_run_pipeline_execution` on the full dataset asynchronously. Status is exposed via `/api/pipelines/{dataset}/full-execution/{job}`.
- Full runs and previews share the same execution path, so any node that respects split metadata behaves identically across modes.

## Outputs & signals
- `build_data_snapshot_response` packages the transformed rows, column stats, `PipelinePreviewSignals`, and any full-execution status into the API response.
- Clients can set `PipelinePreviewRequest.include_signals=False` when they only need the tabular preview; the response will then omit the `signals` payload entirely.
- The response records every applied step in order, making it easy to audit what happened before and after each split-aware transformation.
