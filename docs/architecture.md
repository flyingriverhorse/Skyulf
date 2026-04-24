# Architecture

Skyulf is split into three pieces with strict boundaries:

1. **skyulf-core** (this docs set focuses on this)
   - A standalone Python ML library.
   - Implements a strict **Calculator → Applier** pattern for every node.
   - Depends on Pandas/Numpy/Scikit-Learn (and a few optional ML utilities).
2. **backend**
   - FastAPI + Celery orchestration layer.
   - Handles ingestion, jobs, persistence, and exposes REST APIs.
3. **frontend**
   - React + TypeScript UI (ML canvas).
   - Builds pipeline configs and talks to the backend.

## The Calculator → Applier Pattern

Skyulf-core separates learning from transformation:

- **Calculator**: `fit(data, config) -> params`
  - Learns statistics / encoders / models.
  - Returns a serializable `params` dictionary.
- **Applier**: `apply(data, params) -> transformed_data`
  - Stateless transformer.
  - Applies learned parameters.

This makes pipelines easier to persist and safer to run in production:

- Learning happens on train.
- The learned state is explicit.
- Applying is pure and repeatable.

## Hybrid Engine (Polars & Pandas)

Skyulf employs a **Hybrid Engine** architecture to maximize performance:

-   **Polars**: Used for high-performance data ingestion (ETL) and stateless transformations (Scaling, Imputation, Encoding) where possible.
-   **Pandas/Numpy**: Used for stateful learning (Calculators) and compatibility with Scikit-Learn models.

The system automatically detects the input data type (`pd.DataFrame` or `pl.DataFrame`) and dispatches to the appropriate optimized path.

## Node Registry

Skyulf uses a **Registry Pattern** to decouple the pipeline orchestrator from specific node implementations.

- **Registration**: Nodes self-register using the `@NodeRegistry.register("NodeName", ApplierClass)` decorator on the Calculator class.
- **Discovery**: The pipeline dynamically looks up the Calculator and Applier classes by name at runtime.
- **Extensibility**: New nodes can be added simply by creating a new file and decorating the class; no changes to `pipeline.py` are required.

## Data Catalog

To decouple data loading from the execution engine, Skyulf uses a **Data Catalog** pattern.

- **Interface**: `DataCatalog` (in `skyulf-core`) defines the contract for loading data by identifier.
- **Implementation**: `FileSystemCatalog` (in `backend`) implements this interface to load files from the local filesystem.
- **Usage**: The `PipelineEngine` is injected with a catalog instance. Nodes request data by ID (or path), and the catalog handles the retrieval.

## Pipeline Data Flow

At runtime, `SkyulfPipeline` orchestrates:

1. **Preprocessing**: `FeatureEngineer`
   - Executes a list of steps (each step is a transformer).
   - Some steps change the data structure (e.g., splitters) and are handled specially.
2. **Modeling**: `StatefulEstimator`
   - Trains a model on the train split.
   - Optionally evaluates on test/validation.

High-level flow:

```
Raw DataFrame
  └─ FeatureEngineer.fit_transform(...)  -> DataFrame or SplitDataset
        └─ (optionally) SplitDataset.train / test / validation
              └─ StatefulEstimator.fit_predict(...) -> predictions
```

## Avoiding Data Leakage

If you split first (or provide a `SplitDataset`), calculators should learn only on the train split.
This prevents leakage of statistics from test/validation.

See the User Guide section "SplitDataset & Leakage" for recommended patterns.

## Multi-Path Execution & Merge

*New in v0.3.0*

Training nodes can receive inputs from **multiple upstream branches**. The engine collects all inputs via `_resolve_all_inputs()` and combines them using `_merge_inputs()`:

- **Column-wise concat** when row counts match (parallel preprocessing paths).
- **Row-wise concat** when column schemas match (data augmentation).

Inputs are merged in topological order. Duplicate columns are deduplicated. Dead-end branches (nodes not on a path to any terminal) are pruned from execution.

```
Dataset → Scaling   ──┐
                      ├──→ Training Node (merge)
Dataset → Encoding  ──┘
```

## Parallel Experiment Execution

*New in v0.4.0*

When a canvas has multiple training/tuning nodes, `partition_parallel_pipeline()` splits the graph into independent sub-pipelines:

1. **Multiple terminals** — Each training node gets its own sub-pipeline via BFS ancestry tracing (`_collect_ancestors()`).
2. **Single terminal, parallel mode** — When `execution_mode=parallel`, each incoming branch becomes a separate job.

The API returns `job_ids: List[str]` — one per branch. Shared prefix nodes (e.g., dataset) are included in each sub-pipeline. When `target_node_id` is set, only the branch containing that node executes.

```
Dataset → Scaling → RandomForest (Train)   → Job 1
    │
    └──→ Encoding → XGBoost (Train)         → Job 2
```

## Topological Execution Order (Kahn's Algorithm)

*Hardened in v0.5.1*

Sub-pipelines produced by partitioning must be executed in **topological order** —
every parent must finish (and write its artifact) before any child reads it.
Earlier versions used a reversed-BFS in `_collect_ancestors()`, which silently
produced an incorrect order for diamond-shaped graphs and caused
`FileNotFoundError: Artifact not found` at run time.

`_collect_ancestors()` now uses **Kahn's algorithm** restricted to the ancestor
subgraph of the requested node:

1. **Discover** the ancestor set (the node plus everything reachable backwards
   through `inputs`) via BFS.
2. **Build in-degree map** counting how many parents each ancestor has *within
   the subgraph*.
3. **Pop ready nodes** (in-degree 0) one at a time, append to the result, and
   decrement the in-degree of each child. Any child whose in-degree drops to 0
   becomes ready.
4. **Cycle detection** — if the result is shorter than the discovered set, the
   subgraph contains a cycle; we log a warning and fall back to discovery
   order.

> **BFS in one line:** *Breadth-First Search* explores a graph level-by-level using a FIFO queue — visit a node, enqueue its neighbours,repeat. Contrast with DFS which uses a stack and goes deep first. We use BFS in step 1 because we only need the *set* of ancestors (level order is irrelevant); the actual execution order comes from step 3.

### Why the diamond case broke reversed-BFS

```
Dataset ──► Split ──► Scaler ──► Train
   │                    ▲
   └────────────────────┘   (shortcut edge)
```

Scaler has two parents (`Split` and `Dataset`). Reversed-BFS starting from
Train enqueues parents in the order it visits children, so for some traversal
orders Scaler ended up emitted **before** Split. The engine then ran Scaler →
tried to load Split's artifact → file did not exist.

Kahn's prevents this because Scaler's in-degree (2) cannot reach 0 until both
Split and Dataset have been emitted.

### Preview-specific partitioning

`partition_for_preview()` (also v0.5.1) reuses `_collect_ancestors()` to split
a graph by **data leaves** rather than by training terminals. A canvas with
several parallel preprocessing chains and no training node now renders one
preview tab per leaf — see `branch_previews` in the `/api/pipeline/preview`
response and the branch tab bar in `ResultsPanel.tsx`.
