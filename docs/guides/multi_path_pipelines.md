# Multi-Path Pipelines

Skyulf supports building pipelines with **multiple branches** that merge into a single training node or fan out into separate experiments. This guide covers both patterns.

---

## Merge: Combining Multiple Branches

When a training node has 2+ incoming edges, Skyulf **automatically merges** the upstream DataFrames before training.

### How It Works

```
Dataset → Scaling    ──┐
                       ├──→ Training Node (⊕ Merge)
Dataset → Encoding   ──┘
```

The training node collects **all** upstream branch outputs via `_resolve_all_inputs()` and combines them using `_merge_inputs()`.

### Merge Strategy (Auto-Detected)

| Condition | Strategy | Example |
|---|---|---|
| Same row count, different columns | **Column-wise concat** | Parallel preprocessing branches |
| Same columns, different rows | **Row-wise concat** | Data augmentation |
| No common columns, different shapes | **Error** | Incompatible inputs |

- Duplicate columns are automatically deduplicated after merging.
- Inputs are merged in deterministic **topological order** based on the pipeline graph.

### Merge Badge

Nodes with 2+ incoming edges display a blue **⊕ Merge** badge in the header showing the input count. Hover over it for a tooltip: *"Merge: combining data from N upstream sources"*.

### Connection Validation

Model-to-model connections (e.g., training → training) are **blocked** with an alert. Training nodes accept inputs from preprocessing nodes only.

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| "Empty DataFrame from upstream branch" | A preprocessing branch produced no rows | Check filters/cleaning nodes upstream |
| "No common columns" | Branches have incompatible schemas | Ensure branches produce compatible columns |

---

## Parallel: Running Separate Experiments

When you have 2+ training nodes on the canvas connected to **separate branches**, each one runs as an independent experiment.

### How It Works

```
Dataset → Scaling → Random Forest (Train)
    │
    └──→ Encoding → XGBoost (Train)
```

Each training node has its own **Train** button. Clicking it runs **only that branch** — the backend uses `target_node_id` filtering to isolate the sub-pipeline.

### Run All Experiments

When 2+ training nodes are connected on separate branches, a **"Run All Experiments"** button (🚀 Rocket icon) appears in the toolbar. Clicking it queues **all branches** at once, returning a list of `job_ids`.

### Merge/Parallel Toggle

Training nodes with 2+ incoming connections show a **Merge / Parallel** toggle:

- **Merge** (default): Combines upstream data before training.
- **Parallel**: Treats each incoming branch as a separate experiment and creates independent jobs.

The toggle is **user-controlled** — you decide based on your intent. The choice is stored as `execution_mode` on the node and passed to the backend during execution.

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| **Ctrl+C** (Cmd+C on Mac) | Copy selected nodes and their internal edges |
| **Ctrl+V** (Cmd+V on Mac) | Paste copied nodes with a position offset |

Supports multi-select. Each paste increments the offset so nodes don't stack.

---

## Pipeline Partitioning (Backend)

The backend function `partition_parallel_pipeline()` in `graph_utils.py` handles splitting:

1. **Multiple terminals**: If the graph has 2+ training/tuning nodes, each gets its own sub-pipeline via BFS ancestor tracing (`_collect_ancestors()`).
2. **Single terminal with parallel mode**: If one training node has `execution_mode=parallel`, each incoming branch becomes a separate sub-pipeline.

Shared prefix nodes (e.g., a dataset node used by both branches) are duplicated into each sub-pipeline so they can execute independently.
