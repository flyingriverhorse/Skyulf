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

- Each output column appears once. Different-named columns are kept from both branches;
  overlapping columns follow the ownership and merge-strategy rules below.
- Inputs are merged in the order their edges enter the node, except that an input which is an
  ancestor of another input is always applied first. For sibling branches this is saved incoming-edge
  order, not their position on the canvas or completion time. `last_wins` is the default;
  `first_wins` selects the first version of a conflicting column instead.
- **Per-column ownership:** when a shared-ancestor baseline frame is available, the engine compares
  each branch against it. A column changed by only one branch keeps that branch's value under
  either strategy. Differing changes to the same column follow the configured strategy; branches
  that agree do not introduce a conflicting version. Without a usable baseline, overlapping
  columns follow the strategy directly. The strategy selects entire columns, not a whole branch
  or individual cells from different versions.

<a id="after-a-split-order-decides-everything"></a>

#### After a Split: Merge Strategy Still Applies

!!! warning
    When branches fork **after Train-Test Split**, its split artifact is not a baseline frame,
    so per-column ownership is generally unavailable. The configured strategy still applies:
    `last_wins` (default) keeps the last input's overlapping column, while `first_wins` keeps
    the first. Different-named columns from both branches survive.

    Train, Validation and Test merge separately; this does not concatenate the partitions.
    For X/y inputs, y is retained from the first branch regardless of the feature strategy.
    Every branch must preserve the same rows, row order and aligned target. First/last wins
    does not repair alignment or establish leakage safety.

    The practical consequence: under default `last_wins`, an earlier branch's encoding can be
    discarded if the last branch still carries that column raw. Training then fails fast with
    *"training frame contains N non-numeric column(s): …"*.

    Design post-split merges accordingly:

    - Make branches emit **disjoint columns**, or
    - Make **every** branch emit a **fully-numeric frame** (each branch imputes/encodes/scales on
      its own), so whichever branch wins, the merged result is model-ready.
    - Text pipelines: after a vectorizer, set `drop_original=True` (or drop the raw text column)
      so the raw text does not survive into the merge.

### Merge Badge

Nodes with 2+ incoming edges display a blue **⊕ Merge** badge in the header showing the input count. Hover over it for a tooltip: *"Merge: combining data from N upstream sources"*.

### Connection Validation

Model-to-model connections (e.g., training → training) are **blocked** with an alert. Training nodes accept inputs from preprocessing nodes only.

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| "Empty DataFrame from upstream branch" | A preprocessing branch produced no rows | Check filters/cleaning nodes upstream |
| "No common columns" | Branches have incompatible schemas | Ensure branches produce compatible columns |
| "training frame contains N non-numeric column(s)" | After Train-Test Split, the configured strategy retained a raw column from the winning input | Encode the column on the winning branch, drop it, or make branches disjoint / fully numeric |

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
