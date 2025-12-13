# ML Pipeline: Metrics & Artifacts Architecture

This document explains how the Skyulf ML Pipeline handles execution state, artifacts, and performance metrics.

## 1. Architecture Overview

The pipeline execution involves three main layers:

1.  **Pipeline Engine (`engine.py`)**: The high-level orchestrator. It receives a `PipelineConfig`, runs each node, and reports status.
2.  **Feature Engineer (`pipeline.py`)**: The specialized processor for data transformation steps. It runs a sequence of transformers on a DataFrame.
3.  **Transformers (`base.py`, `cleaning.py`, etc.)**: Individual units of logic (e.g., `Deduplicate`, `StandardScaler`).

## 2. Artifacts (State Persistence)

Artifacts are files saved to disk that preserve the **state** of a transformer after it has "learned" from the data.

-   **Format**: Python objects serialized using `joblib`.
-   **Location**: 
    -   **Preview Mode**: A temporary directory (e.g., `/tmp/skyulf_preview_xyz/`). Deleted after the request.
    -   **Training Mode**: A persistent directory associated with the model version.
-   **Content**: The `params` dictionary returned by the `Calculator.fit()` method.

### Examples of Artifact Content

| Node Type | What is Saved in Artifact? | Example Content |
| :--- | :--- | :--- |
| **Feature Selection** | The list of columns to *keep*. | `{'selected_columns': ['age', 'income'], 'threshold': 0.5}` |
| **Imputer** | The mean/median values for each column. | `{'statistics': {'age': 35.5, 'income': 50000}}` |
| **Deduplicate** | The configuration used. | `{'subset': ['id'], 'keep': 'first'}` |
| **Drop Rows** | The configuration used. | `{'missing_threshold': 50}` |

**Key Insight**: Artifacts store *what the model learned* (parameters), not necessarily *what happened to the data* (metrics).

## 3. Metrics (Execution Feedback)

Metrics describe the *result* of an operation (e.g., "5 rows removed", "3 columns dropped"). There are two ways these are generated:

### A. Dynamic Calculation (In-Memory)
**Location**: `core/ml_pipeline/preprocessing/pipeline.py`

This is the preferred method for "delta" metrics. The `FeatureEngineer` compares the DataFrame **before** and **after** a step runs.

```python
# Pseudo-code from pipeline.py
rows_before = len(df)
df = transformer.transform(df)
rows_after = len(df)

metrics["rows_removed"] = rows_before - rows_after
```

**Why use this?**
1.  **Universal**: Works for any node that changes data shape.
2.  **Fast**: No disk I/O required.
3.  **Necessary for some nodes**: `Deduplicate` and `DropRows` do *not* know how many rows they will drop until they actually run. They don't save this count in their artifact.

### B. Artifact Inspection (Post-Process)
**Location**: `core/ml_pipeline/execution/engine.py`

The engine can load the saved artifact to extract details that are intrinsic to the algorithm.

```python
# Pseudo-code from engine.py
params = artifact_store.load(node_id)
if "selected_columns" in params:
    metrics["selected_columns"] = params["selected_columns"]
```

**Why use this?**
1.  **Detailed Info**: Good for retrieving complex learned state (e.g., "which specific features were selected?").

## 4. The "Feature Selection" Case Study

You might wonder: *If Feature Selection saves `selected_columns` in its artifact, why do we also calculate `dropped_columns` in `pipeline.py`?*

```python
# In pipeline.py
if transformer_type == "feature_selection":
    dropped_cols = cols_before - cols_after
    metrics["dropped_columns"] = list(dropped_cols)
```

**Reasons for Redundancy:**
1.  **Standardization**: It ensures `FeatureEngineer` returns a complete `metrics` dictionary for all steps, regardless of type. The caller doesn't need to know *how* to fetch the data.
2.  **Performance**: Calculating the set difference in memory is faster than loading the `joblib` file from disk.
3.  **Completeness**: The artifact stores what was *kept* (`selected_columns`). The frontend often wants to know what was *dropped*. While one can be derived from the other, calculating it explicitly in the pipeline is cleaner.

## 5. Data Flow Summary

1.  **Frontend** sends graph.
2.  **Engine** starts `FeatureEngineer`.
3.  **FeatureEngineer**:
    -   Runs `FeatureSelection`.
    -   Saves `selected_columns` to disk (Artifact).
    -   Calculates `dropped_columns` in memory (Dynamic Metric).
    -   Returns `(data, metrics)`.
4.  **Engine**:
    -   Receives `metrics` (containing `dropped_columns`).
    -   *Optionally* loads artifact to get `selected_columns`.
    -   Merges them.
5.  **Frontend** receives combined metrics and displays them.
