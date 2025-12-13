# Data Splitting Strategies in Skyulf Pipeline

This document explains the two supported workflows for splitting data within the Skyulf ML Pipeline and their implications for **Data Leakage**.

## Overview

In machine learning, there are two common sequences for preparing data for modeling:
1.  **Separate X/y first**, then split into Train/Test sets.
2.  **Split rows into Train/Test first**, then separate X/y.

Both methods are mathematically valid, but they have different implications for data safety and pipeline architecture. Skyulf supports both.

---

## Workflow 1: The "Scripting" Standard (X/y First)

**Sequence:** `Feature-Target Split` $\rightarrow$ `Train-Test Split`

This is the standard approach seen in most Scikit-Learn tutorials and scripts.

1.  **Feature-Target Split:** You separate the entire dataset into a Features dataframe ($X$) and a Target series ($y$).
2.  **Train-Test Split:** You pass both $X$ and $y$ into the splitter. The splitter shuffles them in unison and produces four outputs: `X_train`, `X_test`, `y_train`, `y_test`.

### Pros
*   **Familiarity:** Matches standard Python/Pandas/Scikit-Learn coding patterns.
*   **Simplicity:** You only have to select the target column once.

### Cons (Risk of Data Leakage)
*   **Leakage Risk:** If you perform feature engineering (like scaling, imputation, or encoding) *before* this split, you are calculating statistics (mean, variance, etc.) on the **entire dataset**. This "leaks" information from the Test set into the Training set, leading to overly optimistic performance estimates.
*   **Mitigation:** You must ensure all feature engineering happens *after* the split, which can be awkward if your pipeline structure expects a single dataframe input.

---

## Workflow 2: The "Pipeline" Standard (Row Split First)

**Sequence:** `Train-Test Split` $\rightarrow$ `Feature-Target Split`

This is the preferred approach for robust production pipelines and ETL processes.

1.  **Train-Test Split:** You split the raw rows of your dataset into a "Training Table" and a "Testing Table". All columns (features + target) remain together.
2.  **Feature Engineering:** You apply transformations. The pipeline automatically "fits" on the Training Table and "applies" to the Testing Table.
3.  **Feature-Target Split:** Right before modeling, you separate $X$ and $y$ for each table.

### Pros
*   **Leakage Prevention:** By splitting rows first, you physically isolate the Test data. It is impossible for a scaler or imputer to "see" the test rows because they exist in a separate container (`SplitDataset`).
*   **Robustness:** Ensures that all transformations are learned strictly from training data.

### Cons
*   **Complexity:** Requires the pipeline engine to handle complex data structures (like `SplitDataset`) flowing between nodes, rather than simple DataFrames.

---

## How Skyulf Handles This

The Skyulf engine (`core/ml_pipeline`) is designed to support **both** workflows seamlessly.

*   **Smart Splitters:** The `TrainTestSplitter` and `FeatureTargetSplitter` nodes detect the shape of the incoming data.
    *   If they receive a Table, they split it.
    *   If they receive a Tuple $(X, y)$, they split both.
    *   If they receive a `SplitDataset` (Train/Test tables), they apply the operation to each table individually.

### Recommendation

For the safest and most robust ML pipelines, we recommend **Workflow 2**:
1.  **Dataset Node**
2.  **Train-Test Split Node** (Isolate your test data early)
3.  **Preprocessing Nodes** (Impute, Scale, Encode - safe from leakage)
4.  **Feature-Target Split Node** (Prepare for modeling)
5.  **Model Training Node**
