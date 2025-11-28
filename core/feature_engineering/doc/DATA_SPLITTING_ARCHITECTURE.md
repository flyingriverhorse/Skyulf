# Data Splitting and Feature/Target Separation Architecture

This document details the architecture used in the Skyulf feature engineering pipeline for handling dataset splits (Train/Test/Validation) and Feature/Target separation ($X$ and $y$).

## 1. Core Philosophy: "Tagging" vs. "Cutting"

In many simple ML scripts, datasets are physically split into four or more variables early in the process:
```python
# Traditional Script Approach (NOT used here)
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Now you have 4 separate dataframes to manage
```

**Skyulf uses a "Tagging" strategy.** We keep the data in a single DataFrame for as long as possible, using a special metadata column to designate which split a row belongs to.

### Benefits
*   **Guaranteed Row Alignment**: Since $X$ and $y$ stay in the same row, they can never become misaligned.
*   **Unified Preprocessing**: Operations like "Drop Missing Rows" automatically apply to both features and target simultaneously.
*   **Simplified Pipeline**: Nodes pass a single DataFrame object, reducing complexity.

---

## 2. The Workflow

### Step 1: Dataset Split Node (Logical Splitting)
**Location**: `core/feature_engineering/preprocessing/split/dataset_split.py`

When the **Train/Test Split** node is executed, it does **not** break the dataframe apart. Instead, it uses `sklearn.model_selection.train_test_split` to generate indices and applies a tag.

1.  **Input**: Raw DataFrame.
2.  **Action**:
    *   Uses `sklearn` to calculate split indices (supporting stratification if configured).
    *   Creates a reserved column: `__split_type__`.
    *   Populates this column with values: `"train"`, `"test"`, or `"validation"`.
3.  **Output**: The same DataFrame, now with the `__split_type__` column.

### Step 2: Feature/Target Split Node (Metadata Identification)
**Location**: `core/feature_engineering/preprocessing/split/feature_target_split.py`

This node identifies the roles of columns but, again, keeps the DataFrame intact.

1.  **Input**: DataFrame (potentially with `__split_type__`).
2.  **Action**:
    *   Validates the existence of the selected **Target Column** ($y$).
    *   Identifies all other columns as **Feature Columns** ($X$).
    *   Updates the pipeline signal/metadata to record these roles.
3.  **Output**: The DataFrame remains unchanged physically, but the system now knows which column is the target.

### Step 3: Preprocessing (Target-Blind Transformations)
**Location**: `core/feature_engineering/preprocessing/*`

Transformers (Scaling, Imputation, Encoding) operate on the DataFrame.

*   **Split Awareness**: Most transformers (like Scalers) are "Split Aware". They will:
    *   **Fit** only on rows where `__split_type__ == "train"`.
    *   **Transform** all rows (train, test, and validation).
    *   This prevents **Data Leakage** (e.g., calculating the mean for imputation using Test data).
*   **Target Safety**: Transformers are generally "Target Blind".
    *   **Warning**: If "Auto Detect" is enabled, a numeric target might be scaled or imputed.
    *   **Best Practice**: Explicitly exclude the target column in the node settings for operations like Scaling or Imputation to preserve the raw target values.

### Step 4: Model Training (Just-in-Time Separation)
**Location**: `core/feature_engineering/modeling/training/tasks.py`

The physical separation of $X$ and $y$ happens **only** at the moment of training, inside the asynchronous training task.

1.  **Input**: The fully preprocessed DataFrame.
2.  **Action**:
    *   **Row Filtering**:
        ```python
        train_df = df[df["__split_type__"] == "train"]
        test_df  = df[df["__split_type__"] == "test"]
        ```
    *   **Column Separation**:
        ```python
        y_train = train_df[target_column]
        X_train = train_df.drop([target_column, "__split_type__"], axis=1)
        ```
3.  **Result**: Clean, isolated numpy arrays/dataframes are passed to the model (`.fit(X_train, y_train)`).

---

## 3. Verification of Correctness

This architecture ensures correctness by design:

1.  **Standard Algorithms**: We use `sklearn.model_selection.train_test_split` for the randomization logic.
2.  **Leakage Prevention**: By fitting transformers only on the `"train"` tag, we ensure test data statistics do not influence the training process.
3.  **Integrity**: By keeping $X$ and $y$ together until the final step, we eliminate the risk of shuffling one without the other.

## 4. Summary Diagram

```mermaid
graph TD
    A[Raw Dataset] --> B[Dataset Split Node]
    B -- Adds '__split_type__' column --> C[Feature/Target Split Node]
    C -- Identifies 'target' column --> D[Preprocessing Nodes]
    D -- Fits on 'train', Transforms all --> E[Processed DataFrame]
    E --> F[Model Training Task]
    
    subgraph "Model Training Task (Internal)"
        F --> G{Filter Rows}
        G -- split_type='train' --> H[Train Rows]
        G -- split_type='test' --> I[Test Rows]
        
        H --> J{Separate Cols}
        J --> K[X_train]
        J --> L[y_train]
        
        I --> M{Separate Cols}
        M --> N[X_test]
        M --> O[y_test]
        
        K & L --> P[model.fit()]
    end
```
