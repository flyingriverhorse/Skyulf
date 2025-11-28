# Feature Engineering Recommendation System

This document outlines the architecture and logic behind the automated recommendation system in the Skyulf feature engineering pipeline.

## Overview

The recommendation system analyzes the current state of the dataset (after applying any existing pipeline steps) and suggests appropriate transformations. It is designed to guide users towards better feature engineering practices while preventing common pitfalls like data leakage or incorrect handling of target variables.

## Core Principles

1.  **Context-Awareness**: Recommendations are generated based on the *current* state of the data, respecting the graph of operations applied so far.
2.  **Safety First**: Critical columns, especially the **Target Column**, are protected from accidental transformations.
3.  **Explainability**: Every recommendation includes a reason, a score (confidence), and a status to help the user understand *why* a transformation is suggested.

## Recommendation Categories

### 1. Drop Columns (`/drop-columns`)

Identifies columns that provide little to no value for modeling or pose a risk.

*   **Criteria**:
    *   **High Missingness**: Columns with > 95% missing values.
    *   **Zero Variance**: Columns with a single unique value (constants).
    *   **ID Columns**: Columns detected as sequential identifiers or high-cardinality unique strings (potential leakage/noise).
    *   **Target Exclusion**: The configured target column is *never* recommended for dropping.

### 2. Categorical Encoding

Suggests the best encoding strategy for text/categorical columns.

#### Label Encoding (`/label-encoding`)
*   **Logic**: Suitable for ordinal data or when using tree-based models that can handle integer-coded categories.
*   **Target Handling**:
    *   **Special Case**: If the target column is categorical, it is **strongly recommended** for Label Encoding.
    *   **Safety**: The system forces a warning: *"Target Column detected. You MUST select 'Replace Original' to use the encoded values for training."* This ensures the downstream training step finds the correct numeric target.

#### One-Hot Encoding (`/one-hot-encoding`)
*   **Logic**: Best for low-cardinality nominal data (e.g., < 20 categories).
*   **Exclusions**: High-cardinality columns are flagged as "Caution" or "Not Recommended" to prevent dimensionality explosion.
*   **Target Exclusion**: The target column is excluded to prevent splitting the target into multiple binary columns.

#### Other Encodings
*   **Ordinal, Hash, Target, Dummy**: Similar logic applies, with specific thresholds for cardinality. The target column is generally excluded from these to favor Label Encoding for classification targets.

### 3. Numerical Transformations

#### Scaling (`/scaling`)
*   **Logic**: Recommends scaling for numeric features with high variance or different scales.
*   **Methods**:
    *   **Standard**: Default for Gaussian-like data.
    *   **Robust**: Suggested if outliers are detected.
    *   **MinMax**: Suggested for bounded data.
*   **Target Exclusion**: The target column is excluded. Scaling the target (especially for regression) is usually handled by the model or a specific target transformer, not a general feature scaler.

#### Outlier Detection (`/outliers`)
*   **Logic**: Identifies columns with values far from the central tendency (using IQR or Z-Score).
*   **Target Exclusion**: The target column is excluded. Removing rows based on "outlier" target values can bias the model and remove valid edge cases.

#### Skewness (`/skewness`)
*   **Logic**: Detects highly skewed distributions and suggests Log or Box-Cox transformations.
*   **Target Exclusion**: The target column is excluded.

#### Binning (`/binning`)
*   **Logic**: Suggests binning for continuous variables that might benefit from discretization (e.g., Age, Income).
*   **Target Exclusion**: The target column is excluded.

## Architecture

### Data Flow
1.  **Request**: Frontend sends a request with `dataset_source_id`, `graph`, and `target_node_id`.
2.  **Graph Resolution**: The backend reconstructs the pipeline graph and identifies the **Target Column** via the `feature_target_split` node configuration.
3.  **Preview Generation**: The `FeatureEngineeringEDAService` generates a sample of the data *after* applying the graph operations.
4.  **Profiling**: The data sample is profiled (missingness, cardinality, distribution, etc.).
5.  **Filtering**:
    *   The **Target Column** is extracted from the graph context.
    *   Recommendation builders generate candidate lists.
    *   A final filter pass removes the target column from unsafe recommendations (Scaling, Outliers, etc.).
    *   For Label Encoding, the target is preserved and flagged with a specific warning.
6.  **Response**: The filtered and scored recommendations are sent back to the frontend.

## Manual Configuration vs. Recommendations

While the system provides powerful automated recommendations, **Manual Configuration** is fully supported and often necessary for domain experts.

### Why Manual Mode?
1.  **Domain Knowledge**: You might know that "Age > 120" is an error, even if the statistical outlier detector (IQR) says "Age > 85" is an outlier.
2.  **Specific Requirements**: A downstream model might strictly require Min-Max scaling (0-1 range), even if the recommender suggests Standard Scaling.
3.  **Fine-Tuning**: You may want to adjust the sensitivity of outlier detection (e.g., changing Z-Score threshold from 3.0 to 4.0).

### How to Use Manual Mode
All feature engineering nodes support a "Manual" or "Custom" configuration mode in the UI, which maps to specific backend parameters:

*   **Scaling**:
    *   Disable `auto_detect`.
    *   Explicitly select columns in the `columns` list.
    *   Set `default_method` (e.g., "minmax") or map specific methods to specific columns via `column_methods`.

*   **Outlier Removal**:
    *   **Manual Bounds**: Select the "Manual bounds" method to set explicit `lower_bound` and `upper_bound` values for a column.
    *   **Parameter Tuning**: Adjust parameters like `threshold` (Z-Score), `multiplier` (IQR), or `contamination` (Elliptic Envelope) to control aggressiveness.
    *   **Winsorization**: Choose to **Cap** values (Winsorize) instead of removing rows to preserve data quantity.

The backend is designed to prioritize explicit user configuration over automated defaults whenever provided.

## Scalability & Big Data

The recommendation system is designed to be **scalable** and responsive even for very large datasets (GBs or TBs).

### Sampling Strategy
Instead of processing the entire dataset (which would be slow and memory-intensive), the system operates on a **representative sample**.

1.  **Default Sample Size**: By default, recommendations are generated based on the first **10,000 rows**.
2.  **Efficient Loading**: For CSV files, the system uses optimized partial reading (e.g., `nrows` in pandas) to load only the required sample into memory, avoiding the overhead of reading the full file.
3.  **Statistical Validity**: For most feature engineering tasks (distribution analysis, cardinality checks, missingness), a sample of 10,000 rows provides a statistically significant proxy for the full dataset.

### Implications
*   **Performance**: Recommendations are generated in seconds, regardless of the total dataset size.
*   **Edge Cases**: Extremely rare categories (appearing < 1 in 10,000 rows) might be missed in the sample. This is generally acceptable as features that rare are often noise.
*   **Override**: The `sample_size` parameter in the API allows the frontend to request a larger sample (e.g., 50,000 rows) if higher precision is needed for a specific analysis.

## Future Improvements

Potential enhancements to the recommendation engine:

1.  **Correlation & Leakage**: Implement the "High Correlation" check in `DropColumnRecommendationBuilder` to flag features that are too highly correlated with the target (leakage) or with each other (collinearity).
2.  **Imbalance Detection**: For classification targets, analyze class balance and recommend resampling (SMOTE, undersampling) if classes are heavily skewed.
3.  **Interaction Features**: Analyze feature importance or correlation residuals to suggest polynomial features or interaction terms (`Feature A * Feature B`).
4.  **Time-Series Awareness**: If a datetime column is detected as the index or sorting key, suggest lag features or rolling window statistics.
5.  **Model-Specific Recommendations**: Allow the user to select their intended model type (e.g., "XGBoost" vs. "Linear Regression") and tailor recommendations accordingly (e.g., Linear Regression needs One-Hot/Scaling more than XGBoost).
