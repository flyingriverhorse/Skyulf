# Hyperparameter Missing Value Configuration Analysis

## Context
The current implementation of Skyulf does not expose specific configurations for handling missing values in the UI, even though these parameters are available in the underlying libraries. This means the models are currently running with default library behaviors, which may not always be optimal for specific datasets.

## Current Model Behavior vs. Library Capabilities

| Model | Library Configuration | Skyulf UI Parameters | Observations |
| :--- | :--- | :--- | :--- |
| **LightGBM** | `use_missing` (default: True), `zero_as_missing` (default: False) | `n_estimators`, `num_leaves`, `learning_rate`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `boosting_type` | Neither `use_missing` nor `zero_as_missing` are currently visible or configurable. |
| **XGBoost** | `missing` (defines which value is treated as missing, default: NaN) | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` | The `missing` parameter is absent from the UI. |
| **HistGradientBoosting** | No specific parameter (NaN is always treated as missing) | `max_iter`, `learning_rate`, `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`, `max_bins` | No configurable missing value parameters exist in the library. |

## Analysis
Currently, all models operate on default library behaviors:
- **LightGBM:** Missing values are used; zeros are not treated as missing.
- **XGBoost:** NaN values are treated as missing.
- **HistGradientBoosting:** NaN values are always treated as missing.

While these are reasonable defaults, the inability to switch these behaviors limits flexibility for users who need custom handling (e.g., treating zero as a missing value in LightGBM).

## Proposed Improvements
1. **LightGBM:** Add two boolean form fields for `use_missing` and `zero_as_missing`. This is a low-effort, high-impact improvement that provides necessary control.
2. **XGBoost:** The `missing` parameter typically accepts a numerical value (NaN). Placing this in a numerical form field can be awkward in the UI. For now, we can skip adding it unless a cleaner UI solution is identified.
3. **HistGradientBoosting:** No action required as there are no configurable parameters.

## Next Steps
- Implement the `use_missing` and `zero_as_missing` boolean fields for the LightGBM model configuration.
