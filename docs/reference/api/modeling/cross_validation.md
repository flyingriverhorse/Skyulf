# API: modeling.cross_validation

Cross-validation module providing five strategies: K-Fold, Stratified K-Fold, Shuffle Split, Time Series Split, and Nested CV.

The main entry point is `perform_cross_validation()`, which dispatches to the appropriate sklearn splitter (or the custom nested CV loop). For usage details and examples, see the [Cross-Validation Guide](../../../user_guide/cross_validation.md).

**Key functions:**

- `perform_cross_validation()` — Run CV with any of the five methods.
- `_sort_by_time()` — Auto-sort data chronologically for Time Series Split.
- `_build_splitter()` — Build a sklearn splitter from a `cv_type` string.
- `_perform_nested_cv()` — Dual-loop nested CV (outer evaluation + inner HP stability).
- `_aggregate_metrics()` — Compute mean/std/min/max across folds.

::: skyulf.modeling.cross_validation
