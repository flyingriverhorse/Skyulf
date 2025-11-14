# Modeling Workflow Notes

This document describes how the training node handles validation splits and cross-validation.

## Data Splits

- Upstream nodes (notably the Train/Test Split node) tag each record with a `split_type` column.
- `_prepare_training_data` trims the incoming DataFrame into train, validation, and test subsets based on that column.
- Each split is converted into numeric feature matrices and encoded target arrays.
- When no validation rows exist, the validation variables stay `None`, and downstream steps simply skip validation-specific work.

## Cross-Validation

- `_parse_cross_validation_config` reads the train-model node settings and produces a `CrossValidationConfig` dataclass.
- `_run_cross_validation` builds either `KFold` or `StratifiedKFold` depending on the problem type/strategy and loops over folds using only the **training** rows.
- Fold metrics are aggregated into mean/std summaries and attached to the job metrics as `cross_validation`.
- If CV is enabled but cannot run (for example, too few rows), the status in the payload is set to `skipped` with a reason.

## Final Model Fit

- `_train_and_save_model` trains the production model after cross-validation concludes.
- When `cv_refit_strategy` is `train_plus_validation` **and** the validation split has rows, the validation data is concatenated back into the training set before the final fit. Otherwise, only the original training rows are used.
- The boolean `validation_used_for_training` in the metrics indicates whether the validation rows participated in the final fit.

## Metrics and Metadata

- Train, validation (when present), and test metrics are computed with helpers `_classification_metrics` or `_regression_metrics`.
- The `metrics.cross_validation` block includes fold summaries, and the row counts expose how many records were available for each split.
- Job metadata mirrors the CV configuration so the UI can display strategy, folds, shuffle, random state, and refit behavior alongside the results.

## Frontend Integration

- The frontend surfaces these controls through `ModelTrainingSection`, which renders toggles and inputs for the CV parameters.
- Training payloads include the CV fields so the worker receives the exact configuration that users set in the UI.
