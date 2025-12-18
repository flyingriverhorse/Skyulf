# Overview

`skyulf-core` is a standalone ML pipeline library designed for reproducible feature engineering and modeling.

## Key idea: explicit learned state

Skyulf-core uses a strict **Calculator → Applier** pattern:

- A **Calculator** learns from data and returns a `params` dictionary.
- An **Applier** takes `params` and transforms data.

This differs from scikit-learn’s stateful estimators where learned values are stored inside the object instance.
In Skyulf-core, learned values are explicit and can be inspected or persisted.

## Two ways to use the library

### 1) Pipeline way (recommended)

Use `SkyulfPipeline` to run preprocessing + modeling end-to-end.

### 2) Component way (low-level)

Call calculators/appliers directly for debugging, testing, or custom scripts.

## Where things live

- `skyulf.preprocessing`: feature engineering nodes (imputation, encoding, scaling, …)
- `skyulf.modeling`: estimators (classification/regression + tuning)
- `skyulf.data`: `SplitDataset` for safe train/test/validation flows
