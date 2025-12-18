# Overview

`skyulf-core` is a standalone ML pipeline library designed for reproducible feature engineering and modeling.

## Key idea: explicit learned state

Skyulf-core uses a strict **Calculator → Applier** pattern:

- A **Calculator** learns from data and returns a `params` dictionary.
- An **Applier** takes `params` and transforms data.

This differs from scikit-learn’s default pattern:

- In scikit-learn, `fit()` **mutates** the estimator/transformer object (e.g. `self.mean_`, `self.categories_`), and `transform()` uses those hidden internal attributes.
- In Skyulf-core, `fit()` returns the learned state explicitly as a plain `params` dictionary (ideally JSON-serializable; sometimes pickled for complex sklearn objects), and `apply()` uses only that dictionary.

Practically, this makes learned state easier to inspect, persist, and apply consistently across train/test/inference.

## Two ways to use the library

### 1) Pipeline way (recommended)

Use `SkyulfPipeline` to run preprocessing + modeling end-to-end.

### 2) Component way (low-level)

Call calculators/appliers directly for debugging, testing, or custom scripts.

This is also where you’ll see `StatefulEstimator` used: it’s a small convenience wrapper that
keeps a fitted model artifact in memory and can run `fit_predict()` on a `SplitDataset`.
`SkyulfPipeline` uses the same underlying idea internally.

## How `fit` / `transform` works in Skyulf

At a high level:

- `SkyulfPipeline.fit(data, target_column=...)`
	- Runs preprocessing in order.
	- For each preprocessing step: Calculator learns params (typically from train only), then Applier applies those params to train/test/validation.
	- Trains the model and reports metrics.

- `SkyulfPipeline.predict(df)`
	- Applies the already-learned preprocessing params (no re-fitting).
	- Skips steps that only make sense during training (e.g., splitters / resampling).
	- Runs the trained model to produce predictions.

If you want reproducible “proof-style” checks (sklearn-style `X/y` split + leakage demonstration), see:

- Validation vs scikit-learn

## Why splitting matters (leakage)

Many preprocessing nodes *learn* from data (means, categories, bin edges, vocabularies…).
If you learn those from the full dataset and then evaluate on a test set, you leak information.

The safe pattern is:

- Split first (or provide a `SplitDataset`).
- Fit preprocessing on `train` only.
- Reuse the learned `params` to transform `test` / new inference data.

## Where things live

- `skyulf.preprocessing`: feature engineering nodes (imputation, encoding, scaling, …)
- `skyulf.modeling`: estimators (classification/regression + tuning)
- `skyulf.data`: `SplitDataset` for safe train/test/validation flows
