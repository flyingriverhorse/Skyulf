# Architecture

Skyulf is split into three pieces with strict boundaries:

1. **skyulf-core** (this docs set focuses on this)
   - A standalone Python ML library.
   - Implements a strict **Calculator → Applier** pattern for every node.
   - Depends on Pandas/Numpy/Scikit-Learn (and a few optional ML utilities).
2. **backend**
   - FastAPI + Celery orchestration layer.
   - Handles ingestion, jobs, persistence, and exposes REST APIs.
3. **frontend**
   - React + TypeScript UI (ML canvas).
   - Builds pipeline configs and talks to the backend.

## The Calculator → Applier Pattern

Skyulf-core separates learning from transformation:

- **Calculator**: `fit(data, config) -> params`
  - Learns statistics / encoders / models.
  - Returns a serializable `params` dictionary.
- **Applier**: `apply(data, params) -> transformed_data`
  - Stateless transformer.
  - Applies learned parameters.

This makes pipelines easier to persist and safer to run in production:

- Learning happens on train.
- The learned state is explicit.
- Applying is pure and repeatable.

## Pipeline Data Flow

At runtime, `SkyulfPipeline` orchestrates:

1. **Preprocessing**: `FeatureEngineer`
   - Executes a list of steps (each step is a transformer).
   - Some steps change the data structure (e.g., splitters) and are handled specially.
2. **Modeling**: `StatefulEstimator`
   - Trains a model on the train split.
   - Optionally evaluates on test/validation.

High-level flow:

```
Raw DataFrame
  └─ FeatureEngineer.fit_transform(...)  -> DataFrame or SplitDataset
        └─ (optionally) SplitDataset.train / test / validation
              └─ StatefulEstimator.fit_predict(...) -> predictions
```

## Avoiding Data Leakage

If you split first (or provide a `SplitDataset`), calculators should learn only on the train split.
This prevents leakage of statistics from test/validation.

See the User Guide section “SplitDataset & Leakage” for recommended patterns.
