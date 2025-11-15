# Shared Modeling Toolkit

This directory hosts helpers that both the training and hyperparameter-tuning flows rely on. Each module focuses on one concern so Celery tasks can stay thin.

| Module | Responsibility |
| --- | --- |
| `common.py` | Infrastructure primitives (Celery app, DB bootstrapping), cross-validation configuration, splitter factories, dataset/feature preparation utilities, and warning extraction.
| `inputs.py` | Loader for modeling jobs. Resolves datasets, pipeline graphs, problem-type hints, and exposes the `ModelingInputs` container used by Celery tasks.
| `evaluation.py` | Metric calculators for classification/regression splits. Keeps train/validation/test evaluations consistent across workflows.
| `results.py` | Serialization helpers for search results (flattening `cv_results_` and handling numpy/scalar conversions).
| `search.py` | Normalizes search strategies, search spaces, and CV policies for hyperparameter tuning. See `search.md` for deeper guidance.
| `artifacts.py` | Artifact persistence helpers shared by training and tuning, metadata builders, and transformer debug snapshot writer.
| `__init__.py` | Re-exports the public surface so tasks can import from `core.feature_engineering.modeling.shared` without worrying about module boundaries.

When introducing a new modeling workflow, prefer adding primitives here instead of duplicating logic inside task modules. The tests in `tests/test_shared_*.py` document expected behavior for the most critical helpers.
