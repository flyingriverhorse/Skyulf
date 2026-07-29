# Task 4 Report: Bound and Report Clustering Silhouette Sampling

Status: implemented, pending acceptance.
Base commit: da98e7a1

## Summary
- Added `DEFAULT_SILHOUETTE_SAMPLE_SIZE = 10_000` and
  `DEFAULT_SILHOUETTE_RANDOM_STATE = 42` in
  `skyulf-core/skyulf/modeling/_evaluation/metrics.py`.
- Extended `calculate_clustering_metrics()` with keyword-only
  `silhouette_sample_size` and `random_state` parameters without changing
  existing caller behavior.
- Added `silhouette_sample_size` to reported metrics whenever silhouette is
  defined, while leaving Calinski-Harabasz and Davies-Bouldin on full input.
- Documented the >10k deterministic silhouette sampling behavior in
  `docs/user_guide/segmentation.md`.

## Explicit cap policy
- `silhouette_sample_size < 2` raises
  `ValueError("silhouette_sample_size must be at least 2")` before any
  degenerate-label short-circuiting.
- If silhouette is undefined because labels have fewer than 2 clusters or as
  many clusters as rows, no clustering-quality metrics are emitted.
- If total rows are `<= silhouette_sample_size`, silhouette scores all rows and
  does not pass sklearn sampling kwargs; reported
  `silhouette_sample_size == total_rows`.
- If total rows are `> silhouette_sample_size`, silhouette passes exactly
  `sample_size=silhouette_sample_size` and `random_state=random_state`; reported
  `silhouette_sample_size == silhouette_sample_size`.
- Highly imbalanced sampled subsets continue to surface sklearn's own
  silhouette error rather than falling back to a misleading reported sample
  size.

## Files changed
- `skyulf-core/skyulf/modeling/_evaluation/metrics.py`
- `skyulf-core/tests/test_evaluation_clustering.py`
- `skyulf-core/tests/test_modeling_clustering.py`
- `docs/user_guide/segmentation.md`
- `.superpowers/sdd/core-safety-task-4-report.md`
- `.superpowers/sdd/progress.md`

## Validation
- `source .venv/bin/activate && pytest skyulf-core/tests/test_evaluation_clustering.py -q` -> `11 passed`
- `source .venv/bin/activate && pytest skyulf-core/tests/test_evaluation_clustering.py skyulf-core/tests/test_modeling_clustering.py skyulf-core/tests/test_modeling_base.py -q` -> `65 passed`
- `source .venv/bin/activate && ruff check .` -> `All checks passed!`
- `source .venv/bin/activate && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py` -> `569 files already formatted`
- `source .venv/bin/activate && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py` -> `All checks passed!`

## Notes
- Regression coverage now captures custom sampled-silhouette kwargs, the
  small-input no-sampling path, invalid caps (including before the degenerate
  guard), and the existing evaluator/raw-metrics path.
- Codacy CLI analysis was attempted after each edit, but the repository-local
  Codacy wrapper is malformed and fails before analysis starts.
