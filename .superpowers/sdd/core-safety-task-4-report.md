# Task 4 Report: Bound and Report Clustering Silhouette Sampling

Status: reviewer follow-up implemented, pending review.
Base commit: da98e7a1

## Summary
- Added `DEFAULT_SILHOUETTE_SAMPLE_SIZE = 10_000` and
  `DEFAULT_SILHOUETTE_RANDOM_STATE = 42` in
  `skyulf-core/skyulf/modeling/_evaluation/metrics.py`.
- Extended `calculate_clustering_metrics()` with keyword-only
  `silhouette_sample_size` and `random_state` parameters without changing
  existing caller behavior.
- Replaced sklearn's internal silhouette subsampling with a deterministic
  Skyulf-managed representative sample that guarantees every predicted
  cluster is present before scoring.
- Added `silhouette_sample_size` to reported metrics as the actual number of
  rows supplied to silhouette scoring, while leaving Calinski-Harabasz and
  Davies-Bouldin on full input.
- Added sampled-cap validation that rejects impossible silhouette requests
  with a clear `ValueError` before sklearn can raise an opaque cardinality
  error.
- Documented the >10k deterministic representative silhouette sampling
  behavior in
  `docs/user_guide/segmentation.md`.

## Explicit cap policy
- `silhouette_sample_size < 2` raises
  `ValueError("silhouette_sample_size must be at least 2")` before any
  degenerate-label short-circuiting.
- Mismatched feature/label row counts raise
  `ValueError("X and labels must have the same number of rows")` before any
  metric-specific work.
- If silhouette is undefined because labels have fewer than 2 clusters or as
  many clusters as rows, no clustering-quality metrics are emitted.
- If total rows are `<= silhouette_sample_size`, silhouette scores all rows and
  reports `silhouette_sample_size == total_rows`.
- If total rows are `> silhouette_sample_size`, Skyulf builds one deterministic
  random permutation from `random_state`, takes the first occurrence of each
  cluster label from that order so every cluster is represented, then fills the
  remaining slots from the rest of the same permutation without replacement.
  Silhouette is computed on that bounded subset directly (no sklearn
  `sample_size` kwargs); reported
  `silhouette_sample_size == silhouette_sample_size`.
- If sampled scoring would need to represent `n_clusters` labels in a cap of
  `<= n_clusters` rows, Skyulf raises
  `ValueError("silhouette_sample_size=<cap> is too small for <n_clusters> clusters; increase it above the number of clusters when scoring datasets larger than the cap")`.
- Extremely imbalanced valid clusterings (including a single rare-label row in
  a `>10_000` input) still stay bounded and score deterministically because the
  rare cluster's representative row is forced into the capped sample.

## Files changed
- `skyulf-core/skyulf/modeling/_evaluation/metrics.py`
- `skyulf-core/tests/test_evaluation_clustering.py`
- `skyulf-core/tests/test_modeling_clustering.py`
- `docs/user_guide/segmentation.md`
- `.superpowers/sdd/core-safety-task-4-report.md`
- `.superpowers/sdd/progress.md`

## Validation
- `source .venv/bin/activate && pytest skyulf-core/tests/test_evaluation_clustering.py -q` -> `13 passed`
- `source .venv/bin/activate && pytest skyulf-core/tests/test_evaluation_clustering.py skyulf-core/tests/test_modeling_clustering.py -q && ruff check . && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py` -> `18 passed`, `All checks passed!`, `569 files already formatted`, `All checks passed!`

## Notes
- Regression coverage now captures deterministic bounded silhouette inputs, the
  small-input no-sampling path, invalid caps (including the new
  sampled-cardinality guard and the pre-existing `<2` guard), and the existing
  evaluator/raw-metrics path.
- Codacy CLI analysis was attempted after each edit, but the repository-local
  Codacy wrapper is malformed and fails before analysis starts.
