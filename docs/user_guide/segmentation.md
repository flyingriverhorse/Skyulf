# Segmentation (Clustering)

Segmentation groups similar rows together (customer segments, behavior
clusters, anomalies) without needing a target column. Skyulf ships this as a
regular, trainable model — `kmeans` — that plugs into the same
Calculator/Applier pipeline as classification and regression.

## Quick start

```python
from skyulf.modeling.clustering import KMeansCalculator, KMeansApplier
from skyulf.modeling.base import StatefulEstimator

estimator = StatefulEstimator(KMeansCalculator(), KMeansApplier(), node_id="segments")

# No target column needed — pass "" (or omit it).
estimator.fit_predict(split_dataset, target_column="", config={"params": {"n_clusters": 4}})

# Evaluate cluster quality + get cluster sizes/centroids
report = estimator.evaluate(split_dataset, target_column="", job_id="job1")
train_eval = report["splits"]["train"]

print(train_eval.metrics)              # {"silhouette_score": 0.61, "n_clusters": 4, ...}
print(train_eval.clustering.n_clusters)      # 4
print(train_eval.clustering.cluster_sizes)   # {"0": 120, "1": 95, ...}
for c in train_eval.clustering.centroids:
    print(c.cluster_id, c.size, c.percentage, c.center)  # mean feature values per cluster
```

That's it — no separate clustering API to learn. `fit_predict`/`evaluate`
work exactly like every other model; the only difference is the empty
`target_column`.

## Via the pipeline / node graph

In the UI, drag a **Segmentation** node onto the canvas (found alongside
other Modeling nodes) instead of Basic Training. It reuses the same
model/hyperparameter picker, minus the target-column selector and
Cross-Validation section (neither applies to unsupervised clustering).
Results appear in a dedicated **Segmentation** tab on the Experiments page:
cluster-quality scores (silhouette / Calinski-Harabasz / Davies-Bouldin),
a cluster-size bar chart, and per-cluster centroid cards.

Same thing via a pipeline config:

```python
{
    "node_id": "segment_customers",
    "step_type": "basic_training",
    "params": {
        "target_column": "",          # sentinel for "no target"
        "model_type": "kmeans",
        "hyperparameters": {"n_clusters": 4},
        "evaluate": True,
    },
}
```

## Notes

- **Only K-Means today.** It's the only clustering algorithm with genuine
  out-of-sample `.predict()`, so a fitted model can score new/held-out rows
  the same way a classifier or regressor does. Density-based algorithms
  (DBSCAN, Agglomerative) don't support this and aren't wired in yet.
- **Metrics need real clusters.** Silhouette/Calinski-Harabasz/Davies-Bouldin
  are only computed when there's more than 1 cluster and fewer clusters than
  rows — otherwise they're omitted rather than raising.
- **Advanced Tuning doesn't apply.** Hyperparameter tuning scores candidates
  with a supervised metric (accuracy, R², etc.), which clustering has no
  equivalent of — the model dropdown there excludes `kmeans`.
