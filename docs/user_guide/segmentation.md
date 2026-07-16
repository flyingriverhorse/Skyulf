# Segmentation (Clustering)

Segmentation groups similar rows together (customer segments, behavior
clusters, anomalies) without needing a target column. Skyulf ships this as a
regular, trainable model that plugs into the same Calculator/Applier
pipeline as classification and regression, with four algorithms available:
`kmeans`, `minibatch_kmeans`, `gaussian_mixture`, and `birch`.

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
    print(c.cluster_id, c.size, c.percentage, c.center, c.profile)
    # c.profile is an auto-generated label, e.g. "High income, Low age"
```

That's it — no separate clustering API to learn. `fit_predict`/`evaluate`
work exactly like every other model; the only difference is the empty
`target_column`.

## Available algorithms

| `model_type`        | Class                          | Good for                                              |
|----------------------|---------------------------------|--------------------------------------------------------|
| `kmeans`             | `KMeansCalculator`/`KMeansApplier` | General-purpose, roughly spherical clusters. Default choice. |
| `minibatch_kmeans`   | `MiniBatchKMeansCalculator`/`MiniBatchKMeansApplier` | Same as K-Means but faster on large datasets (fits on random mini-batches). |
| `gaussian_mixture`   | `GaussianMixtureCalculator`/`GaussianMixtureApplier` | Elongated/overlapping clusters — models each cluster as its own Gaussian shape (`covariance_type`). |
| `birch`              | `BirchCalculator`/`BirchApplier`   | Very large datasets — builds a memory-efficient tree summary before clustering. |

All four support genuine out-of-sample `.predict()`, so a fitted model can
score new/held-out rows the same way a classifier or regressor does — that's
also why they're the only clustering algorithms wired in: density-based
algorithms (DBSCAN, Agglomerative, OPTICS) only implement `fit_predict()` on
the training data and can't be deployed for inference.

Swap the calculator/applier class to switch algorithms — the rest of the
code above is identical:

```python
from skyulf.modeling.clustering import GaussianMixtureCalculator, GaussianMixtureApplier

estimator = StatefulEstimator(GaussianMixtureCalculator(), GaussianMixtureApplier(), node_id="segments")
estimator.fit_predict(split_dataset, target_column="", config={"params": {"n_components": 4, "covariance_type": "full"}})
```

## Interpreting clusters

Cluster IDs (`0`, `1`, `2`, ...) are arbitrary — the model has no idea a
cluster corresponds to, say, "high-value customers" or "setosa". Two ways
to make sense of them, both computed automatically:

**1. Auto-generated profile.** Every centroid gets a `profile` string (e.g.
`"High petal_length, High petal_width"`) — the top features that most set
that cluster apart from the dataset average (a z-score comparison). No
extra config needed; it's just part of `centroid.profile`.

**2. Reference column (optional).** If you have a known label that you
*don't* want used as a training feature (e.g. species name in the Iris
dataset, or a customer segment you already know), pass it as
`reference_column` — it's excluded from the model entirely, but a
cluster → label breakdown is computed for you:

```python
estimator.fit_predict(
    split_dataset, target_column="", config={"params": {"n_clusters": 3}, "reference_column": "species"}
)
report = estimator.evaluate(split_dataset, target_column="", reference_column="species")
print(report["splits"]["train"].clustering.reference_crosstab)
# {"0": {"setosa": 46, "versicolor": 2}, "1": {"versicolor": 44}, "2": {"virginica": 50}}
```

This tells you e.g. "Cluster 0 is 92% setosa" — without the model ever
seeing the `species` column.

## Deployment & inference

When a Segmentation job finishes, the bundled artifact stores the exact list
of `feature_columns` the model was fit on — this is what the deployment
service and the manual-prediction UI use to know which columns to send at
predict time (instead of guessing from `model.feature_names_in_`, which
`SklearnBridge` doesn't populate since it fits on a bare numpy array).

If you set a `reference_column`, it's dropped from that `feature_columns`
list too, for the same reason it's dropped from training: the model never
saw it, so inference shouldn't require it either. Concretely, for the Iris
example above (`reference_column="species"`), the deployed model expects
only the 4 numeric measurement columns — **not** `species` — so a prediction
request just needs `sepal_length`, `sepal_width`, `petal_length`, and
`petal_width`. Sending `species` as well is harmless (it's ignored), but
omitting it is what makes the model usable on genuinely new/unlabeled rows.

This is the fix for a prediction-time mismatch (`"X has N features, but
KMeans is expecting N-1 features"`) that used to happen if a reference-style
column got included in the feature set at fit time but wasn't available (or
had a different name) when calling `predict()` later — excluding it from both
training *and* the persisted `feature_columns` keeps fit/predict consistent.

Note this `feature_columns`-exclusion mechanism is generic in the backend
runner (`exclude_columns` on `_bundle_model_with_transformers`) but in
practice only ever fires for clustering: it's populated from
`reference_column`, which only Segmentation nodes expose. Classification and
Regression models never pass anything here, so this has no effect on them.

## Via the pipeline / node graph

In the UI, drag a **Segmentation** node onto the canvas (found alongside
other Modeling nodes) instead of Basic Training. It has its own
model/hyperparameter picker (all four algorithms above), an optional
**Reference Column** selector, and no target-column/CV UI at all (neither
applies to unsupervised clustering). Results appear in a dedicated
**Segmentation** tab on the Experiments page: cluster-quality scores
(silhouette / Calinski-Harabasz / Davies-Bouldin), a cluster-size bar chart,
per-cluster centroid cards (with the auto-generated profile), and — if a
reference column was set — a breakdown of that column's values per cluster.

Same thing via a pipeline config:

```python
{
    "node_id": "segment_customers",
    "step_type": "basic_training",
    "params": {
        "target_column": "",          # sentinel for "no target"
        "model_type": "gaussian_mixture",
        "hyperparameters": {"n_components": 4, "covariance_type": "full"},
        "reference_column": "known_segment",  # optional
        "evaluate": True,
    },
}
```

## Notes

- **Only genuinely deployable algorithms are included.** K-Means,
  Mini-Batch K-Means, Gaussian Mixture, and Birch all support real
  out-of-sample `.predict()`. Density-based algorithms (DBSCAN,
  Agglomerative, OPTICS) don't support this and aren't wired in.
- **Metrics need real clusters.** Silhouette/Calinski-Harabasz/Davies-Bouldin
  are only computed when there's more than 1 cluster and fewer clusters than
  rows — otherwise they're omitted rather than raising.
- **Advanced Tuning doesn't apply.** Hyperparameter tuning scores candidates
  with a supervised metric (accuracy, R², etc.), which clustering has no
  equivalent of — the model dropdown there excludes all clustering models.
- **Reference column dtype doesn't matter.** Whether it's text (species
  name) or numeric (a segment code), it's excluded from training by name,
  not just by non-numeric filtering.
