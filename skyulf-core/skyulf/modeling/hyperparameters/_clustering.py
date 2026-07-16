"""Clustering (unsupervised) hyperparameters."""

from ._field import HyperparameterField

KMEANS_PARAMS = [
    HyperparameterField(
        name="n_clusters",
        label="Number of Clusters",
        type="number",
        default=3,
        min=2,
        max=50,
        step=1,
        description="Number of segments to partition the data into.",
    ),
    HyperparameterField(
        name="n_init",
        label="Number of Initializations",
        type="number",
        default=10,
        min=1,
        max=50,
        step=1,
        description="Number of times K-Means runs with different centroid seeds; the best result is kept.",
    ),
    HyperparameterField(
        name="random_state",
        label="Random State",
        type="number",
        default=42,
        min=0,
        max=10000,
        step=1,
        description="Seed for centroid initialization, for reproducible results.",
    ),
]
