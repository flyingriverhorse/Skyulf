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

MINIBATCH_KMEANS_PARAMS = [
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
        name="batch_size",
        label="Batch Size",
        type="number",
        default=1024,
        min=32,
        max=10000,
        step=32,
        description="Number of rows per random mini-batch used to update centroids.",
    ),
    HyperparameterField(
        name="n_init",
        label="Number of Initializations",
        type="number",
        default=10,
        min=1,
        max=50,
        step=1,
        description="Number of times the algorithm runs with different centroid seeds; the best result is kept.",
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

GAUSSIAN_MIXTURE_PARAMS = [
    HyperparameterField(
        name="n_components",
        label="Number of Clusters",
        type="number",
        default=3,
        min=2,
        max=50,
        step=1,
        description="Number of Gaussian components (segments) to fit.",
    ),
    HyperparameterField(
        name="covariance_type",
        label="Covariance Type",
        type="select",
        default="full",
        options=[
            {"label": "Full (each cluster has its own shape)", "value": "full"},
            {"label": "Tied (all clusters share one shape)", "value": "tied"},
            {"label": "Diagonal (axis-aligned clusters)", "value": "diag"},
            {"label": "Spherical (round clusters)", "value": "spherical"},
        ],
        description="Shape constraint for each cluster's Gaussian distribution.",
    ),
    HyperparameterField(
        name="random_state",
        label="Random State",
        type="number",
        default=42,
        min=0,
        max=10000,
        step=1,
        description="Seed for initialization, for reproducible results.",
    ),
]

BIRCH_PARAMS = [
    HyperparameterField(
        name="n_clusters",
        label="Number of Clusters",
        type="number",
        default=3,
        min=2,
        max=50,
        step=1,
        description="Number of segments to reduce the tree summary into (final clustering step).",
    ),
    HyperparameterField(
        name="threshold",
        label="Threshold",
        type="number",
        default=0.5,
        min=0.01,
        max=5.0,
        step=0.01,
        description="Max radius of a sub-cluster before it's split — lower values create more, tighter sub-clusters.",
    ),
    HyperparameterField(
        name="branching_factor",
        label="Branching Factor",
        type="number",
        default=50,
        min=2,
        max=200,
        step=1,
        description="Max number of sub-clusters allowed per tree node before it splits.",
    ),
]
