"""K-Nearest Neighbors (Classifier and Regressor share the same param set)."""

from ._field import HyperparameterField

KNN_PARAMS = [
    HyperparameterField(
        name="n_neighbors",
        label="Number of Neighbors",
        type="number",
        default=5,
        min=1,
        max=50,
        step=1,
        description="Number of neighbors to use.",
    ),
    HyperparameterField(
        name="weights",
        label="Weights",
        type="select",
        default="uniform",
        options=[
            {"label": "Uniform", "value": "uniform"},
            {"label": "Distance", "value": "distance"},
        ],
        description="Weight function used in prediction.",
    ),
    HyperparameterField(
        name="algorithm",
        label="Algorithm",
        type="select",
        default="auto",
        options=[
            {"label": "Auto", "value": "auto"},
            {"label": "Ball Tree", "value": "ball_tree"},
            {"label": "KD Tree", "value": "kd_tree"},
            {"label": "Brute", "value": "brute"},
        ],
        description="Algorithm used to compute the nearest neighbors.",
    ),
]
