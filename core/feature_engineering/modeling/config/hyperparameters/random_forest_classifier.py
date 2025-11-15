"""Hyperparameter definitions for RandomForestClassifier models."""

from .base import HyperparameterField

RANDOM_FOREST_CLASSIFIER_PARAMS = [
    HyperparameterField(
        name="n_estimators",
        label="Number of Trees",
        type="number",
        default=100,
        description="Number of trees in the forest",
        min=10,
        max=1000,
        step=10,
    ),
    HyperparameterField(
        name="max_depth",
        label="Max Depth",
        type="number",
        default=None,
        description="Maximum depth of trees (empty = no limit)",
        min=1,
        max=100,
        step=1,
        nullable=True,
    ),
    HyperparameterField(
        name="min_samples_split",
        label="Min Samples Split",
        type="number",
        default=2,
        description="Minimum samples required to split a node",
        min=2,
        max=20,
        step=1,
    ),
    HyperparameterField(
        name="min_samples_leaf",
        label="Min Samples Leaf",
        type="number",
        default=1,
        description="Minimum samples required at a leaf node",
        min=1,
        max=20,
        step=1,
    ),
    HyperparameterField(
        name="max_features",
        label="Max Features",
        type="select",
        default="sqrt",
        description="Number of features to consider for best split",
        options=[
            {"value": "sqrt", "label": "Square Root"},
            {"value": "log2", "label": "Log2"},
            {"value": "None", "label": "All Features"},
        ],
    ),
    HyperparameterField(
        name="random_state",
        label="Random State",
        type="number",
        default=42,
        description="Seed for reproducibility",
        min=0,
        max=9999,
        step=1,
    ),
]
