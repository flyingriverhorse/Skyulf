"""Naive Bayes variants."""

from ._field import HyperparameterField

GAUSSIAN_NB_PARAMS = [
    HyperparameterField(
        name="var_smoothing",
        label="Var Smoothing",
        type="number",
        default=1e-9,
        min=1e-12,
        max=1.0,
        description=(
            "Portion of the largest variance of all features that is added "
            "to variances for calculation stability."
        ),
    ),
]
