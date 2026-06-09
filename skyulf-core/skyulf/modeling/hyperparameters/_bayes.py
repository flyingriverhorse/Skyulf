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

MULTINOMIAL_NB_PARAMS = [
    HyperparameterField(
        name="alpha",
        label="Alpha (Laplace smoothing)",
        type="number",
        default=1.0,
        min=0.0,
        max=10.0,
        description="Additive (Laplace/Lidstone) smoothing parameter. 0 = no smoothing.",
    ),
    HyperparameterField(
        name="fit_prior",
        label="Fit Class Priors",
        type="boolean",
        default=True,
        description="Whether to learn class prior probabilities. If False, uniform priors are used.",
    ),
]

BERNOULLI_NB_PARAMS = [
    HyperparameterField(
        name="alpha",
        label="Alpha (Laplace smoothing)",
        type="number",
        default=1.0,
        min=0.0,
        max=10.0,
        description="Additive (Laplace/Lidstone) smoothing parameter.",
    ),
    HyperparameterField(
        name="binarize",
        label="Binarize Threshold",
        type="number",
        default=0.0,
        min=0.0,
        max=1.0,
        description=(
            "Threshold for binarizing features. Features above this value are set to 1, "
            "others to 0. Set to None to skip binarization (assume features are already binary)."
        ),
    ),
    HyperparameterField(
        name="fit_prior",
        label="Fit Class Priors",
        type="boolean",
        default=True,
        description="Whether to learn class prior probabilities.",
    ),
]
