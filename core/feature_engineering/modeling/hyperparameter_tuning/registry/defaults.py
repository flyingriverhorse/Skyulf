"""Default strategy definitions and supported implementations."""

from typing import Dict, Sequence, Set

DEFAULT_TUNING_STRATEGIES: Sequence[Dict[str, object]] = (
    {
        "value": "random",
        "label": "Random search",
        "description": "Sample candidate hyperparameters uniformly at random.",
        "impl": "random",
        "aliases": ("random_search",),
    },
    {
        "value": "grid",
        "label": "Grid search",
        "description": "Evaluate every combination in the search space.",
        "impl": "grid",
        "aliases": ("grid_search",),
    },
    {
        "value": "halving",
        "label": "Successive halving (grid)",
        "description": "Successively allocate resources to the best grid candidates.",
        "impl": "halving",
        "aliases": ("successive_halving", "halving_grid"),
    },
    {
        "value": "halving_random",
        "label": "Successive halving (random)",
        "description": "Random sampling with successive halving to prune weak candidates.",
        "impl": "halving_random",
        "aliases": ("successive_halving_random", "halving_search"),
    },
    {
        "value": "optuna",
        "label": "Optuna (TPE)",
        "description": "Bayesian optimisation with pruning via Optuna.",
        "impl": "optuna",
        "aliases": ("bayesian", "optuna_tpe"),
    },
)

SUPPORTED_IMPLS: Set[str] = {"random", "grid", "halving", "halving_random", "optuna"}
