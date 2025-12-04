"""Tuning configuration schemas."""

from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    strategy: Literal["grid", "random", "optuna", "halving_grid", "halving_random"] = "random"
    metric: str = "accuracy" # or 'mse', 'f1', etc.
    n_trials: int = 10
    timeout: Optional[int] = None # Seconds
    search_space: Dict[str, List[Any]] = field(default_factory=dict) # e.g. {"C": [0.1, 1.0, 10.0]}
    cv_folds: int = 5
    random_state: int = 42

@dataclass
class TuningResult:
    """Result of a tuning session."""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    trials: List[Dict[str, Any]] # List of {params, score}
