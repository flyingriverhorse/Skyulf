"""Hyperparameter-tuning package: the tuning engine and all its plumbing.

Exports a deliberately small public pair — :class:`TuningCalculator`, which
tunes a wrapped base-model calculator and refits it with the best parameters,
and :class:`TuningConfig`, the strategy/metric/search-space schema. The
machinery lives in the leaf modules (``engine`` orchestration, ``params``,
``splitters``, ``metrics``, the ``grid_random`` fold loop, ``refit``,
``fold_pipeline``, ``reporter``) and in the ``strategies/`` subpackage
(successive halving, Optuna, and the shared searcher runner).
"""

from .engine import TuningCalculator
from .schemas import TuningConfig

__all__ = ["TuningCalculator", "TuningConfig"]
