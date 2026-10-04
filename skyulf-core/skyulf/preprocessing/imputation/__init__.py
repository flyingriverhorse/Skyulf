"""Imputation nodes package.

Split from a single 416-LOC module into per-imputer files:
  _common.py    — shared helpers (column resolution, polars fill values, sklearn transform)
  simple.py     — SimpleImputer
  group.py      — GroupImputer (per-group statistic with global fallback)
  knn.py        — KNNImputer
  iterative.py  — IterativeImputer (MICE)

All public names are re-exported here so existing imports such as
``from skyulf.preprocessing.imputation import SimpleImputerCalculator``
continue to work unchanged.
"""

from .group import GroupImputerApplier, GroupImputerCalculator
from .iterative import IterativeImputerApplier, IterativeImputerCalculator
from .knn import KNNImputerApplier, KNNImputerCalculator
from .simple import SimpleImputerApplier, SimpleImputerCalculator

__all__ = [
    "GroupImputerApplier",
    "GroupImputerCalculator",
    "IterativeImputerApplier",
    "IterativeImputerCalculator",
    "KNNImputerApplier",
    "KNNImputerCalculator",
    "SimpleImputerApplier",
    "SimpleImputerCalculator",
]
