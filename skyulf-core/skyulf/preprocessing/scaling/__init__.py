"""Scaling nodes package.

Split from a single 496-LOC module into per-scaler files:
  _common.py   — shared subset-selection helpers
  standard.py  — StandardScaler
  minmax.py    — MinMaxScaler
  robust.py    — RobustScaler
  maxabs.py    — MaxAbsScaler

All public names are re-exported here so existing imports such as
``from skyulf.preprocessing.scaling import StandardScalerCalculator``
continue to work unchanged.
"""

from .maxabs import MaxAbsScalerApplier, MaxAbsScalerCalculator
from .minmax import MinMaxScalerApplier, MinMaxScalerCalculator
from .robust import RobustScalerApplier, RobustScalerCalculator
from .standard import StandardScalerApplier, StandardScalerCalculator

__all__ = [
    "StandardScalerApplier",
    "StandardScalerCalculator",
    "MinMaxScalerApplier",
    "MinMaxScalerCalculator",
    "RobustScalerApplier",
    "RobustScalerCalculator",
    "MaxAbsScalerApplier",
    "MaxAbsScalerCalculator",
]
