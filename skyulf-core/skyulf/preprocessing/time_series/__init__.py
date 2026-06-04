"""Time-series preprocessing nodes.

Each module registers its node via ``@NodeRegistry.register`` at import time;
importing them here wires the nodes into the registry and re-exports the
public Calculator/Applier pairs.

  lag.py            — LagFeatures (shift columns by N rows)
  rolling.py        — RollingAggregate (rolling mean/sum/min/max/std/median)
  date_features.py  — DateFeatures (calendar parts from datetime columns)
"""

from .date_features import DateFeaturesApplier, DateFeaturesCalculator
from .lag import LagFeaturesApplier, LagFeaturesCalculator
from .rolling import RollingAggregateApplier, RollingAggregateCalculator

__all__ = [
    "LagFeaturesApplier",
    "LagFeaturesCalculator",
    "RollingAggregateApplier",
    "RollingAggregateCalculator",
    "DateFeaturesApplier",
    "DateFeaturesCalculator",
]
