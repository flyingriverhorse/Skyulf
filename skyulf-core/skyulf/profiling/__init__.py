"""Exploratory data analysis, dataset expectations and drift detection.

This package is the public profiling surface: the EDA analyzer and its
terminal/matplotlib visualizer, the ``expect_*`` dataset assertions, and the
drift calculator with its report models. The Pydantic schemas those return are
re-exported here, so ``from skyulf.profiling import EDAAnalyzer`` is the
intended contract.

Use ``EDAProfile`` for the analyzer's result schema. ``DatasetProfile`` remains
an identical compatibility alias; the registered ``"DatasetProfile"`` pipeline
node is a separate inspection operation.
"""

from .analyzer import EDAAnalyzer
from .drift import ColumnDrift, DriftCalculator, DriftMetric, DriftReport
from .expect import (
    ExpectationError,
    expect_columns_exist,
    expect_no_nulls,
    expect_unique,
    expect_value_range,
)
from .schemas import Alert, ColumnProfile, DatasetProfile
from .schemas import DatasetProfile as EDAProfile
from .visualizer import EDAVisualizer

__all__ = [
    "Alert",
    "ColumnDrift",
    "ColumnProfile",
    "DatasetProfile",
    "DriftCalculator",
    "DriftMetric",
    "DriftReport",
    "EDAAnalyzer",
    "EDAProfile",
    "EDAVisualizer",
    "ExpectationError",
    "expect_columns_exist",
    "expect_no_nulls",
    "expect_unique",
    "expect_value_range",
]
