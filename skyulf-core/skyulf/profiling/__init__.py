"""Exploratory data analysis, dataset expectations and drift detection.

This package is the public profiling surface: the EDA analyzer and its
terminal/matplotlib visualizer, the ``expect_*`` dataset assertions, and the
drift calculator with its report models. The Pydantic schemas those return are
re-exported here, so ``from skyulf.profiling import EDAAnalyzer`` is the
intended contract.
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
    "EDAVisualizer",
    "ExpectationError",
    "expect_columns_exist",
    "expect_no_nulls",
    "expect_unique",
    "expect_value_range",
]
