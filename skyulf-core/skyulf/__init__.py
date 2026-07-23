"""
Skyulf Core SDK
"""

from importlib.metadata import PackageNotFoundError, version

from .data.dataset import SplitDataset
from .leakage import validate_leakage_safety
from .pipeline import SkyulfPipeline
from .preprocessing.pipeline import FeatureEngineer
from .profiling import (
    DatasetProfile,
    DriftCalculator,
    EDAAnalyzer,
    EDAVisualizer,
    expect_columns_exist,
    expect_no_nulls,
    expect_unique,
    expect_value_range,
)
from .registry import NodeRegistry

try:
    # Single source of truth is setup.py's `version=`; setuptools records it in
    # the installed package's metadata, so we read it back here instead of
    # hardcoding a second copy that can drift out of sync.
    __version__ = version("skyulf-core")
except PackageNotFoundError:  # pragma: no cover - package not installed (e.g. raw checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "SkyulfPipeline",
    "SplitDataset",
    "FeatureEngineer",
    "NodeRegistry",
    "EDAAnalyzer",
    "EDAVisualizer",
    "DatasetProfile",
    "DriftCalculator",
    "expect_columns_exist",
    "expect_no_nulls",
    "expect_unique",
    "expect_value_range",
]
