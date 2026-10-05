"""Compatibility alias for :mod:`skyulf.integrations.mlflow.spark._spark_output`."""

import sys

from skyulf.integrations.mlflow.spark import _spark_output as _implementation

sys.modules[__name__] = _implementation
