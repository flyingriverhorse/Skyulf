"""Compatibility alias for :mod:`skyulf.integrations.mlflow.spark._spark_environment`."""

import sys

from skyulf.integrations.mlflow.spark import _spark_environment as _implementation

sys.modules[__name__] = _implementation
