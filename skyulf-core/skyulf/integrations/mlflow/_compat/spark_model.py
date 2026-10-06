"""Compatibility alias for :mod:`skyulf.integrations.mlflow.spark.spark_model`."""

import sys

from skyulf.integrations.mlflow.spark import spark_model as _implementation

sys.modules[__name__] = _implementation
