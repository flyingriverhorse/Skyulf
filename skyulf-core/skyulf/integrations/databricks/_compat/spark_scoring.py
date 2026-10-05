"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.batch.spark_scoring`."""

import sys

from skyulf.integrations.databricks.scoring.batch import spark_scoring as _implementation

sys.modules[__name__] = _implementation
