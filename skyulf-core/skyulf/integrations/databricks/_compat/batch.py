"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.batch.batch`."""

import sys

from skyulf.integrations.databricks.scoring.batch import batch as _implementation

sys.modules[__name__] = _implementation
