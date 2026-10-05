"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.batch.local_batch`."""

import sys

from skyulf.integrations.databricks.scoring.batch import local_batch as _implementation

sys.modules[__name__] = _implementation
