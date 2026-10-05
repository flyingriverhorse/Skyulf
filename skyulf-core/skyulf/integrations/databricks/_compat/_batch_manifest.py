"""Compatibility alias for :mod:`skyulf.integrations.databricks.shared._batch_manifest`."""

import sys

from skyulf.integrations.databricks.shared import _batch_manifest as _implementation

sys.modules[__name__] = _implementation
