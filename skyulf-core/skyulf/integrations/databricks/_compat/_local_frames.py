"""Compatibility alias for :mod:`skyulf.integrations.databricks.shared._local_frames`."""

import sys

from skyulf.integrations.databricks.shared import _local_frames as _implementation

sys.modules[__name__] = _implementation
