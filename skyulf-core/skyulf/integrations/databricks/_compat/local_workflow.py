"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle.local_workflow`."""

import sys

from skyulf.integrations.databricks.lifecycle import local_workflow as _implementation

sys.modules[__name__] = _implementation
